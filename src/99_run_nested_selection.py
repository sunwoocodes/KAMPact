"""
Nested 선택: val 안에서 (분위수 q, 연속 알람 k) 를 고르고 test 에는 선택된 조합만 적용

배경
    지금까지 q, k 는 test 결과를 보면서 골랐기 때문에 성능이 낙관적이다.
    이 스크립트는 fold 마다 다음 절차를 따른다.

      1) train fold 의 정상 윈도우로 Mahalanobis 학습  (6번과 동일)
      2) val fold 의 점수만으로 후보 (q, k) 들을 평가하고 하나를 선택
             - q : val 정상 윈도우 점수의 q-분위수를 threshold 로 사용
             - k : 연속 k 개 윈도우가 threshold 이상일 때 알람
      3) 선택된 (q, k) 를 test fold 에 적용  -> 이 값이 "정직한" 성능

    같은 fold/시드/모델로 모든 고정 조합의 test 성능도 함께 계산해서
    "nested 선택" 과 "test 를 보고 고른 최선" 의 차이(낙관성)를 볼 수 있다.
    선택 과정은 test 결과를 전혀 사용하지 않는다.

선택 규칙 (--select-rule)
    lexi    (기본) val 이벤트 탐지율이 가장 높은 조합 중 val 오탐 사이클 비율이 가장 낮은 것
    utility val 탐지율 - lambda * val 오탐 사이클 비율 이 가장 큰 것 (--lambda-fa)
    동률이면 k 가 작은 것, 그다음 q 가 작은 것(더 민감한 것)

주의
    - val 오탐은 threshold 를 정한 같은 val 정상 데이터에서 측정하므로 다소 낙관적이다.
      (q 가 클수록 val 오탐이 줄어드는 단조 관계라서 선택 규칙은 그대로 동작한다)
    - val fold 의 fault 이벤트는 3~4 개뿐이라 선택 자체가 흔들릴 수 있다.
      어떤 조합이 얼마나 자주 선택됐는지 함께 출력한다.

사용 예 (PowerShell 에서는 한 줄로):
    python ./src/6_run_nested_selection.py --multiscale result/modeling_dataset_1.0_0.1 result/modeling_dataset_0.5_0.1
    python ./src/6_run_nested_selection.py --inputs result/modeling_dataset_1.0_0.1 --select-rule utility --lambda-fa 5
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ALARM_COL = "alarm_k_consecutive"


# ============================================================
# 모듈 로드
# ============================================================

def load_module(path: Path, name: str):

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)

    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


# ============================================================
# 후보 조합
# ============================================================

def cand_name(cand: tuple) -> str:

    kind, q, k = cand

    if kind == "q":
        return f"q={q:g},k={k}"

    return f"f1,k={k}"


def build_candidates(args) -> tuple[list, list]:
    """(선택 대상 q,k 조합, 참고용 f1 조합)"""

    selectable = [
        ("q", q, k)
        for q in args.q_grid
        for k in args.k_grid
    ]

    reference = [
        ("f1", None, k)
        for k in sorted(set(args.k_grid) & {1, 2})
    ]

    return selectable, reference


def rank_key(rule: str, detect: float, fa: float, k: int, q: float, lam: float):
    """작을수록 좋은 키"""

    if rule == "utility":
        return (-(detect - lam * fa), k, q)

    return (-detect, fa, k, q)


# ============================================================
# fold 프레임 (val 을 test 로 복제해서 val 점수도 얻는다)
# ============================================================

def make_frame(mod6, sc, fold_map, fold, n_splits):

    work = mod6.make_fold_frame(sc, fold_map, fold, n_splits)

    work["is_val_copy"] = False

    val_copy = work[work["split"] == "val"].copy()

    val_copy["group_id"] = val_copy["group_id"].astype(str) + "@val"
    val_copy["split"] = "test"
    val_copy["is_val_copy"] = True

    return pd.concat([work, val_copy], ignore_index=True)


def apply_alarm(mod5, df, threshold, k):

    alarm = mod5.apply_k_consecutive(df, threshold, k)

    return alarm.reindex(df.index).to_numpy(dtype=bool)


# ============================================================
# fold x scale 1회 평가
# ============================================================

def evaluate_fold_scale(
    mod5,
    mod6,
    sc,
    work,
    args,
    selectable,
    reference,
    multi,
):

    _, test_work, _ = mod5.run_once(
        df=work,
        feature_cols=args.feature_cols,
        covariance=args.covariance,
        threshold_mode="normal_quantile",
        normal_quantile=0.995,
        k_consecutive=1,
        train_seed=None,
        train_fraction=1.0,
    )

    flag = test_work["is_val_copy"].astype(bool)

    real = test_work[~flag]
    val = test_work[flag]

    val_scores = val["score"].to_numpy(dtype=float)

    # ---- threshold (val 점수만 사용) ----
    thr_q = {
        q: mod5.choose_threshold_normal_quantile(
            val,
            val_scores,
            q,
        )[0]
        for q in args.q_grid
    }

    thr_f1 = mod5.choose_threshold_f1(
        val["label"].astype(int).to_numpy(),
        val_scores,
    )[0]

    results = {}

    for cand in selectable + reference:

        kind, q, k = cand

        thr = thr_q[q] if kind == "q" else thr_f1

        # ---------- val 성능 (선택에만 사용) ----------
        v = val.copy()
        v[ALARM_COL] = apply_alarm(mod5, val, thr, k)

        ev_v = mod5.calculate_event_details(v, ALARM_COL)

        det_v = ev_v["detected"].astype(bool)

        fa_v = mod6.false_alarm_episodes(
            v,
            ALARM_COL,
            sc.step_sec,
        )

        val_detect = float(det_v.mean()) if len(det_v) else 0.0

        val_fa = (
            fa_v["n_fa_groups"] / fa_v["n_groups"]
            if fa_v["n_groups"]
            else 0.0
        )

        # ---------- test 성능 (보고용) ----------
        tw = real.copy()
        tw[ALARM_COL] = apply_alarm(mod5, real, thr, k)

        ev_t = mod5.calculate_event_details(tw, ALARM_COL)

        part_full, ev_full = mod6.compute_part(
            tw,
            ev_t,
            sc.step_sec,
            sc.window_sec,
            None,
        )

        part_routed = ev_routed = None

        if multi:
            part_routed, ev_routed = mod6.compute_part(
                tw,
                ev_t,
                sc.step_sec,
                sc.window_sec,
                sc.routed,
            )

        results[cand] = {
            "threshold": float(thr),
            "val_detect": val_detect,
            "val_fa_cycle": float(val_fa),
            "part_full": part_full,
            "ev_full": ev_full,
            "part_routed": part_routed,
            "ev_routed": ev_routed,
        }

    return results


# ============================================================
# 누적기
# ============================================================

def new_acc(scale_names):

    return {
        "full_part": {n: mod6_empty() for n in scale_names},
        "full_ev": {n: [] for n in scale_names},
        "routed_part": mod6_empty(),
        "routed_ev": [],
    }


_MOD6 = None


def mod6_empty():
    return _MOD6.empty_part()


def add_result(acc, mod6, sc, res, fold, cand_label, multi):

    mod6.add_part(acc["full_part"][sc.name], res["part_full"])

    ev = res["ev_full"].copy()
    ev["fold"] = fold
    ev["scale"] = sc.name
    ev["threshold"] = res["threshold"]

    acc["full_ev"][sc.name].append(ev)

    if multi:

        mod6.add_part(acc["routed_part"], res["part_routed"])

        ev_r = res["ev_routed"].copy()
        ev_r["fold"] = fold
        ev_r["scale"] = sc.name
        ev_r["threshold"] = res["threshold"]

        acc["routed_ev"].append(ev_r)


# ============================================================
# 시스템 하나 평가
# ============================================================

def evaluate_system(mod5, mod6, paths, args):

    scales = [
        mod6.load_scale(mod5, p, args.feature_cols)
        for p in paths
    ]

    scales.sort(key=lambda s: -s.window_sec)

    mod6.route_scales(scales)

    multi = len(scales) > 1

    system_id = (
        "MULTI[" + "+".join(s.name for s in scales) + "]"
        if multi
        else scales[0].name
    )

    n_fault = len(
        set().union(
            *[
                set(
                    s.df.loc[s.df["source"] == "fault", "group_id"]
                )
                for s in scales
            ]
        )
    )

    if n_fault < args.n_splits:
        raise ValueError("fault 이벤트가 fold 수보다 적습니다.")

    selectable, reference = build_candidates(args)

    print()
    print(f"[SYSTEM] {system_id}")

    for sc in scales:
        print(
            f"  scale {sc.name} (W={sc.window_sec:.1f}s): "
            f"평가 대상 fault {len(set(sc.df.loc[sc.df['source'] == 'fault', 'group_id']) & sc.routed) if multi else n_fault}"
        )

    print(
        f"  후보 q={args.q_grid}, k={args.k_grid} "
        f"({len(selectable)}개) | 선택 규칙: {args.select_rule}"
        + (
            f" (lambda={args.lambda_fa})"
            if args.select_rule == "utility"
            else ""
        )
    )

    scale_names = [s.name for s in scales]

    order = (
        ["NESTED"]
        + [cand_name(c) for c in selectable]
        + [cand_name(c) for c in reference]
    )

    rows = []
    event_frames = []
    choice_rows = []

    for r in range(args.repeats):

        seed = args.seed + r

        fold_map = mod6.build_fold_map(
            scales,
            args.n_splits,
            seed,
        )

        acc = {name: new_acc(scale_names) for name in order}

        for i in range(args.n_splits):

            for sc in scales:

                work = make_frame(
                    mod6,
                    sc,
                    fold_map,
                    i,
                    args.n_splits,
                )

                results = evaluate_fold_scale(
                    mod5,
                    mod6,
                    sc,
                    work,
                    args,
                    selectable,
                    reference,
                    multi,
                )

                # ---- val 성능만으로 선택 (test 미사용) ----
                best = min(
                    selectable,
                    key=lambda c: rank_key(
                        args.select_rule,
                        results[c]["val_detect"],
                        results[c]["val_fa_cycle"],
                        c[2],
                        c[1],
                        args.lambda_fa,
                    ),
                )

                choice_rows.append({
                    "system_id": system_id,
                    "scale": sc.name,
                    "repeat": r,
                    "fold": i,
                    "q": best[1],
                    "k": best[2],
                    "config": cand_name(best),
                    "val_detect": results[best]["val_detect"],
                    "val_fa_cycle": results[best]["val_fa_cycle"],
                    "threshold": results[best]["threshold"],
                })

                for cand in selectable + reference:

                    add_result(
                        acc[cand_name(cand)],
                        mod6,
                        sc,
                        results[cand],
                        i,
                        cand_name(cand),
                        multi,
                    )

                add_result(
                    acc["NESTED"],
                    mod6,
                    sc,
                    results[best],
                    i,
                    "NESTED",
                    multi,
                )

        # ---- 반복 1회 행 생성 ----
        def emit(config, system, kind, part, ev_list, W, step, scale_text):

            events = pd.concat(ev_list, ignore_index=True)

            row = mod6.finalize_row(part, events, None)

            row.update({
                "system_id": system_id,
                "system": system,
                "kind": kind,
                "scales": scale_text,
                "config": config,
                "repeat": r,
                "seed": seed,
                "W_sec": W,
                "step_sec": step,
                "n_events_total": scales[0].n_events_total,
            })

            events["system_id"] = system_id
            events["system"] = system
            events["config"] = config
            events["repeat"] = r

            rows.append(row)
            event_frames.append(events)

        for config in order:

            a = acc[config]

            for sc in scales:

                emit(
                    config,
                    sc.name,
                    "single",
                    a["full_part"][sc.name],
                    a["full_ev"][sc.name],
                    sc.window_sec,
                    sc.step_sec,
                    sc.name,
                )

            if multi:

                emit(
                    config,
                    system_id,
                    "multi",
                    a["routed_part"],
                    a["routed_ev"],
                    float("nan"),
                    float("nan"),
                    "+".join(scale_names),
                )

        nested = [
            x for x in rows[-len(order) * (len(scales) + (1 if multi else 0)):]
            if x["config"] == "NESTED"
            and (x["kind"] == "multi" or not multi)
        ]

        n = nested[-1] if nested else None

        if n is not None:
            print(
                f"  [{system_id}] repeat {r + 1}/{args.repeats} NESTED "
                f"events={n['detected_events']}/{n['n_events']} "
                f"delay={n['delay_mean_sec']:.3f}s "
                f"F1={n['pooled_f1']:.4f} "
                f"FAcycle={n['fa_cycle_rate'] * 100:.1f}% "
                f"idleFAcycle={n['idle_fa_cycle_rate'] * 100:.1f}%"
            )

    return (
        pd.DataFrame(rows),
        pd.concat(event_frames, ignore_index=True),
        pd.DataFrame(choice_rows),
        order,
    )


# ============================================================
# 집계 / 출력
# ============================================================

SUMMARY_COLS = [
    "event_detection_rate",
    "delay_mean_sec",
    "pooled_f1",
    "pooled_precision",
    "pooled_recall",
    "fa_episodes",
    "fa_cycle_rate",
    "idle_fa_cycle_rate",
]


def summarize(repeat_df: pd.DataFrame) -> pd.DataFrame:

    out = []

    for (sid, system, kind, config), g in repeat_df.groupby(
        ["system_id", "system", "kind", "config"],
        sort=False,
    ):

        first = g.iloc[0]

        row = {
            "system_id": sid,
            "system": system,
            "kind": kind,
            "config": config,
            "n_events": int(first["n_events"]),
            "normal_cycles": int(first["normal_cycles"]),
            "idle_cycles": int(first["idle_cycles"]),
            "repeats": int(len(g)),
        }

        for col in SUMMARY_COLS:

            row[f"{col}_mean"] = float(g[col].mean())

            row[f"{col}_std"] = (
                float(g[col].std(ddof=1))
                if len(g) > 1
                else 0.0
            )

        out.append(row)

    return pd.DataFrame(out)


def print_report(summary, choices, args):

    def fmt(r, col, scale=1.0, nd=3):

        m = r[f"{col}_mean"]

        if pd.isna(m):
            return "-"

        return f"{m * scale:.{nd}f}±{r[f'{col}_std'] * scale:.{nd}f}"

    top = summary[
        (summary["kind"] == "multi")
        | (summary["system"] == summary["system_id"])
    ]

    for sid, g in top.groupby("system_id", sort=False):

        print()
        print("=" * 108)
        print(f"[{sid}] Nested 선택 vs 고정 조합 (mean ± std over repeats)")
        print("=" * 108)

        table = pd.DataFrame({
            "config": g["config"],
            "event%": g.apply(
                lambda r: fmt(r, "event_detection_rate", 100, 1),
                axis=1,
            ),
            "delay(s)": g.apply(
                lambda r: fmt(r, "delay_mean_sec"),
                axis=1,
            ),
            "F1": g.apply(
                lambda r: fmt(r, "pooled_f1"),
                axis=1,
            ),
            "FA episodes": g.apply(
                lambda r: fmt(r, "fa_episodes", 1, 1),
                axis=1,
            ),
            "FA cycle%": g.apply(
                lambda r: fmt(r, "fa_cycle_rate", 100, 2),
                axis=1,
            ),
            "idle FA cycle%": g.apply(
                lambda r: fmt(r, "idle_fa_cycle_rate", 100, 2),
                axis=1,
            ),
        })

        print(table.to_string(index=False))

        # ---- 낙관성: test 를 보고 고른 최선 vs nested ----
        fixed = g[
            g["config"].str.startswith("q=")
        ].copy()

        def key_row(r):

            k = int(r["config"].split("k=")[1])
            q = float(r["config"].split(",")[0].split("=")[1])

            return rank_key(
                args.select_rule,
                r["event_detection_rate_mean"],
                r["fa_cycle_rate_mean"],
                k,
                q,
                args.lambda_fa,
            )

        fixed["_key"] = fixed.apply(key_row, axis=1)

        best_fixed = fixed.sort_values("_key").iloc[0]

        nested = g[g["config"] == "NESTED"].iloc[0]

        print()
        print("낙관성 점검 (같은 규칙으로 test 에서 고른 최선 vs nested)")
        print(
            f"  test 를 보고 고른 최선 [{best_fixed['config']}]: "
            f"event {best_fixed['event_detection_rate_mean'] * 100:.1f}%  "
            f"F1 {best_fixed['pooled_f1_mean']:.3f}  "
            f"FA cycle {best_fixed['fa_cycle_rate_mean'] * 100:.2f}%"
        )
        print(
            f"  val 로만 선택한 NESTED            : "
            f"event {nested['event_detection_rate_mean'] * 100:.1f}%  "
            f"F1 {nested['pooled_f1_mean']:.3f}  "
            f"FA cycle {nested['fa_cycle_rate_mean'] * 100:.2f}%"
        )

        ch = choices[choices["system_id"] == sid]

        print()
        print("선택된 (q, k) 분포  (fold x 반복 x 스케일)")

        dist = (
            ch.groupby(["scale", "config"])
            .size()
            .rename("count")
            .reset_index()
            .sort_values(["scale", "count"], ascending=[True, False])
        )

        dist["ratio"] = (
            dist["count"]
            / dist.groupby("scale")["count"].transform("sum")
            * 100
        ).round(0).astype(int).astype(str) + "%"

        print(dist.to_string(index=False))

    print()
    print("* NESTED       : fold 마다 val 점수만으로 고른 (q,k) 를 test 에 적용한 결과 (보고용)")
    print("* q=..,k=..    : 모든 fold 에 같은 조합을 고정했을 때의 test 결과 (참고)")
    print("* f1,k=..      : val 의 fault 라벨로 F1 최대 threshold 를 잡은 참고용 결과")
    print("* delay        : 세그먼트 시작 -> 첫 알람 윈도우 종료 (실제 고장 발생 지연 아님)")


# ============================================================
# Main
# ============================================================

def main() -> None:

    global _MOD6

    here = Path(__file__).parent

    p = argparse.ArgumentParser(
        description="val 안에서 (q, k) 를 고르는 nested 선택"
    )

    p.add_argument("--inputs", nargs="*", default=None)
    p.add_argument("--multiscale", nargs="+", action="append", default=None)
    p.add_argument("--base-script", default=str(here / "5_run_mahalanobis.py"))
    p.add_argument("--cv-script", default=str(here / "5_2_run_mahalanobis.py"))
    p.add_argument("--output-dir", default="outputs/6_nested_selection")

    p.add_argument("--q-grid", type=float, nargs="+",
                   default=[0.99, 0.995, 0.998, 0.999, 0.9995])
    p.add_argument("--k-grid", type=int, nargs="+", default=[1, 2, 3])

    p.add_argument("--select-rule", choices=["lexi", "utility"], default="lexi")
    p.add_argument("--lambda-fa", type=float, default=5.0,
                   help="utility 규칙에서 오탐 사이클 비율의 가중치")

    p.add_argument("--n-splits", type=int, default=4)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--covariance", default="ledoitwolf",
                   choices=["empirical", "ledoitwolf", "oas"])

    args = p.parse_args()

    if args.n_splits < 3:
        raise ValueError("--n-splits 는 3 이상이어야 합니다.")

    if any(k < 1 for k in args.k_grid):
        raise ValueError("--k-grid 는 1 이상이어야 합니다.")

    mod5 = load_module(Path(args.base_script), "mahalanobis_base")
    mod6 = load_module(Path(args.cv_script), "mahalanobis_cv")

    _MOD6 = mod6

    mod5.choose_threshold_f1 = mod6.fast_choose_threshold_f1

    args.feature_cols = mod5.DEFAULT_FEATURES.copy()

    if args.inputs is None and args.multiscale is None:
        args.inputs = ["result/modeling_dataset_1.0_0.1"]

    systems = []

    for text in (args.inputs or []):

        path = mod6.resolve_windows_path(text)

        if not path.exists():
            print(f"[SKIP] 파일 없음: {path}")
            continue

        systems.append([path])

    for group in (args.multiscale or []):

        paths = [mod6.resolve_windows_path(t) for t in group]

        if any(not q.exists() for q in paths) or len(paths) < 2:
            print(f"[SKIP] 다중 스케일 입력 확인 필요: {paths}")
            continue

        systems.append(paths)

    if not systems:
        raise RuntimeError("평가할 시스템이 없습니다.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repeat_frames, event_frames, choice_frames = [], [], []

    for paths in systems:

        rep, ev, ch, _ = evaluate_system(mod5, mod6, paths, args)

        repeat_frames.append(rep)
        event_frames.append(ev)
        choice_frames.append(ch)

    repeat_all = pd.concat(repeat_frames, ignore_index=True)
    events_all = pd.concat(event_frames, ignore_index=True)
    choices = pd.concat(choice_frames, ignore_index=True)

    summary = summarize(repeat_all)

    repeat_all.to_csv(out_dir / "nested_repeat_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "nested_summary.csv", index=False, encoding="utf-8-sig")
    events_all.to_csv(out_dir / "nested_event_details.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(out_dir / "nested_choices.csv", index=False, encoding="utf-8-sig")

    print_report(summary, choices, args)

    # ---- nested 에서 놓친 이벤트 ----
    nested_ev = events_all[
        (events_all["config"] == "NESTED")
        & (
            (events_all["system"] == events_all["system_id"])
            | events_all["system_id"].str.startswith("MULTI")
        )
    ].copy()

    nested_ev["detected"] = nested_ev["detected"].astype(bool)

    freq = (
        nested_ev.groupby(["system_id", "system", "group_id"], sort=False)["detected"]
        .mean()
        .rename("detect_rate")
        .reset_index()
    )

    missed = freq[
        (freq["detect_rate"] < 1.0)
        & (
            (freq["system"] == freq["system_id"])
            | freq["system_id"].str.startswith("MULTI")
        )
    ]

    print()
    print("=" * 108)
    print("NESTED 에서 탐지율 100% 미만인 이벤트")
    print("=" * 108)

    if missed.empty:
        print("없음")
    else:
        print(missed.to_string(index=False))

    print()
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()