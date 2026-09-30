"""
Mahalanobis 이벤트 단위 교차검증 (Group K-Fold)
  + 다중 스케일(multi-scale) 평가
  + 사이클 단위 오탐률
  + Idle 상태 오탐 측정

5_run_mahalanobis.py 의 함수(run_once 등)를 그대로 재사용한다.

[1] Group K-Fold
    group_id(=프레스 사이클 세그먼트) 단위로 fold 를 나눈다.
    fault / normal / idle 세그먼트를 각각 섞어서 배분한다.
    fold i = test, fold (i+1)%K = val, 나머지 = train.
    모든 fault 이벤트가 정확히 한 번씩 test 에 들어간다.

[2] 다중 스케일 (--multiscale)
    W 가 다른 데이터셋 여러 개를 큰 W -> 작은 W 순으로 지정한다.
      예) 1.0s 모델 + 0.5s 모델
    각 세그먼트는 "window 가 만들어지는 가장 큰 W" 의 모델에서만 평가된다.
      - 1.0s 이상인 세그먼트 -> 1.0s 모델
      - 1.0s 미만인 짧은 세그먼트 -> 0.5s 모델
    스케일별 모델은 같은 fold 배정으로 각자 학습/threshold 선택을 하고,
    "합쳐진 시스템" 성능은 세그먼트를 한 번씩만 세어 계산한다.
    -> 짧아서 버려지던 fault 이벤트를 평가에 포함시킬 수 있다.

[3] Idle 오탐 측정
    4번이 만든 idle_windows.csv 를 학습/threshold 에는 쓰지 않고
    test fold 에서 "정상인데 알람이 나는지" 만 측정한다.

[4] 지표
    - 이벤트 탐지율 / delay (세그먼트 시작 -> 첫 알람 윈도우 종료)
    - 사이클당 오탐률 (fa_cycle_rate): 오탐이 1번이라도 난 정상 사이클 비율
    - 오탐 에피소드 수 (연속 오탐 윈도우를 1건으로 묶음)
    - Idle 사이클 오탐률

참고: fault 데이터는 전 행이 Equipment_state=1 이므로 fault_onset_time 은
      "세그먼트 첫 행" 이다. delay 는 실제 고장 발생 지연이 아니라
      "세그먼트 시작 후 첫 알람까지의 시간" 이다.

사용 예:
    # 단일 스케일 (기존 방식)
    python ./src/5_2_run_mahalanobis.py --inputs result/modeling_dataset_1.0_0.1

    # 다중 스케일: 1.0s + 0.5s
    python ./src/5_2_run_mahalanobis.py --multiscale result/modeling_dataset_1.0_0.1 result/modeling_dataset_0.5_0.1

    # 단일 스케일과 다중 스케일을 함께 비교
    python ./src/5_2_run_mahalanobis.py \\
        --inputs result/modeling_dataset_1.0_0.1 \\
        --multiscale result/modeling_dataset_1.0_0.1 result/modeling_dataset_0.5_0.1
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 기본 설정
# ============================================================

DEFAULT_INPUTS = [
    "result/modeling_dataset_1.0_0.5",
    "result/modeling_dataset_1.0_0.1",
    "result/modeling_dataset_0.9_0.1",
    "result/modeling_dataset_0.8_0.1",
]

ALARM_COL = "alarm_k_consecutive"

# fold 배정 시 source 처리 순서 (fault/normal 의 RNG 순서를 고정)
FOLD_SOURCE_ORDER = ["fault", "normal", "idle"]


# ============================================================
# 5_run_mahalanobis.py 로드
# ============================================================

def load_base_module(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"기준 스크립트를 찾을 수 없습니다: {path}\n"
            f"--base-script 로 5_run_mahalanobis.py 경로를 지정하세요."
        )

    spec = importlib.util.spec_from_file_location(
        "mahalanobis_base",
        str(path),
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ============================================================
# 빠른 F1 threshold (5번 choose_threshold_f1 과 동일한 규칙)
#   1. F1 최대  2. 동률이면 precision 최대  3. 동률이면 낮은 threshold
# ============================================================

def fast_choose_threshold_f1(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    finite = np.isfinite(scores)

    y = y_true[finite]
    s = scores[finite]

    if len(s) == 0:
        raise ValueError("No finite validation scores.")

    order = np.argsort(-s, kind="stable")

    s = s[order]
    y = y[order]

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)

    last = np.r_[
        np.flatnonzero(np.diff(s) != 0),
        len(s) - 1,
    ]

    tp = tp[last]
    fp = fp[last]
    thr = s[last]

    positives = int(y.sum())
    fn = positives - tp

    denom_f1 = 2 * tp + fp + fn

    f1 = np.divide(
        2 * tp,
        denom_f1,
        out=np.zeros(len(denom_f1), dtype=float),
        where=denom_f1 > 0,
    )

    denom_p = tp + fp

    precision = np.divide(
        tp,
        denom_p,
        out=np.zeros(len(denom_p), dtype=float),
        where=denom_p > 0,
    )

    idx = np.lexsort(
        (
            thr,
            -np.round(precision, 12),
            -np.round(f1, 12),
        )
    )[0]

    return float(thr[idx]), float(f1[idx])


# ============================================================
# Scale (W 하나에 해당하는 데이터셋)
# ============================================================

@dataclass
class Scale:
    name: str
    path: Path
    df: pd.DataFrame
    idle: pd.DataFrame | None
    step_sec: float
    window_sec: float
    n_events_total: int | None
    routed: set = field(default_factory=set)


def resolve_windows_path(text: str) -> Path:

    path = Path(text)

    if path.is_dir():
        path = path / "model_windows.csv"

    return path


def infer_step_and_window(
    df: pd.DataFrame,
) -> tuple[float, float]:

    ordered = df.sort_values(
        ["group_id", "window_start"]
    )

    deltas = (
        ordered.groupby("group_id")["window_start"]
        .diff()
        .dt.total_seconds()
    )

    positive = deltas[deltas > 0]

    step_sec = (
        float(positive.median())
        if len(positive)
        else 0.1
    )

    window_sec = float(
        (
            df["window_end"]
            - df["window_start"]
        )
        .dt.total_seconds()
        .median()
    )

    return step_sec, window_sec


def load_scale(
    mod,
    path: Path,
    feature_cols: list[str],
) -> Scale:

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    missing = [
        c
        for c in mod.REQUIRED_META + ["source"]
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path}: 필수 컬럼 누락 {missing}"
        )

    df = mod.prepare_data(df, feature_cols)
    df = df[df["label"].isin([0, 1])].copy()

    step_sec, window_sec = infer_step_and_window(df)

    # ---- Idle window (없으면 Idle 평가 생략) ----
    idle = None
    idle_path = path.with_name("idle_windows.csv")

    if idle_path.exists():

        idle_raw = pd.read_csv(idle_path)

        if len(idle_raw) > 0:
            idle = mod.prepare_data(idle_raw, feature_cols)
            idle = idle[idle["label"].isin([0, 1])].copy()

            if len(idle) == 0:
                idle = None
    else:
        print(
            f"  [WARN] {idle_path.name} 없음 -> Idle 평가 생략 "
            f"(4번 스크립트를 다시 실행하세요)"
        )

    # ---- 전체 이벤트 수 (커버리지 표시용) ----
    n_events_total = None
    em_path = path.with_name("event_manifest.csv")

    if em_path.exists():
        n_events_total = int(len(pd.read_csv(em_path)))

    name = path.parent.name.replace(
        "modeling_dataset_",
        "",
    )

    return Scale(
        name=name,
        path=path,
        df=df,
        idle=idle,
        step_sec=step_sec,
        window_sec=window_sec,
        n_events_total=n_events_total,
    )


def route_scales(scales: list[Scale]) -> None:
    """
    큰 W 부터 순서대로, 아직 배정되지 않은 group 을 해당 스케일에 배정한다.
    (window 가 만들어지는 가장 큰 W 의 모델이 그 세그먼트를 평가)
    """

    assigned: set = set()

    for sc in scales:

        groups = set(sc.df["group_id"].unique())

        if sc.idle is not None:
            groups |= set(sc.idle["group_id"].unique())

        sc.routed = groups - assigned
        assigned |= groups


# ============================================================
# Fold 배정 (스케일 간 공통)
# ============================================================

def build_fold_map(
    scales: list[Scale],
    n_splits: int,
    seed: int,
) -> dict:

    groups_by_source: dict[str, set] = {}

    for sc in scales:

        for src, ids in sc.df.groupby("source")["group_id"]:
            groups_by_source.setdefault(src, set()).update(
                ids.unique()
            )

        if sc.idle is not None:
            groups_by_source.setdefault("idle", set()).update(
                sc.idle["group_id"].unique()
            )

    order = [
        s for s in FOLD_SOURCE_ORDER
        if s in groups_by_source
    ] + sorted(
        s for s in groups_by_source
        if s not in FOLD_SOURCE_ORDER
    )

    rng = np.random.default_rng(seed)

    fold_of = {}

    for source in order:

        groups = np.array(
            sorted(groups_by_source[source])
        )

        rng.shuffle(groups)

        for i, group_id in enumerate(groups):
            fold_of[group_id] = i % n_splits

    return fold_of


def make_fold_frame(
    sc: Scale,
    fold_map: dict,
    fold: int,
    n_splits: int,
) -> pd.DataFrame:

    work = sc.df.copy()

    f = work["group_id"].map(fold_map)

    work["split"] = np.where(
        f == fold,
        "test",
        np.where(
            f == (fold + 1) % n_splits,
            "val",
            "train",
        ),
    )

    work["is_idle"] = False

    # Idle 은 학습/threshold 에 쓰지 않고 test fold 에서 오탐만 측정
    if sc.idle is not None:

        fi = sc.idle["group_id"].map(fold_map)

        idle = sc.idle[fi == fold].copy()

        if len(idle) > 0:
            idle["split"] = "test"
            idle["is_idle"] = True
            idle["source"] = "idle"

            work = pd.concat(
                [work, idle],
                ignore_index=True,
            )

    return work


# ============================================================
# 오탐 에피소드 / 사이클 단위 오탐
# ============================================================

def false_alarm_episodes(
    test_df: pd.DataFrame,
    alarm_col: str,
    step_sec: float,
) -> dict:
    """
    n_episodes   : 연속된 오탐 윈도우를 1건으로 센 횟수
    exposure_sec : 정상 구간이 관측된 시간 (윈도우 구간의 합집합)
    n_groups     : 정상 구간 윈도우가 있는 사이클 수
    n_fa_groups  : 그중 오탐이 1번이라도 난 사이클 수

    정상 구간 = fault 의 (onset 이후) 구간을 제외한 모든 윈도우
    """

    res = {
        "n_episodes": 0,
        "exposure_sec": 0.0,
        "n_groups": 0,
        "n_fa_groups": 0,
    }

    if test_df.empty:
        return res

    in_fault_zone = (
        test_df["source"].eq("fault")
        & test_df["fault_onset_time"].notna()
        & (test_df["window_end"] >= test_df["fault_onset_time"])
    )

    zone = test_df.loc[~in_fault_zone]

    if zone.empty:
        return res

    zone = zone.sort_values(["group_id", "window_start"])

    for _, g in zone.groupby("group_id", sort=False):

        start = (
            g["window_start"]
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
            / 1e9
        )

        end = (
            g["window_end"]
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
            / 1e9
        )

        alarm = g[alarm_col].to_numpy(dtype=bool)

        contiguous = np.r_[
            False,
            np.diff(start) <= step_sec * 1.5,
        ]

        prev_alarm = np.r_[False, alarm[:-1]]

        new_episode = alarm & ~(prev_alarm & contiguous)

        n_ep = int(new_episode.sum())

        res["n_episodes"] += n_ep
        res["n_groups"] += 1
        res["n_fa_groups"] += int(n_ep > 0)

        prev_end = np.r_[
            -np.inf,
            np.maximum.accumulate(end)[:-1],
        ]

        res["exposure_sec"] += float(
            np.clip(
                end - np.maximum(start, prev_end),
                0,
                None,
            ).sum()
        )

    return res


# ============================================================
# fold 1회 결과를 (group 부분집합 기준으로) 집계
# ============================================================

NUMERIC_PART_KEYS = [
    "tp", "fp", "tn", "fn",
    "fa_episodes", "exposure_sec", "cycles", "fa_cycles",
    "idle_windows", "idle_alarm_windows",
    "idle_cycles", "idle_fa_cycles",
    "idle_fa_episodes", "idle_exposure_sec",
]


def empty_part() -> dict:
    return {k: 0.0 for k in NUMERIC_PART_KEYS}


def add_part(total: dict, part: dict) -> None:
    for k in NUMERIC_PART_KEYS:
        total[k] += part[k]


def compute_part(
    test_work: pd.DataFrame,
    event_details: pd.DataFrame,
    step_sec: float,
    window_sec: float,
    selected: set | None,
) -> tuple[dict, pd.DataFrame]:

    tw = test_work
    ev = event_details

    if selected is not None:
        tw = tw[tw["group_id"].isin(selected)]
        ev = ev[ev["group_id"].isin(selected)]

    is_idle = tw["is_idle"].astype(bool)

    main = tw[~is_idle]
    idle = tw[is_idle]

    y = main["label"].astype(int).to_numpy()
    pred = main[ALARM_COL].astype(int).to_numpy()

    fa = false_alarm_episodes(main, ALARM_COL, step_sec)
    fa_idle = false_alarm_episodes(idle, ALARM_COL, step_sec)

    part = {
        "tp": int(((y == 1) & (pred == 1)).sum()),
        "fp": int(((y == 0) & (pred == 1)).sum()),
        "tn": int(((y == 0) & (pred == 0)).sum()),
        "fn": int(((y == 1) & (pred == 0)).sum()),
        "fa_episodes": fa["n_episodes"],
        "exposure_sec": fa["exposure_sec"],
        "cycles": fa["n_groups"],
        "fa_cycles": fa["n_fa_groups"],
        "idle_windows": int(len(idle)),
        "idle_alarm_windows": int(
            idle[ALARM_COL].astype(bool).sum()
        ),
        "idle_cycles": fa_idle["n_groups"],
        "idle_fa_cycles": fa_idle["n_fa_groups"],
        "idle_fa_episodes": fa_idle["n_episodes"],
        "idle_exposure_sec": fa_idle["exposure_sec"],
    }

    ev = ev.copy()
    ev["W_sec"] = window_sec

    return part, ev


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def finalize_row(
    part: dict,
    events: pd.DataFrame,
    thresholds: list[float] | None,
) -> dict:

    detected = events["detected"].astype(bool)

    ev_d = events[
        detected & events["detection_delay_sec"].notna()
    ]

    delay = ev_d["detection_delay_sec"].to_numpy(dtype=float)

    delay_minus_w = (
        ev_d["detection_delay_sec"] - ev_d["W_sec"]
    ).to_numpy(dtype=float)

    tp, fp, tn, fn = (
        part["tp"], part["fp"], part["tn"], part["fn"]
    )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    hours = part["exposure_sec"] / 3600.0

    return {
        "n_events": int(len(events)),
        "detected_events": int(detected.sum()),
        "event_detection_rate": (
            float(detected.mean()) if len(events) else float("nan")
        ),
        "delay_mean_sec": (
            float(delay.mean()) if len(delay) else float("nan")
        ),
        "delay_median_sec": (
            float(np.median(delay)) if len(delay) else float("nan")
        ),
        "delay_minus_W_mean_sec": (
            float(delay_minus_w.mean())
            if len(delay_minus_w)
            else float("nan")
        ),
        "pooled_f1": float(f1),
        "pooled_precision": float(precision),
        "pooled_recall": float(recall),
        "normal_window_fpr": safe_div(fp, fp + tn),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "normal_cycles": int(part["cycles"]),
        "fa_cycles": int(part["fa_cycles"]),
        "fa_cycle_rate": safe_div(
            part["fa_cycles"], part["cycles"]
        ),
        "fa_episodes": int(part["fa_episodes"]),
        "normal_exposure_min": float(
            part["exposure_sec"] / 60.0
        ),
        "fa_per_hour_observed": safe_div(
            part["fa_episodes"], hours
        ),
        "idle_cycles": int(part["idle_cycles"]),
        "idle_fa_cycles": int(part["idle_fa_cycles"]),
        "idle_fa_cycle_rate": safe_div(
            part["idle_fa_cycles"], part["idle_cycles"]
        ),
        "idle_window_fpr": safe_div(
            part["idle_alarm_windows"], part["idle_windows"]
        ),
        "idle_fa_episodes": int(part["idle_fa_episodes"]),
        "threshold_mean": (
            float(np.mean(thresholds))
            if thresholds
            else float("nan")
        ),
    }


# ============================================================
# 시스템 하나 (단일 스케일 또는 다중 스케일) 평가
# ============================================================

def evaluate_system(
    mod,
    paths: list[Path],
    args,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:

    scales = [
        load_scale(mod, p, feature_cols)
        for p in paths
    ]

    # 큰 W -> 작은 W
    scales.sort(key=lambda s: -s.window_sec)

    route_scales(scales)

    multi = len(scales) > 1

    system_name = (
        "MULTI[" + "+".join(s.name for s in scales) + "]"
        if multi
        else scales[0].name
    )

    n_fault_groups = len(
        set().union(
            *[
                set(
                    s.df.loc[
                        s.df["source"] == "fault",
                        "group_id",
                    ].unique()
                )
                for s in scales
            ]
        )
    )

    if n_fault_groups < args.n_splits:
        raise ValueError(
            f"fault 이벤트({n_fault_groups}개)가 "
            f"--n-splits({args.n_splits})보다 적습니다."
        )

    # ---- 라우팅 요약 ----
    print()
    print(f"[SYSTEM] {system_name}")

    for sc in scales:

        fault_ids = set(
            sc.df.loc[
                sc.df["source"] == "fault",
                "group_id",
            ]
        )

        normal_ids = set(
            sc.df.loc[
                sc.df["source"] == "normal",
                "group_id",
            ]
        )

        idle_ids = (
            set(sc.idle["group_id"])
            if sc.idle is not None
            else set()
        )

        print(
            f"  scale {sc.name:>8s} (W={sc.window_sec:.1f}s, "
            f"step={sc.step_sec:.1f}s) | "
            f"전체: fault {len(fault_ids)}, normal {len(normal_ids)}, "
            f"idle {len(idle_ids)}"
            + (
                f" | 이 스케일이 평가: fault "
                f"{len(fault_ids & sc.routed)}, "
                f"normal {len(normal_ids & sc.routed)}, "
                f"idle {len(idle_ids & sc.routed)}"
                if multi
                else ""
            )
        )

    n_events_total = scales[0].n_events_total

    repeat_rows = []
    event_frames = []

    for mode in args.threshold_modes:

        for r in range(args.repeats):

            seed = args.seed + r

            fold_map = build_fold_map(
                scales,
                args.n_splits,
                seed,
            )

            full_part = {sc.name: empty_part() for sc in scales}
            full_events = {sc.name: [] for sc in scales}
            full_thr = {sc.name: [] for sc in scales}

            routed_part = empty_part()
            routed_events = []

            for i in range(args.n_splits):

                for sc in scales:

                    work = make_fold_frame(
                        sc,
                        fold_map,
                        i,
                        args.n_splits,
                    )

                    metrics, test_work, event_details = mod.run_once(
                        df=work,
                        feature_cols=feature_cols,
                        covariance=args.covariance,
                        threshold_mode=mode,
                        normal_quantile=args.normal_quantile,
                        k_consecutive=args.k_consecutive,
                        train_seed=None,
                        train_fraction=1.0,
                    )

                    # ---- 이 스케일이 가진 모든 group 기준 (단일 스케일 성능) ----
                    part, ev = compute_part(
                        test_work,
                        event_details,
                        sc.step_sec,
                        sc.window_sec,
                        None,
                    )

                    add_part(full_part[sc.name], part)

                    ev["fold"] = i
                    ev["threshold"] = metrics["threshold"]
                    ev["scale"] = sc.name

                    full_events[sc.name].append(ev)
                    full_thr[sc.name].append(metrics["threshold"])

                    # ---- 라우팅된 group 만 (합쳐진 시스템 성능) ----
                    if multi:

                        part_r, ev_r = compute_part(
                            test_work,
                            event_details,
                            sc.step_sec,
                            sc.window_sec,
                            sc.routed,
                        )

                        add_part(routed_part, part_r)

                        ev_r["fold"] = i
                        ev_r["threshold"] = metrics["threshold"]
                        ev_r["scale"] = sc.name

                        routed_events.append(ev_r)

            # ---- 반복 1회 결과 행 생성 ----
            def emit(system, kind, part, ev_list, thr, W, step, scale_text):

                events = pd.concat(ev_list, ignore_index=True)

                row = finalize_row(part, events, thr)

                row.update({
                    "system": system,
                    "kind": kind,
                    "scales": scale_text,
                    "threshold_mode": mode,
                    "repeat": r,
                    "seed": seed,
                    "W_sec": W,
                    "step_sec": step,
                    "n_events_total": n_events_total,
                })

                events["system"] = system
                events["threshold_mode"] = mode
                events["repeat"] = r

                repeat_rows.append(row)
                event_frames.append(events)

                return row

            last_row = None

            for sc in scales:

                last_row = emit(
                    sc.name,
                    "single",
                    full_part[sc.name],
                    full_events[sc.name],
                    full_thr[sc.name],
                    sc.window_sec,
                    sc.step_sec,
                    sc.name,
                )

            if multi:

                last_row = emit(
                    system_name,
                    "multi",
                    routed_part,
                    routed_events,
                    None,
                    float("nan"),
                    float("nan"),
                    "+".join(s.name for s in scales),
                )

            print(
                f"  [{system_name} | {mode}] "
                f"repeat {r + 1}/{args.repeats} "
                f"events={last_row['detected_events']}/"
                f"{last_row['n_events']} "
                f"delay={last_row['delay_mean_sec']:.3f}s "
                f"F1={last_row['pooled_f1']:.4f} "
                f"FAcycle={last_row['fa_cycle_rate'] * 100:.1f}% "
                f"idleFAcycle={last_row['idle_fa_cycle_rate'] * 100:.1f}%"
            )

    return (
        pd.DataFrame(repeat_rows),
        pd.concat(event_frames, ignore_index=True),
    )


# ============================================================
# 집계 / 출력
# ============================================================

SUMMARY_COLS = [
    "event_detection_rate",
    "delay_mean_sec",
    "delay_minus_W_mean_sec",
    "pooled_f1",
    "pooled_precision",
    "pooled_recall",
    "normal_window_fpr",
    "fa_episodes",
    "fa_cycle_rate",
    "fa_per_hour_observed",
    "idle_fa_cycle_rate",
    "idle_window_fpr",
]


def summarize(repeat_df: pd.DataFrame) -> pd.DataFrame:

    out = []

    for (system, mode), g in repeat_df.groupby(
        ["system", "threshold_mode"],
        sort=False,
    ):

        first = g.iloc[0]

        row = {
            "system": system,
            "kind": first["kind"],
            "threshold_mode": mode,
            "W_sec": first["W_sec"],
            "n_events": int(first["n_events"]),
            "n_events_total": first["n_events_total"],
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


def print_comparison(summary: pd.DataFrame) -> None:

    def fmt(row, col, scale=1.0, nd=3):

        m = row[f"{col}_mean"]
        s = row[f"{col}_std"]

        if pd.isna(m):
            return "-"

        return f"{m * scale:.{nd}f}±{s * scale:.{nd}f}"

    print()
    print("=" * 110)
    print("[표 1] 탐지 성능  (mean ± std over repeats)")
    print("=" * 110)

    t1 = pd.DataFrame({
        "system": summary["system"],
        "mode": summary["threshold_mode"],
        "events": summary.apply(
            lambda r: (
                f"{r['n_events']}"
                + (
                    f"/{int(r['n_events_total'])}"
                    if pd.notna(r["n_events_total"])
                    else ""
                )
            ),
            axis=1,
        ),
        "event%": summary.apply(
            lambda r: fmt(r, "event_detection_rate", 100, 1),
            axis=1,
        ),
        "delay(s)": summary.apply(
            lambda r: fmt(r, "delay_mean_sec"),
            axis=1,
        ),
        "delay-W": summary.apply(
            lambda r: fmt(r, "delay_minus_W_mean_sec"),
            axis=1,
        ),
        "F1": summary.apply(
            lambda r: fmt(r, "pooled_f1"),
            axis=1,
        ),
        "prec": summary.apply(
            lambda r: fmt(r, "pooled_precision"),
            axis=1,
        ),
        "recall": summary.apply(
            lambda r: fmt(r, "pooled_recall"),
            axis=1,
        ),
    })

    print(t1.to_string(index=False))

    print()
    print("=" * 110)
    print("[표 2] 오탐  (정상 사이클 / Idle 사이클)")
    print("=" * 110)

    t2 = pd.DataFrame({
        "system": summary["system"],
        "mode": summary["threshold_mode"],
        "normal cycles": summary["normal_cycles"],
        "FA episodes": summary.apply(
            lambda r: fmt(r, "fa_episodes", 1, 1),
            axis=1,
        ),
        "FA cycle%": summary.apply(
            lambda r: fmt(r, "fa_cycle_rate", 100, 2),
            axis=1,
        ),
        "FA/h(obs)": summary.apply(
            lambda r: fmt(r, "fa_per_hour_observed", 1, 1),
            axis=1,
        ),
        "idle cycles": summary["idle_cycles"],
        "idle FA cycle%": summary.apply(
            lambda r: fmt(r, "idle_fa_cycle_rate", 100, 2),
            axis=1,
        ),
        "idle win FPR%": summary.apply(
            lambda r: fmt(r, "idle_window_fpr", 100, 2),
            axis=1,
        ),
    })

    print(t2.to_string(index=False))

    print()
    print("* events       : 평가된 fault 이벤트 수 / 전체 이벤트 수")
    print("* delay(s)     : 세그먼트 시작 -> 첫 알람 윈도우 종료 (실제 고장 발생 지연 아님)")
    print("* delay-W      : delay 에서 (해당 스케일의) 윈도우 길이를 뺀 값")
    print("* FA cycle%    : 오탐이 1번이라도 난 정상 사이클 비율")
    print("* FA/h(obs)    : 관측된 정상 데이터 1시간당 오탐 에피소드 (벽시계 시간 아님)")
    print("* MULTI[...]   : 세그먼트를 window 가 생기는 가장 큰 W 모델로 한 번씩만 평가")
    print("* idle         : 학습/threshold 에 쓰지 않은 Idle 세그먼트에서의 오탐")


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Mahalanobis Group K-Fold CV + 다중 스케일 + Idle 오탐"
        )
    )

    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help=(
            "단일 스케일 평가: 데이터셋 폴더 또는 model_windows.csv "
            "(여러 개 가능, 각각 독립 평가)"
        ),
    )

    parser.add_argument(
        "--multiscale",
        nargs="+",
        action="append",
        default=None,
        help=(
            "다중 스케일 평가: 데이터셋 폴더들 (큰 W 와 작은 W). "
            "여러 번 지정하면 각각 별도 시스템."
        ),
    )

    parser.add_argument(
        "--base-script",
        default=str(
            Path(__file__).with_name("5_run_mahalanobis.py")
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/5_2_mahalanobis_cv",
    )

    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--covariance",
        choices=["empirical", "ledoitwolf", "oas"],
        default="ledoitwolf",
    )

    parser.add_argument(
        "--threshold-modes",
        nargs="+",
        choices=["f1", "normal_quantile"],
        default=["f1", "normal_quantile"],
    )

    parser.add_argument(
        "--normal-quantile",
        type=float,
        default=0.995,
    )

    parser.add_argument(
        "--k-consecutive",
        type=int,
        choices=[1, 2],
        default=1,
    )

    args = parser.parse_args()

    if args.n_splits < 3:
        raise ValueError("--n-splits 는 3 이상이어야 합니다.")

    if args.repeats < 1:
        raise ValueError("--repeats 는 1 이상이어야 합니다.")

    if args.inputs is None and args.multiscale is None:
        args.inputs = DEFAULT_INPUTS

    mod = load_base_module(Path(args.base_script))

    # 느린 원본 threshold 함수를 동일 규칙의 빠른 버전으로 교체
    mod.choose_threshold_f1 = fast_choose_threshold_f1

    feature_cols = mod.DEFAULT_FEATURES.copy()

    systems: list[list[Path]] = []

    for text in (args.inputs or []):

        path = resolve_windows_path(text)

        if not path.exists():
            print(f"[SKIP] 파일 없음: {path}")
            continue

        systems.append([path])

    for group in (args.multiscale or []):

        paths = [resolve_windows_path(t) for t in group]

        missing = [p for p in paths if not p.exists()]

        if missing:
            print(f"[SKIP] 다중 스케일: 파일 없음 {missing}")
            continue

        if len(paths) < 2:
            print("[SKIP] --multiscale 는 2개 이상의 폴더가 필요합니다.")
            continue

        systems.append(paths)

    if not systems:
        raise RuntimeError("평가할 시스템이 없습니다.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repeat_frames = []
    event_frames = []

    for paths in systems:

        repeat_df, event_df = evaluate_system(
            mod,
            paths,
            args,
            feature_cols,
        )

        repeat_frames.append(repeat_df)
        event_frames.append(event_df)

    repeat_all = pd.concat(repeat_frames, ignore_index=True)
    events_all = pd.concat(event_frames, ignore_index=True)

    # 같은 단일 스케일이 여러 시스템에 중복 포함될 수 있으므로 제거
    repeat_all = repeat_all.drop_duplicates(
        subset=["system", "kind", "threshold_mode", "repeat"],
        keep="first",
    ).reset_index(drop=True)

    events_all = events_all.drop_duplicates(
        subset=[
            "system", "threshold_mode", "repeat",
            "scale", "group_id",
        ],
        keep="first",
    ).reset_index(drop=True)

    summary = summarize(repeat_all)

    event_freq = (
        events_all
        .assign(detected=events_all["detected"].astype(bool))
        .groupby(
            ["system", "threshold_mode", "group_id"],
            sort=False,
        )
        .agg(
            detect_rate=("detected", "mean"),
            delay_mean_sec=("detection_delay_sec", "mean"),
            scale=("scale", "first"),
        )
        .reset_index()
    )

    repeat_all.to_csv(
        out_dir / "cv_repeat_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        out_dir / "cv_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    events_all.to_csv(
        out_dir / "cv_event_details.csv",
        index=False,
        encoding="utf-8-sig",
    )

    event_freq.to_csv(
        out_dir / "cv_event_detection_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print_comparison(summary)

    missed = event_freq[event_freq["detect_rate"] < 1.0]

    print()
    print("=" * 110)
    print("탐지율 100% 미만인 이벤트")
    print("=" * 110)

    if missed.empty:
        print("없음 (모든 설정에서 모든 이벤트를 탐지)")
    else:
        print(
            missed[
                [
                    "system",
                    "threshold_mode",
                    "group_id",
                    "scale",
                    "detect_rate",
                ]
            ].to_string(index=False)
        )

    print()
    print("Saved:")

    for name in [
        "cv_comparison.csv",
        "cv_repeat_results.csv",
        "cv_event_details.csv",
        "cv_event_detection_frequency.csv",
    ]:
        print(out_dir / name)


if __name__ == "__main__":
    main()