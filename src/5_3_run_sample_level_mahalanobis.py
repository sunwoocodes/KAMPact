"""
샘플 단위 Mahalanobis + 인과적(causal) 시간 평활 + Group K-Fold

윈도우 통계(std/ptp/slope/corr ...)를 만들지 않고 원본 샘플(0.1s 간격)에
직접 Mahalanobis 거리를 계산한다. 그다음 세그먼트(=프레스 사이클) 안에서
"지금까지 본 마지막 k개 샘플의 점수"를 평균(또는 최댓값)해서 알람을 낸다.

장점
  - 세그먼트가 아무리 짧아도(3~4행 포함) 21개 fault 이벤트 전부 사용
  - 511개 정상 사이클 전부 사용 (W 보다 짧아서 버려지는 사이클 없음)
  - 첫 샘플부터 점수가 나오므로 delay 가 윈도우 길이 W 에 묶이지 않음
단점
  - 윈도우 통계가 없어 정보량이 적음 (raw_diff_roll3 로 일부 보완)

평가 방식은 5_2번(교차검증 CV)과 동일하다.
  - fault / normal / idle 세그먼트를 각각 섞어 fold 배정 (누수 없음)
  - fold i = test, fold (i+1)%K = val, 나머지 = train
  - 학습: train fold 의 비-Idle 정상 샘플만
  - Idle 세그먼트: 학습/threshold 에 쓰지 않고 test fold 에서 오탐만 측정
  - 오탐은 "정상 사이클에서 알람이 1번이라도 난 비율" 로 보고

주의: delay 는 "세그먼트 첫 샘플 -> 알람을 만든 샘플" 의 시간차이다.
      (fault 데이터는 전부 Equipment_state=1 이므로 실제 고장 발생 시점은 알 수 없음)
      W=1.0 윈도우 모델의 "delay ≈ 1.0s" 는 이 정의로는 약 0.9s(샘플 10개) 에 해당한다.

사용 예:
    python ./src/5_3_run_sample_level_mahalanobis.py \\
        --normal-path data/press_data_normal_with_idle.csv \\
        --fault-path  data/outlier_data.csv

    # 평활 길이 여러 개 비교
    python ./src/5_3_run_sample_level_mahalanobis.py --agg-k 1 3 5 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS


# ============================================================
# 설정
# ============================================================

TIME_COL = "TimeStamp"
IDLE_COL = "Idle"
STATE_COL = "Equipment_state"

SENSOR_COLS = [
    "AI0_Vibration",
    "AI1_Vibration",
    "AI2_Current",
]

GAP_THRESHOLD_SEC = 0.5

FOLD_SOURCE_ORDER = ["fault", "normal", "idle"]


# ============================================================
# 데이터 로드 / 세그먼트 / 특징
# ============================================================

def load_raw(path: str, source: str) -> pd.DataFrame:

    df = pd.read_csv(path)

    need = [TIME_COL, STATE_COL] + SENSOR_COLS
    missing = [c for c in need if c not in df.columns]

    if missing:
        raise ValueError(f"{path}: 컬럼 누락 {missing}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

    df = (
        df.dropna(subset=[TIME_COL])
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )

    for col in SENSOR_COLS + [STATE_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if IDLE_COL in df.columns:
        df[IDLE_COL] = pd.to_numeric(df[IDLE_COL], errors="coerce").fillna(0)
    else:
        df[IDLE_COL] = 0

    gap = df[TIME_COL].diff().dt.total_seconds()

    brk = gap > GAP_THRESHOLD_SEC
    brk.iloc[0] = True

    seg = brk.cumsum() - 1

    df["source"] = source
    df["segment_id"] = seg
    df["group_id"] = source + "_" + seg.astype(str)

    idle_ratio = (
        (df[IDLE_COL] == 1)
        .groupby(df["group_id"])
        .transform("mean")
    )

    df["kind"] = np.where(
        source == "fault",
        "fault",
        np.where(idle_ratio >= 0.5, "idle", "normal"),
    )

    df["label"] = (df[STATE_COL].fillna(0) >= 1).astype(int)

    return df


def build_features(
    df: pd.DataFrame,
    feature_set: str,
) -> tuple[np.ndarray, list[str]]:

    cols = list(SENSOR_COLS)
    parts = [df[SENSOR_COLS].to_numpy(dtype=float)]

    if feature_set in ("raw_diff", "raw_diff_roll3"):

        diff = (
            df.groupby("group_id")[SENSOR_COLS]
            .diff()
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        parts.append(diff)
        cols += [f"d_{c}" for c in SENSOR_COLS]

    if feature_set == "raw_diff_roll3":

        for c in SENSOR_COLS:

            roll = (
                df.groupby("group_id")[c]
                .transform(
                    lambda s: s.rolling(3, min_periods=2).std()
                )
                .fillna(0.0)
                .to_numpy(dtype=float)
            )

            parts.append(roll.reshape(-1, 1))
            cols.append(f"std3_{c}")

    return np.hstack(parts), cols


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


# ============================================================
# Mahalanobis / threshold / 평활
# ============================================================

def get_covariance(name: str):

    if name == "empirical":
        return EmpiricalCovariance(assume_centered=False)

    if name == "oas":
        return OAS(assume_centered=False)

    return LedoitWolf(assume_centered=False)


def fit_mahalanobis(
    x_train: np.ndarray,
    covariance: str,
):

    est = get_covariance(covariance)
    est.fit(x_train)

    mean = est.location_
    precision = est.precision_

    def score(x: np.ndarray) -> np.ndarray:
        d = x - mean
        return np.einsum("ij,jk,ik->i", d, precision, d)

    return score


def causal_smooth(
    scores: np.ndarray,
    group_positions: list[np.ndarray],
    k: int,
    min_periods: int,
    how: str,
) -> np.ndarray:
    """세그먼트 안에서 마지막 k 샘플 평균/최댓값 (미래 샘플 사용 안 함)"""

    out = np.full(len(scores), np.nan)

    for idx in group_positions:

        roll = pd.Series(scores[idx]).rolling(
            k,
            min_periods=min_periods,
        )

        out[idx] = (
            roll.max() if how == "max" else roll.mean()
        ).to_numpy()

    return out


def fast_choose_threshold_f1(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    finite = np.isfinite(scores)

    y = y_true[finite]
    s = scores[finite]

    if len(s) == 0 or y.sum() == 0:
        raise ValueError("검증 데이터에 양성 샘플이 없습니다.")

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

    fn = int(y.sum()) - tp

    den = 2 * tp + fp + fn

    f1 = np.divide(
        2 * tp,
        den,
        out=np.zeros(len(den), dtype=float),
        where=den > 0,
    )

    den_p = tp + fp

    prec = np.divide(
        tp,
        den_p,
        out=np.zeros(len(den_p), dtype=float),
        where=den_p > 0,
    )

    idx = np.lexsort(
        (
            thr,
            -np.round(prec, 12),
            -np.round(f1, 12),
        )
    )[0]

    return float(thr[idx]), float(f1[idx])


def choose_threshold(
    mode: str,
    y_val: np.ndarray,
    s_val: np.ndarray,
    quantile: float,
) -> float:

    if mode == "f1":
        thr, _ = fast_choose_threshold_f1(y_val, s_val)
        return thr

    normal_scores = s_val[(y_val == 0) & np.isfinite(s_val)]

    if len(normal_scores) == 0:
        raise ValueError("검증 데이터에 정상 샘플이 없습니다.")

    return float(np.quantile(normal_scores, quantile))


# ============================================================
# Fold 배정
# ============================================================

def build_fold_map(
    df: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> dict:

    rng = np.random.default_rng(seed)

    fold_of = {}

    for kind in FOLD_SOURCE_ORDER:

        groups = np.array(
            sorted(
                df.loc[
                    df["kind"] == kind,
                    "group_id",
                ].unique()
            )
        )

        rng.shuffle(groups)

        for i, group_id in enumerate(groups):
            fold_of[group_id] = i % n_splits

    return fold_of


# ============================================================
# 알람 -> 이벤트 / 오탐 집계
# ============================================================

def count_episodes(alarm: np.ndarray) -> int:
    prev = np.r_[False, alarm[:-1]]
    return int((alarm & ~prev).sum())


def evaluate_fold(
    alarm: np.ndarray,
    df: pd.DataFrame,
    test_groups: list[str],
    group_index: dict,
    dt: float,
    acc: dict,
    events: list,
) -> None:

    for gid in test_groups:

        idx = group_index[gid]

        kind = df["kind"].iat[idx[0]]

        a = alarm[idx]
        y = df["label"].to_numpy()[idx]

        if kind == "idle":

            n_ep = count_episodes(a)

            acc["idle_cycles"] += 1
            acc["idle_fa_cycles"] += int(n_ep > 0)
            acc["idle_fa_episodes"] += n_ep
            acc["idle_samples"] += len(idx)
            acc["idle_alarm_samples"] += int(a.sum())

            continue

        acc["tp"] += int(((y == 1) & a).sum())
        acc["fp"] += int(((y == 0) & a).sum())
        acc["tn"] += int(((y == 0) & ~a).sum())
        acc["fn"] += int(((y == 1) & ~a).sum())

        if kind == "fault":

            t = df["t_sec"].to_numpy()[idx]

            hit = np.flatnonzero(a)

            events.append({
                "group_id": gid,
                "n_rows": len(idx),
                "duration_sec": float(t[-1] - t[0]),
                "detected": bool(len(hit) > 0),
                "samples_to_alarm": (
                    int(hit[0] + 1) if len(hit) else np.nan
                ),
                "delay_sec": (
                    float(t[hit[0]] - t[0]) if len(hit) else np.nan
                ),
            })

        else:

            n_ep = count_episodes(a)

            acc["cycles"] += 1
            acc["fa_cycles"] += int(n_ep > 0)
            acc["fa_episodes"] += n_ep
            acc["exposure_sec"] += len(idx) * dt


def empty_acc() -> dict:
    return {
        k: 0.0
        for k in [
            "tp", "fp", "tn", "fn",
            "cycles", "fa_cycles", "fa_episodes", "exposure_sec",
            "idle_cycles", "idle_fa_cycles", "idle_fa_episodes",
            "idle_samples", "idle_alarm_samples",
        ]
    }


def safe_div(a, b) -> float:
    return float(a / b) if b else float("nan")


def make_row(
    acc: dict,
    events_df: pd.DataFrame,
    compare_min_duration: float,
) -> dict:

    det = events_df["detected"].astype(bool)
    ev_d = events_df[det]

    long_ev = events_df[
        events_df["duration_sec"] >= compare_min_duration - 1e-9
    ]

    tp, fp, tn, fn = acc["tp"], acc["fp"], acc["tn"], acc["fn"]

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision
        and recall == recall
        and (precision + recall) > 0
        else 0.0
    )

    hours = acc["exposure_sec"] / 3600.0

    return {
        "n_events": int(len(events_df)),
        "event_detection_rate": float(det.mean()),
        "n_events_long": int(len(long_ev)),
        "event_detection_rate_long": (
            float(long_ev["detected"].astype(bool).mean())
            if len(long_ev)
            else float("nan")
        ),
        "delay_mean_sec": (
            float(ev_d["delay_sec"].mean())
            if len(ev_d)
            else float("nan")
        ),
        "delay_median_sec": (
            float(ev_d["delay_sec"].median())
            if len(ev_d)
            else float("nan")
        ),
        "samples_to_alarm_mean": (
            float(ev_d["samples_to_alarm"].mean())
            if len(ev_d)
            else float("nan")
        ),
        "sample_f1": float(f1),
        "sample_precision": precision,
        "sample_recall": recall,
        "normal_sample_fpr": safe_div(fp, fp + tn),
        "normal_cycles": int(acc["cycles"]),
        "fa_cycles": int(acc["fa_cycles"]),
        "fa_cycle_rate": safe_div(acc["fa_cycles"], acc["cycles"]),
        "fa_episodes": int(acc["fa_episodes"]),
        "fa_per_hour_observed": safe_div(acc["fa_episodes"], hours),
        "idle_cycles": int(acc["idle_cycles"]),
        "idle_fa_cycle_rate": safe_div(
            acc["idle_fa_cycles"], acc["idle_cycles"]
        ),
        "idle_sample_fpr": safe_div(
            acc["idle_alarm_samples"], acc["idle_samples"]
        ),
    }


# ============================================================
# 설정 1개 평가
# ============================================================

def evaluate_config(
    df: pd.DataFrame,
    x_all: np.ndarray,
    group_index: dict,
    group_positions_all: list[np.ndarray],
    dt: float,
    args,
    agg_k: int,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    min_periods = (
        args.min_periods
        if args.min_periods is not None
        else agg_k
    )

    min_periods = min(min_periods, agg_k)

    config = (
        f"{args.feature_set}|{args.agg}{agg_k}"
        f"(mp={min_periods})"
    )

    group_fold_cache = None

    kind = df["kind"].to_numpy()
    label = df["label"].to_numpy()
    group_arr = df["group_id"].to_numpy()

    rows = []
    event_frames = []

    for r in range(args.repeats):

        seed = args.seed + r

        fold_map = build_fold_map(df, args.n_splits, seed)

        fold_arr = np.array(
            [fold_map[g] for g in group_arr]
        )

        acc = empty_acc()
        events: list = []

        for i in range(args.n_splits):

            val_fold = (i + 1) % args.n_splits

            train_mask = (
                (kind == "normal")
                & (fold_arr != i)
                & (fold_arr != val_fold)
            )

            val_mask = (
                (kind != "idle")
                & (fold_arr == val_fold)
            )

            score_fn = fit_mahalanobis(
                x_all[train_mask],
                args.covariance,
            )

            raw = score_fn(x_all)

            smooth = causal_smooth(
                raw,
                group_positions_all,
                agg_k,
                min_periods,
                args.agg,
            )

            thr = choose_threshold(
                mode,
                label[val_mask],
                smooth[val_mask],
                args.normal_quantile,
            )

            alarm = np.nan_to_num(smooth, nan=-np.inf) >= thr

            test_groups = sorted(
                g for g, f in fold_map.items() if f == i
            )

            evaluate_fold(
                alarm,
                df,
                test_groups,
                group_index,
                dt,
                acc,
                events,
            )

        events_df = pd.DataFrame(events)

        row = make_row(
            acc,
            events_df,
            args.compare_min_duration,
        )

        row.update({
            "config": config,
            "threshold_mode": mode,
            "repeat": r,
            "seed": seed,
        })

        events_df["config"] = config
        events_df["threshold_mode"] = mode
        events_df["repeat"] = r

        rows.append(row)
        event_frames.append(events_df)

        print(
            f"  [{config} | {mode}] repeat {r + 1}/{args.repeats} "
            f"events={int(events_df['detected'].sum())}/{len(events_df)} "
            f"delay={row['delay_mean_sec']:.3f}s "
            f"FAcycle={row['fa_cycle_rate'] * 100:.1f}% "
            f"idleFAcycle={row['idle_fa_cycle_rate'] * 100:.1f}%"
        )

    return (
        pd.DataFrame(rows),
        pd.concat(event_frames, ignore_index=True),
    )


# ============================================================
# 집계 / 출력
# ============================================================

SUMMARY_COLS = [
    "event_detection_rate",
    "event_detection_rate_long",
    "delay_mean_sec",
    "samples_to_alarm_mean",
    "sample_f1",
    "sample_precision",
    "sample_recall",
    "fa_episodes",
    "fa_cycle_rate",
    "fa_per_hour_observed",
    "idle_fa_cycle_rate",
    "idle_sample_fpr",
]


def summarize(repeat_df: pd.DataFrame) -> pd.DataFrame:

    out = []

    for (config, mode), g in repeat_df.groupby(
        ["config", "threshold_mode"],
        sort=False,
    ):

        first = g.iloc[0]

        row = {
            "config": config,
            "threshold_mode": mode,
            "n_events": int(first["n_events"]),
            "n_events_long": int(first["n_events_long"]),
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


def print_summary(summary: pd.DataFrame, compare_min: float) -> None:

    def fmt(r, col, scale=1.0, nd=3):

        m = r[f"{col}_mean"]

        if pd.isna(m):
            return "-"

        return f"{m * scale:.{nd}f}±{r[f'{col}_std'] * scale:.{nd}f}"

    print()
    print("=" * 112)
    print("[표 1] 탐지 성능  (mean ± std over repeats)")
    print("=" * 112)

    t1 = pd.DataFrame({
        "config": summary["config"],
        "mode": summary["threshold_mode"],
        f"event%(all {summary['n_events'].iloc[0]})": summary.apply(
            lambda r: fmt(r, "event_detection_rate", 100, 1), axis=1
        ),
        f"event%(>={compare_min:.1f}s, {summary['n_events_long'].iloc[0]})": summary.apply(
            lambda r: fmt(r, "event_detection_rate_long", 100, 1), axis=1
        ),
        "delay(s)": summary.apply(
            lambda r: fmt(r, "delay_mean_sec"), axis=1
        ),
        "samples": summary.apply(
            lambda r: fmt(r, "samples_to_alarm_mean", 1, 1), axis=1
        ),
        "F1(sample)": summary.apply(
            lambda r: fmt(r, "sample_f1"), axis=1
        ),
        "prec": summary.apply(
            lambda r: fmt(r, "sample_precision"), axis=1
        ),
        "recall": summary.apply(
            lambda r: fmt(r, "sample_recall"), axis=1
        ),
    })

    print(t1.to_string(index=False))

    print()
    print("=" * 112)
    print("[표 2] 오탐  (정상 사이클 / Idle 사이클)")
    print("=" * 112)

    t2 = pd.DataFrame({
        "config": summary["config"],
        "mode": summary["threshold_mode"],
        "normal cycles": summary["normal_cycles"],
        "FA episodes": summary.apply(
            lambda r: fmt(r, "fa_episodes", 1, 1), axis=1
        ),
        "FA cycle%": summary.apply(
            lambda r: fmt(r, "fa_cycle_rate", 100, 2), axis=1
        ),
        "FA/h(obs)": summary.apply(
            lambda r: fmt(r, "fa_per_hour_observed", 1, 1), axis=1
        ),
        "idle cycles": summary["idle_cycles"],
        "idle FA cycle%": summary.apply(
            lambda r: fmt(r, "idle_fa_cycle_rate", 100, 2), axis=1
        ),
    })

    print(t2.to_string(index=False))

    print()
    print("* delay(s)  : 세그먼트 첫 샘플 -> 알람을 만든 샘플의 시간 (0.1s 단위)")
    print("* samples   : 알람까지 필요했던 샘플 수 (윈도우 모델의 W=1.0 은 10개에 해당)")
    print("* F1(sample): 샘플 단위 지표라 6번의 윈도우 단위 F1 과 직접 비교할 수 없음")
    print("* 비교 기준 : event%(>=X s) 는 윈도우 모델(W=1.0)이 평가하던 이벤트와 같은 집합")


# ============================================================
# Main
# ============================================================

def main() -> None:

    p = argparse.ArgumentParser(
        description="샘플 단위 Mahalanobis + causal 평활 + Group K-Fold"
    )

    p.add_argument(
        "--normal-path",
        default="data/press_data_normal_with_idle.csv",
    )

    p.add_argument(
        "--fault-path",
        default="data/outlier_data.csv",
    )

    p.add_argument(
        "--output-dir",
        default="outputs/5_3_sample_level_cv",
    )

    p.add_argument(
        "--feature-set",
        choices=["raw", "raw_diff", "raw_diff_roll3"],
        default="raw_diff",
    )

    p.add_argument(
        "--agg",
        choices=["mean", "max"],
        default="mean",
        help="마지막 k 샘플 점수의 집계 방식",
    )

    p.add_argument(
        "--agg-k",
        type=int,
        nargs="+",
        default=[3],
        help="평활 길이(샘플 수). 여러 개면 각각 비교",
    )

    p.add_argument(
        "--min-periods",
        type=int,
        default=None,
        help="점수를 내기 위한 최소 샘플 수 (기본값: agg-k)",
    )

    p.add_argument(
        "--no-log",
        action="store_true",
        help="signed log1p 변환 끄기",
    )

    p.add_argument("--n-splits", type=int, default=4)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--covariance",
        choices=["empirical", "ledoitwolf", "oas"],
        default="ledoitwolf",
    )

    p.add_argument(
        "--threshold-modes",
        nargs="+",
        choices=["f1", "normal_quantile"],
        default=["f1", "normal_quantile"],
    )

    p.add_argument(
        "--normal-quantile",
        type=float,
        default=0.999,
        help="샘플 단위이므로 윈도우 모델(0.995)보다 높게 설정",
    )

    p.add_argument(
        "--compare-min-duration",
        type=float,
        default=1.0,
        help="윈도우 모델(W=1.0)과 같은 이벤트 집합으로 비교하기 위한 길이 기준",
    )

    args = p.parse_args()

    if args.n_splits < 3:
        raise ValueError("--n-splits 는 3 이상이어야 합니다.")

    # ---- 데이터 로드 ----
    normal = load_raw(args.normal_path, "normal")
    fault = load_raw(args.fault_path, "fault")

    df = pd.concat([normal, fault], ignore_index=True)

    df["t_sec"] = (
        df[TIME_COL] - df[TIME_COL].min()
    ).dt.total_seconds()

    n_before = len(df)

    df = df.dropna(subset=SENSOR_COLS).reset_index(drop=True)

    if len(df) < n_before:
        print(f"[INFO] 센서 NaN 행 제거: {n_before - len(df):,}")

    x_all, feat_names = build_features(df, args.feature_set)

    if not args.no_log:
        x_all = signed_log1p(x_all)

    group_index = df.groupby("group_id").indices

    group_positions_all = list(group_index.values())

    diffs = (
        df.groupby("group_id")[TIME_COL]
        .diff()
        .dt.total_seconds()
        .dropna()
    )

    dt = float(diffs[diffs > 0].median())

    print("=" * 80)
    print("데이터 요약")
    print("=" * 80)
    print(f"feature set : {args.feature_set} ({len(feat_names)}개) {feat_names}")
    print(f"샘플 간격   : {dt:.3f}s | signed_log1p: {not args.no_log}")

    summary_tbl = (
        df.groupby("kind")
        .agg(
            segments=("group_id", "nunique"),
            samples=("group_id", "size"),
        )
    )

    print(summary_tbl.to_string())

    n_fault = int((summary_tbl.loc["fault", "segments"]))

    if n_fault < args.n_splits:
        raise ValueError("fault 이벤트가 fold 수보다 적습니다.")

    # ---- 평가 ----
    repeat_frames = []
    event_frames = []

    print()

    for agg_k in args.agg_k:

        for mode in args.threshold_modes:

            rep, ev = evaluate_config(
                df,
                x_all,
                group_index,
                group_positions_all,
                dt,
                args,
                agg_k,
                mode,
            )

            repeat_frames.append(rep)
            event_frames.append(ev)

    repeat_all = pd.concat(repeat_frames, ignore_index=True)
    events_all = pd.concat(event_frames, ignore_index=True)

    summary = summarize(repeat_all)

    freq = (
        events_all
        .groupby(["config", "threshold_mode", "group_id"], sort=False)
        .agg(
            detect_rate=("detected", "mean"),
            delay_mean_sec=("delay_sec", "mean"),
            n_rows=("n_rows", "first"),
            duration_sec=("duration_sec", "first"),
        )
        .reset_index()
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repeat_all.to_csv(out_dir / "sample_cv_repeat_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "sample_cv_comparison.csv", index=False, encoding="utf-8-sig")
    events_all.to_csv(out_dir / "sample_cv_event_details.csv", index=False, encoding="utf-8-sig")
    freq.to_csv(out_dir / "sample_cv_event_detection_frequency.csv", index=False, encoding="utf-8-sig")

    print_summary(summary, args.compare_min_duration)

    missed = freq[freq["detect_rate"] < 1.0]

    print()
    print("=" * 112)
    print("탐지율 100% 미만인 이벤트")
    print("=" * 112)

    if missed.empty:
        print("없음 (모든 설정에서 모든 이벤트를 탐지)")
    else:
        print(
            missed[
                [
                    "config", "threshold_mode", "group_id",
                    "n_rows", "duration_sec", "detect_rate",
                ]
            ].to_string(index=False)
        )

    print()
    print("Saved:")

    for name in [
        "sample_cv_comparison.csv",
        "sample_cv_repeat_results.csv",
        "sample_cv_event_details.csv",
        "sample_cv_event_detection_frequency.csv",
    ]:
        print(out_dir / name)


if __name__ == "__main__":
    main()