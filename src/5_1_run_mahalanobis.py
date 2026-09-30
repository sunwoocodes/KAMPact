"""
Mahalanobis 이벤트 단위 교차검증 (Group K-Fold) + 오탐 에피소드 집계

5_run_mahalanobis.py 의 함수(run_once, 이벤트 평가 등)를 그대로 재사용하고,
아래 세 가지만 추가한다.

1) Group K-Fold 교차검증
   - group_id(세그먼트) 단위로 fold를 나눈다. fault 세그먼트(=이벤트)와
     normal 세그먼트를 각각 따로 섞어서 배분하므로 모든 fold에 이벤트가 들어간다.
   - fold i 를 test, fold (i+1) % K 를 val, 나머지를 train 으로 사용한다.
   - 이벤트 16개가 모두 정확히 한 번씩 test 에 들어간다.
   - --repeats 로 fold 배정을 여러 번 바꿔가며 반복 -> 평균 / 표준편차 산출.

2) 오탐 에피소드 집계
   - 연속된 오탐 윈도우를 1건으로 묶어 "정상 운전 1시간당 오탐 횟수"를 계산한다.
   - step=0.1s 일 때 윈도우가 90% 겹쳐서 윈도우 단위 FA율이 부풀려지는 문제를 보정.
   - fault 세그먼트에서도 onset 이전 구간의 알람은 오탐으로 센다.

3) delay - W
   - delay 는 정의상 W 이상이 되기 쉬우므로 W 를 뺀 값도 함께 보고한다.
   - alarm 윈도우가 onset 이전에 시작(transition window)했는지도 기록한다.

사용 예:
    python 6_run_mahalanobis_cv.py
    python 6_run_mahalanobis_cv.py --inputs result/modeling_dataset_0.9_0.1/model_windows.csv
    python 6_run_mahalanobis_cv.py --n-splits 4 --repeats 10 --threshold-modes f1
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 기본 설정
# ============================================================

DEFAULT_INPUTS = [
    "result/modeling_dataset_1.0_0.5/model_windows.csv",
    "result/modeling_dataset_1.0_0.1/model_windows.csv",
    "result/modeling_dataset_0.9_0.1/model_windows.csv",
    "result/modeling_dataset_0.8_0.1/model_windows.csv",
]

ALARM_COL = "alarm_k_consecutive"


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
# 빠른 F1 threshold (5번의 choose_threshold_f1 과 동일한 규칙)
#
#   1. F1 최대
#   2. 동률이면 precision 최대
#   3. 그래도 동률이면 더 낮은 threshold
#
# 원본은 unique score 마다 sklearn f1_score 를 호출해서 느리므로
# 누적합으로 한 번에 계산한다. (CV 로 수백 번 호출하기 때문)
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

    # 같은 score 는 한 덩어리로 (score >= threshold 이므로 동점은 함께 양성)
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
# Fold 배정
# ============================================================

def assign_folds(
    df: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> pd.Series:
    """
    source(normal / fault)별로 group_id 를 섞어서 fold 번호를 부여한다.
    같은 group_id 의 모든 윈도우는 같은 fold 에 들어간다. (leakage 방지)
    """

    rng = np.random.default_rng(seed)

    fold_of = {}

    for source in sorted(df["source"].unique()):

        groups = np.array(
            sorted(
                df.loc[
                    df["source"] == source,
                    "group_id",
                ].unique()
            )
        )

        rng.shuffle(groups)

        for i, group_id in enumerate(groups):
            fold_of[group_id] = i % n_splits

    return df["group_id"].map(fold_of)


# ============================================================
# 오탐 에피소드 / 정상 노출 시간
# ============================================================

def false_alarm_episodes(
    test_df: pd.DataFrame,
    alarm_col: str,
    step_sec: float,
) -> tuple[int, float]:
    """
    Returns
    -------
    n_episodes : 연속된 오탐 윈도우를 1건으로 센 오탐 횟수
    exposure_sec : 정상 구간이 실제로 관측된 시간 (윈도우 구간의 합집합)

    정상 구간의 정의:
        - normal 세그먼트 전체
        - fault 세그먼트에서 window_end < fault_onset_time 인 윈도우
          (window_end >= onset 이면 고장 구간/전이 구간이므로 오탐이 아님)
    """

    work = test_df.copy()

    in_fault_zone = (
        work["source"].eq("fault")
        & work["fault_onset_time"].notna()
        & (work["window_end"] >= work["fault_onset_time"])
    )

    normal_zone = work.loc[~in_fault_zone].copy()

    if normal_zone.empty:
        return 0, 0.0

    normal_zone = normal_zone.sort_values(
        ["group_id", "window_start"]
    )

    n_episodes = 0
    exposure_sec = 0.0

    for _, g in normal_zone.groupby("group_id", sort=False):

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

        # 인접 윈도우인지 (시간 간격이 step 이내)
        contiguous = np.r_[
            False,
            np.diff(start) <= step_sec * 1.5,
        ]

        prev_alarm = np.r_[False, alarm[:-1]]

        new_episode = alarm & ~(prev_alarm & contiguous)

        n_episodes += int(new_episode.sum())

        # 윈도우 구간 합집합 길이
        prev_end = np.r_[
            -np.inf,
            np.maximum.accumulate(end)[:-1],
        ]

        covered = np.clip(
            end - np.maximum(start, prev_end),
            0,
            None,
        ).sum()

        exposure_sec += float(covered)

    return n_episodes, exposure_sec


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


# ============================================================
# 1회 반복 (K fold 전체)
# ============================================================

def run_cv_repeat(
    mod,
    df: pd.DataFrame,
    feature_cols: list[str],
    covariance: str,
    threshold_mode: str,
    normal_quantile: float,
    k_consecutive: int,
    n_splits: int,
    seed: int,
    step_sec: float,
    window_sec: float,
) -> tuple[dict, pd.DataFrame]:

    fold = assign_folds(df, n_splits, seed)

    event_frames = []
    fold_f1 = []
    thresholds = []

    tp = fp = tn = fn = 0
    fa_episodes = 0
    exposure_sec = 0.0

    for i in range(n_splits):

        work = df.copy()

        work["split"] = np.where(
            fold == i,
            "test",
            np.where(
                fold == (i + 1) % n_splits,
                "val",
                "train",
            ),
        )

        metrics, test_work, event_details = mod.run_once(
            df=work,
            feature_cols=feature_cols,
            covariance=covariance,
            threshold_mode=threshold_mode,
            normal_quantile=normal_quantile,
            k_consecutive=k_consecutive,
            train_seed=None,
            train_fraction=1.0,
        )

        tp += metrics["tp"]
        fp += metrics["fp"]
        tn += metrics["tn"]
        fn += metrics["fn"]

        fold_f1.append(metrics["f1"])
        thresholds.append(metrics["threshold"])

        n_ep, exp_sec = false_alarm_episodes(
            test_work,
            ALARM_COL,
            step_sec,
        )

        fa_episodes += n_ep
        exposure_sec += exp_sec

        ev = event_details.copy()

        ev["fold"] = i
        ev["threshold"] = metrics["threshold"]

        # alarm 이 onset 이전에 시작한 윈도우(transition)에서 났는지
        ev["transition_detect"] = (
            ev["detected"].astype(bool)
            & (
                pd.to_datetime(ev["alarm_window_start"])
                < pd.to_datetime(ev["fault_onset_time"])
            )
        )

        event_frames.append(ev)

    events = pd.concat(
        event_frames,
        ignore_index=True,
    )

    detected = events["detected"].astype(bool)

    delays = events.loc[
        detected,
        "detection_delay_sec",
    ].dropna().to_numpy(dtype=float)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    exposure_hours = exposure_sec / 3600.0

    row = {
        "seed": seed,
        "n_events": int(len(events)),
        "detected_events": int(detected.sum()),
        "event_detection_rate": float(detected.mean()),
        "delay_mean_sec": (
            float(delays.mean()) if len(delays) else np.nan
        ),
        "delay_median_sec": (
            float(np.median(delays)) if len(delays) else np.nan
        ),
        "delay_minus_W_mean_sec": (
            float(delays.mean() - window_sec)
            if len(delays)
            else np.nan
        ),
        "transition_detect_rate": float(
            events["transition_detect"].mean()
        ),
        "pooled_f1": float(f1),
        "pooled_precision": float(precision),
        "pooled_recall": float(recall),
        "fold_f1_mean": float(np.mean(fold_f1)),
        "normal_window_fpr": (
            float(fp / (fp + tn)) if (fp + tn) else np.nan
        ),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "fa_episodes": int(fa_episodes),
        "normal_exposure_min": float(exposure_sec / 60.0),
        "fa_per_hour": (
            float(fa_episodes / exposure_hours)
            if exposure_hours > 0
            else np.nan
        ),
        "threshold_mean": float(np.mean(thresholds)),
        "threshold_std": float(np.std(thresholds, ddof=0)),
    }

    return row, events


# ============================================================
# 데이터 1개 x threshold mode 1개
# ============================================================

def evaluate_config(
    mod,
    input_path: Path,
    config_name: str,
    args,
    threshold_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = pd.read_csv(input_path)

    missing = [
        c
        for c in mod.REQUIRED_META + ["source"]
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{input_path}: 필수 컬럼 누락 {missing}"
        )

    feature_cols = mod.DEFAULT_FEATURES.copy()

    df = mod.prepare_data(df, feature_cols)

    df = df[df["label"].isin([0, 1])].copy()

    n_fault_groups = df.loc[
        df["source"] == "fault",
        "group_id",
    ].nunique()

    if n_fault_groups < args.n_splits:
        raise ValueError(
            f"fault 이벤트({n_fault_groups}개)가 "
            f"--n-splits({args.n_splits})보다 적습니다."
        )

    step_sec, window_sec = infer_step_and_window(df)

    rows = []
    event_frames = []

    for r in range(args.repeats):

        seed = args.seed + r

        row, events = run_cv_repeat(
            mod=mod,
            df=df,
            feature_cols=feature_cols,
            covariance=args.covariance,
            threshold_mode=threshold_mode,
            normal_quantile=args.normal_quantile,
            k_consecutive=args.k_consecutive,
            n_splits=args.n_splits,
            seed=seed,
            step_sec=step_sec,
            window_sec=window_sec,
        )

        row.update({
            "config": config_name,
            "threshold_mode": threshold_mode,
            "repeat": r,
            "W_sec": window_sec,
            "step_sec": step_sec,
        })

        events["config"] = config_name
        events["threshold_mode"] = threshold_mode
        events["repeat"] = r

        rows.append(row)
        event_frames.append(events)

        print(
            f"  [{config_name} | {threshold_mode}] "
            f"repeat {r + 1}/{args.repeats} "
            f"event={row['event_detection_rate'] * 100:5.1f}% "
            f"delay={row['delay_mean_sec']:.3f}s "
            f"F1={row['pooled_f1']:.4f} "
            f"FA/h={row['fa_per_hour']:.1f}"
        )

    return (
        pd.DataFrame(rows),
        pd.concat(event_frames, ignore_index=True),
    )


# ============================================================
# 집계
# ============================================================

SUMMARY_COLS = [
    "event_detection_rate",
    "delay_mean_sec",
    "delay_minus_W_mean_sec",
    "transition_detect_rate",
    "pooled_f1",
    "pooled_precision",
    "pooled_recall",
    "normal_window_fpr",
    "fa_episodes",
    "fa_per_hour",
]


def summarize(repeat_df: pd.DataFrame) -> pd.DataFrame:

    out = []

    for (config, mode), g in repeat_df.groupby(
        ["config", "threshold_mode"],
        sort=False,
    ):

        row = {
            "config": config,
            "threshold_mode": mode,
            "W_sec": float(g["W_sec"].iloc[0]),
            "step_sec": float(g["step_sec"].iloc[0]),
            "n_events": int(g["n_events"].iloc[0]),
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

    print()
    print("=" * 100)
    print("GROUP K-FOLD 비교 (mean ± std over repeats)")
    print("=" * 100)

    def fmt(row, col, scale=1.0, nd=3):
        return (
            f"{row[f'{col}_mean'] * scale:.{nd}f}"
            f"±{row[f'{col}_std'] * scale:.{nd}f}"
        )

    table = pd.DataFrame({
        "config": summary["config"],
        "mode": summary["threshold_mode"],
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
        "FA episodes": summary.apply(
            lambda r: fmt(r, "fa_episodes", 1, 1),
            axis=1,
        ),
        "FA/hour": summary.apply(
            lambda r: fmt(r, "fa_per_hour", 1, 1),
            axis=1,
        ),
    })

    print(table.to_string(index=False))

    print()
    print("* FA episodes : 연속된 오탐 윈도우를 1건으로 묶은 횟수 (전체 이벤트/정상 구간 합계)")
    print("* FA/hour     : 정상 구간 관측 1시간당 오탐 에피소드 수")
    print("* delay-W     : 탐지 지연에서 윈도우 길이를 뺀 값")


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Mahalanobis Group K-Fold CV "
            "+ false alarm episode 집계"
        )
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="model_windows.csv 경로들 (여러 개 가능)",
    )

    parser.add_argument(
        "--base-script",
        default=str(
            Path(__file__).with_name("5_run_mahalanobis.py")
        ),
        help="5_run_mahalanobis.py 경로",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/mahalanobis_cv",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=4,
        help="fold 수 (이벤트 16개 기준 4 권장)",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="fold 배정을 바꿔가며 반복하는 횟수",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

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
        raise ValueError(
            "--n-splits 는 3 이상이어야 합니다. (train/val/test 분리)"
        )

    if args.repeats < 1:
        raise ValueError("--repeats 는 1 이상이어야 합니다.")

    mod = load_base_module(Path(args.base_script))

    # 느린 원본 threshold 함수를 동일 규칙의 빠른 버전으로 교체
    mod.choose_threshold_f1 = fast_choose_threshold_f1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repeat_frames = []
    event_frames = []

    for input_str in args.inputs:

        input_path = Path(input_str)

        if not input_path.exists():
            print(f"[SKIP] 파일 없음: {input_path}")
            continue

        config_name = input_path.parent.name.replace(
            "modeling_dataset_",
            "",
        )

        print()
        print(f"[CONFIG] {config_name}  ({input_path})")

        for mode in args.threshold_modes:

            repeat_df, event_df = evaluate_config(
                mod=mod,
                input_path=input_path,
                config_name=config_name,
                args=args,
                threshold_mode=mode,
            )

            repeat_frames.append(repeat_df)
            event_frames.append(event_df)

    if not repeat_frames:
        raise RuntimeError("평가된 입력 파일이 없습니다.")

    repeat_all = pd.concat(repeat_frames, ignore_index=True)
    events_all = pd.concat(event_frames, ignore_index=True)

    summary = summarize(repeat_all)

    # 이벤트별 탐지 빈도 (반복 전체에서 몇 %나 탐지됐는가)
    event_freq = (
        events_all
        .assign(detected=events_all["detected"].astype(bool))
        .groupby(
            ["config", "threshold_mode", "group_id"],
            sort=False,
        )
        .agg(
            detect_rate=("detected", "mean"),
            delay_mean_sec=("detection_delay_sec", "mean"),
            transition_detect_rate=("transition_detect", "mean"),
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

    # 항상 놓치는 / 가끔 놓치는 이벤트
    missed = event_freq[event_freq["detect_rate"] < 1.0]

    print()
    print("=" * 100)
    print("탐지율 100% 미만인 이벤트")
    print("=" * 100)

    if missed.empty:
        print("없음 (모든 설정에서 모든 이벤트를 탐지)")
    else:
        print(
            missed[
                [
                    "config",
                    "threshold_mode",
                    "group_id",
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