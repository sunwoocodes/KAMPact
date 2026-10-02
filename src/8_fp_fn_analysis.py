"""
KAMPact - FP / FN Condition Analysis
=====================================

3단계: False Positive / False Negative 원인 분석

분석 대상
----------
Final Ensemble:
    OR_3 + Persistence P2
    (= OR_3_P2)

[FP]
정상 운전 중 최종 OR_3_P2 alarm이 발생한 구간의
window feature를 정상 전체와 비교한다.

[FN]
fault event별
    - 지속시간
    - sample 수
    - 1.0s window 생성 수
    - 0.5s window 생성 수
    - repeat별 탐지 횟수
    - repeat별 미탐 횟수
    - 탐지율
을 분석한다.

출력
----
outputs/8_fp_fn_analysis/
    fp_windows_1.0s.csv
    fp_feature_summary.csv
    source_detector_fp_windows_1.0s.csv
    fp_window_audit.csv

    fn_event_summary.csv
    fn_duration_summary.csv
    fn_event_repeat_details.csv

    analysis_report.txt

실행 예
--------
python src/8_fp_fn_analysis.py --normal-path data/press_data_normal_with_idle.csv --fault-path data/outlier_data.csv --window-1.0 result/modeling_dataset_1.0_0.1 --window-0.5 result/modeling_dataset_0.5_0.1 --n-splits 5 --repeats 5 --seed 0 --window-threshold-mode normal_quantile --window-quantile 0.9999 --window-k 2 --window-covariance ledoitwolf --sample-threshold-mode normal_quantile --sample-quantile 0.9999 --sample-agg-k 3 --sample-agg mean --sample-feature-set raw_diff --sample-covariance ledoitwolf --output-dir outputs/8_fp_fn_analysis
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

TIME_COL = "TimeStamp"
STATE_COL = "Equipment_state"
IDLE_COL = "Idle"

SENSOR_COLS = [
    "AI0_Vibration",
    "AI1_Vibration",
    "AI2_Current",
]

GAP_THRESHOLD_SEC = 0.5

FOLD_SOURCE_ORDER = [
    "fault",
    "normal",
    "idle",
]


# ============================================================
# 모듈 로드
# ============================================================

def load_module(
    path: Path,
    name: str,
):
    if not path.exists():
        raise FileNotFoundError(
            f"스크립트를 찾을 수 없습니다: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"모듈 로드 실패: {path}"
        )

    module = importlib.util.module_from_spec(spec)

    # 중요:
    # exec_module() 전에 sys.modules에 등록해야
    # @dataclass 등에서 module.__module__을
    # 정상적으로 찾을 수 있다.
    import sys

    sys.modules[name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise

    return module


# ============================================================
# Raw 데이터 로드
# ============================================================

def load_raw(
    path: Path,
    source: str,
) -> pd.DataFrame:

    df = pd.read_csv(path)

    required = [
        TIME_COL,
        STATE_COL,
        *SENSOR_COLS,
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path}: 필수 컬럼 누락: {missing}"
        )

    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL],
        errors="coerce",
    )

    df = (
        df
        .dropna(subset=[TIME_COL])
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )

    for c in [
        *SENSOR_COLS,
        STATE_COL,
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    if IDLE_COL in df.columns:

        df[IDLE_COL] = (
            pd.to_numeric(
                df[IDLE_COL],
                errors="coerce",
            )
            .fillna(0)
        )

    else:

        df[IDLE_COL] = 0

    # --------------------------------------------------------
    # Segment
    # --------------------------------------------------------

    gap = (
        df[TIME_COL]
        .diff()
        .dt.total_seconds()
    )

    segment_break = gap > GAP_THRESHOLD_SEC

    segment_break.iloc[0] = True

    df["segment_id"] = (
        segment_break
        .cumsum()
        - 1
    )

    df["source"] = source

    df["group_id"] = (
        source
        + "_"
        + df["segment_id"].astype(str)
    )

    # --------------------------------------------------------
    # Idle / Normal / Fault
    # --------------------------------------------------------

    idle_ratio = (
        (df[IDLE_COL] == 1)
        .groupby(df["group_id"])
        .transform("mean")
    )

    df["kind"] = np.where(
        source == "fault",
        "fault",
        np.where(
            idle_ratio >= 0.5,
            "idle",
            "normal",
        ),
    )

    df["label"] = (
        pd.to_numeric(
            df[STATE_COL],
            errors="coerce",
        )
        .fillna(0)
        .ge(1)
        .astype(int)
    )

    return df


# ============================================================
# Fold map
# 6_compare_three_detectors.py와 동일한 방식
# ============================================================

def build_fold_map(
    raw_df: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> dict[str, int]:

    rng = np.random.default_rng(seed)

    fold_of: dict[str, int] = {}

    groups_by_kind = {
        "fault": set(),
        "normal": set(),
        "idle": set(),
    }

    for kind, group in raw_df.groupby("kind"):

        groups_by_kind.setdefault(
            kind,
            set(),
        ).update(
            group["group_id"]
            .astype(str)
            .unique()
        )

    for kind in FOLD_SOURCE_ORDER:

        groups = np.array(
            sorted(
                groups_by_kind.get(
                    kind,
                    set(),
                )
            )
        )

        rng.shuffle(groups)

        for i, gid in enumerate(groups):

            fold_of[str(gid)] = (
                i % n_splits
            )

    return fold_of


# ============================================================
# Window feature 자동 추론
# ============================================================

META_EXACT = {
    "Unnamed: 0",
    "TimeStamp",
    "window_start",
    "window_end",
    "fault_onset_time",
    "group_id",
    "segment_id",
    "source",
    "kind",
    "split",
    "label",
    "fault_ratio",
    "normal_ratio",
    "idle_ratio",
    "n_samples",
    "start_idx",
    "end_idx",
    "window_id",
    "window_index",
    "duration_sec",
    "Equipment_state",
    "Idle",
}


def infer_window_feature_cols(
    paths: list[Path],
) -> list[str]:

    if not paths:
        raise ValueError(
            "model_windows.csv 경로가 없습니다."
        )

    frames = [
        pd.read_csv(
            path,
            nrows=200,
        )
        for path in paths
    ]

    common = [
        c
        for c in frames[0].columns
        if all(
            c in df.columns
            for df in frames[1:]
        )
    ]

    numeric_common = []

    for c in common:

        if c in META_EXACT:
            continue

        if all(
            pd.api.types.is_numeric_dtype(
                df[c]
            )
            for df in frames
        ):
            numeric_common.append(c)

    preferred = []

    for c in numeric_common:

        cl = c.lower()

        if (
            cl.startswith("ai0")
            or cl.startswith("ai1")
            or cl.startswith("ai2")
            or "corr" in cl
            or "correlation" in cl
        ):
            preferred.append(c)

    if len(preferred) == 16:
        return preferred

    if len(numeric_common) == 16:
        return numeric_common

    if len(preferred) >= 16:
        return preferred[:16]

    raise ValueError(
        "16개 feature를 자동 추론하지 못했습니다.\n"
        f"후보 feature: {numeric_common}"
    )


# ============================================================
# Window alarm -> raw timeline
# Causal mapping
# ============================================================

def apply_window_alarm_to_raw(
    raw: pd.DataFrame,
    test_work: pd.DataFrame,
    alarm_col: str,
) -> np.ndarray:

    alarm_raw = np.zeros(
        len(raw),
        dtype=bool,
    )

    if test_work.empty:
        return alarm_raw

    tw = test_work.copy()

    tw["window_start"] = pd.to_datetime(
        tw["window_start"],
        errors="coerce",
    )

    tw["window_end"] = pd.to_datetime(
        tw["window_end"],
        errors="coerce",
    )

    raw_time = raw[
        TIME_COL
    ].to_numpy(
        dtype="datetime64[ns]"
    )

    raw_group = raw[
        "group_id"
    ].astype(str).to_numpy()

    for _, row in tw.iterrows():

        if not bool(row[alarm_col]):
            continue

        gid = str(
            row["group_id"]
        )

        start = np.datetime64(
            row["window_start"]
        )

        end = np.datetime64(
            row["window_end"]
        )

        duration = (
            end - start
        )

        shifted_start = (
            start + duration
        )

        shifted_end = (
            end + duration
        )

        mask = (
            (raw_group == gid)
            & (
                raw_time
                >= shifted_start
            )
            & (
                raw_time
                < shifted_end
            )
        )

        alarm_raw[mask] = True

    return alarm_raw


# ============================================================
# Window alarm timeline -> FP context window
#
# 최종 OR_3_P2 alarm이 발생한 정상 raw 시간대와
# causal shifted window 구간이 겹치는 window를 찾는다.
# ============================================================

def mark_final_alarm_overlap(
    windows: pd.DataFrame,
    raw: pd.DataFrame,
    final_alarm: np.ndarray,
) -> pd.DataFrame:

    out = windows.copy()

    out["final_or3p2_overlap"] = False

    if out.empty:
        return out

    final_normal = (
        final_alarm
        & raw["kind"]
        .eq("normal")
        .to_numpy()
    )

    raw_time = raw[
        TIME_COL
    ].to_numpy(
        dtype="datetime64[ns]"
    )

    raw_group = raw[
        "group_id"
    ].astype(str).to_numpy()

    normal_alarm_times: dict[
        str,
        np.ndarray,
    ] = {}

    normal_groups = (
        raw.loc[
            final_normal,
            "group_id",
        ]
        .astype(str)
        .unique()
    )

    for gid in normal_groups:

        mask = (
            final_normal
            & (raw_group == gid)
        )

        times = np.sort(
            raw_time[mask]
        )

        normal_alarm_times[
            gid
        ] = times

    flags = np.zeros(
        len(out),
        dtype=bool,
    )

    for i, (_, row) in enumerate(
        out.iterrows()
    ):

        gid = str(
            row["group_id"]
        )

        alarm_times = (
            normal_alarm_times.get(gid)
        )

        if (
            alarm_times is None
            or len(alarm_times) == 0
        ):
            continue

        start = pd.Timestamp(
            row["window_start"]
        )

        end = pd.Timestamp(
            row["window_end"]
        )

        duration = end - start

        shifted_start = (
            start + duration
        )

        shifted_end = (
            end + duration
        )

        lo = np.datetime64(
            shifted_start
        )

        hi = np.datetime64(
            shifted_end
        )

        left = np.searchsorted(
            alarm_times,
            lo,
            side="left",
        )

        right = np.searchsorted(
            alarm_times,
            hi,
            side="left",
        )

        flags[i] = (
            right > left
        )

    out[
        "final_or3p2_overlap"
    ] = flags

    return out


# ============================================================
# Robust FP Feature Summary
# ============================================================

def make_fp_feature_summary(
    normal_df: pd.DataFrame,
    fp_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:

    rows = []

    for feature in feature_cols:

        normal = pd.to_numeric(
            normal_df[feature],
            errors="coerce",
        ).dropna()

        fp = pd.to_numeric(
            fp_df[feature],
            errors="coerce",
        ).dropna()

        if normal.empty:
            continue

        normal_median = float(
            normal.median()
        )

        normal_q25 = float(
            normal.quantile(0.25)
        )

        normal_q75 = float(
            normal.quantile(0.75)
        )

        normal_iqr = (
            normal_q75
            - normal_q25
        )

        if fp.empty:

            fp_median = np.nan
            fp_q25 = np.nan
            fp_q75 = np.nan

        else:

            fp_median = float(
                fp.median()
            )

            fp_q25 = float(
                fp.quantile(0.25)
            )

            fp_q75 = float(
                fp.quantile(0.75)
            )

        median_delta = (
            fp_median
            - normal_median
            if np.isfinite(fp_median)
            else np.nan
        )

        delta_pct = (
            100.0
            * median_delta
            / abs(normal_median)
            if (
                np.isfinite(median_delta)
                and abs(normal_median) > 1e-12
            )
            else np.nan
        )

        # IQR -> robust std 근사
        robust_std = (
            normal_iqr / 1.349
            if normal_iqr > 0
            else np.nan
        )

        robust_effect = (
            median_delta
            / robust_std
            if (
                np.isfinite(median_delta)
                and np.isfinite(robust_std)
                and robust_std > 0
            )
            else np.nan
        )

        if (
            not np.isfinite(
                median_delta
            )
            or abs(median_delta) < 1e-12
        ):
            direction = "≈"

        elif median_delta > 0:
            direction = "↑"

        else:
            direction = "↓"

        rows.append({
            "feature": feature,

            "normal_count": int(
                len(normal)
            ),

            "fp_count": int(
                len(fp)
            ),

            "normal_median": normal_median,
            "normal_q25": normal_q25,
            "normal_q75": normal_q75,
            "normal_iqr": normal_iqr,

            "fp_median": fp_median,
            "fp_q25": fp_q25,
            "fp_q75": fp_q75,

            "median_delta": median_delta,

            "delta_pct_of_abs_normal": (
                delta_pct
            ),

            "robust_effect": (
                robust_effect
            ),

            "direction": direction,
        })

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result["abs_robust_effect"] = (
        result["robust_effect"]
        .abs()
    )

    result = (
        result
        .sort_values(
            [
                "abs_robust_effect",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# FP 자동 해석 문장
# ============================================================

def make_fp_sentence(
    summary: pd.DataFrame,
    top_n: int = 5,
) -> str:

    valid = summary.dropna(
        subset=[
            "robust_effect"
        ]
    ).copy()

    if valid.empty:

        return (
            "FP feature 비교에서 "
            "유효한 차이를 계산하지 못했습니다."
        )

    top = valid.head(
        top_n
    )

    parts = []

    for _, row in top.iterrows():

        pct = row[
            "delta_pct_of_abs_normal"
        ]

        if np.isfinite(pct):

            parts.append(
                f"{row['feature']} "
                f"{row['direction']} "
                f"(정상 중앙값 대비 "
                f"{pct:+.1f}%)"
            )

        else:

            parts.append(
                f"{row['feature']} "
                f"{row['direction']} "
                f"(robust effect="
                f"{row['robust_effect']:+.2f})"
            )

    return (
        "오경보가 발생한 정상 구간은 "
        "다음 feature에서 정상 전체와 "
        "상대적으로 큰 차이를 보였습니다: "
        + ", ".join(parts)
        + "."
    )


# ============================================================
# Duration bin
# ============================================================

def add_duration_bin(
    duration_sec: pd.Series,
) -> pd.Series:

    return pd.cut(
        duration_sec,
        bins=[
            -np.inf,
            0.5,
            1.0,
            np.inf,
        ],
        right=False,
        labels=[
            "<0.5s",
            "0.5-<1.0s",
            ">=1.0s",
        ],
    )


# ============================================================
# FN event summary
# ============================================================

def build_fn_event_summary(
    fault_rows: pd.DataFrame,
    window_audit: pd.DataFrame,
) -> pd.DataFrame:

    base = (
        fault_rows
        .groupby("group_id")
        .agg(
            total_repeats=(
                "detected",
                "count",
            ),

            detected_count=(
                "detected",
                "sum",
            ),

            event_duration_sec=(
                "event_duration_sec",
                "first",
            ),

            n_samples=(
                "n_samples",
                "first",
            ),

            delay_mean_sec=(
                "delay_sec",
                "mean",
            ),

            delay_median_sec=(
                "delay_sec",
                "median",
            ),
        )
        .reset_index()
    )

    base["missed_count"] = (
        base["total_repeats"]
        - base["detected_count"]
    )

    base["detection_rate_pct"] = (
        100.0
        * base["detected_count"]
        / base["total_repeats"]
    )

    base["duration_bin"] = (
        add_duration_bin(
            base[
                "event_duration_sec"
            ]
        )
    )

    # --------------------------------------------------------
    # 실제 fault event 안에서 생성된 window 개수
    # --------------------------------------------------------

    for window_sec in [
        1.0,
        0.5,
    ]:

        tag = (
            f"{window_sec:.1f}"
            .replace(".", "p")
        )

        subset = window_audit[
            window_audit[
                "window_sec"
            ]
            .sub(window_sec)
            .abs()
            .lt(1e-6)
        ]

        # Fault event의 label=1 window만 count
        counts = (
            subset[
                subset["label"].eq(1)
            ]
            .groupby("group_id")
            .size()
            .rename(
                f"n_windows_{tag}s"
            )
        )

        base = base.merge(
            counts,
            on="group_id",
            how="left",
        )

    for c in [
        "n_windows_1p0s",
        "n_windows_0p5s",
    ]:

        if c in base.columns:

            base[c] = (
                base[c]
                .fillna(0)
                .astype(int)
            )

    # --------------------------------------------------------
    # 정렬
    # --------------------------------------------------------

    base = (
        base
        .sort_values(
            [
                "missed_count",
                "event_duration_sec",
                "group_id",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    return base


# ============================================================
# Duration별 탐지율
# ============================================================

def build_duration_summary(
    event_summary: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        event_summary
        .groupby(
            "duration_bin",
            observed=False,
        )
        .agg(
            n_events=(
                "group_id",
                "count",
            ),

            mean_duration_sec=(
                "event_duration_sec",
                "mean",
            ),

            min_duration_sec=(
                "event_duration_sec",
                "min",
            ),

            max_duration_sec=(
                "event_duration_sec",
                "max",
            ),

            total_repeat_evaluations=(
                "total_repeats",
                "sum",
            ),

            detected_repeat_evaluations=(
                "detected_count",
                "sum",
            ),

            missed_repeat_evaluations=(
                "missed_count",
                "sum",
            ),
        )
        .reset_index()
    )

    result["detection_rate_pct"] = (
        100.0
        * result[
            "detected_repeat_evaluations"
        ]
        / result[
            "total_repeat_evaluations"
        ]
    )

    return result


# ============================================================
# FN 해석 문장
# ============================================================

def make_fn_sentences(
    event_summary: pd.DataFrame,
    duration_summary: pd.DataFrame,
) -> list[str]:

    notes = []

    # --------------------------------------------------------
    # 모든 repeat에서 미탐
    # --------------------------------------------------------

    always_missed = (
        event_summary[
            event_summary[
                "detected_count"
            ].eq(0)
        ]
    )

    if len(always_missed):

        names = ", ".join(
            always_missed[
                "group_id"
            ]
            .astype(str)
        )

        notes.append(
            "모든 repeat에서 미탐된 "
            f"fault event: {names}."
        )

    # --------------------------------------------------------
    # 짧은 event
    # --------------------------------------------------------

    short_events = event_summary[
        event_summary[
            "event_duration_sec"
        ] < 1.0
    ]

    if len(short_events):

        total_eval = (
            short_events[
                "total_repeats"
            ]
            .sum()
        )

        detected_eval = (
            short_events[
                "detected_count"
            ]
            .sum()
        )

        rate = (
            100.0
            * detected_eval
            / total_eval
            if total_eval > 0
            else np.nan
        )

        notes.append(
            f"1.0초 미만 fault event의 "
            f"repeat 기준 탐지율은 "
            f"{rate:.2f}%입니다."
        )

    # --------------------------------------------------------
    # duration 해석
    # --------------------------------------------------------

    if len(duration_summary) >= 2:

        notes.append(
            "Event duration 구간별 탐지율을 비교하여 "
            "짧은 fault event에서 관측 가능한 window 수가 "
            "제한되는지 확인할 수 있습니다. "
            "다만 지속시간만으로 미탐 원인을 단정하지 않고 "
            "signal deviation 및 detector 구성과 함께 해석해야 합니다."
        )

    return notes


# ============================================================
# 1회 repeat 재실행
# ============================================================

def replay_repeat(
    *,
    raw: pd.DataFrame,
    x_sample: np.ndarray,
    feature_cols: list[str],
    mod5: object,
    mod6: object,
    window_paths: list[Path],
    n_splits: int,
    seed: int,
    window_threshold_mode: str,
    window_quantile: float,
    window_k: int,
    window_covariance: str,
    sample_threshold_mode: str,
    sample_quantile: float,
    sample_agg_k: int,
    sample_agg_how: str,
    sample_feature_set: str,
    sample_covariance: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    # 6번 코드의 run_repeat과 동일하게
    # fold별 detector alarm을 재구성한다.

    scales = [
        mod5.load_scale(
            mod6.mod5_base,
            p,
            feature_cols,
        )
        for p in window_paths
    ]

    scales.sort(
        key=lambda x: -x.window_sec
    )

    fold_map = build_fold_map(
        raw,
        n_splits=n_splits,
        seed=seed,
    )

    group_index = (
        raw
        .groupby("group_id")
        .indices
    )

    all_one = np.zeros(
        len(raw),
        dtype=bool,
    )

    all_half = np.zeros(
        len(raw),
        dtype=bool,
    )

    all_sample = np.zeros(
        len(raw),
        dtype=bool,
    )

    window_audit_parts = []

    for fold in range(
        n_splits
    ):

        fold_one = np.zeros(
            len(raw),
            dtype=bool,
        )

        fold_half = np.zeros(
            len(raw),
            dtype=bool,
        )

        # ----------------------------------------------------
        # Window detectors
        # ----------------------------------------------------

        for sc in scales:

            work = mod5.make_fold_frame(
                sc,
                fold_map,
                fold,
                n_splits,
            )

            metrics, test_work, _ = (
                mod6.mod5_base.run_once(
                    df=work,
                    feature_cols=feature_cols,
                    covariance=window_covariance,
                    threshold_mode=(
                        window_threshold_mode
                    ),
                    normal_quantile=(
                        window_quantile
                    ),
                    k_consecutive=window_k,
                    train_seed=None,
                    train_fraction=1.0,
                )
            )

            alarm_raw = (
                apply_window_alarm_to_raw(
                    raw,
                    test_work,
                    mod5.ALARM_COL,
                )
            )

            if sc.window_sec >= 0.99:

                fold_one |= alarm_raw

            else:

                fold_half |= alarm_raw

            audit = test_work.copy()

            audit["repeat_seed"] = seed
            audit["fold"] = fold
            audit["scale"] = sc.name
            audit["window_sec"] = (
                float(sc.window_sec)
            )
            audit["threshold"] = (
                float(metrics["threshold"])
            )

            audit["source_detector_alarm"] = (
                audit[
                    mod5.ALARM_COL
                ]
                .astype(bool)
            )

            window_audit_parts.append(
                audit
            )

        all_one |= fold_one
        all_half |= fold_half

        # ----------------------------------------------------
        # Sample detector
        # ----------------------------------------------------

        sample_alarm, _, _ = (
            mod6.make_sample_alarm_for_fold(
                raw=raw,
                x_all=x_sample,
                group_index=group_index,
                fold_map=fold_map,
                test_fold=fold,
                n_splits=n_splits,
                covariance=sample_covariance,
                feature_set=sample_feature_set,
                agg_k=sample_agg_k,
                agg_how=sample_agg_how,
                threshold_mode=sample_threshold_mode,
                normal_quantile=sample_quantile,
            )
        )

        all_sample |= sample_alarm

    # --------------------------------------------------------
    # Final OR_3_P2
    # --------------------------------------------------------

    or3 = (
        all_one
        | all_half
        | all_sample
    )

    or3_p2 = (
        mod6.apply_group_persistence(
            or3,
            list(group_index.values()),
            p=2,
        )
    )

    # --------------------------------------------------------
    # Window audit
    # --------------------------------------------------------

    window_audit = pd.concat(
        window_audit_parts,
        ignore_index=True,
    )

    window_audit = (
        mark_final_alarm_overlap(
            window_audit,
            raw,
            or3_p2,
        )
    )

    window_audit[
        "fp_context"
    ] = (
        window_audit[
            "final_or3p2_overlap"
        ]
        & window_audit[
            "label"
        ].eq(0)
    )

    # Detector 자체 FP
    window_audit[
        "source_detector_fp"
    ] = (
        window_audit[
            "source_detector_alarm"
        ]
        & window_audit[
            "label"
        ].eq(0)
    )

    # --------------------------------------------------------
    # Fault event details
    # --------------------------------------------------------

    fault_groups = sorted(
        raw.loc[
            raw["kind"].eq("fault"),
            "group_id",
        ]
        .astype(str)
        .unique()
    )

    fault_rows = []

    for gid in fault_groups:

        idx = raw.index[
            raw["group_id"].eq(gid)
        ].to_numpy()

        onset = raw.loc[
            idx[0],
            TIME_COL,
        ]

        end = raw.loc[
            idx[-1],
            TIME_COL,
        ]

        duration = (
            end - onset
        ).total_seconds()

        final_hits = idx[
            or3_p2[idx]
        ]

        detected = (
            len(final_hits) > 0
        )

        if detected:

            first_alarm_time = raw.loc[
                final_hits[0],
                TIME_COL,
            ]

            delay = (
                first_alarm_time
                - onset
            ).total_seconds()

        else:

            first_alarm_time = pd.NaT
            delay = np.nan

        fault_rows.append({
            "group_id": gid,

            "fault_onset_time": (
                onset
            ),

            "fault_end_time": (
                end
            ),

            "event_duration_sec": (
                float(duration)
            ),

            "n_samples": int(
                len(idx)
            ),

            "detected": bool(
                detected
            ),

            "first_alarm_time": (
                first_alarm_time
            ),

            "delay_sec": (
                float(delay)
                if detected
                else np.nan
            ),

            "repeat_seed": seed,
        })

    fault_df = pd.DataFrame(
        fault_rows
    )

    return (
        window_audit,
        fault_df,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "KAMPact FP/FN condition analysis"
        )
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    parser.add_argument(
        "--normal-path",
        default=(
            "data/"
            "press_data_normal_with_idle.csv"
        ),
    )

    parser.add_argument(
        "--fault-path",
        default=(
            "data/"
            "outlier_data.csv"
        ),
    )

    parser.add_argument(
        "--window-1.0",
        dest="window_10",
        default=(
            "result/"
            "modeling_dataset_1.0_0.1"
        ),
    )

    parser.add_argument(
        "--window-0.5",
        dest="window_05",
        default=(
            "result/"
            "modeling_dataset_0.5_0.1"
        ),
    )

    parser.add_argument(
        "--base-script",
        default=(
            "src/"
            "5_run_mahalanobis.py"
        ),
    )

    parser.add_argument(
        "--script-5-2",
        default=(
            "src/"
            "5_2_run_mahalanobis.py"
        ),
    )

    parser.add_argument(
        "--script-6",
        default=(
            "src/"
            "6_compare_three_detectors.py"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/"
            "8_fp_fn_analysis"
        ),
    )

    # --------------------------------------------------------
    # CV
    # --------------------------------------------------------

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    # --------------------------------------------------------
    # Window
    # --------------------------------------------------------

    parser.add_argument(
        "--window-threshold-mode",
        choices=[
            "f1",
            "normal_quantile",
        ],
        default="normal_quantile",
    )

    parser.add_argument(
        "--window-quantile",
        type=float,
        default=0.9999,
    )

    parser.add_argument(
        "--window-k",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--window-covariance",
        choices=[
            "empirical",
            "ledoitwolf",
            "oas",
        ],
        default="ledoitwolf",
    )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    parser.add_argument(
        "--sample-threshold-mode",
        choices=[
            "f1",
            "normal_quantile",
        ],
        default="normal_quantile",
    )

    parser.add_argument(
        "--sample-quantile",
        type=float,
        default=0.9999,
    )

    parser.add_argument(
        "--sample-agg-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--sample-agg",
        choices=[
            "mean",
            "max",
        ],
        default="mean",
    )

    parser.add_argument(
        "--sample-feature-set",
        choices=[
            "raw",
            "raw_diff",
            "raw_diff_roll3",
        ],
        default="raw_diff",
    )

    parser.add_argument(
        "--sample-covariance",
        choices=[
            "empirical",
            "ledoitwolf",
            "oas",
        ],
        default="ledoitwolf",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if args.n_splits < 2:
        raise ValueError(
            "--n-splits must be >= 2"
        )

    if args.repeats < 1:
        raise ValueError(
            "--repeats must be >= 1"
        )

    root = Path.cwd()

    normal_path = (
        root / args.normal_path
    )

    fault_path = (
        root / args.fault_path
    )

    window_1_path = (
        root / args.window_10
        / "model_windows.csv"
    )

    window_05_path = (
        root / args.window_05
        / "model_windows.csv"
    )

    if not normal_path.exists():
        raise FileNotFoundError(
            normal_path
        )

    if not fault_path.exists():
        raise FileNotFoundError(
            fault_path
        )

    if not window_1_path.exists():
        raise FileNotFoundError(
            window_1_path
        )

    if not window_05_path.exists():
        raise FileNotFoundError(
            window_05_path
        )

    # --------------------------------------------------------
    # Module load
    # --------------------------------------------------------

    mod5_base = load_module(
        root / args.base_script,
        "kamp5_base_fp_fn",
    )

    mod5 = load_module(
        root / args.script_5_2,
        "kamp5_2_fp_fn",
    )

    mod6 = load_module(
        root / args.script_6,
        "kamp6_fp_fn",
    )

    # 6번 코드가 internally 사용하는 global 설정
    mod6.window_covariance = (
        args.window_covariance
    )

    mod6.sample_covariance = (
        args.sample_covariance
    )

    # 5번 base module을 6번 모듈의 global에 연결
    mod6.mod5_base = mod5_base

    # --------------------------------------------------------
    # Raw load
    # --------------------------------------------------------

    raw_normal = load_raw(
        normal_path,
        "normal",
    )

    raw_fault = load_raw(
        fault_path,
        "fault",
    )

    raw = (
        pd.concat(
            [
                raw_normal,
                raw_fault,
            ],
            ignore_index=True,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Sample-level feature
    # 6번과 동일
    # --------------------------------------------------------

    x_sample, _ = (
        mod6.build_sample_features(
            raw,
            args.sample_feature_set,
        )
    )

    x_sample = (
        mod6.signed_log1p(
            x_sample
        )
    )

    # --------------------------------------------------------
    # Window feature
    # --------------------------------------------------------

    feature_cols = (
        infer_window_feature_cols(
            [
                window_1_path,
                window_05_path,
            ]
        )
    )

    print()
    print("=" * 100)
    print("KAMPact - FP / FN Condition Analysis")
    print("=" * 100)
    print()
    print(
        f"Window features ({len(feature_cols)}):"
    )

    for c in feature_cols:
        print(f"  - {c}")

    print()

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = (
        root / args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_window_audit = []
    all_fault_rows = []

    # ========================================================
    # Repeat
    # ========================================================

    for r in range(
        args.repeats
    ):

        seed = (
            args.seed + r
        )

        print(
            f"[Repeat {r + 1}/{args.repeats}] "
            f"seed={seed}"
        )

        window_audit, fault_rows = (
            replay_repeat(
                raw=raw,
                x_sample=x_sample,
                feature_cols=feature_cols,
                mod5=mod5,
                mod6=mod6,
                window_paths=[
                    window_1_path,
                    window_05_path,
                ],
                n_splits=args.n_splits,
                seed=seed,
                window_threshold_mode=(
                    args.window_threshold_mode
                ),
                window_quantile=(
                    args.window_quantile
                ),
                window_k=(
                    args.window_k
                ),
                window_covariance=(
                    args.window_covariance
                ),
                sample_threshold_mode=(
                    args.sample_threshold_mode
                ),
                sample_quantile=(
                    args.sample_quantile
                ),
                sample_agg_k=(
                    args.sample_agg_k
                ),
                sample_agg_how=(
                    args.sample_agg
                ),
                sample_feature_set=(
                    args.sample_feature_set
                ),
                sample_covariance=(
                    args.sample_covariance
                ),
            )
        )

        all_window_audit.append(
            window_audit
        )

        all_fault_rows.append(
            fault_rows
        )

    # ========================================================
    # Merge
    # ========================================================

    window_audit_all = pd.concat(
        all_window_audit,
        ignore_index=True,
    )

    fault_rows_all = pd.concat(
        all_fault_rows,
        ignore_index=True,
    )

    # ========================================================
    # FP 분석
    # ========================================================

    # 1.0s window 기준
    # 서로 다른 window scale의 feature를 섞지 않는다.

    normal_windows = (
        window_audit_all[
            window_audit_all[
                "window_sec"
            ]
            .sub(1.0)
            .abs()
            .lt(1e-6)
            & window_audit_all[
                "label"
            ].eq(0)
        ]
        .copy()
    )

    fp_windows = (
        normal_windows[
            normal_windows[
                "fp_context"
            ]
        ]
        .copy()
    )

    # 동일 logical window가
    # repeat마다 반복되므로 unique window 기준으로
    # feature distribution을 비교한다.

    dedup_cols = [
        "group_id",
        "window_start",
        "window_end",
    ]

    normal_unique = (
        normal_windows
        .sort_values(
            dedup_cols
        )
        .drop_duplicates(
            dedup_cols
        )
        .copy()
    )

    fp_unique = (
        fp_windows
        .sort_values(
            dedup_cols
        )
        .drop_duplicates(
            dedup_cols
        )
        .copy()
    )

    fp_summary = (
        make_fp_feature_summary(
            normal_unique,
            fp_unique,
            feature_cols,
        )
    )

    # --------------------------------------------------------
    # FP output 1
    # --------------------------------------------------------

    fp_unique.to_csv(
        output_dir
        / "fp_windows_1.0s.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # FP output 2
    # --------------------------------------------------------

    fp_summary.to_csv(
        output_dir
        / "fp_feature_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Detector-native FP
    # --------------------------------------------------------

    source_fp = (
        window_audit_all[
            window_audit_all[
                "source_detector_fp"
            ]
            & window_audit_all[
                "window_sec"
            ]
            .sub(1.0)
            .abs()
            .lt(1e-6)
        ]
        .sort_values(
            [
                "group_id",
                "window_start",
                "window_end",
            ]
        )
        .drop_duplicates(
            dedup_cols
        )
    )

    source_fp.to_csv(
        output_dir
        / "source_detector_fp_windows_1.0s.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # FP audit
    # --------------------------------------------------------

    fp_audit_cols = [
        "repeat_seed",
        "fold",
        "scale",
        "window_sec",
        "group_id",
        "window_start",
        "window_end",
        "label",
        "source_detector_alarm",
        "final_or3p2_overlap",
        "fp_context",
        *feature_cols,
    ]

    fp_audit_cols = [
        c
        for c in fp_audit_cols
        if c in window_audit_all.columns
    ]

    (
        window_audit_all[
            window_audit_all[
                "fp_context"
            ]
        ][fp_audit_cols]
        .to_csv(
            output_dir
            / "fp_window_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
    )

    # ========================================================
    # FN 분석
    # ========================================================

    event_summary = (
        build_fn_event_summary(
            fault_rows_all,
            window_audit_all,
        )
    )

    duration_summary = (
        build_duration_summary(
            event_summary
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    event_summary.to_csv(
        output_dir
        / "fn_event_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    duration_summary.to_csv(
        output_dir
        / "fn_duration_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fault_rows_all.to_csv(
        output_dir
        / "fn_event_repeat_details.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Report
    # ========================================================

    fp_sentence = (
        make_fp_sentence(
            fp_summary,
            top_n=5,
        )
    )

    fn_notes = (
        make_fn_sentences(
            event_summary,
            duration_summary,
        )
    )

    report = []

    report.append(
        "KAMPact - FP/FN Condition Analysis"
    )

    report.append(
        "=" * 80
    )

    report.append("")

    # --------------------------------------------------------
    # FP
    # --------------------------------------------------------

    report.append(
        "[1] False Positive Analysis"
    )

    report.append(
        "-" * 80
    )

    report.append(
        f"Unique normal 1.0s windows : "
        f"{len(normal_unique):,}"
    )

    report.append(
        f"Unique FP context windows  : "
        f"{len(fp_unique):,}"
    )

    report.append("")

    report.append(
        fp_sentence
    )

    report.append("")

    report.append(
        "Top FP feature differences:"
    )

    if not fp_summary.empty:

        report.append(
            fp_summary[
                [
                    "feature",
                    "normal_median",
                    "fp_median",
                    "median_delta",
                    "delta_pct_of_abs_normal",
                    "robust_effect",
                    "direction",
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

    else:

        report.append(
            "No FP feature summary."
        )

    report.append("")

    # --------------------------------------------------------
    # FN
    # --------------------------------------------------------

    report.append(
        "[2] False Negative Analysis"
    )

    report.append(
        "-" * 80
    )

    report.append(
        f"Fault events : "
        f"{event_summary['group_id'].nunique()}"
    )

    report.append("")

    if not event_summary.empty:

        report.append(
            event_summary[
                [
                    "group_id",
                    "event_duration_sec",
                    "n_samples",
                    "n_windows_1p0s",
                    "n_windows_0p5s",
                    "detected_count",
                    "total_repeats",
                    "missed_count",
                    "detection_rate_pct",
                    "delay_mean_sec",
                ]
            ]
            .to_string(
                index=False
            )
        )

    else:

        report.append(
            "No fault event summary."
        )

    report.append("")

    report.append(
        "[3] Duration-bin Summary"
    )

    report.append(
        "-" * 80
    )

    if not duration_summary.empty:

        report.append(
            duration_summary.to_string(
                index=False
            )
        )

    else:

        report.append(
            "No duration summary."
        )

    report.append("")

    report.append(
        "[4] Interpretation Notes"
    )

    report.append(
        "-" * 80
    )

    for note in fn_notes:

        report.append(
            f"- {note}"
        )

    report.append("")

    report.append(
        "[5] Important Interpretation"
    )

    report.append(
        "-" * 80
    )

    report.append(
        "FP feature comparison is descriptive. "
        "A feature having a higher median in FP windows "
        "does not by itself prove that the feature causally "
        "generated the false alarm."
    )

    report.append(
        "FN duration analysis identifies association between "
        "event duration and detection opportunity, but "
        "duration alone should not be interpreted as the "
        "sole cause of missed detection."
    )

    report.append("")

    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------

    report_path = (
        output_dir
        / "analysis_report.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # ========================================================
    # Console output
    # ========================================================

    print()
    print("=" * 100)
    print("FP / FN ANALYSIS COMPLETE")
    print("=" * 100)

    print()
    print("[FP]")

    print(
        f"Normal 1.0s windows : "
        f"{len(normal_unique):,}"
    )

    print(
        f"FP context windows  : "
        f"{len(fp_unique):,}"
    )

    print()

    print(
        fp_sentence
    )

    print()
    print("[Top FP features]")

    if not fp_summary.empty:

        print(
            fp_summary[
                [
                    "feature",
                    "normal_median",
                    "fp_median",
                    "median_delta",
                    "delta_pct_of_abs_normal",
                    "robust_effect",
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

    print()
    print("[FN]")

    if not event_summary.empty:

        print(
            event_summary[
                [
                    "group_id",
                    "event_duration_sec",
                    "n_samples",
                    "n_windows_1p0s",
                    "n_windows_0p5s",
                    "detected_count",
                    "total_repeats",
                    "missed_count",
                    "detection_rate_pct",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print()
    print("[Duration]")

    if not duration_summary.empty:

        print(
            duration_summary.to_string(
                index=False
            )
        )

    print()
    print("Output files:")

    for filename in [
        "fp_windows_1.0s.csv",
        "fp_feature_summary.csv",
        "source_detector_fp_windows_1.0s.csv",
        "fp_window_audit.csv",
        "fn_event_summary.csv",
        "fn_duration_summary.csv",
        "fn_event_repeat_details.csv",
        "analysis_report.txt",
    ]:

        print(
            f"  - {output_dir / filename}"
        )

    print()


if __name__ == "__main__":
    main()