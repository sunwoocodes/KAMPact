from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_FEATURES = [
    "AI0_Vibration_mean",
    "AI0_Vibration_std",
    "AI0_Vibration_rms",
    "AI0_Vibration_ptp",
    "AI0_Vibration_slope",
    "AI1_Vibration_mean",
    "AI1_Vibration_std",
    "AI1_Vibration_rms",
    "AI1_Vibration_ptp",
    "AI1_Vibration_slope",
    "AI2_Current_mean",
    "AI2_Current_std",
    "AI2_Current_rms",
    "AI2_Current_ptp",
    "AI2_Current_slope",
    "AI0_AI1_corr",
]

REQUIRED_META = [
    "window_start",
    "window_end",
    "fault_onset_time",
    "label",
    "group_id",
    "split",
]


# ============================================================
# Transform / covariance / score
# ============================================================

def signed_log1p(x: np.ndarray) -> np.ndarray:
    """Signed log1p transform for positive/negative features."""
    return np.sign(x) * np.log1p(np.abs(x))


def get_covariance(kind: str):
    if kind == "empirical":
        return EmpiricalCovariance()

    if kind == "ledoitwolf":
        return LedoitWolf()

    if kind == "oas":
        return OAS()

    raise ValueError(f"Unknown covariance: {kind}")


def mahalanobis_scores(
    model,
    X: np.ndarray
) -> np.ndarray:
    """Squared Mahalanobis distance."""
    diff = X - model.location_

    return np.einsum(
        "ij,jk,ik->i",
        diff,
        model.precision_,
        diff
    )


# ============================================================
# Threshold
# ============================================================

def choose_threshold_f1(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    """
    Validation threshold maximizing F1.

    Tie-break:
    1. Higher F1
    2. Higher precision
    3. Lower threshold
    """

    finite = np.isfinite(scores)

    y_true = y_true[finite]
    scores = scores[finite]

    if len(scores) == 0:
        raise ValueError(
            "No finite validation scores."
        )

    best = None

    for threshold in np.unique(scores):

        pred = (
            scores >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            pred,
            zero_division=0,
        )

        precision = precision_score(
            y_true,
            pred,
            zero_division=0,
        )

        key = (
            float(f1),
            float(precision),
            -float(threshold),
        )

        if best is None or key > best[0]:

            best = (
                key,
                float(threshold),
                float(f1),
            )

    return best[1], best[2]


def choose_threshold_normal_quantile(
    val_df: pd.DataFrame,
    val_scores: np.ndarray,
    q: float,
) -> tuple[float, float]:
    """
    Threshold using only normal validation scores.
    """

    normal_mask = (
        val_df["label"]
        .eq(0)
        .to_numpy()
    )

    normal_scores = val_scores[
        normal_mask
    ]

    normal_scores = normal_scores[
        np.isfinite(normal_scores)
    ]

    if len(normal_scores) == 0:
        raise ValueError(
            "No finite normal validation scores."
        )

    threshold = float(
        np.quantile(
            normal_scores,
            q
        )
    )

    val_pred = (
        val_scores >= threshold
    ).astype(int)

    val_f1 = f1_score(
        val_df["label"].astype(int),
        val_pred,
        zero_division=0,
    )

    return threshold, float(val_f1)


# ============================================================
# Consecutive alarm
# ============================================================

def apply_k_consecutive(
    df: pd.DataFrame,
    threshold: float,
    k: int,
    step_tolerance: float = 1e-6,
) -> pd.Series:
    """
    Alarm on kth consecutive high-score window.

    Expected step is inferred separately for each group
    from the median positive window_start difference.
    """

    if k <= 1:

        return (
            df["score"]
            >= threshold
        ).astype(bool)

    work = df.copy()

    work["window_start"] = pd.to_datetime(
        work["window_start"],
        errors="coerce",
    )

    work["high"] = (
        work["score"]
        >= threshold
    )

    alarm = pd.Series(
        False,
        index=work.index,
        dtype=bool,
    )

    grouped = (
        work
        .sort_values(
            [
                "group_id",
                "window_start"
            ]
        )
        .groupby(
            "group_id",
            sort=False
        )
    )

    for _, g in grouped:

        if len(g) == 0:
            continue

        idx = g.index.to_numpy()

        times = (
            g["window_start"]
            .to_numpy(
                dtype="datetime64[ns]"
            )
        )

        high = (
            g["high"]
            .to_numpy(
                dtype=bool
            )
        )

        deltas = (
            np.diff(times)
            .astype("timedelta64[ns]")
            .astype(np.int64)
            / 1e9
        )

        positive = (
            deltas[deltas > 0]
        )

        expected = (
            float(
                np.median(positive)
            )
            if len(positive)
            else 0.0
        )

        run = 0

        for j in range(len(idx)):

            contiguous = True

            if j > 0 and expected > 0:

                dt = float(
                    deltas[j - 1]
                )

                contiguous = (
                    abs(
                        dt - expected
                    )
                    <= max(
                        step_tolerance,
                        expected * 0.02
                    )
                )

            if high[j] and contiguous:

                run += 1

            else:

                run = (
                    1
                    if high[j]
                    else 0
                )

            if run >= k:

                alarm.loc[
                    idx[j]
                ] = True

    return alarm


# ============================================================
# Event-level detection delay
# ============================================================

def calculate_event_details(
    test_df: pd.DataFrame,
    alarm_col: str,
) -> pd.DataFrame:
    """
    Calculate one row per fault event.

    Detection delay:

        fault_onset_time
        ->
        first alarm window_end

    Important:
        ALL windows in the event are searched.

        Therefore a transition window whose
        label is still 0 can count as a valid
        early detection if its alarm becomes
        actionable at/after fault_onset_time.
    """

    work = test_df.copy()

    for col in [
        "window_start",
        "window_end",
        "fault_onset_time",
    ]:

        work[col] = pd.to_datetime(
            work[col],
            errors="coerce",
        )

    fault_groups = (
        work.loc[
            work["source"].eq("fault"),
            "group_id",
        ]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    rows = []

    for group_id in fault_groups:

        g = (
            work.loc[
                work["group_id"].eq(group_id)
            ]
            .sort_values(
                "window_start"
            )
            .copy()
        )

        if g.empty:
            continue

        onset_values = (
            g["fault_onset_time"]
            .dropna()
        )

        if onset_values.empty:

            rows.append({
                "group_id": group_id,
                "fault_onset_time": pd.NaT,
                "detected": False,
                "alarm_window_start": pd.NaT,
                "alarm_window_end": pd.NaT,
                "detection_delay_sec": np.nan,
                "note": (
                    "missing_fault_onset_time"
                ),
            })

            continue

        true_onset = (
            onset_values.iloc[0]
        )

        # ----------------------------------------------------
        # Valid detection windows
        # ----------------------------------------------------
        #
        # window_start < onset도 허용.
        #
        # 중요한 조건은 window가 실제로
        # 완료되는 시점(window_end)이
        # onset 이후여야 한다는 것.
        #
        # 즉:
        #
        # window_start < onset
        # window_end   >= onset
        #
        # 인 transition window도 탐지에 포함.
        # ----------------------------------------------------

        valid_windows = g[
            g["window_end"]
            >= true_onset
        ].copy()

        hits = (
            valid_windows[
                valid_windows[
                    alarm_col
                ].astype(bool)
            ]
            .sort_values(
                "window_end"
            )
        )

        if hits.empty:

            rows.append({
                "group_id": group_id,
                "fault_onset_time": true_onset,
                "detected": False,
                "alarm_window_start": pd.NaT,
                "alarm_window_end": pd.NaT,
                "detection_delay_sec": np.nan,
                "note": "not_detected",
            })

            continue

        first_hit = hits.iloc[0]

        alarm_start = (
            first_hit["window_start"]
        )

        alarm_end = (
            first_hit["window_end"]
        )

        detection_delay = (
            alarm_end - true_onset
        ).total_seconds()

        rows.append({
            "group_id": group_id,
            "fault_onset_time": true_onset,
            "detected": True,
            "alarm_window_start": alarm_start,
            "alarm_window_end": alarm_end,
            "detection_delay_sec": max(
                0.0,
                float(detection_delay)
            ),
            "note": "detected",
        })

    return pd.DataFrame(rows)


def event_metrics(
    test_df: pd.DataFrame,
    alarm_col: str,
) -> tuple[dict, pd.DataFrame]:
    """
    Aggregate event-level detection metrics.
    """

    details = calculate_event_details(
        test_df,
        alarm_col
    )

    if details.empty:

        return (
            {
                "event_detection_rate": np.nan,
                "mean_detection_delay_sec": np.nan,
                "median_detection_delay_sec": np.nan,
                "min_detection_delay_sec": np.nan,
                "max_detection_delay_sec": np.nan,
                "detected_events": 0,
                "total_events": 0,
                "valid_onset_events": 0,
            },
            details,
        )

    total_events = len(details)

    detected_mask = (
        details["detected"]
        .eq(True)
    )

    detected_events = int(
        detected_mask.sum()
    )

    delay_values = (
        details.loc[
            detected_mask,
            "detection_delay_sec",
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    valid_onset_events = int(
        details[
            "fault_onset_time"
        ]
        .notna()
        .sum()
    )

    metrics = {

        "event_detection_rate": (
            detected_events
            / total_events
            if total_events > 0
            else np.nan
        ),

        "mean_detection_delay_sec": (
            float(
                np.mean(
                    delay_values
                )
            )
            if len(delay_values)
            else np.nan
        ),

        "median_detection_delay_sec": (
            float(
                np.median(
                    delay_values
                )
            )
            if len(delay_values)
            else np.nan
        ),

        "min_detection_delay_sec": (
            float(
                np.min(
                    delay_values
                )
            )
            if len(delay_values)
            else np.nan
        ),

        "max_detection_delay_sec": (
            float(
                np.max(
                    delay_values
                )
            )
            if len(delay_values)
            else np.nan
        ),

        "detected_events": (
            detected_events
        ),

        "total_events": (
            total_events
        ),

        "valid_onset_events": (
            valid_onset_events
        ),
    }

    return metrics, details


# ============================================================
# Window-level evaluation
# ============================================================

def evaluate_window_metrics(
    test_df: pd.DataFrame,
    alarm_col: str,
) -> dict:
    """
    Calculate window-level metrics.
    """

    y = (
        test_df["label"]
        .astype(int)
        .to_numpy()
    )

    pred = (
        test_df[alarm_col]
        .astype(int)
        .to_numpy()
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    normal = test_df[
        test_df["label"].eq(0)
    ]

    normal_fpr = (
        float(
            normal[alarm_col].mean()
        )
        if len(normal)
        else np.nan
    )

    return {

        "f1": float(
            f1_score(
                y,
                pred,
                zero_division=0,
            )
        ),

        "precision": float(
            precision_score(
                y,
                pred,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y,
                pred,
                zero_division=0,
            )
        ),

        "normal_false_alarm_rate": (
            normal_fpr
        ),

        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


# ============================================================
# Data preparation
# ============================================================

def prepare_data(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Parse metadata and clean feature columns.
    """

    out = df.copy()

    for col in [
        "window_start",
        "window_end",
        "fault_onset_time",
    ]:

        out[col] = pd.to_datetime(
            out[col],
            errors="coerce",
        )

    for col in feature_cols:

        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        )

        out[col] = out[col].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    out["label"] = pd.to_numeric(
        out["label"],
        errors="coerce",
    )

    return out


# ============================================================
# Single run
# ============================================================

def run_once(
    df: pd.DataFrame,
    feature_cols: list[str],
    covariance: str,
    threshold_mode: str,
    normal_quantile: float,
    k_consecutive: int,
    train_seed: int | None,
    train_fraction: float,
) -> tuple[
    dict,
    pd.DataFrame,
    pd.DataFrame,
]:

    # --------------------------------------------------------
    # Train = normal only
    # --------------------------------------------------------

    train = (
        df[
            df["split"].eq("train")
            & df["label"].eq(0)
        ]
        .copy()
    )

    val = (
        df[
            df["split"].eq("val")
        ]
        .copy()
    )

    test = (
        df[
            df["split"].eq("test")
        ]
        .copy()
    )

    if (
        train.empty
        or val.empty
        or test.empty
    ):

        raise ValueError(
            "train/val/test split "
            "is missing or empty."
        )

    # --------------------------------------------------------
    # Train subsampling
    # --------------------------------------------------------

    if not 0 < train_fraction <= 1:

        raise ValueError(
            "--train-fraction must "
            "be in (0, 1]."
        )

    if train_fraction < 1.0:

        rng = np.random.default_rng(
            train_seed
        )

        n = max(
            2,
            int(
                round(
                    len(train)
                    * train_fraction
                )
            )
        )

        n = min(
            n,
            len(train)
        )

        sample_idx = rng.choice(
            len(train),
            size=n,
            replace=False,
        )

        train = (
            train.iloc[
                sample_idx
            ]
            .copy()
        )

    # --------------------------------------------------------
    # Median imputation
    # ONLY train normal
    # --------------------------------------------------------

    med = (
        train[
            feature_cols
        ]
        .median()
    )

    train_X = (
        train[
            feature_cols
        ]
        .fillna(med)
        .to_numpy(
            dtype=float
        )
    )

    val_X = (
        val[
            feature_cols
        ]
        .fillna(med)
        .to_numpy(
            dtype=float
        )
    )

    test_X = (
        test[
            feature_cols
        ]
        .fillna(med)
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Signed log
    # --------------------------------------------------------

    train_X = signed_log1p(
        train_X
    )

    val_X = signed_log1p(
        val_X
    )

    test_X = signed_log1p(
        test_X
    )

    # --------------------------------------------------------
    # Covariance fitting
    # --------------------------------------------------------

    cov = get_covariance(
        covariance
    )

    cov.fit(
        train_X
    )

    # --------------------------------------------------------
    # Mahalanobis score
    # --------------------------------------------------------

    val_scores = (
        mahalanobis_scores(
            cov,
            val_X
        )
    )

    test_scores = (
        mahalanobis_scores(
            cov,
            test_X
        )
    )

    val_work = val.copy()

    val_work["score"] = (
        val_scores
    )

    test_work = test.copy()

    test_work["score"] = (
        test_scores
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    if threshold_mode == "f1":

        threshold, val_f1 = (
            choose_threshold_f1(
                val_work[
                    "label"
                ]
                .astype(int)
                .to_numpy(),

                val_scores,
            )
        )

    elif threshold_mode == "normal_quantile":

        threshold, val_f1 = (
            choose_threshold_normal_quantile(
                val_work,
                val_scores,
                normal_quantile,
            )
        )

    else:

        raise ValueError(
            "threshold_mode must be "
            "'f1' or 'normal_quantile'."
        )

    # --------------------------------------------------------
    # Immediate alarm
    # --------------------------------------------------------

    test_work[
        "alarm_immediate"
    ] = (
        test_work["score"]
        >= threshold
    )

    # --------------------------------------------------------
    # Consecutive alarm
    # --------------------------------------------------------

    if k_consecutive > 1:

        test_work[
            "alarm_k_consecutive"
        ] = (
            apply_k_consecutive(
                test_work,
                threshold,
                k_consecutive,
            )
        )

        alarm_col = (
            "alarm_k_consecutive"
        )

    else:

        test_work[
            "alarm_k_consecutive"
        ] = (
            test_work[
                "alarm_immediate"
            ]
        )

        alarm_col = (
            "alarm_immediate"
        )

    # --------------------------------------------------------
    # Window metrics
    # --------------------------------------------------------

    metrics = (
        evaluate_window_metrics(
            test_work,
            alarm_col,
        )
    )

    # --------------------------------------------------------
    # Event metrics
    # --------------------------------------------------------

    event_metric_dict, event_details = (
        event_metrics(
            test_work,
            alarm_col,
        )
    )

    metrics.update(
        event_metric_dict
    )

    # --------------------------------------------------------
    # Config metrics
    # --------------------------------------------------------

    metrics.update({

        "threshold": float(
            threshold
        ),

        "validation_f1": float(
            val_f1
        ),

        "covariance": (
            covariance
        ),

        "threshold_mode": (
            threshold_mode
        ),

        "normal_quantile": float(
            normal_quantile
        ),

        "k_consecutive": int(
            k_consecutive
        ),

        "train_rows_used": int(
            len(train)
        ),

        "seed": train_seed,
    })

    # --------------------------------------------------------
    # Stable ordering
    # --------------------------------------------------------

    test_work = (
        test_work
        .sort_values(
            [
                "group_id",
                "window_start",
            ]
        )
        .copy()
    )

    return (
        metrics,
        test_work,
        event_details,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Mahalanobis(log) anomaly detection "
            "using model_windows.csv"
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "result/modeling_dataset_0.8_0.1/"
            "model_windows.csv"
        ),
        help=(
            "Path to model_windows.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/"
            "mahalanobis_model_windows"
        ),
        help=(
            "Output directory"
        ),
    )

    parser.add_argument(
        "--covariance",
        choices=[
            "empirical",
            "ledoitwolf",
            "oas",
        ],
        default="ledoitwolf",
    )

    parser.add_argument(
        "--threshold-mode",
        choices=[
            "f1",
            "normal_quantile",
        ],
        default="f1",
    )

    parser.add_argument(
        "--normal-quantile",
        type=float,
        default=0.995,
        help=(
            "Quantile for normal-only threshold."
        ),
    )

    parser.add_argument(
        "--k-consecutive",
        type=int,
        choices=[
            1,
            2,
        ],
        default=1,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help=(
            "Number of repeated runs."
        ),
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of train-normal rows."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Argument validation
    # --------------------------------------------------------

    if args.seeds < 1:

        raise ValueError(
            "--seeds must be >= 1."
        )

    if not 0 < args.train_fraction <= 1:

        raise ValueError(
            "--train-fraction must "
            "be in (0, 1]."
        )

    if not 0 < args.normal_quantile < 1:

        raise ValueError(
            "--normal-quantile must "
            "be between 0 and 1."
        )

    input_path = Path(
        args.input
    )

    out_dir = Path(
        args.output_dir
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{input_path}"
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(
        input_path
    )

    # --------------------------------------------------------
    # Required metadata
    # --------------------------------------------------------

    missing_meta = [
        col
        for col in REQUIRED_META
        if col not in df.columns
    ]

    if missing_meta:

        raise ValueError(
            "Missing required metadata "
            f"columns: {missing_meta}"
        )

    # --------------------------------------------------------
    # Required features
    # --------------------------------------------------------

    missing_features = [
        col
        for col in DEFAULT_FEATURES
        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing feature columns: "
            f"{missing_features}"
        )

    feature_cols = (
        DEFAULT_FEATURES.copy()
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    df = prepare_data(
        df,
        feature_cols,
    )

    df = (
        df[
            df["label"].isin(
                [0, 1]
            )
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print(
        "[INFO] Input:",
        input_path
    )

    print(
        "[INFO] Shape:",
        df.shape
    )

    print(
        "[INFO] Features:",
        len(feature_cols)
    )

    print(
        "[INFO] Split counts:\n",
        df[
            "split"
        ].value_counts()
    )

    print(
        "[INFO] Label counts:\n",
        df[
            "label"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Fault onset coverage
    # --------------------------------------------------------

    fault_rows = (
        df[
            df["label"].eq(1)
        ]
    )

    if not fault_rows.empty:

        total_fault_events = (
            fault_rows[
                "group_id"
            ]
            .nunique()
        )

        onset_fault_events = (
            fault_rows.loc[
                fault_rows[
                    "fault_onset_time"
                ].notna(),
                "group_id",
            ]
            .nunique()
        )

        coverage = (
            100.0
            * onset_fault_events
            / total_fault_events
            if total_fault_events > 0
            else 0.0
        )

        print()
        print(
            "[INFO] Fault onset coverage"
        )

        print(
            f"  Fault windows : "
            f"{len(fault_rows):,}"
        )

        print(
            f"  Fault events  : "
            f"{total_fault_events:,}"
        )

        print(
            f"  Onset events  : "
            f"{onset_fault_events:,}"
        )

        print(
            f"  Coverage      : "
            f"{coverage:.1f}%"
        )

    # --------------------------------------------------------
    # Repeat runs
    # --------------------------------------------------------

    all_metrics = []

    first_scores = None
    first_event_details = None

    for seed in range(
        args.seeds
    ):

        effective_seed = (
            seed
            if args.train_fraction < 1.0
            else None
        )

        (
            metrics,
            score_df,
            event_details,
        ) = run_once(
            df=df,
            feature_cols=feature_cols,
            covariance=args.covariance,
            threshold_mode=args.threshold_mode,
            normal_quantile=args.normal_quantile,
            k_consecutive=args.k_consecutive,
            train_seed=effective_seed,
            train_fraction=args.train_fraction,
        )

        all_metrics.append(
            metrics
        )

        if first_scores is None:

            first_scores = (
                score_df
            )

        if first_event_details is None:

            first_event_details = (
                event_details
            )

        print(
            f"seed={seed} | "
            f"F1={metrics['f1']:.4f} | "
            f"precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"event={metrics['event_detection_rate'] * 100:.1f}% | "
            f"delay_mean={metrics['mean_detection_delay_sec']:.3f}s | "
            f"delay_median={metrics['median_detection_delay_sec']:.3f}s | "
            f"normal_FA={metrics['normal_false_alarm_rate'] * 100:.3f}% | "
            f"threshold={metrics['threshold']:.6f}"
        )

    # --------------------------------------------------------
    # Seed results
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        all_metrics
    )

    seed_results_path = (
        out_dir
        / "seed_results.csv"
    )

    result_df.to_csv(
        seed_results_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    numeric_cols = [
        "f1",
        "precision",
        "recall",
        "event_detection_rate",
        "mean_detection_delay_sec",
        "median_detection_delay_sec",
        "min_detection_delay_sec",
        "max_detection_delay_sec",
        "normal_false_alarm_rate",
    ]

    summary = {}

    for col in numeric_cols:

        summary[
            f"{col}_mean"
        ] = float(
            result_df[col]
            .mean()
        )

        summary[
            f"{col}_std"
        ] = (
            float(
                result_df[col]
                .std(
                    ddof=1
                )
            )
            if len(result_df) > 1
            else 0.0
        )

    summary.update({

        "input": str(
            input_path
        ),

        "n_rows": int(
            len(df)
        ),

        "n_features": int(
            len(feature_cols)
        ),

        "covariance": (
            args.covariance
        ),

        "threshold_mode": (
            args.threshold_mode
        ),

        "normal_quantile": float(
            args.normal_quantile
        ),

        "k_consecutive": int(
            args.k_consecutive
        ),

        "seeds": int(
            args.seeds
        ),

        "train_fraction": float(
            args.train_fraction
        ),

        "delay_definition": (
            "fault_onset_time -> "
            "first alarm window_end"
        ),

        "onset_definition": (
            "first timestamp with "
            "Equipment_state >= 1 "
            "within each fault segment"
        ),
    })

    # --------------------------------------------------------
    # Infer global window step
    # --------------------------------------------------------

    starts = pd.to_datetime(
        df[
            "window_start"
        ],
        errors="coerce"
    )

    starts = (
        starts
        .sort_values()
    )

    deltas = (
        starts
        .diff()
        .dt.total_seconds()
    )

    positive_deltas = (
        deltas[
            deltas > 0
        ]
    )

    if len(positive_deltas):

        summary[
            "median_global_window_step_sec"
        ] = float(
            positive_deltas
            .median()
        )

    else:

        summary[
            "median_global_window_step_sec"
        ] = np.nan

    summary_path = (
        out_dir
        / "summary.csv"
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Test score export
    # --------------------------------------------------------

    test_scores_path = (
        out_dir
        / "test_scores.csv"
    )

    if first_scores is not None:

        export_cols = [

            "window_start",
            "window_end",
            "fault_onset_time",

            "label",
            "fault_ratio",

            "group_id",
            "segment_id",
            "split",
            "source",

            "score",

            "alarm_immediate",
            "alarm_k_consecutive",
        ]

        export_cols = [
            col
            for col in export_cols
            if col in first_scores.columns
        ]

        first_scores[
            export_cols
        ].to_csv(
            test_scores_path,
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------
    # Event detail export
    # --------------------------------------------------------

    event_details_path = (
        out_dir
        / "event_detection_details.csv"
    )

    if first_event_details is not None:

        first_event_details.to_csv(
            event_details_path,
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------
    # Config export
    # --------------------------------------------------------

    config = vars(args).copy()

    config["feature_cols"] = (
        feature_cols
    )

    config["delay_definition"] = {

        "detection_delay": (
            "fault_onset_time -> "
            "first alarm window_end"
        ),

        "onset_definition": (
            "first timestamp with "
            "Equipment_state >= 1 "
            "in each fault segment"
        ),

        "transition_window_rule": (
            "A window may start before "
            "fault onset, but its window_end "
            "must be >= fault onset."
        ),
    }

    config_path = (
        out_dir
        / "config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("MAHALANOBIS EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"Covariance     : "
        f"{args.covariance}"
    )

    print(
        f"Threshold mode : "
        f"{args.threshold_mode}"
    )

    print(
        f"k consecutive  : "
        f"{args.k_consecutive}"
    )

    print(
        f"Train fraction : "
        f"{args.train_fraction}"
    )

    if not np.isnan(
        summary[
            "median_global_window_step_sec"
        ]
    ):

        print(
            f"Median window step : "
            f"{summary['median_global_window_step_sec']:.3f}s"
        )

    print()
    print("Saved:")
    print(
        seed_results_path
    )
    print(
        summary_path
    )
    print(
        test_scores_path
    )
    print(
        event_details_path
    )
    print(
        config_path
    )


if __name__ == "__main__":
    main()