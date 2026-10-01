from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_INPUT = "result/modeling_dataset_1.0_0.1/model_windows.csv"
DEFAULT_OUTPUT_DIR = "outputs/7_isolation_forest_baseline"

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
    "source",
]


# ============================================================
# Transform
# ============================================================

def signed_log1p(x: np.ndarray) -> np.ndarray:
    """Same signed-log transform used by the current Mahalanobis baseline."""
    return np.sign(x) * np.log1p(np.abs(x))


# ============================================================
# Threshold
# ============================================================

def choose_threshold_f1(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    """Validation threshold maximizing F1.

    Tie-break:
      1. Higher F1
      2. Higher precision
      3. Lower threshold
    """
    finite = np.isfinite(scores)
    y_true = np.asarray(y_true).astype(int)[finite]
    scores = np.asarray(scores, dtype=float)[finite]

    if len(scores) == 0:
        raise ValueError("No finite validation scores.")

    best = None

    for threshold in np.unique(scores):
        pred = (scores >= threshold).astype(int)

        f1 = f1_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)

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
    """Threshold from normal validation scores only."""
    normal_mask = val_df["label"].eq(0).to_numpy()
    normal_scores = val_scores[normal_mask]
    normal_scores = normal_scores[np.isfinite(normal_scores)]

    if len(normal_scores) == 0:
        raise ValueError("No finite normal validation scores.")

    threshold = float(np.quantile(normal_scores, q))
    val_pred = (val_scores >= threshold).astype(int)
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
    """Alarm on the kth consecutive high-score window in each group."""
    if k <= 1:
        return (df["score"] >= threshold).astype(bool)

    work = df.copy()
    work["window_start"] = pd.to_datetime(
        work["window_start"],
        errors="coerce",
    )
    work["high"] = work["score"] >= threshold

    alarm = pd.Series(
        False,
        index=work.index,
        dtype=bool,
    )

    for _, g in (
        work.sort_values(["group_id", "window_start"])
        .groupby("group_id", sort=False)
    ):
        if len(g) == 0:
            continue

        idx = g.index.to_numpy()
        times = g["window_start"].to_numpy(dtype="datetime64[ns]")
        high = g["high"].to_numpy(dtype=bool)

        deltas = (
            np.diff(times)
            .astype("timedelta64[ns]")
            .astype(np.int64)
            / 1e9
        )
        positive = deltas[deltas > 0]
        expected = float(np.median(positive)) if len(positive) else 0.0

        run = 0
        for j in range(len(idx)):
            contiguous = True

            if j > 0 and expected > 0:
                dt = float(deltas[j - 1])
                contiguous = (
                    abs(dt - expected)
                    <= max(step_tolerance, expected * 0.02)
                )

            if high[j] and contiguous:
                run += 1
            else:
                run = 1 if high[j] else 0

            if run >= k:
                alarm.loc[idx[j]] = True

    return alarm


# ============================================================
# Event-level metrics
# ============================================================

def calculate_event_details(
    test_df: pd.DataFrame,
    alarm_col: str,
) -> pd.DataFrame:
    """Same event/delay definition as current 5_run_mahalanobis.py.

    Valid detection windows satisfy window_end >= fault_onset_time.
    A transition window may therefore count as an early detection.
    Delay = first valid alarm window_end - fault_onset_time.
    """
    work = test_df.copy()

    for col in ["window_start", "window_end", "fault_onset_time"]:
        work[col] = pd.to_datetime(work[col], errors="coerce")

    fault_groups = (
        work.loc[work["source"].eq("fault"), "group_id"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    rows = []

    for group_id in fault_groups:
        g = (
            work.loc[work["group_id"].eq(group_id)]
            .sort_values("window_start")
            .copy()
        )

        if g.empty:
            continue

        onset_values = g["fault_onset_time"].dropna()
        if onset_values.empty:
            rows.append({
                "group_id": group_id,
                "fault_onset_time": pd.NaT,
                "detected": False,
                "alarm_window_start": pd.NaT,
                "alarm_window_end": pd.NaT,
                "detection_delay_sec": np.nan,
                "note": "missing_fault_onset_time",
            })
            continue

        true_onset = onset_values.iloc[0]

        valid_windows = g[g["window_end"] >= true_onset].copy()
        hits = (
            valid_windows[valid_windows[alarm_col].astype(bool)]
            .sort_values("window_end")
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
        alarm_start = first_hit["window_start"]
        alarm_end = first_hit["window_end"]
        detection_delay = (
            alarm_end - true_onset
        ).total_seconds()

        rows.append({
            "group_id": group_id,
            "fault_onset_time": true_onset,
            "detected": True,
            "alarm_window_start": alarm_start,
            "alarm_window_end": alarm_end,
            "detection_delay_sec": max(0.0, float(detection_delay)),
            "note": "detected",
        })

    return pd.DataFrame(rows)


def event_metrics(
    test_df: pd.DataFrame,
    alarm_col: str,
) -> tuple[dict, pd.DataFrame]:
    details = calculate_event_details(test_df, alarm_col)

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

    detected_mask = details["detected"].eq(True)
    detected_events = int(detected_mask.sum())
    total_events = len(details)

    delay_values = (
        details.loc[detected_mask, "detection_delay_sec"]
        .dropna()
        .to_numpy(dtype=float)
    )

    metrics = {
        "event_detection_rate": detected_events / total_events,
        "mean_detection_delay_sec": (
            float(np.mean(delay_values))
            if len(delay_values)
            else np.nan
        ),
        "median_detection_delay_sec": (
            float(np.median(delay_values))
            if len(delay_values)
            else np.nan
        ),
        "min_detection_delay_sec": (
            float(np.min(delay_values))
            if len(delay_values)
            else np.nan
        ),
        "max_detection_delay_sec": (
            float(np.max(delay_values))
            if len(delay_values)
            else np.nan
        ),
        "detected_events": detected_events,
        "total_events": total_events,
        "valid_onset_events": int(details["fault_onset_time"].notna().sum()),
    }

    return metrics, details


# ============================================================
# Window-level metrics
# ============================================================

def evaluate_window_metrics(
    test_df: pd.DataFrame,
    alarm_col: str,
) -> dict:
    y = test_df["label"].astype(int).to_numpy()
    pred = test_df[alarm_col].astype(int).to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    normal = test_df[test_df["label"].eq(0)]
    normal_fpr = (
        float(normal[alarm_col].mean())
        if len(normal)
        else np.nan
    )

    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "normal_false_alarm_rate": normal_fpr,
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
    out = df.copy()

    for col in ["window_start", "window_end", "fault_onset_time"]:
        out[col] = pd.to_datetime(out[col], errors="coerce")

    out["label"] = pd.to_numeric(
        out["label"],
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

    return out


# ============================================================
# Single run
# ============================================================

def run_once(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_estimators: int,
    max_samples: str | int | float,
    max_features: float,
    contamination: str | float,
    threshold_mode: str,
    normal_quantile: float,
    k_consecutive: int,
    random_state: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:

    train = df[
        df["split"].eq("train")
        & df["label"].eq(0)
    ].copy()
    val = df[df["split"].eq("val")].copy()
    test = df[df["split"].eq("test")].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError("train/val/test split is missing or empty.")

    # --------------------------------------------------------
    # Median imputation: train normal only
    # --------------------------------------------------------
    med = train[feature_cols].median()

    train_X = train[feature_cols].fillna(med).to_numpy(dtype=float)
    val_X = val[feature_cols].fillna(med).to_numpy(dtype=float)
    test_X = test[feature_cols].fillna(med).to_numpy(dtype=float)

    if not np.isfinite(train_X).all():
        raise ValueError("Train features still contain NaN/Inf after imputation.")
    if not np.isfinite(val_X).all():
        raise ValueError("Validation features still contain NaN/Inf after imputation.")
    if not np.isfinite(test_X).all():
        raise ValueError("Test features still contain NaN/Inf after imputation.")

    # Same transform as current Mahalanobis baseline.
    train_X = signed_log1p(train_X)
    val_X = signed_log1p(val_X)
    test_X = signed_log1p(test_X)

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_X)

    # score_samples: higher = more normal.
    # Negate so that higher score = more anomalous,
    # matching the Mahalanobis score convention.
    val_scores = -model.score_samples(val_X)
    test_scores = -model.score_samples(test_X)

    val_work = val.copy()
    val_work["score"] = val_scores

    test_work = test.copy()
    test_work["score"] = test_scores

    # --------------------------------------------------------
    # Threshold selection
    # --------------------------------------------------------
    if threshold_mode == "f1":
        threshold, val_f1 = choose_threshold_f1(
            val_work["label"].astype(int).to_numpy(),
            val_scores,
        )
    elif threshold_mode == "normal_quantile":
        threshold, val_f1 = choose_threshold_normal_quantile(
            val_work,
            val_scores,
            normal_quantile,
        )
    else:
        raise ValueError(
            "threshold_mode must be 'f1' or 'normal_quantile'."
        )

    # --------------------------------------------------------
    # Alarm
    # --------------------------------------------------------
    test_work["alarm_immediate"] = (
        test_work["score"] >= threshold
    )

    if k_consecutive > 1:
        test_work["alarm_k_consecutive"] = apply_k_consecutive(
            test_work,
            threshold,
            k_consecutive,
        )
        alarm_col = "alarm_k_consecutive"
    else:
        test_work["alarm_k_consecutive"] = test_work["alarm_immediate"]
        alarm_col = "alarm_immediate"

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    metrics = evaluate_window_metrics(test_work, alarm_col)
    event_metric_dict, event_details = event_metrics(test_work, alarm_col)
    metrics.update(event_metric_dict)

    metrics.update({
        "threshold": float(threshold),
        "validation_f1": float(val_f1),
        "threshold_mode": threshold_mode,
        "normal_quantile": float(normal_quantile),
        "k_consecutive": int(k_consecutive),
        "n_estimators": int(n_estimators),
        "max_samples": max_samples,
        "max_features": float(max_features),
        "contamination": contamination,
        "train_rows_used": int(len(train)),
        "seed": int(random_state),
        "n_features": int(len(feature_cols)),
        "input_rows": int(len(df)),
    })

    test_work = (
        test_work
        .sort_values(["group_id", "window_start"])
        .copy()
    )

    return metrics, test_work, event_details


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "KAMPact Isolation Forest baseline using "
            "the same 16 window features and evaluation protocol "
            "as the current Mahalanobis baseline."
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to model_windows.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--max-samples",
        default="auto",
        help="IsolationForest max_samples; e.g. auto, 1.0, 500",
    )
    parser.add_argument(
        "--max-features",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--contamination",
        default="auto",
        help="Used by IsolationForest internals; threshold is selected separately.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["f1", "normal_quantile"],
        default="f1",
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
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of independent Isolation Forest runs.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"입력 파일을 찾을 수 없습니다: {input_path}"
        )

    df = pd.read_csv(input_path)

    missing_meta = [
        c for c in REQUIRED_META
        if c not in df.columns
    ]
    if missing_meta:
        raise ValueError(
            f"필수 metadata 컬럼 누락: {missing_meta}"
        )

    missing_features = [
        c for c in DEFAULT_FEATURES
        if c not in df.columns
    ]
    if missing_features:
        raise ValueError(
            f"필수 16개 feature 컬럼 누락: {missing_features}"
        )

    df = prepare_data(df, DEFAULT_FEATURES)
    df = df[df["label"].isin([0, 1])].copy()

    # Ensure model data is exactly the same feature set.
    feature_cols = DEFAULT_FEATURES.copy()

    print("=" * 90)
    print("KAMPact - Isolation Forest Baseline")
    print("=" * 90)
    print(f"Input       : {input_path}")
    print(f"Output      : {out_dir}")
    print(f"Rows        : {len(df):,}")
    print(f"Features    : {len(feature_cols)}")
    print(f"Estimators  : {args.n_estimators}")
    print(f"Threshold   : {args.threshold_mode}")
    print(f"Persistence : {args.k_consecutive}")
    print()

    print("[Split counts]")
    print(df["split"].value_counts().sort_index().to_string())
    print("\n[Label counts]")
    print(df["label"].value_counts().sort_index().to_string())
    print()

    all_metrics = []
    first_test_scores = None
    first_event_details = None

    if args.seeds < 1:
        raise ValueError("--seeds must be >= 1")

    for i in range(args.seeds):
        seed = args.seed_start + i

        metrics, test_scores, event_details = run_once(
            df=df,
            feature_cols=feature_cols,
            n_estimators=args.n_estimators,
            max_samples=args.max_samples,
            max_features=args.max_features,
            contamination=args.contamination,
            threshold_mode=args.threshold_mode,
            normal_quantile=args.normal_quantile,
            k_consecutive=args.k_consecutive,
            random_state=seed,
        )

        all_metrics.append(metrics)

        if first_test_scores is None:
            first_test_scores = test_scores
            first_event_details = event_details

        print(
            f"seed={seed} | "
            f"F1={metrics['f1']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"Event={metrics['event_detection_rate'] * 100:.1f}% | "
            f"Delay={metrics['mean_detection_delay_sec']:.3f}s | "
            f"Normal_FA={metrics['normal_false_alarm_rate'] * 100:.3f}% | "
            f"Threshold={metrics['threshold']:.6f}"
        )

    result_df = pd.DataFrame(all_metrics)
    result_df.to_csv(
        out_dir / "seed_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    numeric_cols = [
        "f1",
        "precision",
        "recall",
        "event_detection_rate",
        "mean_detection_delay_sec",
        "median_detection_delay_sec",
        "normal_false_alarm_rate",
    ]

    summary = {
        "input": str(input_path),
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "n_train": int((df["split"] == "train").sum()),
        "n_val": int((df["split"] == "val").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "n_events_total_in_test": int(
            result_df["total_events"].iloc[0]
        ),
        "n_estimators": int(args.n_estimators),
        "max_samples": args.max_samples,
        "max_features": float(args.max_features),
        "contamination": args.contamination,
        "threshold_mode": args.threshold_mode,
        "normal_quantile": float(args.normal_quantile),
        "k_consecutive": int(args.k_consecutive),
        "seeds": int(args.seeds),
        "seed_start": int(args.seed_start),
    }

    for col in numeric_cols:
        summary[col + "_mean"] = float(result_df[col].mean())
        summary[col + "_std"] = (
            float(result_df[col].std(ddof=1))
            if len(result_df) > 1
            else 0.0
        )

    pd.DataFrame([summary]).to_csv(
        out_dir / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if first_test_scores is not None:
        export_cols = [
            "window_start",
            "window_end",
            "fault_onset_time",
            "label",
            "group_id",
            "split",
            "source",
            "score",
            "alarm_immediate",
            "alarm_k_consecutive",
        ]
        export_cols = [
            c for c in export_cols
            if c in first_test_scores.columns
        ]

        first_test_scores[export_cols].to_csv(
            out_dir / "test_scores.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if first_event_details is not None:
        first_event_details.to_csv(
            out_dir / "test_event_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

    with open(
        out_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            vars(args) | {"feature_cols": feature_cols},
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print()
    print("=" * 90)
    print("Final summary")
    print("=" * 90)
    print(
        f"F1                 : {result_df['f1'].mean():.4f}"
    )
    print(
        f"Precision          : {result_df['precision'].mean():.4f}"
    )
    print(
        f"Recall             : {result_df['recall'].mean():.4f}"
    )
    print(
        f"Event Detection    : {result_df['event_detection_rate'].mean() * 100:.1f}%"
    )
    print(
        f"Detection Delay    : {result_df['mean_detection_delay_sec'].mean():.3f}s"
    )
    print(
        f"Normal FA          : {result_df['normal_false_alarm_rate'].mean() * 100:.3f}%"
    )
    print()
    print("[Saved]")
    print(out_dir / "seed_results.csv")
    print(out_dir / "summary.csv")
    print(out_dir / "test_scores.csv")
    print(out_dir / "test_event_results.csv")
    print(out_dir / "config.json")


if __name__ == "__main__":
    main()
