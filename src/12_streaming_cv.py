"""
KAMPact - StreamingDetector Cross-Validation
---------------------------------------------
Evaluate the exact detector used by 11_realtime_dashboard.py under a
segment-aware train / calibration / test protocol.

Protocol per fold
-----------------
- Train: normal-operation segments only, excluding validation/test folds.
- Calibration: normal-operation segments only, validation fold.
- Test: the held-out fold, including normal + fault + idle segments.
- Idle is never used for fitting/calibration, but is evaluated for false alarms.
- StreamingDetector is run causally, one raw sample at a time.
- Detector state is reset at each segment boundary.
- Thresholds are calibrated with normal-only q-quantiles.

Outputs
-------
outputs/12_streaming_cv/
    streaming_cv_summary.csv
    streaming_cv_fold_metrics.csv
    streaming_cv_event_details.csv
    streaming_cv_thresholds.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

try:
    from core_detector import StreamingDetector
except ImportError:
    from src.core_detector import StreamingDetector


SENSORS = [
    "AI0_Vibration",
    "AI1_Vibration",
    "AI2_Current",
]
TIME_COL = "TimeStamp"
STATE_COL = "Equipment_state"
IDLE_COL = "Idle"
GAP_THRESHOLD_SEC = 0.5
FOLD_SOURCE_ORDER = ["fault", "normal", "idle"]


def preprocess_raw(df: pd.DataFrame, source: str) -> pd.DataFrame:
    required = [TIME_COL, *SENSORS, STATE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing columns: {missing}")

    work = df.copy()
    work[TIME_COL] = pd.to_datetime(work[TIME_COL], errors="coerce")
    work = work.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)

    for col in SENSORS + [STATE_COL]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if work[SENSORS].isna().any().any():
        raise ValueError(f"{source}: sensor data contain NaN.")

    if IDLE_COL in work.columns:
        work[IDLE_COL] = pd.to_numeric(work[IDLE_COL], errors="coerce").fillna(0)
    else:
        work[IDLE_COL] = 0

    gap = work[TIME_COL].diff().dt.total_seconds()
    brk = gap > GAP_THRESHOLD_SEC
    brk.iloc[0] = True
    work["segment_id"] = (brk.cumsum() - 1).astype(int)
    work["source"] = source
    work["group_id"] = source + "_" + work["segment_id"].astype(str)

    if source == "fault":
        work["kind"] = "fault"
    else:
        idle_ratio = work[IDLE_COL].eq(1).groupby(work["group_id"]).transform("mean")
        work["kind"] = np.where(idle_ratio.ge(0.5), "idle", "normal")

    work["label"] = work[STATE_COL].ge(1).astype(int)
    return work


def build_fold_map(raw: pd.DataFrame, n_splits: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    fold_map: dict[str, int] = {}

    for kind in FOLD_SOURCE_ORDER:
        groups = np.array(
            sorted(raw.loc[raw["kind"].eq(kind), "group_id"].astype(str).unique()),
            dtype=str,
        )
        rng.shuffle(groups)
        for i, gid in enumerate(groups):
            fold_map[gid] = i % n_splits

    return fold_map


def split_by_groups(
    raw: pd.DataFrame,
    fold_map: dict[str, int],
    test_fold: int,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val_fold = (test_fold + 1) % n_splits
    groups = raw["group_id"].astype(str)
    folds = groups.map(fold_map)

    train_mask = raw["kind"].eq("normal") & folds.ne(test_fold) & folds.ne(val_fold)
    cal_mask = raw["kind"].eq("normal") & folds.eq(val_fold)
    test_mask = folds.eq(test_fold)

    return (
        raw.loc[train_mask].copy(),
        raw.loc[cal_mask].copy(),
        raw.loc[test_mask].copy(),
    )


def run_streaming_test(
    detector: StreamingDetector,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run causally by segment and return one row per raw sample."""
    results: list[dict[str, object]] = []

    for gid, group in test_df.groupby("group_id", sort=True):
        group = group.sort_values(TIME_COL)
        detector.reset_stream()

        for idx, row in group.iterrows():
            result = detector.step(
                pd.Timestamp(row[TIME_COL]),
                float(row[SENSORS[0]]),
                float(row[SENSORS[1]]),
                float(row[SENSORS[2]]),
            )

            results.append(
                {
                    "raw_index": int(idx),
                    "group_id": str(gid),
                    "kind": str(row["kind"]),
                    "label": int(row["label"]),
                    "TimeStamp": pd.Timestamp(row[TIME_COL]),
                    "final_alarm": bool(result["final_alarm"]),
                    "or3_base": bool(result["or3_base"]),
                    "sample_alarm": bool(result["sample_alarm"]),
                    "win05_alarm": bool(result["win05_alarm"]),
                    "win1_alarm": bool(result["win1_alarm"]),
                    "segment_reset": bool(result["segment_reset"]),
                    "gap_seconds": float(result["gap_seconds"]),
                }
            )

    out = pd.DataFrame(results)
    if out.empty:
        raise ValueError("No streaming test results were produced.")
    return out.sort_values("raw_index").reset_index(drop=True)


def count_episodes(alarm: np.ndarray) -> int:
    alarm = np.asarray(alarm, dtype=bool)
    prev = np.r_[False, alarm[:-1]]
    return int((alarm & ~prev).sum())


def evaluate_metrics(
    test_results: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    r = test_results.set_index("raw_index")
    aligned = test_df.join(r[["final_alarm"]], how="inner")

    non_idle = ~aligned["kind"].eq("idle")
    y = aligned.loc[non_idle, "label"].to_numpy(dtype=int)
    pred = aligned.loc[non_idle, "final_alarm"].to_numpy(dtype=bool).astype(int)

    metrics: dict[str, float | int] = {
        "sample_f1": float(f1_score(y, pred, zero_division=0)),
        "sample_precision": float(precision_score(y, pred, zero_division=0)),
        "sample_recall": float(recall_score(y, pred, zero_division=0)),
    }

    fault_rows: list[dict[str, object]] = []
    for gid, group in test_df[test_df["kind"].eq("fault")].groupby("group_id", sort=True):
        idx = group.index.to_numpy()
        hit = r.loc[r.index.intersection(idx), "final_alarm"].to_numpy(dtype=bool)
        hit_idx = np.flatnonzero(hit)

        if len(hit_idx):
            alarm_raw_index = int(r.index.intersection(idx)[hit_idx[0]])
            onset = pd.Timestamp(group[TIME_COL].iloc[0])
            alarm_time = pd.Timestamp(test_df.loc[alarm_raw_index, TIME_COL])
            delay = max(0.0, (alarm_time - onset).total_seconds())
            detected = True
        else:
            delay = np.nan
            detected = False

        fault_rows.append(
            {
                "group_id": str(gid),
                "detected": detected,
                "delay_sec": delay,
                "n_samples": int(len(group)),
            }
        )

    events = pd.DataFrame(fault_rows)
    if len(events):
        metrics["event_detection_rate"] = float(events["detected"].mean())
        metrics["detected_events"] = int(events["detected"].sum())
        metrics["n_events"] = int(len(events))
        metrics["delay_mean_sec"] = float(events.loc[events["detected"], "delay_sec"].mean()) if events["detected"].any() else np.nan
    else:
        metrics.update({
            "event_detection_rate": np.nan,
            "detected_events": 0,
            "n_events": 0,
            "delay_mean_sec": np.nan,
        })

    normal_fa = []
    for gid, group in test_df[test_df["kind"].eq("normal")].groupby("group_id", sort=True):
        a = r.loc[r.index.intersection(group.index), "final_alarm"].to_numpy(dtype=bool)
        normal_fa.append({"group_id": str(gid), "fa": bool(a.any()), "episodes": count_episodes(a)})
    normal_df = pd.DataFrame(normal_fa)
    metrics["normal_fa_cycle_rate"] = float(normal_df["fa"].mean()) if len(normal_df) else np.nan
    metrics["normal_fa_cycles"] = int(normal_df["fa"].sum()) if len(normal_df) else 0
    metrics["normal_cycles"] = int(len(normal_df))
    metrics["normal_fa_episodes"] = int(normal_df["episodes"].sum()) if len(normal_df) else 0

    idle_fa = []
    for gid, group in test_df[test_df["kind"].eq("idle")].groupby("group_id", sort=True):
        a = r.loc[r.index.intersection(group.index), "final_alarm"].to_numpy(dtype=bool)
        idle_fa.append({"group_id": str(gid), "fa": bool(a.any()), "episodes": count_episodes(a)})
    idle_df = pd.DataFrame(idle_fa)
    metrics["idle_fa_cycle_rate"] = float(idle_df["fa"].mean()) if len(idle_df) else np.nan
    metrics["idle_fa_cycles"] = int(idle_df["fa"].sum()) if len(idle_df) else 0
    metrics["idle_cycles"] = int(len(idle_df))
    metrics["idle_fa_episodes"] = int(idle_df["episodes"].sum()) if len(idle_df) else 0

    return metrics, events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-path", default="data/press_data_normal_with_idle.csv")
    parser.add_argument("--fault-path", default="data/outlier_data.csv")
    parser.add_argument("--output-dir", default="outputs/12_streaming_cv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quantile", type=float, default=0.9999)
    parser.add_argument(
    "--persistence-p",
    type=int,
    default=2,
    choices=[1, 2, 3],
    help="Persistence 조건: 1, 2, 3",
)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    args = parser.parse_args()

    if args.n_splits < 3:
        raise ValueError("--n-splits must be >= 3")
    if not 0.0 < args.quantile < 1.0:
        raise ValueError("--quantile must be between 0 and 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normal = preprocess_raw(pd.read_csv(args.normal_path), "normal")
    fault = preprocess_raw(pd.read_csv(args.fault_path), "fault")
    raw = pd.concat([normal, fault], ignore_index=True)
    raw = raw.sort_values(TIME_COL).reset_index(drop=True)

    all_metrics: list[dict[str, object]] = []
    all_events: list[pd.DataFrame] = []
    all_thresholds: list[dict[str, object]] = []
    all_folds: list[dict[str, object]] = []

    print("=" * 88)
    print("KAMPact - StreamingDetector Cross-Validation")
    print("=" * 88)
    print(f"Rows          : {len(raw):,}")
    print(f"Fault events  : {raw.loc[raw['kind'].eq('fault'), 'group_id'].nunique():,}")
    print(f"Folds/repeats : {args.n_splits}/{args.repeats}")
    print(f"Quantile      : {args.quantile}")

    for repeat in range(args.repeats):
        seed = args.seed + repeat
        fold_map = build_fold_map(raw, args.n_splits, seed)
        repeat_results: list[pd.DataFrame] = []

        for fold in range(args.n_splits):
            train_df, cal_df, test_df = split_by_groups(raw, fold_map, fold, args.n_splits)

            detector = StreamingDetector(
                win1_size=10,
                win05_size=5,
                sample_agg_k=3,
                persistence_p=args.persistence_p,
                gap_threshold_sec=GAP_THRESHOLD_SEC,
                quantile_threshold=args.quantile,
                random_state=seed,
            )
            fit_info = detector.fit(train_df, calibration_df=cal_df)

            test_results = run_streaming_test(detector, test_df)
            metrics, events = evaluate_metrics(test_results, test_df)

            row = {
                "repeat": repeat,
                "seed": seed,
                "fold": fold,
                "train_rows": fit_info["train_rows"],
                "calibration_rows": fit_info["calibration_rows"],
                "test_rows": len(test_df),
                "train_segments": fit_info["train_segments"],
                "calibration_segments": fit_info["calibration_segments"],
                "test_segments": int(test_df["group_id"].nunique()),
                **metrics,
            }
            all_folds.append(row)
            repeat_results.append(test_results)

            for scale, threshold in fit_info["thresholds"].items():
                all_thresholds.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "fold": fold,
                        "detector": scale,
                        "threshold": threshold,
                        "quantile": args.quantile,
                    }
                )

            if not events.empty:
                events["repeat"] = repeat
                events["seed"] = seed
                events["fold"] = fold
                all_events.append(events)

        # One repeat = every group tested exactly once.
        repeat_results_df = pd.concat(repeat_results, ignore_index=True)
        test_rows = raw.loc[raw["group_id"].astype(str).isin(repeat_results_df["group_id"].astype(str))].copy()
        # Reconstruct metrics with the held-out alarm rows. Duplicate indices are impossible because folds partition groups.
        test_results_all = repeat_results_df.drop_duplicates("raw_index").sort_values("raw_index")
        repeat_metrics, repeat_events = evaluate_metrics(test_results_all, test_rows)
        repeat_metrics.update({"repeat": repeat, "seed": seed})
        all_metrics.append(repeat_metrics)

        print(
            f"[repeat {repeat + 1}/{args.repeats}] seed={seed} "
            f"event={repeat_metrics['event_detection_rate'] * 100:.2f}% "
            f"delay={repeat_metrics['delay_mean_sec']:.3f}s "
            f"normal_FA={repeat_metrics['normal_fa_cycle_rate'] * 100:.2f}% "
            f"idle_FA={repeat_metrics['idle_fa_cycle_rate'] * 100:.2f}%"
        )

    repeat_df = pd.DataFrame(all_metrics)
    fold_df = pd.DataFrame(all_folds)
    event_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    threshold_df = pd.DataFrame(all_thresholds)

    numeric_cols = [
        "event_detection_rate",
        "delay_mean_sec",
        "sample_f1",
        "sample_precision",
        "sample_recall",
        "normal_fa_cycle_rate",
        "normal_fa_episodes",
        "idle_fa_cycle_rate",
        "idle_fa_episodes",
    ]

    summary_row = {
        "repeats": len(repeat_df),
        "n_events": int(repeat_df["n_events"].iloc[0]) if len(repeat_df) else 0,
        "event_detection_rate_mean": repeat_df["event_detection_rate"].mean(),
        "event_detection_rate_std": repeat_df["event_detection_rate"].std(ddof=1),
        "delay_mean_sec_mean": repeat_df["delay_mean_sec"].mean(),
        "delay_mean_sec_std": repeat_df["delay_mean_sec"].std(ddof=1),
        "sample_f1_mean": repeat_df["sample_f1"].mean(),
        "sample_precision_mean": repeat_df["sample_precision"].mean(),
        "sample_recall_mean": repeat_df["sample_recall"].mean(),
        "normal_fa_cycle_rate_mean": repeat_df["normal_fa_cycle_rate"].mean(),
        "normal_fa_cycle_rate_std": repeat_df["normal_fa_cycle_rate"].std(ddof=1),
        "normal_fa_episodes_mean": repeat_df["normal_fa_episodes"].mean(),
        "idle_fa_cycle_rate_mean": repeat_df["idle_fa_cycle_rate"].mean(),
        "idle_fa_cycle_rate_std": repeat_df["idle_fa_cycle_rate"].std(ddof=1),
        "idle_fa_episodes_mean": repeat_df["idle_fa_episodes"].mean(),
        "quantile": args.quantile,
        "protocol": "StreamingDetector; normal-only train/calibration; held-out segment test",
    }

    pd.DataFrame([summary_row]).to_csv(
        output_dir / "streaming_cv_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    repeat_df.to_csv(output_dir / "streaming_cv_repeat_metrics.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(output_dir / "streaming_cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    threshold_df.to_csv(output_dir / "streaming_cv_thresholds.csv", index=False, encoding="utf-8-sig")
    if not event_df.empty:
        event_df.to_csv(output_dir / "streaming_cv_event_details.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 88)
    print("STREAMING CV SUMMARY")
    print("=" * 88)
    print(pd.DataFrame([summary_row]).to_string(index=False))
    print(f"\n[SAVED] {output_dir}")


if __name__ == "__main__":
    main()
