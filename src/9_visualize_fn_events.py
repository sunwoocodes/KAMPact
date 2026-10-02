"""
KAMPact - FN Event Visualization
=================================

fault_5 / fault_19를 대상으로

    Raw Sensors
        +
    1.0s Window Detector
    0.5s Window Detector
    Sample-level Detector
    OR_3
    OR_3_P2

를 동일 시간축에 시각화한다.

특히 fault_19는 repeat별 결과가 다르므로

    repeat 0 -> Missed
    repeat 3 -> Detected

를 비교한다.

출력
----
outputs/9_fn_visualization/
    fault_5_seed0.png
    fault_19_seed0_missed.png
    fault_19_seed3_detected.png

실행
----
python src/9_visualize_fn_events.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
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

FAULT_EVENTS = [
    "fault_5",
    "fault_19",
]

# 비교할 repeat
PLOT_CASES = [
    ("fault_5", 0, "fault_5 - Missed"),
    ("fault_19", 0, "fault_19 - Missed"),
    ("fault_19", 3, "fault_19 - Detected"),
]


# ============================================================
# Dynamic module loader
# ============================================================

def load_module(
    path: Path,
    name: str,
):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"모듈 로드 실패: {path}"
        )

    module = importlib.util.module_from_spec(spec)

    # dataclass가 있는 5_2 때문에 반드시 등록
    sys.modules[name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise

    return module


# ============================================================
# Raw data
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
            f"{path}: 필수 컬럼 누락 {missing}"
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

    for col in [
        *SENSOR_COLS,
        STATE_COL,
    ]:
        df[col] = pd.to_numeric(
            df[col],
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

    # Segment
    gap = (
        df[TIME_COL]
        .diff()
        .dt.total_seconds()
    )

    brk = gap > GAP_THRESHOLD_SEC
    brk.iloc[0] = True

    df["segment_id"] = (
        brk.cumsum() - 1
    )

    df["source"] = source

    df["group_id"] = (
        source
        + "_"
        + df["segment_id"].astype(str)
    )

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
        df[STATE_COL]
        .fillna(0)
        .ge(1)
        .astype(int)
    )

    return df


# ============================================================
# Window feature inference
# ============================================================

META_COLS = {
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
    "score",
    "high",
    "alarm_immediate",
    "alarm_k_consecutive",
}


def infer_window_feature_cols(
    paths: list[Path],
) -> list[str]:

    frames = [
        pd.read_csv(
            p,
            nrows=200,
        )
        for p in paths
    ]

    common = [
        c
        for c in frames[0].columns
        if all(
            c in f.columns
            for f in frames[1:]
        )
    ]

    numeric_common = []

    for c in common:

        if c in META_COLS:
            continue

        if all(
            pd.api.types.is_numeric_dtype(
                f[c]
            )
            for f in frames
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
        "16개 window feature를 찾지 못했습니다.\n"
        f"후보: {numeric_common}"
    )


# ============================================================
# Fold map
# 6_compare_three_detectors.py와 동일
# ============================================================

def build_fold_map(
    raw_df: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> dict[str, int]:

    rng = np.random.default_rng(seed)

    fold_map = {}

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

    for kind in [
        "fault",
        "normal",
        "idle",
    ]:

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

            fold_map[str(gid)] = (
                i % n_splits
            )

    return fold_map


# ============================================================
# Window alarm -> raw timeline
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

    work = test_work.copy()

    work["window_start"] = pd.to_datetime(
        work["window_start"],
        errors="coerce",
    )

    work["window_end"] = pd.to_datetime(
        work["window_end"],
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

    for _, row in work.iterrows():

        if not bool(
            row[alarm_col]
        ):
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

        duration = end - start

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
# Reconstruct one repeat
# ============================================================

def reconstruct_repeat(
    *,
    raw: pd.DataFrame,
    feature_cols: list[str],
    mod5_base,
    mod5,
    mod6,
    window_1_path: Path,
    window_05_path: Path,
    seed: int,
    n_splits: int,
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
):

    # --------------------------------------------------------
    # Scale load
    # --------------------------------------------------------

    scales = [
        mod5.load_scale(
            mod5_base,
            window_1_path,
            feature_cols,
        ),
        mod5.load_scale(
            mod5_base,
            window_05_path,
            feature_cols,
        ),
    ]

    scales.sort(
        key=lambda x: -x.window_sec
    )

    # --------------------------------------------------------
    # Fold
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Sample features
    # --------------------------------------------------------

    x_sample, _ = (
        mod6.build_sample_features(
            raw,
            sample_feature_set,
        )
    )

    x_sample = (
        mod6.signed_log1p(
            x_sample
        )
    )

    # --------------------------------------------------------
    # Arrays
    # --------------------------------------------------------

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

    window_logs = []

    # --------------------------------------------------------
    # Fold evaluation
    # --------------------------------------------------------

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

        for scale in scales:

            work = mod5.make_fold_frame(
                scale,
                fold_map,
                fold,
                n_splits,
            )

            metrics, test_work, _ = (
                mod5_base.run_once(
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

            if scale.window_sec >= 0.99:

                fold_one |= alarm_raw

            else:

                fold_half |= alarm_raw

            window_logs.append({
                "fold": fold,
                "scale": scale.name,
                "window_sec": (
                    scale.window_sec
                ),
                "threshold": (
                    metrics["threshold"]
                ),
            })

        all_one |= fold_one
        all_half |= fold_half

        # ----------------------------------------------------
        # Sample-level
        # ----------------------------------------------------

        sample_alarm, sample_thr, _ = (
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
    # Ensemble
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

    alarms = {
        "1.0s Window": all_one,
        "0.5s Window": all_half,
        "Sample-level": all_sample,
        "OR_3": or3,
        "OR_3_P2": or3_p2,
    }

    return alarms, window_logs


# ============================================================
# Fault event information
# ============================================================

def get_fault_event(
    raw: pd.DataFrame,
    group_id: str,
) -> pd.DataFrame:

    event = (
        raw[
            raw["group_id"]
            .astype(str)
            .eq(group_id)
        ]
        .copy()
        .sort_values(TIME_COL)
        .reset_index()
    )

    if event.empty:
        raise ValueError(
            f"{group_id}를 찾지 못했습니다."
        )

    return event


# ============================================================
# Plot
# ============================================================

def plot_fault_event(
    *,
    raw: pd.DataFrame,
    alarms: dict[str, np.ndarray],
    group_id: str,
    title: str,
    output_path: Path,
):

    event = get_fault_event(
        raw,
        group_id,
    )

    original_index = (
        event["index"]
        .to_numpy()
    )

    onset = event[
        TIME_COL
    ].iloc[0]

    t = (
        event[TIME_COL]
        - onset
    ).dt.total_seconds()

    # --------------------------------------------------------
    # First detection
    # --------------------------------------------------------

    first_detection = {}

    for name, alarm in alarms.items():

        local_alarm = alarm[
            original_index
        ]

        hits = np.flatnonzero(
            local_alarm
        )

        if len(hits):

            first_detection[name] = (
                float(t.iloc[hits[0]])
            )

        else:

            first_detection[name] = None

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(13, 11),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                1.7,
                1.7,
                1.7,
                0.65,
                0.65,
                0.65,
                1.0,
            ]
        },
    )

    # --------------------------------------------------------
    # Raw sensors
    # --------------------------------------------------------

    sensor_labels = [
        "AI0_Vibration",
        "AI1_Vibration",
        "AI2_Current",
    ]

    for ax, sensor, label in zip(
        axes[:3],
        SENSOR_COLS,
        sensor_labels,
    ):

        ax.plot(
            t,
            event[sensor],
            linewidth=1.5,
            label=label,
        )

        ax.axvline(
            0.0,
            linestyle="--",
            linewidth=1.2,
            label="Fault onset",
        )

        ax.set_ylabel(
            label,
            fontsize=9,
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        ax.legend(
            loc="upper right",
            fontsize=8,
        )

    # --------------------------------------------------------
    # Alarm tracks
    # --------------------------------------------------------

    alarm_order = [
        "1.0s Window",
        "0.5s Window",
        "Sample-level",
    ]

    for ax, name in zip(
        axes[3:6],
        alarm_order,
    ):

        local_alarm = (
            alarms[name][
                original_index
            ]
            .astype(int)
        )

        ax.step(
            t,
            local_alarm,
            where="post",
            linewidth=2.0,
        )

        ax.set_ylim(
            -0.05,
            1.1,
        )

        ax.set_yticks(
            [0, 1]
        )

        ax.set_ylabel(
            name,
            fontsize=9,
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        detection = (
            first_detection[name]
        )

        if detection is not None:

            ax.text(
                0.99,
                0.78,
                f"detected @ {detection:.2f}s",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=8,
            )

        else:

            ax.text(
                0.99,
                0.78,
                "MISSED",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=8,
            )

    # --------------------------------------------------------
    # Final ensemble
    # --------------------------------------------------------

    ax = axes[6]

    local_alarm = (
        alarms["OR_3_P2"][
            original_index
        ]
        .astype(int)
    )

    ax.step(
        t,
        local_alarm,
        where="post",
        linewidth=3.0,
        label="OR_3_P2",
    )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_ylim(
        -0.05,
        1.1,
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_ylabel(
        "OR_3_P2",
        fontsize=9,
    )

    ax.set_xlabel(
        "Time since fault onset (s)"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    detection = (
        first_detection["OR_3_P2"]
    )

    if detection is not None:

        ax.text(
            0.99,
            0.78,
            f"FINAL DETECTION @ {detection:.2f}s",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )

    else:

        ax.text(
            0.99,
            0.78,
            "FINAL MISS",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    duration = (
        event[TIME_COL].iloc[-1]
        - event[TIME_COL].iloc[0]
    ).total_seconds()

    fig.suptitle(
        (
            f"{title}\n"
            f"duration={duration:.3f}s, "
            f"samples={len(event)}"
        ),
        fontsize=15,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Console info
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(
        f"Duration : {duration:.3f} sec"
    )

    print(
        f"Samples  : {len(event)}"
    )

    print()

    for name in alarms:

        detection = (
            first_detection[name]
        )

        if detection is None:

            print(
                f"{name:20s}: MISSED"
            )

        else:

            print(
                f"{name:20s}: "
                f"{detection:.3f} sec"
            )

    print()
    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

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
            "9_fn_visualization"
        ),
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
    )

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

    root = Path.cwd()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    normal_path = (
        root / args.normal_path
    )

    fault_path = (
        root / args.fault_path
    )

    window_1_path = (
        root
        / args.window_10
        / "model_windows.csv"
    )

    window_05_path = (
        root
        / args.window_05
        / "model_windows.csv"
    )

    # --------------------------------------------------------
    # Load modules
    # --------------------------------------------------------

    mod5_base = load_module(
        root / args.base_script,
        "kamp5_base_visualization",
    )

    mod5 = load_module(
        root / args.script_5_2,
        "kamp5_2_visualization",
    )

    mod6 = load_module(
        root / args.script_6,
        "kamp6_visualization",
    )

    # 6번 코드 내부 global
    mod6.mod5_base = mod5_base
    mod6.window_covariance = (
        args.window_covariance
    )
    mod6.sample_covariance = (
        args.sample_covariance
    )

    # --------------------------------------------------------
    # Raw
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
    # Feature columns
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
    print("=" * 80)
    print(
        f"Window features ({len(feature_cols)})"
    )
    print("=" * 80)

    print(
        feature_cols
    )

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

    # ========================================================
    # Reconstruct and plot
    # ========================================================

    for (
        event_id,
        repeat_index,
        title,
    ) in PLOT_CASES:

        seed = (
            repeat_index
        )

        print()
        print(
            "#" * 80
        )
        print(
            f"{event_id} / repeat {repeat_index}"
        )
        print(
            "#" * 80
        )

        alarms, _ = (
            reconstruct_repeat(
                raw=raw,
                feature_cols=feature_cols,
                mod5_base=mod5_base,
                mod5=mod5,
                mod6=mod6,
                window_1_path=window_1_path,
                window_05_path=window_05_path,
                seed=seed,
                n_splits=args.n_splits,
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

        safe_event_name = (
            event_id
            .replace(
                " ",
                "_",
            )
        )

        output_path = (
            output_dir
            / (
                f"{safe_event_name}"
                f"_seed{repeat_index}.png"
            )
        )

        plot_fault_event(
            raw=raw,
            alarms=alarms,
            group_id=event_id,
            title=title,
            output_path=output_path,
        )

    print()
    print("=" * 80)
    print("Visualization complete")
    print("=" * 80)
    print(
        f"Output directory: {output_dir}"
    )


if __name__ == "__main__":
    main()