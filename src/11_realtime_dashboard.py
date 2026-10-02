
from __future__ import annotations

import argparse
import platform
from collections import deque
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from core_detector import StreamingDetector
except ImportError:
    from src.core_detector import StreamingDetector


# ---------------------------------------------------------------------
# Korean font
# ---------------------------------------------------------------------
system_os = platform.system()

if system_os == "Windows":
    target_font = "Malgun Gothic"
elif system_os == "Darwin":
    target_font = "AppleGothic"
else:
    target_font = "NanumGothic"

font_names = {f.name for f in fm.fontManager.ttflist}

if target_font in font_names:
    plt.rcParams["font.family"] = target_font
else:
    fallbacks = [
        f.name
        for f in fm.fontManager.ttflist
        if any(token in f.name for token in ["Gothic", "Malgun", "Nanum", "Apple"])
    ]
    if fallbacks:
        plt.rcParams["font.family"] = fallbacks[0]

plt.rcParams["axes.unicode_minus"] = False


SENSORS = [
    "AI0_Vibration",
    "AI1_Vibration",
    "AI2_Current",
]

GAP_THRESHOLD_SEC = 0.5
DEFAULT_OUTPUT_DIR = "outputs/11_realtime_dashboard"


def preprocess_raw(
    df: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    required = ["TimeStamp", *SENSORS]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing columns: {missing}")

    work = df.copy()

    work["TimeStamp"] = pd.to_datetime(
        work["TimeStamp"],
        errors="coerce",
    )
    work = (
        work
        .dropna(subset=["TimeStamp"])
        .sort_values("TimeStamp")
        .reset_index(drop=True)
    )

    for col in SENSORS:
        work[col] = pd.to_numeric(
            work[col],
            errors="coerce",
        )

    if work[SENSORS].isna().any().any():
        raise ValueError(f"{source}: sensor data contain NaN.")

    if "Idle" in work.columns:
        work["Idle"] = pd.to_numeric(
            work["Idle"],
            errors="coerce",
        ).fillna(0)
    else:
        work["Idle"] = 0

    if "Equipment_state" in work.columns:
        work["Equipment_state"] = pd.to_numeric(
            work["Equipment_state"],
            errors="coerce",
        ).fillna(0)
    else:
        work["Equipment_state"] = 1 if source == "fault" else 0

    # Segment by the same gap rule as the offline pipeline.
    gap = work["TimeStamp"].diff().dt.total_seconds()
    break_mask = gap > GAP_THRESHOLD_SEC
    break_mask.iloc[0] = True

    work["segment_id"] = (
        break_mask.cumsum() - 1
    ).astype(int)

    work["source"] = source
    work["group_id"] = (
        source
        + "_"
        + work["segment_id"].astype(str)
    )

    if source == "fault":
        work["ground_truth"] = 1
        work["kind"] = "fault"
    else:
        work["ground_truth"] = (
            work["Equipment_state"].ge(1).astype(int)
        )

        idle_ratio = (
            work["Idle"].eq(1)
            .groupby(work["group_id"])
            .transform("mean")
        )

        work["kind"] = np.where(
            idle_ratio.ge(0.5),
            "idle",
            "normal",
        )

    return work


def select_demo_normal(
    normal_df: pd.DataFrame,
    n_samples: int,
) -> pd.DataFrame:
    """Use only non-Idle normal operation for the visual demo."""
    usable = normal_df[
        normal_df["kind"].eq("normal")
        & normal_df["Idle"].eq(0)
    ].copy()

    if usable.empty:
        raise ValueError("No non-Idle normal samples are available for the demo.")

    # Keep the latest normal samples, preserving real segment gaps.
    return usable.tail(max(1, int(n_samples))).copy()


def build_calibration_split(
    normal_df: pd.DataFrame,
    exclude_group_ids: set[str],
    fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = normal_df[
        normal_df["kind"].eq("normal")
        & normal_df["Idle"].eq(0)
        & ~normal_df["group_id"].isin(exclude_group_ids)
    ].copy()

    if usable.empty:
        raise ValueError("No normal operation remains after excluding demo groups.")

    groups = np.asarray(
        sorted(usable["group_id"].astype(str).unique()),
        dtype=str,
    )

    if len(groups) < 2:
        raise ValueError("At least two normal segments are needed for train/calibration.")

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n_cal = max(1, int(np.ceil(len(groups) * fraction)))
    n_cal = min(n_cal, len(groups) - 1)

    cal_groups = set(groups[:n_cal])

    train = usable[
        ~usable["group_id"].isin(cal_groups)
    ].copy()

    calibration = usable[
        usable["group_id"].isin(cal_groups)
    ].copy()

    return train, calibration


def make_info_text(
    now: pd.Timestamp,
    res: dict,
    detector: StreamingDetector,
    demo_fault_onset: pd.Timestamp | None,
    first_alarm_time: pd.Timestamp | None,
    current_ground_truth: int,
) -> str:
    threshold_s = detector.thresholds["sample"]
    threshold_05 = detector.thresholds["win05"]
    threshold_10 = detector.thresholds["win1"]

    status = "ALARM" if res["final_alarm"] else (
        "WARNING" if res["or3_base"] else "NORMAL"
    )

    active = ", ".join(res["active_detectors"]) if res["active_detectors"] else "None"

    lines = [
        "KAMPact | REAL-TIME DETECTOR",
        "=" * 34,
        f"Time          : {now:%H:%M:%S.%f}"[:-3],
        f"Ground Truth  : {'FAULT' if current_ground_truth else 'NORMAL'}",
        "",
        "[INDIVIDUAL DETECTORS]",
        f"Sample  : {res['sample_score']:8.2f} / {threshold_s:8.2f}"
        f"  {'ALARM' if res['sample_alarm'] else 'NORMAL'}",
        f"Win0.5  : {res['win05_score']:8.2f} / {threshold_05:8.2f}"
        f"  {'ALARM' if res['win05_alarm'] else 'NORMAL'}",
        f"Win1.0  : {res['win1_score']:8.2f} / {threshold_10:8.2f}"
        f"  {'ALARM' if res['win1_alarm'] else 'NORMAL'}",
        "",
        "[ENSEMBLE]",
        f"OR_3 candidate : {'ON' if res['or3_base'] else 'OFF'}",
        f"P2 count       : {res['persistence_count']} / {detector.p}",
        f"Active         : {active}",
        "",
        "[FINAL DECISION]",
        f">>> {status} <<<",
        "",
        "[CALIBRATION]",
        f"Quantile       : {detector.quantile:.4f}",
        f"Train segments : {detector.calibration_info.get('train_segments', '-')}",
        f"Calib segments : {detector.calibration_info.get('calibration_segments', '-')}",
    ]

    if demo_fault_onset is not None:
        lines += [
            "",
            "[DEMO FAULT EVENT]",
            f"Fault onset    : {demo_fault_onset:%H:%M:%S.%f}"[:-3],
        ]

        if first_alarm_time is not None:
            delay = (
                first_alarm_time - demo_fault_onset
            ).total_seconds()
            lines.append(f"Detection delay: {max(0.0, delay):.3f} sec")
        else:
            lines.append("Detection delay: waiting...")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KAMPact OR_3_P2 real-time streaming dashboard"
    )

    parser.add_argument(
        "--normal-path",
        default="data/press_data_normal_with_idle.csv",
    )
    parser.add_argument(
        "--fault-path",
        default="data/outlier_data.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--plot-window",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--demo-normal-samples",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.20,
        help="Fraction of non-Idle normal segments used only for threshold calibration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0.")
    if args.plot_window < 20:
        raise ValueError("--plot-window must be >= 20.")
    if not 0.05 <= args.calibration_fraction < 0.5:
        raise ValueError("--calibration-fraction must be in [0.05, 0.5).")

    normal_path = Path(args.normal_path)
    fault_path = Path(args.fault_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not normal_path.exists():
        raise FileNotFoundError(f"Normal file not found: {normal_path}")
    if not fault_path.exists():
        raise FileNotFoundError(f"Fault file not found: {fault_path}")

    print("=" * 88)
    print("KAMPact - Real-time OR_3_P2 Dashboard")
    print("=" * 88)

    normal_raw = preprocess_raw(
        pd.read_csv(normal_path),
        "normal",
    )
    fault_raw = preprocess_raw(
        pd.read_csv(fault_path),
        "fault",
    )

    demo_normal = select_demo_normal(
        normal_raw,
        args.demo_normal_samples,
    )

    demo_normal_groups = set(
        demo_normal["group_id"].astype(str).unique()
    )

    # Train/calibration are completely separated from the samples shown
    # in the demo stream.
    train_df, calibration_df = build_calibration_split(
        normal_raw,
        exclude_group_ids=demo_normal_groups,
        fraction=args.calibration_fraction,
        seed=args.seed,
    )

    detector = StreamingDetector(
        win1_size=10,          # 1.0 s at nominal 0.1 s sampling
        win05_size=5,          # 0.5 s
        sample_agg_k=3,        # causal mean3
        persistence_p=2,       # OR_3 + P2
        gap_threshold_sec=GAP_THRESHOLD_SEC,
        quantile_threshold=0.9999,
        random_state=args.seed,
    )

    fit_info = detector.fit(
        normal_df=train_df,
        calibration_df=calibration_df,
    )

    print(
        f"[INFO] Train rows      : {fit_info['train_rows']:,}\n"
        f"[INFO] Calibration rows: {fit_info['calibration_rows']:,}\n"
        f"[INFO] Train segments  : {fit_info['train_segments']}\n"
        f"[INFO] Calibration seg.: {fit_info['calibration_segments']}\n"
        f"[INFO] Thresholds      : {fit_info['thresholds']}"
    )

    # One continuous stream: normal demonstration -> fault.
    demo_df = pd.concat(
        [
            demo_normal,
            fault_raw,
        ],
        ignore_index=True,
    )

    if demo_df.empty:
        raise ValueError("Demo stream is empty.")

    demo_df = demo_df.sort_values(
        ["TimeStamp", "source", "segment_id"]
    ).reset_index(drop=True)

    # A large time gap naturally triggers detector.reset_state().
    # Store the actual source order as a visual/demo sequence too.
    first_fault_rows = demo_df[
        demo_df["source"].eq("fault")
    ]

    demo_fault_onset = (
        first_fault_rows["TimeStamp"].iloc[0]
        if not first_fault_rows.empty
        else None
    )

    print(f"[INFO] Demo stream rows : {len(demo_df):,}")
    print(f"[INFO] Demo fault onset : {demo_fault_onset}")
    print("[INFO] Starting dashboard...")

    # -----------------------------------------------------------------
    # Plot buffers
    # -----------------------------------------------------------------
    plot_window = args.plot_window
    x_data = np.arange(plot_window)

    buf_ai0 = deque([np.nan] * plot_window, maxlen=plot_window)
    buf_ai1 = deque([np.nan] * plot_window, maxlen=plot_window)
    buf_ai2 = deque([np.nan] * plot_window, maxlen=plot_window)
    buf_alarm = deque([0] * plot_window, maxlen=plot_window)
    buf_warning = deque([0] * plot_window, maxlen=plot_window)

    fig = plt.figure(figsize=(15, 9))
    try:
        fig.canvas.manager.set_window_title(
            "KAMPact Real-time OR_3_P2 Dashboard"
        )
    except Exception:
        pass

    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[3.0, 1.35],
        height_ratios=[2.0, 2.0, 1.25],
    )

    ax_vib = fig.add_subplot(grid[0, 0])
    ax_cur = fig.add_subplot(grid[1, 0])
    ax_alarm = fig.add_subplot(grid[2, 0])
    ax_info = fig.add_subplot(grid[:, 1])

    line_ai0, = ax_vib.plot(
        x_data,
        buf_ai0,
        label="AI0 Vibration",
        linewidth=1.4,
    )
    line_ai1, = ax_vib.plot(
        x_data,
        buf_ai1,
        label="AI1 Vibration",
        linewidth=1.4,
    )
    ax_vib.set_xlim(0, plot_window - 1)
    ax_vib.set_title("Real-time Sensor Streaming - Vibration")
    ax_vib.legend(loc="upper right")
    ax_vib.grid(True, alpha=0.25)

    line_ai2, = ax_cur.plot(
        x_data,
        buf_ai2,
        label="AI2 Current",
        linewidth=1.4,
    )
    ax_cur.set_xlim(0, plot_window - 1)
    ax_cur.set_title("Real-time Sensor Streaming - Current")
    ax_cur.legend(loc="upper right")
    ax_cur.grid(True, alpha=0.25)

    line_alarm, = ax_alarm.plot(
        x_data,
        buf_alarm,
        linewidth=2.0,
        label="Final Alarm",
    )
    line_warning, = ax_alarm.plot(
        x_data,
        buf_warning,
        linewidth=1.5,
        linestyle="--",
        label="OR_3 Candidate",
    )
    ax_alarm.set_xlim(0, plot_window - 1)
    ax_alarm.set_ylim(-0.15, 1.15)
    ax_alarm.set_yticks([0, 1])
    ax_alarm.set_yticklabels(["Normal", "FAULT"])
    ax_alarm.set_title("Decision Timeline - OR_3 + P2")
    ax_alarm.legend(loc="upper right")
    ax_alarm.grid(True, alpha=0.25)

    ax_info.axis("off")
    info_text = ax_info.text(
        0.03,
        0.98,
        "",
        transform=ax_info.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.92,
        },
    )

    plt.tight_layout()

    first_alarm_time: pd.Timestamp | None = None
    log_rows: list[dict[str, object]] = []

    def update(frame: int):
        nonlocal first_alarm_time

        if frame >= len(demo_df):
            if getattr(update, "event_source", None) is not None:
                update.event_source.stop()
            return (
                line_ai0,
                line_ai1,
                line_ai2,
                line_alarm,
                line_warning,
                info_text,
            )

        row = demo_df.iloc[frame]

        timestamp = pd.Timestamp(row["TimeStamp"])
        ai0 = float(row["AI0_Vibration"])
        ai1 = float(row["AI1_Vibration"])
        ai2 = float(row["AI2_Current"])
        ground_truth = int(row["ground_truth"])

        result = detector.step(
            timestamp,
            ai0,
            ai1,
            ai2,
        )

        if (
            demo_fault_onset is not None
            and timestamp >= demo_fault_onset
            and result["final_alarm"]
            and first_alarm_time is None
        ):
            first_alarm_time = timestamp

        buf_ai0.append(ai0)
        buf_ai1.append(ai1)
        buf_ai2.append(ai2)
        buf_alarm.append(1 if result["final_alarm"] else 0)
        buf_warning.append(1 if result["or3_base"] else 0)

        line_ai0.set_ydata(buf_ai0)
        line_ai1.set_ydata(buf_ai1)
        line_ai2.set_ydata(buf_ai2)
        line_alarm.set_ydata(buf_alarm)
        line_warning.set_ydata(buf_warning)

        # Dynamic axes.
        vib = np.concatenate(
            [
                np.asarray(buf_ai0, dtype=float),
                np.asarray(buf_ai1, dtype=float),
            ]
        )
        vib = vib[np.isfinite(vib)]

        if len(vib):
            vmin = float(vib.min())
            vmax = float(vib.max())
            margin = max((vmax - vmin) * 0.10, 0.05)
            if np.isclose(vmin, vmax):
                margin = max(abs(vmin) * 0.05, 0.05)
            ax_vib.set_ylim(vmin - margin, vmax + margin)

        cur = np.asarray(buf_ai2, dtype=float)
        cur = cur[np.isfinite(cur)]

        if len(cur):
            cmin = float(cur.min())
            cmax = float(cur.max())
            margin = max((cmax - cmin) * 0.10, 1.0)
            if np.isclose(cmin, cmax):
                margin = max(abs(cmin) * 0.05, 1.0)
            ax_cur.set_ylim(cmin - margin, cmax + margin)

        # Current state display.
        status = (
            "ALARM"
            if result["final_alarm"]
            else "WARNING"
            if result["or3_base"]
            else "NORMAL"
        )

        info_text.set_text(
            make_info_text(
                now=timestamp,
                res=result,
                detector=detector,
                demo_fault_onset=demo_fault_onset,
                first_alarm_time=first_alarm_time,
                current_ground_truth=ground_truth,
            )
        )

        # Background indicates only the current final decision.
        if status == "ALARM":
            fig.patch.set_alpha(1.0)
        else:
            fig.patch.set_alpha(1.0)

        log_rows.append(
            {
                "stream_index": int(frame),
                "source": row["source"],
                "TimeStamp": timestamp,
                "ground_truth": ground_truth,
                "AI0_Vibration": ai0,
                "AI1_Vibration": ai1,
                "AI2_Current": ai2,
                "score_0.5s": result["win05_score"],
                "threshold_0.5s": detector.thresholds["win05"],
                "alarm_0.5s": result["win05_alarm"],
                "score_1.0s": result["win1_score"],
                "threshold_1.0s": detector.thresholds["win1"],
                "alarm_1.0s": result["win1_alarm"],
                "sample_score": result["sample_score"],
                "sample_threshold": detector.thresholds["sample"],
                "sample_alarm": result["sample_alarm"],
                "OR_3": result["or3_base"],
                "P2": result["final_alarm"],
                "state": status,
                "active_detectors": ",".join(result["active_detectors"])
                if result["active_detectors"]
                else "",
            }
        )

        return (
            line_ai0,
            line_ai1,
            line_ai2,
            line_alarm,
            line_warning,
            info_text,
        )

    interval_ms = max(1, int(1000 / args.fps))

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(demo_df),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    # Keep a reference because matplotlib may otherwise warn about GC.
    update.event_source = ani.event_source

    try:
        plt.show()
    finally:
        output_path = output_dir / "realtime_log.csv"

        if log_rows:
            log_df = pd.DataFrame(log_rows)
            log_df.to_csv(
                output_path,
                index=False,
                encoding="utf-8-sig",
            )

            print(f"[SAVED] {output_path}")
        else:
            print("[WARN] No realtime log rows were produced.")


if __name__ == "__main__":
    main()
