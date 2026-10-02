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
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

try:
    from core_detector import StreamingDetector, WINDOW_FEATURES, SENSOR_COLS
except ImportError:
    from src.core_detector import StreamingDetector, WINDOW_FEATURES, SENSOR_COLS


# ---------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------
C = {
    "bg": "#0b0f17",
    "panel": "#121926",
    "border": "#27344a",
    "grid": "#202b3d",
    "text": "#e6edf3",
    "muted": "#8b98a9",
    "dim": "#3a4a63",
    "green": "#2ecc71",
    "amber": "#f5a623",
    "red": "#ff4d5e",
    "blue": "#4da3ff",
    "cyan": "#22d3ee",
    "purple": "#b48cff",
    "orange": "#ff9f43",
}

# Timeline strip color codes
CODE_EMPTY, CODE_OK, CODE_WARN, CODE_FAULT, CODE_IDLE = 0, 1, 2, 3, 4
STRIP_CMAP = ListedColormap(
    [C["bg"], C["green"], C["amber"], C["red"], C["blue"]]
)
KIND_TO_CODE = {"normal": CODE_OK, "idle": CODE_IDLE, "fault": CODE_FAULT}

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
    KOREAN_FONT = target_font
else:
    fallbacks = [
        f.name
        for f in fm.fontManager.ttflist
        if any(token in f.name for token in ["Gothic", "Malgun", "Nanum", "Apple"])
    ]
    KOREAN_FONT = fallbacks[0] if fallbacks else None

if KOREAN_FONT:
    plt.rcParams["font.family"] = KOREAN_FONT

# A Korean monospace font is required for the live feature table.
# Prefer a CJK monospace font so Korean glyphs keep the same cell width as numbers.
MONO_FONT = None
for _candidate in ("Noto Sans Mono CJK KR", "NanumGothicCoding"):
    try:
        _path = fm.findfont(_candidate, fallback_to_default=False)
        MONO_FONT = fm.FontProperties(fname=_path).get_name()
        break
    except Exception:
        pass
if MONO_FONT is None:
    MONO_FONT = "DejaVu Sans Mono"

plt.rcParams.update(
    {
        "axes.unicode_minus": False,
        "figure.facecolor": C["bg"],
        "axes.facecolor": C["panel"],
        "axes.edgecolor": C["border"],
        "axes.labelcolor": C["muted"],
        "text.color": C["text"],
        "xtick.color": C["muted"],
        "ytick.color": C["muted"],
        "grid.color": C["grid"],
        "legend.facecolor": C["panel"],
        "legend.edgecolor": C["border"],
    }
)


SENSORS = list(SENSOR_COLS)
GAP_THRESHOLD_SEC = 0.5
DEFAULT_OUTPUT_DIR = "outputs/11_realtime_dashboard"
DEFAULT_QUANTILE = 0.9999
GAUGE_CAP = 3.0  # gauge bars are drawn up to 3x threshold

PREPROCESS_STEPS = [
    "원본 데이터 입력",
    "데이터 검증",
    "타임스탬프 간격 / 세그먼트",
    "유휴 상태 분류",
    "특징 및 기준값 준비",
]


# ---------------------------------------------------------------------
# Raw preprocessing
# ---------------------------------------------------------------------
def preprocess_raw(
    df: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    required = ["TimeStamp", *SENSORS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing columns: {missing}")

    work = df.copy()
    original_rows = len(work)

    work["TimeStamp"] = pd.to_datetime(work["TimeStamp"], errors="coerce")
    invalid_time = int(work["TimeStamp"].isna().sum())
    work = (
        work.dropna(subset=["TimeStamp"])
        .sort_values("TimeStamp")
        .reset_index(drop=True)
    )

    for col in SENSORS:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    invalid_sensor_rows = int(work[SENSORS].isna().any(axis=1).sum())
    if invalid_sensor_rows:
        raise ValueError(f"{source}: sensor data contain {invalid_sensor_rows} invalid rows.")

    if "Idle" in work.columns:
        work["Idle"] = pd.to_numeric(work["Idle"], errors="coerce").fillna(0)
    else:
        work["Idle"] = 0

    if "Equipment_state" in work.columns:
        work["Equipment_state"] = pd.to_numeric(
            work["Equipment_state"], errors="coerce"
        ).fillna(0)
    else:
        work["Equipment_state"] = 1 if source == "fault" else 0

    gap = work["TimeStamp"].diff().dt.total_seconds()
    break_mask = gap > GAP_THRESHOLD_SEC
    break_mask.iloc[0] = True

    work["gap_seconds"] = gap.fillna(0.0)
    work["segment_id"] = (break_mask.cumsum() - 1).astype(int)
    work["source"] = source
    work["group_id"] = source + "_" + work["segment_id"].astype(str)

    if source == "fault":
        work["ground_truth"] = 1
        work["kind"] = "fault"
    else:
        work["ground_truth"] = work["Equipment_state"].ge(1).astype(int)
        idle_ratio = (
            work["Idle"].eq(1)
            .groupby(work["group_id"])
            .transform("mean")
        )
        work["kind"] = np.where(idle_ratio.ge(0.5), "idle", "normal")

    # Metadata shown in the preprocessing stage.
    work.attrs["preprocess_stats"] = {
        "source": source,
        "original_rows": original_rows,
        "valid_rows": len(work),
        "invalid_timestamp": invalid_time,
        "invalid_sensor_rows": invalid_sensor_rows,
        "segments": int(work["group_id"].nunique()),
        "gaps_over_threshold": int((work["gap_seconds"] > GAP_THRESHOLD_SEC).sum()),
        "idle_rows": int(work["Idle"].eq(1).sum()),
        "idle_segments": int(work.loc[work["kind"].eq("idle"), "group_id"].nunique()),
    }

    return work


# ---------------------------------------------------------------------
# Demo data selection
# ---------------------------------------------------------------------
def select_demo_normal(normal_df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    usable = normal_df[
        normal_df["kind"].eq("normal") & normal_df["Idle"].eq(0)
    ].copy()
    if usable.empty:
        raise ValueError("No non-Idle normal samples are available for the demo.")
    return usable.tail(max(1, int(n_samples))).copy()


def select_demo_idle(normal_df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    idle = normal_df[
        normal_df["kind"].eq("idle") & normal_df["Idle"].eq(1)
    ].copy()
    if idle.empty:
        raise ValueError("No Idle samples are available for the demo stream.")

    group_sizes = idle.groupby("group_id").size()
    target_group = str(group_sizes.idxmax())
    selected = idle[idle["group_id"].eq(target_group)].copy()
    return selected.head(max(1, int(n_samples))).copy()


def build_demo_stream(
    blocks: list[tuple[str, pd.DataFrame]],
    join_gap_sec: float = 2.0,
) -> pd.DataFrame:
    if join_gap_sec <= GAP_THRESHOLD_SEC:
        raise ValueError("join_gap_sec must be larger than the detector gap threshold.")

    out: list[pd.DataFrame] = []
    cursor: pd.Timestamp | None = None

    for name, raw_df in blocks:
        df = raw_df.sort_values("TimeStamp").copy().reset_index(drop=True)
        if df.empty:
            continue

        df["orig_TimeStamp"] = df["TimeStamp"]

        if cursor is not None:
            shift = (
                cursor
                + pd.Timedelta(seconds=join_gap_sec)
                - df["TimeStamp"].iloc[0]
            )
            df["TimeStamp"] = df["TimeStamp"] + shift

        cursor = df["TimeStamp"].iloc[-1]
        df["block"] = name
        out.append(df)

    if not out:
        raise ValueError("No non-empty demo blocks were provided.")

    return pd.concat(out, ignore_index=True).reset_index(drop=True)


# ---------------------------------------------------------------------
# Train / calibration split
# ---------------------------------------------------------------------
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

    groups = np.asarray(sorted(usable["group_id"].astype(str).unique()), dtype=str)
    if len(groups) < 2:
        raise ValueError("At least two normal segments are needed for train/calibration.")

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n_cal = max(1, int(np.ceil(len(groups) * fraction)))
    n_cal = min(n_cal, len(groups) - 1)
    cal_groups = set(groups[:n_cal])

    train = usable[~usable["group_id"].isin(cal_groups)].copy()
    calibration = usable[usable["group_id"].isin(cal_groups)].copy()
    return train, calibration


# ---------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------
def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def ts_str(t) -> str:
    return f"{pd.Timestamp(t):%H:%M:%S.%f}"[:-3]


def score_ratio(score: float, thr: float) -> float:
    if not (np.isfinite(score) and np.isfinite(thr)) or thr <= 0 or score <= 0:
        return np.nan
    return float(score / thr)


def short_feature_name(name: str) -> str:
    short = (
        name.replace("AI0_Vibration", "A0")
        .replace("AI1_Vibration", "A1")
        .replace("AI2_Current", "A2")
        .replace("AI0_AI1_corr", "A0-A1 corr")
    )
    for suffix in ("mean", "std", "rms", "ptp", "slope"):
        if short.endswith("_" + suffix):
            short = short[: -(len(suffix) + 1)] + "." + suffix
            break
    return short


def get_feature_snapshot(detector: StreamingDetector) -> dict[str, float]:
    """Read the exact feature implementation used by StreamingDetector."""
    values = list(detector.raw_buffer)
    times = list(detector.time_buffer)
    out: dict[str, float] = {}

    if len(values) >= detector.win05_size:
        arr = np.asarray(values[-detector.win05_size :], dtype=float)
        ts = np.asarray(times[-detector.win05_size :])
        feat = detector._extract_window_features(arr, ts)
        out.update({f"W05_{name}": float(v) for name, v in zip(WINDOW_FEATURES, feat)})

    if len(values) >= detector.win1_size:
        arr = np.asarray(values[-detector.win1_size :], dtype=float)
        ts = np.asarray(times[-detector.win1_size :])
        feat = detector._extract_window_features(arr, ts)
        out.update({f"W10_{name}": float(v) for name, v in zip(WINDOW_FEATURES, feat)})

    return out


def style_card(ax, title: str, subtitle: str | None = None) -> None:
    """Rounded-looking dark card used for all text/gauge panels."""
    ax.set_facecolor(C["panel"])
    for spine in ax.spines.values():
        spine.set_color(C["border"])
        spine.set_linewidth(1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        title, loc="left", fontsize=10.5, fontweight="bold", color=C["text"], pad=7,
        fontname=KOREAN_FONT or "DejaVu Sans",
    )
    if subtitle:
        ax.text(
            1.0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            ha="right",
            fontname=KOREAN_FONT or "DejaVu Sans",
            va="bottom",
            fontsize=8,
            color=C["muted"],
        )


def style_plot(ax, title: str, ylabel: str = "") -> None:
    ax.set_facecolor(C["panel"])
    for spine in ax.spines.values():
        spine.set_color(C["border"])
        spine.set_linewidth(1.1)
    ax.set_title(
        title, loc="left", fontsize=10.5, fontweight="bold", color=C["text"], pad=7,
        fontname=KOREAN_FONT or "DejaVu Sans",
    )
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, fontname=KOREAN_FONT or "DejaVu Sans")
    ax.grid(True, alpha=0.6, linewidth=0.7)
    ax.tick_params(labelsize=8)


def mono_text(ax, x: float, y: float, size: float = 9.0, color: str | None = None):
    return ax.text(
        x,
        y,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=size,
        family=MONO_FONT,
        color=color or C["text"],
        linespacing=1.45,
    )


def make_pill(ax, x: float, w: float, label: str):
    # Header status card: no border, wider vertical padding, white text.
    patch = FancyBboxPatch(
        (x, 0.07),
        w,
        0.84,
        boxstyle="round,pad=0,rounding_size=0.008",
        transform=ax.transAxes,
        facecolor=C["dim"],
        edgecolor="none",
        linewidth=0,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        0.70,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=C["text"],
        fontname=KOREAN_FONT or "DejaVu Sans",
    )
    value = ax.text(
        x + w / 2,
        0.40,
        "-",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=C["text"],
        fontname=KOREAN_FONT or "DejaVu Sans",
    )
    return patch, value


def set_pill(pill, text: str, color: str) -> None:
    from matplotlib.colors import to_rgba

    patch, value = pill
    rgb = to_rgba(color)[:3]
    patch.set_facecolor((*rgb, 0.30))
    patch.set_edgecolor("none")
    value.set_text(text)
    value.set_color(C["text"])


def step_details(
    normal_stats: dict,
    fault_stats: dict,
    train_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    quantile: float,
) -> list[str]:
    return [
        f"정상 {normal_stats['valid_rows']:,} / 고장 {fault_stats['valid_rows']:,}행  |  센서 3개",
        (
            f"잘못된 시간 N{normal_stats['invalid_timestamp']}/F{fault_stats['invalid_timestamp']}"
            f"  |  잘못된 센서 N{normal_stats['invalid_sensor_rows']}/F{fault_stats['invalid_sensor_rows']}"
        ),
        (
            f"간격 > {GAP_THRESHOLD_SEC:.1f}초  |  세그먼트 N{normal_stats['segments']}"
            f" / F{fault_stats['segments']}"
        ),
        (
            f"유휴 {normal_stats['idle_rows']:,}행 / {normal_stats['idle_segments']}개 구간"
            "  |  학습: 운전 상태만"
        ),
        f"학습 {len(train_df):,} / 보정 {len(calibration_df):,}  |  q={quantile:.4f}",
    ]


def make_model_text(
    fit_info: dict,
    detector: StreamingDetector,
    stage_index: int,
) -> str:
    lines = [
        "스트리밍 대기",
        "-" * 30,
    ]
    if stage_index >= 4:
        lines += [
            "모델 준비 완료",
            "",
            f"학습 데이터   {fit_info['train_rows']:>10,}",
            f"보정 데이터   {fit_info['calibration_rows']:>10,}",
            "",
            "샘플 특징     원본 + 1차 차분 (6)",
            "윈도우       0.5초 + 1.0초",
            "윈도우 특징   각 16개",
            "변환         signed-log1p",
            "공분산       Ledoit-Wolf",
            f"기준 분위수   {detector.quantile:.4f}",
            "앙상블       OR_3 + P2",
            "",
            "다음: 정상 > 유휴 > 고장",
        ]
    else:
        lines += [
            "스트리밍 탐지에 사용할",
            "데이터 흐름을 준비합니다.",
            "",
            "원본 > 검증 > 세그먼트",
            "> 유휴/정상 > 특징 추출",
            "> 마할라노비스 > OR_3 > P2",
        ]
    return "\n".join(lines)


def create_feature_panel(ax):
    """Create fixed-position text artists for a stable, perfectly aligned table."""
    font = MONO_FONT
    # All three numeric columns use the same x-coordinate throughout the panel.
    # Values themselves are formatted to a fixed-width, decimal-aligned string.
    COL_X = {
        "label": 0.05,
        "col1": 0.39,
        "col2": 0.64,
    }

    artists = {
        "sample_title": ax.text(COL_X["label"], 0.945, "샘플 (원본 + 차분)", transform=ax.transAxes,
                                 ha="left", va="top", fontsize=8.3, fontname=font, color=C["text"]),
        "sample_header_orig": ax.text(COL_X["col1"], 0.895, "원본", transform=ax.transAxes,
                                       ha="left", va="top", fontsize=8.0, fontname=font, color=C["muted"]),
        "sample_header_diff": ax.text(COL_X["col2"], 0.895, "차분", transform=ax.transAxes,
                                       ha="left", va="top", fontsize=8.0, fontname=font, color=C["muted"]),
        "window_title": ax.text(COL_X["label"], 0.690, "윈도우 특징", transform=ax.transAxes,
                                  ha="left", va="top", fontsize=8.3, fontname=font, color=C["text"]),
        "window_header_05": ax.text(COL_X["col1"], 0.650, "0.5초", transform=ax.transAxes,
                                     ha="left", va="top", fontsize=8.0, fontname=font, color=C["muted"]),
        "window_header_10": ax.text(COL_X["col2"], 0.650, "1.0초", transform=ax.transAxes,
                                     ha="left", va="top", fontsize=8.0, fontname=font, color=C["muted"]),
        "waiting": ax.text(COL_X["label"], 0.605, "", transform=ax.transAxes,
                            ha="left", va="top", fontsize=7.8, fontname=font, color=C["muted"]),
    }

    # Sample rows.  All values are decimal-aligned inside a fixed-width cell.
    sample_y = [0.845, 0.800, 0.755]
    for key, label, y in zip(("ai0", "ai1", "ai2"), ("AI0", "AI1", "AI2"), sample_y):
        artists[f"sample_label_{key}"] = ax.text(COL_X["label"], y, label, transform=ax.transAxes,
                                                   ha="left", va="top", fontsize=8.1, fontname=font, color=C["text"])
        artists[f"sample_orig_{key}"] = ax.text(COL_X["col1"], y, "-", transform=ax.transAxes,
                                                  ha="left", va="top", fontsize=8.1, fontname=font, color=C["text"])
        artists[f"sample_diff_{key}"] = ax.text(COL_X["col2"], y, "-", transform=ax.transAxes,
                                                  ha="left", va="top", fontsize=8.1, fontname=font, color=C["text"])

    # Divider is aligned with the data columns rather than relying on spaces.
    artists["divider"] = ax.plot(
        [COL_X["label"], 0.91], [0.620, 0.620],
        transform=ax.transAxes, color=C["dim"], lw=0.8, ls="--", zorder=2,
    )[0]

    # 16 window-feature rows.
    window_start_y = 0.585
    window_step = 0.034
    for i, name in enumerate(WINDOW_FEATURES):
        y = window_start_y - i * window_step
        artists[f"win_label_{i}"] = ax.text(COL_X["label"], y, short_feature_name(name), transform=ax.transAxes,
                                              ha="left", va="top", fontsize=7.8, fontname=font, color=C["text"])
        artists[f"win05_{i}"] = ax.text(COL_X["col1"], y, "-", transform=ax.transAxes,
                                         ha="left", va="top", fontsize=7.8, fontname=font, color=C["text"])
        artists[f"win10_{i}"] = ax.text(COL_X["col2"], y, "-", transform=ax.transAxes,
                                         ha="left", va="top", fontsize=7.8, fontname=font, color=C["text"])

    return artists


def update_feature_panel(artists, detector: StreamingDetector, feature_snapshot: dict[str, float]) -> None:
    """Update the fixed-position feature table without whitespace-based alignment."""
    current = detector.raw_buffer[-1]
    previous = detector.raw_buffer[-2] if len(detector.raw_buffer) >= 2 else None
    diffs = np.zeros(3, dtype=float) if previous is None else current - previous

    def fixed_num(value: float | None, decimals: int = 4, integer_width: int = 5) -> str:
        if value is None or not np.isfinite(value):
            return "-".ljust(11)

        value = float(value)
        sign = "-" if value < 0 else " "
        abs_value = abs(value)
        integer_part = int(abs_value)
        fractional = int(round((abs_value - integer_part) * (10 ** decimals)))
        if fractional >= 10 ** decimals:
            integer_part += 1
            fractional = 0

        # Fixed integer field + fixed decimals => decimal point is always at
        # the same visual x-position even when values change magnitude/sign.
        return f"{sign}{integer_part:>{integer_width}}.{fractional:0{decimals}d}"

    # Keep every raw/diff value in the same numeric format.
    for key, idx in (("ai0", 0), ("ai1", 1), ("ai2", 2)):
        artists[f"sample_orig_{key}"].set_text(fixed_num(float(current[idx]), 4))
        artists[f"sample_diff_{key}"].set_text(fixed_num(float(diffs[idx]), 4))

    if not feature_snapshot:
        artists["waiting"].set_text(
            f"버퍼 {len(detector.raw_buffer)}/{detector.win1_size} 샘플   |   인과 윈도우 준비 중..."
        )
    else:
        artists["waiting"].set_text("")
        for i, name in enumerate(WINDOW_FEATURES):
            v05 = feature_snapshot.get(f"W05_{name}")
            v10 = feature_snapshot.get(f"W10_{name}")
            artists[f"win05_{i}"].set_text(fixed_num(v05, 4))
            artists[f"win10_{i}"].set_text(fixed_num(v10, 4))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="KAMPact preprocessing-to-realtime OR_3_P2 dashboard"
    )
    parser.add_argument("--normal-path", default="data/press_data_normal_with_idle.csv")
    parser.add_argument("--fault-path", default="data/outlier_data.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--feature-update-every",
        type=int,
        default=3,
        help="Update the 16-feature text panel every N stream frames. Detection still runs every sample.",
    )
    parser.add_argument(
        "--axis-update-every",
        type=int,
        default=5,
        help="Recalculate plot y-limits every N stream frames.",
    )
    parser.add_argument("--plot-window", type=int, default=150)
    parser.add_argument("--demo-normal-samples", type=int, default=800)
    parser.add_argument("--demo-idle-samples", type=int, default=200)
    parser.add_argument("--demo-join-gap", type=float, default=2.0)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preprocess-seconds",
        type=float,
        default=6.0,
        help="Seconds spent showing preprocessing/model preparation before streaming.",
    )
    parser.add_argument(
        "--skip-intro",
        action="store_true",
        help="Skip the visible preprocessing intro and start streaming immediately.",
    )

    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0.")
    if args.feature_update_every < 1:
        raise ValueError("--feature-update-every must be >= 1.")
    if args.axis_update_every < 1:
        raise ValueError("--axis-update-every must be >= 1.")
    if args.plot_window < 20:
        raise ValueError("--plot-window must be >= 20.")
    if args.demo_normal_samples < 1 or args.demo_idle_samples < 1:
        raise ValueError("Demo sample counts must be >= 1.")
    if args.demo_join_gap <= GAP_THRESHOLD_SEC:
        raise ValueError("--demo-join-gap must be > 0.5 sec.")
    if not 0.05 <= args.calibration_fraction < 0.5:
        raise ValueError("--calibration-fraction must be in [0.05, 0.5).")
    if not 0.95 <= args.quantile < 1.0:
        raise ValueError("--quantile must be in [0.95, 1.0).")
    if args.preprocess_seconds < 0:
        raise ValueError("--preprocess-seconds must be >= 0.")

    normal_path = Path(args.normal_path)
    fault_path = Path(args.fault_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not normal_path.exists():
        raise FileNotFoundError(f"Normal file not found: {normal_path}")
    if not fault_path.exists():
        raise FileNotFoundError(f"Fault file not found: {fault_path}")

    print("=" * 96)
    print("KAMPact - Real-time OR_3_P2 Dashboard | PREPROCESSING -> FEATURE -> DETECTION")
    print("=" * 96)

    # ============================================================
    # 1. Actual raw preprocessing
    # ============================================================
    normal_raw = preprocess_raw(pd.read_csv(normal_path), "normal")
    fault_raw = preprocess_raw(pd.read_csv(fault_path), "fault")

    normal_stats = normal_raw.attrs["preprocess_stats"]
    fault_stats = fault_raw.attrs["preprocess_stats"]

    # ============================================================
    # 2. Demo data selection (never used for train/calibration)
    # ============================================================
    demo_normal = select_demo_normal(normal_raw, args.demo_normal_samples)
    demo_idle = select_demo_idle(normal_raw, args.demo_idle_samples)

    demo_normal_groups = set(demo_normal["group_id"].astype(str).unique())
    demo_idle_groups = set(demo_idle["group_id"].astype(str).unique())
    excluded_demo_groups = demo_normal_groups | demo_idle_groups

    train_df, calibration_df = build_calibration_split(
        normal_raw,
        exclude_group_ids=excluded_demo_groups,
        fraction=args.calibration_fraction,
        seed=args.seed,
    )

    # ============================================================
    # 3. Actual model fit / threshold calibration
    # ============================================================
    detector = StreamingDetector(
        win1_size=10,
        win05_size=5,
        sample_agg_k=3,
        persistence_p=2,
        gap_threshold_sec=GAP_THRESHOLD_SEC,
        quantile_threshold=args.quantile,
        random_state=args.seed,
    )

    fit_info = detector.fit(
        normal_df=train_df,
        calibration_df=calibration_df,
    )

    print(f"[INFO] Normal raw rows : {normal_stats['valid_rows']:,}")
    print(f"[INFO] Fault raw rows  : {fault_stats['valid_rows']:,}")
    print(f"[INFO] Normal segments : {normal_stats['segments']}")
    print(f"[INFO] Fault segments  : {fault_stats['segments']}")
    print(f"[INFO] Idle rows       : {normal_stats['idle_rows']:,}")
    print(f"[INFO] Train rows      : {fit_info['train_rows']:,}")
    print(f"[INFO] Calibration rows: {fit_info['calibration_rows']:,}")
    print(f"[INFO] Thresholds      : {fit_info['thresholds']}")

    # ============================================================
    # 4. Explicit visual replay: NORMAL -> IDLE -> FAULT
    # ============================================================
    demo_df = build_demo_stream(
        [
            ("normal", demo_normal),
            ("idle", demo_idle),
            ("fault", fault_raw),
        ],
        join_gap_sec=args.demo_join_gap,
    )

    fault_rows = demo_df[demo_df["source"].eq("fault")]
    demo_fault_onset = fault_rows["TimeStamp"].iloc[0] if not fault_rows.empty else None
    demo_fault_original_onset = (
        fault_rows["orig_TimeStamp"].iloc[0] if not fault_rows.empty else None
    )

    print(
        f"[INFO] Demo stream rows : {len(demo_df):,} "
        f"(normal={len(demo_normal):,}, idle={len(demo_idle):,}, fault={len(fault_raw):,})"
    )
    print(f"[INFO] Demo fault onset : {demo_fault_onset}")
    print("[INFO] Starting dashboard...")

    # ============================================================
    # 5. Figure layout
    # ============================================================
    plot_window = args.plot_window
    x_data = np.arange(plot_window)

    def nan_deque() -> deque:
        return deque([np.nan] * plot_window, maxlen=plot_window)

    def zero_deque() -> deque:
        return deque([CODE_EMPTY] * plot_window, maxlen=plot_window)

    bufs = {
        "ai0": nan_deque(),
        "ai1": nan_deque(),
        "ai2": nan_deque(),
        "r_sample": nan_deque(),
        "r_w05": nan_deque(),
        "r_w1": nan_deque(),
        "truth": zero_deque(),
        "det": zero_deque(),
    }

    def push(**values) -> None:
        for key, buf in bufs.items():
            buf.append(values[key])

    GAP_ENTRY = dict(
        ai0=np.nan,
        ai1=np.nan,
        ai2=np.nan,
        r_sample=np.nan,
        r_w05=np.nan,
        r_w1=np.nan,
        truth=CODE_EMPTY,
        det=CODE_EMPTY,
    )

    fig = plt.figure(figsize=(20, 11), facecolor=C["bg"])
    try:
        fig.canvas.manager.set_window_title("KAMPact - Real-time Anomaly Detection")
    except Exception:
        pass

    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.70, 10],
        hspace=0.11,
        left=0.02,
        right=0.985,
        top=0.975,
        bottom=0.105,
    )
    ax_hdr = fig.add_subplot(outer[0])
    body = outer[1].subgridspec(1, 3, width_ratios=[2.5, 1.05, 1.15], wspace=0.10)
    left = body[0].subgridspec(4, 1, height_ratios=[1.5, 1.0, 1.35, 0.75], hspace=0.40)
    mid = body[1].subgridspec(3, 1, height_ratios=[1.6, 1.3, 1.3], hspace=0.20)
    right = body[2].subgridspec(2, 1, height_ratios=[1.15, 2.4], hspace=0.16)

    ax_vib = fig.add_subplot(left[0])
    ax_cur = fig.add_subplot(left[1], sharex=ax_vib)
    ax_ratio = fig.add_subplot(left[2], sharex=ax_vib)
    ax_strip = fig.add_subplot(left[3], sharex=ax_vib)
    ax_pipe = fig.add_subplot(mid[0])
    ax_live = fig.add_subplot(mid[1])
    ax_event = fig.add_subplot(mid[2])
    ax_det = fig.add_subplot(right[0])
    ax_feat = fig.add_subplot(right[1])

    # ---------------- Header ----------------
    ax_hdr.axis("off")
    ax_hdr.set_xlim(0, 1)
    ax_hdr.set_ylim(0, 1)
    ax_hdr.text(
        0.002, 0.68, "KAMPact", transform=ax_hdr.transAxes,
        fontsize=23, fontweight="bold", color=C["text"], va="center",
    )
    ax_hdr.text(
        0.002, 0.24,
        "프레스 이상 탐지  |  실시간 재생  정상 > 유휴 > 고장",
        transform=ax_hdr.transAxes, fontsize=9.5, color=C["muted"], va="center", fontname=KOREAN_FONT or "DejaVu Sans",
    )
    # Align the two left cards to the DATA PIPELINE width and split them 50/50.
    # The right card exactly matches the DETECTION panel width below.
    hdr_pos = ax_hdr.get_position()
    pipe_pos = ax_pipe.get_position()
    det_pos = ax_det.get_position()

    def fig_x_to_hdr(x_fig: float) -> float:
        return (x_fig - hdr_pos.x0) / hdr_pos.width

    def fig_w_to_hdr(w_fig: float) -> float:
        return w_fig / hdr_pos.width

    status_gap_fig = 0.008
    left_card_w = max(0.01, (pipe_pos.width - status_gap_fig) / 2.0)

    phase_x = fig_x_to_hdr(pipe_pos.x0)
    truth_x = fig_x_to_hdr(pipe_pos.x0 + left_card_w + status_gap_fig)
    det_x = fig_x_to_hdr(det_pos.x0)

    pill_phase = make_pill(ax_hdr, phase_x, fig_w_to_hdr(left_card_w), "단계")
    pill_truth = make_pill(ax_hdr, truth_x, fig_w_to_hdr(left_card_w), "실제 상태")
    pill_det = make_pill(ax_hdr, det_x, fig_w_to_hdr(det_pos.width), "탐지 결과")
    ax_hdr.add_patch(
        Rectangle((0, 0.0), 1, 0.05, transform=ax_hdr.transAxes,
                  facecolor=C["panel"], edgecolor="none")
    )
    progress_bar = Rectangle(
        (0, 0.0), 0, 0.05, transform=ax_hdr.transAxes,
        facecolor=C["blue"], edgecolor="none",
    )
    ax_hdr.add_patch(progress_bar)

    # ---------------- Vibration ----------------
    style_plot(ax_vib, "진동   AI0 / AI1")
    (line_ai0,) = ax_vib.plot(x_data, list(bufs["ai0"]), color=C["cyan"], lw=1.5, label="AI0")
    (line_ai1,) = ax_vib.plot(x_data, list(bufs["ai1"]), color=C["purple"], lw=1.5, label="AI1")
    ax_vib.legend(loc="upper left", ncol=2, fontsize=8, frameon=True, labelcolor=C["text"], prop={"family": KOREAN_FONT or "DejaVu Sans"})
    ax_vib.set_xlim(-0.5, plot_window - 0.5)
    ax_vib.set_autoscale_on(False)
    ax_vib.tick_params(labelbottom=False)

    # ---------------- Current ----------------
    style_plot(ax_cur, "전류   AI2")
    (line_ai2,) = ax_cur.plot(x_data, list(bufs["ai2"]), color=C["orange"], lw=1.5, label="AI2")
    ax_cur.set_autoscale_on(False)
    ax_cur.tick_params(labelbottom=False)

    # ---------------- Score / threshold ratio ----------------
    style_plot(ax_ratio, "이상 점수 / 기준값   (1.0 초과 = 탐지 알람)")
    ax_ratio.set_yscale("log")
    (line_r_sample,) = ax_ratio.plot(x_data, list(bufs["r_sample"]), color=C["cyan"], lw=1.3, label="샘플")
    (line_r_w05,) = ax_ratio.plot(x_data, list(bufs["r_w05"]), color=C["purple"], lw=1.3, label="0.5초 윈도우")
    (line_r_w1,) = ax_ratio.plot(x_data, list(bufs["r_w1"]), color=C["orange"], lw=1.3, label="1.0초 윈도우")
    ax_ratio.axhline(1.0, color=C["red"], lw=1.2, ls="--")
    ax_ratio.axhspan(1.0, 1e9, color=C["red"], alpha=0.06, lw=0)
    ax_ratio.set_ylim(0.05, 10)
    ax_ratio.set_autoscale_on(False)
    ax_ratio.legend(loc="upper left", ncol=3, fontsize=8, labelcolor=C["text"], prop={"family": KOREAN_FONT or "DejaVu Sans"})
    ax_ratio.tick_params(labelbottom=False)

    # ---------------- Timeline strip ----------------
    style_plot(ax_strip, "상태 타임라인   실제 상태 vs 탐지 결과")
    ax_strip.grid(False)
    strip_img = ax_strip.imshow(
        np.zeros((2, plot_window)),
        cmap=STRIP_CMAP,
        vmin=0,
        vmax=4,
        aspect="auto",
        interpolation="nearest",
        extent=(-0.5, plot_window - 0.5, 2, 0),
    )
    ax_strip.set_xlim(-0.5, plot_window - 0.5)
    ax_strip.set_yticks([0.5, 1.5])
    ax_strip.set_yticklabels(
        ["실제 상태", "탐지 결과"],
        fontsize=8,
        fontname=KOREAN_FONT or "DejaVu Sans",
    )
    ax_strip.set_xlabel("스트리밍 샘플  (최신 값이 오른쪽)", fontsize=8.5)
    ax_strip.axhline(1.0, color=C["bg"], lw=2)
    ax_strip.legend(
        handles=[
            Patch(color=C["green"], label="정상"),
            Patch(color=C["blue"], label="유휴"),
            Patch(color=C["amber"], label="경고 (OR_3)"),
            Patch(color=C["red"], label="고장 / 알람"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, -0.48),
        ncol=4,
        fontsize=8,
        frameon=False,
        labelcolor=C["muted"],
        prop={"family": KOREAN_FONT or "DejaVu Sans"},
    )

    shading_store: dict = {ax_vib: [], ax_cur: [], ax_ratio: []}

    def draw_shading(truth_codes) -> None:
        arr = np.asarray(truth_codes)
        for ax, store in shading_store.items():
            for coll in store:
                coll.remove()
            store.clear()
            for code, color in ((CODE_IDLE, C["blue"]), (CODE_FAULT, C["red"])):
                mask = arr == code
                if mask.any():
                    store.append(
                        ax.fill_between(
                            x_data, 0, 1, where=mask,
                            transform=ax.get_xaxis_transform(),
                            color=color, alpha=0.10, linewidth=0, step="mid",
                        )
                    )

    # ---------------- Pipeline stepper ----------------
    style_card(ax_pipe, "데이터 파이프라인", "원본 > 전처리 > 모델")
    n_steps = len(PREPROCESS_STEPS)
    step_ys = np.linspace(0.88, 0.12, n_steps)
    ax_pipe.plot(
        [0.07, 0.07], [step_ys[-1], step_ys[0]],
        transform=ax_pipe.transAxes, color=C["dim"], lw=2, zorder=1,
    )
    step_dots = ax_pipe.scatter(
        [0.07] * n_steps, step_ys, s=300, transform=ax_pipe.transAxes,
        c=[C["dim"]] * n_steps, edgecolors=C["panel"], linewidths=2, zorder=3,
    )
    step_nums, step_labels, step_detail_texts = [], [], []
    for i, (label, y) in enumerate(zip(PREPROCESS_STEPS, step_ys)):
        step_nums.append(
            ax_pipe.text(0.07, y, str(i + 1), transform=ax_pipe.transAxes, ha="center",
                         va="center", fontsize=8.5, fontweight="bold", color=C["muted"], zorder=4, fontname=KOREAN_FONT or "DejaVu Sans")
        )
        step_labels.append(
            ax_pipe.text(0.15, y + 0.035, label, transform=ax_pipe.transAxes, ha="left",
                         va="center", fontsize=9.5, fontweight="bold", color=C["muted"],
                         fontname=KOREAN_FONT or "DejaVu Sans")
        )
        step_detail_texts.append(
            ax_pipe.text(0.15, y - 0.045, "", transform=ax_pipe.transAxes, ha="left",
                         va="center", fontsize=7.4, family=MONO_FONT, color=C["muted"],
                         fontname=MONO_FONT)
        )
    details = step_details(
        normal_stats, fault_stats, train_df, calibration_df, detector.quantile
    )

    # ---------------- Live preprocessing ----------------
    style_card(ax_live, "실시간 입력", "현재 행")
    live_text = mono_text(ax_live, 0.05, 0.94, size=8.8)

    # ---------------- Event ----------------
    style_card(ax_event, "이벤트 / 시스템", "고장 시작 및 탐지 지연")
    event_text = mono_text(ax_event, 0.05, 0.94, size=8.6)
    ax_event.text(
        0.5, 0.33, "탐지 지연", transform=ax_event.transAxes,
        ha="center", va="center", fontsize=8, fontweight="bold", color=C["muted"],
        fontname=KOREAN_FONT or "DejaVu Sans",
    )
    delay_text = ax_event.text(
        0.5, 0.17, "-", transform=ax_event.transAxes, ha="center", va="center",
        fontsize=26, fontweight="bold", color=C["muted"],
        fontname=KOREAN_FONT or "DejaVu Sans",
    )

    # ---------------- Detector gauges ----------------
    style_card(ax_det, "탐지", "점수 / 기준값")
    ax_det.set_xlim(0, GAUGE_CAP)
    ax_det.set_ylim(-1.7, 4.4)
    gauge_y = [3.6, 2.4, 1.2]
    gauge_names = ["샘플", "0.5초", "1.0초"]
    ax_det.set_yticks(gauge_y)
    ax_det.set_yticklabels(gauge_names, fontsize=8.5, color=C["text"], fontname=KOREAN_FONT or "DejaVu Sans")
    ax_det.tick_params(axis="y", length=0)
    ax_det.barh(gauge_y, [GAUGE_CAP] * 3, height=0.46, color=C["bg"], zorder=1)
    gauge_bars = ax_det.barh(gauge_y, [0, 0, 0], height=0.46, color=C["green"], zorder=2)

    # 기준값(1.0) 점선은 각 게이지 바 내부에만 표시한다.
    threshold_lines = []
    for y in gauge_y:
        (line,) = ax_det.plot(
            [1.0, 1.0],
            [y - 0.23, y + 0.23],
            color=C["red"],
            lw=1.5,
            ls="--",
            zorder=4,
            solid_capstyle="butt",
        )
        threshold_lines.append(line)

    # 기준값 라벨은 1.0초 게이지 바로 아래, 점선과 같은 x 위치에 둔다.
    threshold_text = ax_det.text(
        1.0,
        0.82,
        "기준값",
        ha="center",
        va="top",
        fontsize=7.8,
        fontweight="bold",
        color=C["red"],
        fontname=KOREAN_FONT or "DejaVu Sans",
    )

    gauge_labels = [
        ax_det.text(
            0.03,
            y + 0.46,
            "",
            ha="left",
            va="center",
            fontsize=7.8,
            color=C["muted"],
            fontname=KOREAN_FONT or "DejaVu Sans",
        )
        for y in gauge_y
    ]
    or3_text = ax_det.text(
        0.04,
        0.15,
        "OR_3  꺼짐",
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        fontname=KOREAN_FONT or "DejaVu Sans",
        color=C["text"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor=C["dim"], edgecolor="none"),
    )
    p2_label_text = ax_det.text(
        1.55,
        0.15,
        "지속 P2",
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        fontname=KOREAN_FONT or "DejaVu Sans",
        color=C["muted"],
    )
    p_count = int(detector.p)
    p_boxes = []
    # P2 라벨과 겹치지 않도록 상태 박스를 조금 오른쪽으로 이동
    p_box_start_x = 2.05
    p_box_gap = 0.40
    p_box_width = 0.30
    for i in range(p_count):
        box = Rectangle(
            (p_box_start_x + i * p_box_gap, -0.08),
            p_box_width,
            0.46,
            facecolor=C["dim"],
            edgecolor="none",
            zorder=3,
        )
        ax_det.add_patch(box)
        p_boxes.append(box)
    active_text = ax_det.text(
        0.04,
        -0.75,
        "활성 탐지기: 없음",
        ha="left",
        va="center",
        fontsize=8.5,
        fontname=KOREAN_FONT or "DejaVu Sans",
        color=C["muted"],
    )
    final_text = ax_det.text(
        GAUGE_CAP / 2,
        -1.3,
        "정상",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        fontname=KOREAN_FONT or "DejaVu Sans",
        color=C["green"],
    )

    # ---------------- Features ----------------
    style_card(ax_feat, "실시간 처리", "원본 > 특징 추출 > 탐지")
    # Intro text is used only during the preparation phase.
    feat_intro_text = mono_text(ax_feat, 0.05, 0.96, size=8.2)
    feature_artists = create_feature_panel(ax_feat)
    for _artist in feature_artists.values():
        _artist.set_visible(False)

    # ---------------- Shared state ----------------
    first_alarm_time: pd.Timestamp | None = None
    false_alarm_frames = 0
    last_stage = -2
    last_feature_frame = -1
    last_axis_frame = -1
    log_rows: list[dict[str, object]] = []

    def draw_steps(stage_index: int) -> None:
        nonlocal last_stage
        if stage_index == last_stage:
            return
        last_stage = stage_index
        colors = []
        for i in range(n_steps):
            if i < stage_index:
                colors.append(C["green"])
                step_nums[i].set_color(C["bg"])
                step_labels[i].set_color(C["text"])
                step_detail_texts[i].set_text(details[i])
                step_detail_texts[i].set_color(C["muted"])
            elif i == stage_index:
                colors.append(C["amber"])
                step_nums[i].set_color(C["bg"])
                step_labels[i].set_color(C["amber"])
                step_detail_texts[i].set_text(details[i])
                step_detail_texts[i].set_color(C["amber"])
            else:
                colors.append(C["dim"])
                step_nums[i].set_color(C["muted"])
                step_labels[i].set_color(C["muted"])
                step_detail_texts[i].set_text("")
        step_dots.set_facecolor(colors)

    # ============================================================
    # Intro frames: visible preprocessing/model preparation
    # ============================================================
    if args.skip_intro:
        intro_frames = 0
    else:
        intro_frames = max(1, int(round(args.preprocess_seconds * args.fps)))

    stream_total_frames = intro_frames + len(demo_df)
    last_stream_index = len(demo_df) - 1

    def update(frame: int):
        nonlocal first_alarm_time, false_alarm_frames, last_feature_frame, last_axis_frame

        overall = (frame + 1) / max(1, stream_total_frames)
        progress_bar.set_width(min(1.0, overall))

        # --------------------------------------------------------
        # Phase A: preprocessing/model preparation intro
        # --------------------------------------------------------
        if frame < intro_frames:
            progress = frame / max(1, intro_frames - 1)
            stage_index = min(n_steps - 1, int(progress * n_steps))
            draw_steps(stage_index)

            set_pill(pill_phase, "준비중", C["amber"])
            set_pill(pill_truth, "-", C["muted"])
            set_pill(pill_det, "대기", C["muted"])
            progress_bar.set_facecolor(C["amber"])

            live_text.set_text(
                "원본 데이터 요약\n"
                + "-" * 30
                + "\n"
                f"정상 입력    {normal_stats['original_rows']:>9,}행\n"
                f"고장 입력    {fault_stats['original_rows']:>9,}행\n\n"
                "센서\n"
                "  AI0 진동\n"
                "  AI1 진동\n"
                "  AI2 전류"
            )
            event_text.set_text(
                "시스템 초기화\n"
                + "-" * 30
                + "\n"
                f"진행률     {progress * 100:5.1f}%\n\n"
                "학습 / 보정 데이터\n"
                "  운전 상태 정상 데이터만\n"
                "유휴 상태는 오탐 감시에\n"
                "사용합니다."
            )
            feat_intro_text.set_text(make_model_text(fit_info, detector, stage_index))
            feat_intro_text.set_visible(True)
            for _artist in feature_artists.values():
                _artist.set_visible(False)
            return []

        # --------------------------------------------------------
        # Phase B: actual streaming replay
        # --------------------------------------------------------
        draw_steps(n_steps)  # all steps done
        progress_bar.set_facecolor(C["blue"])

        stream_frame = frame - intro_frames
        row = demo_df.iloc[stream_frame]
        timestamp = pd.Timestamp(row["TimeStamp"])
        original_time = pd.Timestamp(row["orig_TimeStamp"])
        block = str(row["block"])
        kind = str(row["kind"])

        ai0 = float(row["AI0_Vibration"])
        ai1 = float(row["AI1_Vibration"])
        ai2 = float(row["AI2_Current"])

        result = detector.step(timestamp, ai0, ai1, ai2)
        feature_snapshot = get_feature_snapshot(detector)

        if (
            demo_fault_onset is not None
            and timestamp >= demo_fault_onset
            and result["final_alarm"]
            and first_alarm_time is None
        ):
            first_alarm_time = timestamp

        if result["final_alarm"] and kind != "fault":
            false_alarm_frames += 1

        # --------------------------------------------------------
        # Ratios, state codes
        # --------------------------------------------------------
        thr = detector.thresholds
        ratios = [
            score_ratio(float(result["sample_score"]), float(thr["sample"])),
            score_ratio(float(result["win05_score"]), float(thr["win05"])),
            score_ratio(float(result["win1_score"]), float(thr["win1"])),
        ]

        if result["final_alarm"]:
            final_state, det_code, det_color = "최종 알람", CODE_FAULT, C["red"]
        elif result["or3_base"]:
            final_state, det_code, det_color = "경고", CODE_WARN, C["amber"]
        else:
            final_state, det_code, det_color = "정상", CODE_OK, C["green"]
        truth_code = KIND_TO_CODE.get(kind, CODE_OK)

        # Real visual break at segment resets
        if result["segment_reset"]:
            push(**GAP_ENTRY)

        push(
            ai0=ai0,
            ai1=ai1,
            ai2=ai2,
            r_sample=ratios[0],
            r_w05=ratios[1],
            r_w1=ratios[2],
            truth=truth_code,
            det=det_code,
        )

        # --------------------------------------------------------
        # Plots
        # --------------------------------------------------------
        line_ai0.set_ydata(np.asarray(bufs["ai0"], dtype=float))
        line_ai1.set_ydata(np.asarray(bufs["ai1"], dtype=float))
        line_ai2.set_ydata(np.asarray(bufs["ai2"], dtype=float))
        line_r_sample.set_ydata(np.asarray(bufs["r_sample"], dtype=float))
        line_r_w05.set_ydata(np.asarray(bufs["r_w05"], dtype=float))
        line_r_w1.set_ydata(np.asarray(bufs["r_w1"], dtype=float))

        strip_img.set_data(
            np.vstack(
                [
                    np.asarray(bufs["truth"], dtype=float),
                    np.asarray(bufs["det"], dtype=float),
                ]
            )
        )

        # Axis autoscaling is intentionally throttled. Recomputing limits on
        # every Matplotlib frame is surprisingly expensive and causes visible
        # stutter even though the streaming detector itself is fast.
        if stream_frame - last_axis_frame >= args.axis_update_every:
            last_axis_frame = stream_frame

            vib = np.concatenate(
                [np.asarray(bufs["ai0"], dtype=float), np.asarray(bufs["ai1"], dtype=float)]
            )
            vib = vib[np.isfinite(vib)]
            if len(vib):
                vmin, vmax = float(vib.min()), float(vib.max())
                margin = max((vmax - vmin) * 0.12, 0.05)
                if np.isclose(vmin, vmax):
                    margin = max(abs(vmin) * 0.05, 0.05)
                ax_vib.set_ylim(vmin - margin, vmax + margin)

            cur = np.asarray(bufs["ai2"], dtype=float)
            cur = cur[np.isfinite(cur)]
            if len(cur):
                cmin, cmax = float(cur.min()), float(cur.max())
                margin = max((cmax - cmin) * 0.12, 1.0)
                if np.isclose(cmin, cmax):
                    margin = max(abs(cmin) * 0.05, 1.0)
                ax_cur.set_ylim(cmin - margin, cmax + margin)

            ratio_all = np.concatenate(
                [
                    np.asarray(bufs["r_sample"], dtype=float),
                    np.asarray(bufs["r_w05"], dtype=float),
                    np.asarray(bufs["r_w1"], dtype=float),
                ]
            )
            ratio_all = ratio_all[np.isfinite(ratio_all)]
            if len(ratio_all):
                ratio_max = max(10.0, float(ratio_all.max()) * 2.5)
                ratio_min = min(0.05, max(1e-4, float(ratio_all.min()) / 2.0))
            else:
                ratio_max, ratio_min = 10.0, 0.05
            ax_ratio.set_ylim(ratio_min, ratio_max)

        # --------------------------------------------------------
        # Header pills
        # --------------------------------------------------------
        set_pill(
            pill_phase,
            "완료" if stream_frame >= last_stream_index else "실시간",
            C["green"] if stream_frame >= last_stream_index else C["blue"],
        )
        truth_label = {"normal": "정상", "idle": "유휴", "fault": "고장"}.get(kind, "-")
        set_pill(
            pill_truth,
            truth_label,
            {"normal": C["green"], "idle": C["blue"], "fault": C["red"]}.get(kind, C["muted"]),
        )
        det_label = {
            "정상": "정상",
            "WARNING": "경고",
            "FINAL ALARM": "최종 알람",
        }.get(final_state, final_state)
        if result["final_alarm"] and kind != "fault":
            det_label = "오탐 경보"
        set_pill(pill_det, det_label, det_color)

        # --------------------------------------------------------
        # Detector gauges
        # --------------------------------------------------------
        scores = [
            float(result["sample_score"]),
            float(result["win05_score"]),
            float(result["win1_score"]),
        ]
        thresholds = [float(thr["sample"]), float(thr["win05"]), float(thr["win1"])]
        alarms = [
            bool(result["sample_alarm"]),
            bool(result["win05_alarm"]),
            bool(result["win1_alarm"]),
        ]
        for bar, label, r, s, t, alarm in zip(
            gauge_bars, gauge_labels, ratios, scores, thresholds, alarms
        ):
            width = 0.0 if not np.isfinite(r) else min(r, GAUGE_CAP)
            bar.set_width(width)
            bar.set_facecolor(C["red"] if alarm else C["green"])
            label.set_text(f"{fmt(s, 2)} / {fmt(t, 2)}   x{fmt(r, 2)}")
            label.set_color(C["red"] if alarm else C["muted"])

        or3_on = bool(result["or3_base"])
        or3_text.set_text("OR_3  켜짐" if or3_on else "OR_3  꺼짐")
        or3_text.get_bbox_patch().set_facecolor(C["amber"] if or3_on else C["dim"])
        or3_text.set_color(C["bg"] if or3_on else C["text"])

        count = int(result["persistence_count"])
        for i, box in enumerate(p_boxes):
            if i < count:
                box.set_facecolor(C["red"] if result["final_alarm"] else C["amber"])
            else:
                box.set_facecolor(C["dim"])

        active = result["active_detectors"]
        active_labels = {
            "sample": "샘플",
            "0.5s": "0.5초",
            "1.0s": "1.0초",
        }
        active_display = [active_labels.get(name, name) for name in active]
        active_text.set_text(f"활성 탐지기: {', '.join(active_display) if active_display else '없음'}")
        active_text.set_color(C["text"] if active else C["muted"])
        final_text.set_text(final_state)
        final_text.set_color(det_color)
        for spine in ax_det.spines.values():
            spine.set_color(det_color if result["final_alarm"] else C["border"])

        # --------------------------------------------------------
        # Text cards
        # --------------------------------------------------------
        live_text.set_text(
            f"구간          {block.upper()}\n"
            f"원본 시간     {ts_str(original_time)}\n"
            f"재생 시간     {ts_str(timestamp)}\n"
            + "-" * 30
            + "\n"
            f"AI0           {ai0: .6f}\n"
            f"AI1           {ai1: .6f}\n"
            f"AI2           {ai2: .3f}\n"
            + "-" * 30
            + "\n"
            f"유휴 {int(row['Idle'])}   상태 {int(row['Equipment_state'])}   세그먼트 {row['group_id']}\n"
            f"간격 {result['gap_seconds']:.3f}초 (> {GAP_THRESHOLD_SEC:.1f}초)  "
            f"초기화 {'예' if result['segment_reset'] else '아니오'}\n"
            f"모델 입력     {'제외' if kind == 'idle' else '사용'}\n"
            f"재생 진행     {stream_frame + 1}/{len(demo_df)}"
        )

        event_lines = [
            f"q={detector.quantile:.4f}  OR_3+P{detector.p}  mean{detector.sample_agg_k}",
            f"윈도우 0.5초/1.0초  초기화 > {detector.gap_threshold_sec:.1f}초",
            "",
        ]
        if demo_fault_onset is not None:
            event_lines += [
                f"고장 시작 (재생)   {ts_str(demo_fault_onset)}",
                (
                    f"고장 시작 (원본)   {ts_str(demo_fault_original_onset)}"
                    if demo_fault_original_onset is not None
                    else "고장 시작 (원본)   -"
                ),
            ]
        event_lines.append(f"오탐 발생 횟수     {false_alarm_frames}")
        event_text.set_text("\n".join(event_lines))

        if demo_fault_onset is not None and first_alarm_time is not None:
            delay = max(0.0, (first_alarm_time - demo_fault_onset).total_seconds())
            delay_text.set_text(f"{delay:.3f} s")
            delay_text.set_color(C["green"])
        elif demo_fault_onset is not None and timestamp >= demo_fault_onset:
            delay_text.set_text("detecting...")
            delay_text.set_color(C["amber"])
        else:
            delay_text.set_text("-")
            delay_text.set_color(C["muted"])

        # The feature table is the heaviest text panel. Keep detection
        # completely sample-by-sample, but refresh this visual every N frames.
        if stream_frame - last_feature_frame >= args.feature_update_every or stream_frame == 0:
            last_feature_frame = stream_frame
            feat_intro_text.set_visible(False)
            for _artist in feature_artists.values():
                _artist.set_visible(True)
            update_feature_panel(feature_artists, detector, feature_snapshot)

        log_rows.append(
            {
                "stream_index": int(stream_frame),
                "source": row["source"],
                "block": block,
                "kind": kind,
                "TimeStamp": timestamp,
                "orig_TimeStamp": original_time,
                "ground_truth": int(row["ground_truth"]),
                "Idle": int(row["Idle"]),
                "Equipment_state": int(row["Equipment_state"]),
                "segment_id": int(row["segment_id"]),
                "group_id": row["group_id"],
                "segment_reset": bool(result["segment_reset"]),
                "gap_seconds": float(result["gap_seconds"]),
                "AI0_Vibration": ai0,
                "AI1_Vibration": ai1,
                "AI2_Current": ai2,
                "score_0.5s": float(result["win05_score"]),
                "threshold_0.5s": float(detector.thresholds["win05"]),
                "alarm_0.5s": bool(result["win05_alarm"]),
                "score_1.0s": float(result["win1_score"]),
                "threshold_1.0s": float(detector.thresholds["win1"]),
                "alarm_1.0s": bool(result["win1_alarm"]),
                "sample_score": float(result["sample_score"]),
                "sample_threshold": float(detector.thresholds["sample"]),
                "sample_alarm": bool(result["sample_alarm"]),
                "OR_3": bool(result["or3_base"]),
                "P2": bool(result["final_alarm"]),
                "state": "ALARM" if result["final_alarm"] else "WARNING" if result["or3_base"] else "정상",
                "active_detectors": ",".join(result["active_detectors"]),
            }
        )

        return []

    total_frames = max(1, stream_total_frames)
    interval_ms = max(1, int(1000 / args.fps))

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    try:
        plt.show()
    finally:
        output_path = output_dir / "realtime_log.csv"
        if log_rows:
            pd.DataFrame(log_rows).to_csv(
                output_path,
                index=False,
                encoding="utf-8-sig",
            )
            print(f"[SAVED] {output_path}")
        else:
            print("[WARN] No realtime log rows were produced.")

    # keep a reference so the animation is not garbage-collected early
    _ = ani


if __name__ == "__main__":
    main()