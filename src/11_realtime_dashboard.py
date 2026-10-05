from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.widgets import Button

try:
    from core_detector import StreamingDetector, WINDOW_FEATURES, SENSOR_COLS
except ImportError:
    from src.core_detector import StreamingDetector, WINDOW_FEATURES, SENSOR_COLS

try:
    from dashboard_stage_monitor import (FigureGallery, StageBoard, TimingStore, find_stage_images,
                                         run_command_live, stage_details)
except ImportError:
    from src.dashboard_stage_monitor import (FigureGallery, StageBoard, TimingStore, find_stage_images,
                                             run_command_live, stage_details)

C = {
    "bg": "#0b0f17", "panel": "#121926", "border": "#27344a",
    "grid": "#202b3d", "text": "#e6edf3", "muted": "#8b98a9",
    "dim": "#3a4a63", "green": "#2ecc71", "amber": "#f5a623",
    "red": "#ff4d5e", "blue": "#4da3ff", "cyan": "#22d3ee",
    "purple": "#b48cff", "orange": "#ff9f43",
}

CODE_EMPTY, CODE_OK, CODE_WARN, CODE_FAULT, CODE_IDLE = 0, 1, 2, 3, 4
STRIP_CMAP = ListedColormap([C["bg"], C["green"], C["amber"], C["red"], C["blue"]])
KIND_TO_CODE = {"normal": CODE_OK, "idle": CODE_IDLE, "fault": CODE_FAULT}

system_os = platform.system()
target_font = "Malgun Gothic" if system_os == "Windows" else ("AppleGothic" if system_os == "Darwin" else "NanumGothic")
font_names = {f.name for f in fm.fontManager.ttflist}
if target_font in font_names:
    KOREAN_FONT = target_font
else:
    fallbacks = [f.name for f in fm.fontManager.ttflist if any(t in f.name for t in ["Gothic", "Malgun", "Nanum", "Apple"])]
    KOREAN_FONT = fallbacks[0] if fallbacks else None

MONO_FONT = None
for candidate in ("Noto Sans Mono CJK KR", "NanumGothicCoding"):
    try:
        path = fm.findfont(candidate, fallback_to_default=False)
        MONO_FONT = fm.FontProperties(fname=path).get_name()
        break
    except Exception:
        pass
if MONO_FONT is None:
    MONO_FONT = "DejaVu Sans Mono"
if KOREAN_FONT:
    plt.rcParams["font.family"] = KOREAN_FONT
plt.rcParams.update({
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
})

SENSORS = list(SENSOR_COLS)
# 최종 배포 detector 설정 (model_config.json 으로도 저장됨).
# 주의: 6~8, 10~11단계 명령줄의 같은 값(window-k, sample-agg-k, quantile, covariance, seed)과 일치해야 합니다.
PIPELINE_CONFIG = {
    "gap_sec": 0.5,
    "win05_size": 5,
    "win1_size": 10,
    "sample_agg_k": 3,
    "persistence_p": 2,
    "quantile": 0.9999,
    "covariance": "ledoitwolf",
    "seed": 0,
}
GAP_THRESHOLD_SEC = PIPELINE_CONFIG["gap_sec"]
DEFAULT_OUTPUT_DIR = "outputs/11_realtime_dashboard"
DEFAULT_QUANTILE = PIPELINE_CONFIG["quantile"]
GAUGE_CAP = 3.0
PREPROCESS_STEPS = [
    "원본 데이터 입력",
    "데이터 검증",
    "타임스탬프 간격 / 세그먼트",
    "유휴 상태 분류",
    "특징 및 탐지",
]


# ---------------------------------------------------------------------
# Offline-only calibration preparation
# ---------------------------------------------------------------------
def preprocess_for_calibration(df: pd.DataFrame, source: str) -> pd.DataFrame:
    required = ["TimeStamp", *SENSORS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing columns: {missing}")

    work = df.copy()
    original_rows = len(work)
    parsed_ts = pd.to_datetime(work["TimeStamp"], errors="coerce")
    invalid_time = int(parsed_ts.isna().sum())
    work["TimeStamp"] = parsed_ts
    work = work.dropna(subset=["TimeStamp"]).sort_values("TimeStamp").reset_index(drop=True)

    for col in SENSORS:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    invalid_sensor_rows = int(work[SENSORS].isna().any(axis=1).sum())
    if invalid_sensor_rows:
        raise ValueError(f"{source}: sensor data contain {invalid_sensor_rows} invalid rows")

    if "Idle" in work.columns:
        work["Idle"] = pd.to_numeric(work["Idle"], errors="coerce").fillna(0).astype(int)
    else:
        work["Idle"] = 0
    if "Equipment_state" in work.columns:
        work["Equipment_state"] = pd.to_numeric(work["Equipment_state"], errors="coerce").fillna(0).astype(int)
    else:
        work["Equipment_state"] = 1 if source == "fault" else 0

    gap = work["TimeStamp"].diff().dt.total_seconds()
    break_mask = gap.gt(GAP_THRESHOLD_SEC)
    if len(work):
        break_mask.iloc[0] = True
    work["gap_seconds"] = gap.fillna(0.0)
    work["segment_id"] = (break_mask.cumsum() - 1).astype(int)
    work["group_id"] = source + "_" + work["segment_id"].astype(str)
    work["source"] = source
    if source == "fault":
        work["ground_truth"] = 1
        work["kind"] = "fault"
    else:
        work["ground_truth"] = work["Equipment_state"].ge(1).astype(int)
        # Only for selecting calibration/demo groups. Never used at runtime.
        work["kind"] = np.where(work["Idle"].eq(1), "idle", "normal")

    work.attrs["preprocess_stats"] = {
        "source": source,
        "original_rows": original_rows,
        "valid_rows": len(work),
        "invalid_timestamp": invalid_time,
        "invalid_sensor_rows": invalid_sensor_rows,
        "segments": int(work["group_id"].nunique()),
        "gaps_over_threshold": int(work["gap_seconds"].gt(GAP_THRESHOLD_SEC).sum()),
        "idle_rows": int(work["Idle"].eq(1).sum()),
        "idle_segments": int(work.loc[work["kind"].eq("idle"), "group_id"].nunique()),
    }
    return work


def select_demo_normal(normal_df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    usable = normal_df[normal_df["Idle"].eq(0)].copy()
    if usable.empty:
        raise ValueError("No non-Idle normal samples are available.")
    return usable.tail(max(1, int(n_samples))).copy()


def select_demo_idle(normal_df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    idle = normal_df[normal_df["Idle"].eq(1)].copy()
    if idle.empty:
        raise ValueError("No Idle samples are available.")
    gap = idle["TimeStamp"].diff().dt.total_seconds()
    local_segment = gap.gt(GAP_THRESHOLD_SEC).fillna(True).cumsum().astype(int)
    idle = idle.assign(_idle_demo_group=local_segment)
    target = idle.groupby("_idle_demo_group").size().idxmax()
    return idle[idle["_idle_demo_group"].eq(target)].head(max(1, int(n_samples))).drop(columns=["_idle_demo_group"])


def build_demo_stream(blocks: list[tuple[str, pd.DataFrame]], join_gap_sec: float = 2.0) -> pd.DataFrame:
    out: list[pd.DataFrame] = []
    cursor: pd.Timestamp | None = None
    for name, raw_df in blocks:
        df = raw_df.copy()
        # fault CSV 는 원본 문자열 그대로 들어오므로 시간 연산 전에 datetime 으로 변환
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
        df = df.dropna(subset=["TimeStamp"]).sort_values("TimeStamp").reset_index(drop=True)
        if df.empty:
            continue
        df["orig_TimeStamp"] = df["TimeStamp"]
        if cursor is not None:
            shift = cursor + pd.Timedelta(seconds=join_gap_sec) - df["TimeStamp"].iloc[0]
            df["TimeStamp"] = df["TimeStamp"] + shift
        cursor = df["TimeStamp"].iloc[-1]
        df["block"] = name
        out.append(df)
    if not out:
        raise ValueError("No non-empty demo blocks were provided")
    return pd.concat(out, ignore_index=True).reset_index(drop=True)


def nominal_interval(ts, default: float = 0.1) -> float:
    """데이터의 공칭 샘플링 간격(초). gap 임계값 이하의 timestamp 차이 중앙값."""
    d = pd.to_datetime(ts).diff().dt.total_seconds().dropna()
    d = d[(d > 0) & (d <= GAP_THRESHOLD_SEC)]
    return float(d.median()) if len(d) else float(default)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _collect_metadata(root, inputs) -> dict:
    """재현성 정보: 패키지 버전, git commit, 입력 파일 해시."""
    meta = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    try:
        import sklearn
        meta["sklearn"] = sklearn.__version__
    except Exception:
        pass
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, timeout=5).stdout.strip()
        meta["code_commit"] = out or "n/a"
    except Exception:
        meta["code_commit"] = "n/a"
    meta["input_sha256_16"] = {Path(p).name: _file_sha256(Path(p)) for p in inputs if Path(p).exists()}
    return meta


def save_final_model(out_dir: Path, detector, fit_info: dict, cfg: dict, args,
                     root: Path | None = None, inputs=()) -> list[str]:
    """최종 배포 모델 설정/기준값/객체를 outputs/final_model/ 에 저장하고 저장 내역을 반환."""
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **cfg,
        "quantile": float(detector.quantile),
        "seed": int(args.seed),
        "calibration_fraction": float(args.calibration_fraction),
        "metadata": _collect_metadata(root or Path("."), inputs),
        "train_rows": int(fit_info["train_rows"]),
        "calibration_rows": int(fit_info["calibration_rows"]),
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    (out_dir / "model_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (out_dir / "thresholds.json").write_text(
        json.dumps(fit_info["thresholds"], ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    saved = ["model_config.json", "thresholds.json"]
    try:
        with open(out_dir / "detector.pkl", "wb") as f:
            pickle.dump(detector, f)
        saved.append("detector.pkl")
    except Exception as exc:  # noqa: BLE001
        saved.append(f"detector.pkl 저장 실패: {exc}")
    return saved


def summarize_realtime(df: pd.DataFrame, nominal_dt: float):
    """realtime_log -> (metrics_df, event_summary_df)."""
    alarm = df["P2"].astype(bool)
    kind = df["kind"]
    mask = kind.eq("fault")
    run = (mask != mask.shift(fill_value=False)).cumsum()
    events = []
    for eid, (_, g) in enumerate(df[mask].groupby(run[mask]), start=1):
        hit = g[g["P2"].astype(bool)]
        row = dict(event_id=eid, fault_start=g["TimeStamp"].iloc[0], fault_rows=len(g),
                   detected=bool(len(hit)), first_alarm=None, delay_sec=np.nan,
                   delay_samples=np.nan, alarm_count=int(g["P2"].astype(bool).sum()))
        if len(hit):
            row["first_alarm"] = hit["TimeStamp"].iloc[0]
            row["delay_sec"] = (hit["TimeStamp"].iloc[0] - g["TimeStamp"].iloc[0]).total_seconds()
            row["delay_samples"] = int(hit["stream_index"].iloc[0] - g["stream_index"].iloc[0])
        events.append(row)
    ev = pd.DataFrame(events, columns=["event_id", "fault_start", "fault_rows", "detected", "first_alarm",
                                       "delay_sec", "delay_samples", "alarm_count"])

    ratios = []
    for s, t in (("sample_score", "sample_threshold"), ("score_0.5s", "threshold_0.5s"), ("score_1.0s", "threshold_1.0s")):
        r = (df[s] / df[t]).replace([np.inf, -np.inf], np.nan)
        ratios.append(r)
    max_ratio = float(pd.concat(ratios, axis=1).max().max())

    def rate(a, b):
        return f"{a / b:.4f}" if b else "n/a"

    n_norm, n_idle, n_fault = int(kind.eq("normal").sum()), int(kind.eq("idle").sum()), int(mask.sum())
    fp_norm, fp_idle = int((alarm & kind.eq("normal")).sum()), int((alarm & kind.eq("idle")).sum())
    tp_frames = int((alarm & mask).sum())
    n_det = int(ev["detected"].sum()) if len(ev) else 0
    metrics = [
        ("streaming_rows", len(df)),
        ("nominal_sampling_sec", round(nominal_dt, 4)),
        ("fault_events", len(ev)),
        ("fault_events_detected", n_det),
        ("event_detection_rate", rate(n_det, len(ev))),
        ("first_detection_delay_sec", "n/a" if not n_det else round(float(ev["delay_sec"].min()), 4)),
        ("first_detection_delay_samples", "n/a" if not n_det else int(ev["delay_samples"].min())),
        ("normal_rows", n_norm), ("normal_false_alarm_frames", fp_norm),
        ("normal_false_alarm_rate", rate(fp_norm, n_norm)),
        ("idle_rows", n_idle), ("idle_false_alarm_frames", fp_idle),
        ("idle_false_alarm_rate", rate(fp_idle, n_idle)),
        ("fault_rows", n_fault), ("fault_alarm_frames", tp_frames),
        ("fault_frame_recall", rate(tp_frames, n_fault)),
        ("final_alarm_frames", int(alarm.sum())),
        ("max_score_ratio", round(max_ratio, 4)),
    ]
    if "infer_ms" in df.columns and df["infer_ms"].notna().any():
        lat = df["infer_ms"].dropna()
        metrics += [("infer_latency_ms_mean", round(float(lat.mean()), 3)),
                    ("infer_latency_ms_p95", round(float(lat.quantile(0.95)), 3)),
                    ("infer_latency_ms_max", round(float(lat.max()), 3))]
    return pd.DataFrame(metrics, columns=["metric", "value"]), ev


def build_calibration_split(normal_df: pd.DataFrame, exclude_group_ids: set[str], fraction: float, seed: int):
    usable = normal_df[
        normal_df["kind"].eq("normal")
        & normal_df["Idle"].eq(0)
        & ~normal_df["group_id"].isin(exclude_group_ids)
    ].copy()
    if usable.empty:
        raise ValueError("No normal operation remains after excluding demo groups")
    groups = np.asarray(sorted(usable["group_id"].astype(str).unique()), dtype=str)
    if len(groups) < 2:
        raise ValueError("At least two normal segments are needed")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_cal = max(1, int(np.ceil(len(groups) * fraction)))
    n_cal = min(n_cal, len(groups) - 1)
    cal_groups = set(groups[:n_cal])
    return usable[~usable["group_id"].isin(cal_groups)].copy(), usable[usable["group_id"].isin(cal_groups)].copy()


# ---------------------------------------------------------------------
# True row-by-row streaming preprocessing
# ---------------------------------------------------------------------
class StreamingPreprocessor:
    """No-look-ahead preprocessor. Only current row + previous timestamp."""

    def __init__(self, gap_threshold_sec: float = 0.5):
        self.gap_threshold_sec = float(gap_threshold_sec)
        self.last_timestamp: pd.Timestamp | None = None
        self.segment_id = -1

    def reset(self):
        self.last_timestamp = None
        self.segment_id = -1

    @staticmethod
    def _finite_number(value, name: str) -> float:
        v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if not np.isfinite(float(v)):
            raise ValueError(f"{name} is invalid")
        return float(v)

    def process(self, row: pd.Series, source: str, block: str, original_timestamp=None) -> dict[str, object]:
        if "TimeStamp" not in row.index:
            raise ValueError("Streaming row is missing TimeStamp")
        for col in SENSORS:
            if col not in row.index:
                raise ValueError(f"Streaming row is missing {col}")

        timestamp = pd.to_datetime(row["TimeStamp"], errors="coerce")
        if pd.isna(timestamp):
            raise ValueError("Streaming TimeStamp is invalid")
        timestamp = pd.Timestamp(timestamp)

        current = np.asarray([self._finite_number(row[c], c) for c in SENSORS], dtype=float)
        gap = 0.0
        segment_reset = False

        if self.last_timestamp is None:
            self.segment_id = 0
        else:
            gap = (timestamp - self.last_timestamp).total_seconds()
            if gap < 0:
                raise ValueError("Streaming timestamps must be non-decreasing")
            if gap > self.gap_threshold_sec:
                self.segment_id += 1
                segment_reset = True
        self.last_timestamp = timestamp

        idle_value = pd.to_numeric(pd.Series([row.get("Idle", 0)]), errors="coerce").iloc[0]
        idle = int(idle_value) if np.isfinite(float(idle_value)) else 0
        state_value = pd.to_numeric(pd.Series([row.get("Equipment_state", 0)]), errors="coerce").iloc[0]
        equipment_state = int(state_value) if np.isfinite(float(state_value)) else 0

        if source == "fault":
            kind = "fault"
            ground_truth = 1
        else:
            # IMPORTANT: no segment-wide idle_ratio; current row only.
            kind = "idle" if idle == 1 else "normal"
            ground_truth = int(equipment_state >= 1)

        return {
            "timestamp": timestamp,
            "orig_timestamp": pd.Timestamp(original_timestamp) if original_timestamp is not None else timestamp,
            "AI0_Vibration": current[0],
            "AI1_Vibration": current[1],
            "AI2_Current": current[2],
            "Idle": idle,
            "Equipment_state": equipment_state,
            "source": source,
            "block": block,
            "kind": kind,
            "ground_truth": ground_truth,
            "segment_id": int(self.segment_id),
            "group_id": f"{source}_{self.segment_id}",
            "gap_seconds": float(gap),
            "segment_reset": bool(segment_reset),
        }


class StreamingPipeline:
    """One raw row -> streaming preprocess -> existing causal detector."""

    def __init__(self, detector: StreamingDetector, gap_threshold_sec: float):
        self.detector = detector
        self.preprocessor = StreamingPreprocessor(gap_threshold_sec)

    def reset(self):
        self.preprocessor.reset()
        self.detector.reset_stream() if hasattr(self.detector, "reset_stream") else self.detector.reset_state()

    def process(self, row: pd.Series, source: str, block: str, original_timestamp=None):
        record = self.preprocessor.process(row, source, block, original_timestamp)
        if record["segment_reset"]:
            self.detector.reset_stream() if hasattr(self.detector, "reset_stream") else self.detector.reset_state()
            # We just reset detector history because this first row belongs to the new segment.
            if hasattr(self.detector, "last_time"):
                self.detector.last_time = None
        result = dict(self.detector.step(
            record["timestamp"],
            float(record["AI0_Vibration"]),
            float(record["AI1_Vibration"]),
            float(record["AI2_Current"]),
        ))
        result["segment_reset"] = bool(record["segment_reset"])
        result["gap_seconds"] = float(record["gap_seconds"])
        result["segment_id"] = int(record["segment_id"])
        return record, result


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

def load_offline_performance(root: Path) -> dict:
    """
    Stage 8에서 계산한 최종 OR_3_P2 Offline CV 성능을 읽는다.
    Replay 성능과 별도로 표시하기 위한 용도.
    """
    path = root / "outputs" / "6_three_detector_ensemble" / "ensemble_summary.csv"

    if not path.exists():
        return {}

    try:
        df = pd.read_csv(path)

        rows = df[
            df["combo"].astype(str).eq("OR_3_P2")
        ]

        if rows.empty:
            return {}

        row = rows.iloc[0]

        return {
            "event_detection_rate": float(
                row["event_detection_rate_mean"]
            ),
            "event_detection_rate_std": float(
                row["event_detection_rate_std"]
            ),
            "delay_mean_sec": float(
                row["delay_mean_sec_mean"]
            ),
            "delay_std_sec": float(
                row["delay_mean_sec_std"]
            ),
            "sample_f1": float(
                row["sample_f1_mean"]
            ),
            "sample_f1_std": float(
                row["sample_f1_std"]
            ),
            "normal_fa_cycle_rate": float(
                row["normal_fa_cycle_rate_mean"]
            ),
            "normal_fa_cycle_rate_std": float(
                row["normal_fa_cycle_rate_std"]
            ),
            "idle_fa_cycle_rate": float(
                row["idle_fa_cycle_rate_mean"]
            ),
            "idle_fa_cycle_rate_std": float(
                row["idle_fa_cycle_rate_std"]
            ),
            "n_events": int(row["n_events"]),
            "repeats": int(row["repeats"]),
        }

    except Exception:
        return {}


def short_feature_name(name: str) -> str:
    short = name.replace("AI0_Vibration", "A0").replace("AI1_Vibration", "A1").replace("AI2_Current", "A2").replace("AI0_AI1_corr", "A0-A1 corr")
    for suffix in ("mean", "std", "rms", "ptp", "slope"):
        if short.endswith("_" + suffix):
            return short[:-(len(suffix)+1)] + "." + suffix
    return short


def get_feature_snapshot(detector: StreamingDetector) -> dict[str, float]:
    """
    현재 StreamingDetector가 보유한 causal 상태를 그대로 읽어
    대시보드의 Feature Table에 표시한다.

    중요:
    실제 detector의 window feature 계산과 동일하게
    detector.time_buffer의 실제 timestamp를 사용한다.
    따라서 화면에 보이는 slope가 실제 추론에 사용되는 slope와 일치한다.
    """
    values = list(detector.raw_buffer)
    times = list(detector.time_buffer)

    out: dict[str, float] = {}

    # -------------------------------------------------------------
    # 0.5초 Window = 최근 5 samples
    # -------------------------------------------------------------
    if len(values) >= detector.win05_size:
        arr = np.asarray(
            values[-detector.win05_size:],
            dtype=float,
        )
        ts = np.asarray(
            times[-detector.win05_size:]
        )

        feat = detector._extract_window_features(
            arr,
            ts,
        )

        out.update({
            f"W05_{name}": float(v)
            for name, v in zip(WINDOW_FEATURES, feat)
        })

    # -------------------------------------------------------------
    # 1.0초 Window = 최근 10 samples
    # -------------------------------------------------------------
    if len(values) >= detector.win1_size:
        arr = np.asarray(
            values[-detector.win1_size:],
            dtype=float,
        )
        ts = np.asarray(
            times[-detector.win1_size:]
        )

        feat = detector._extract_window_features(
            arr,
            ts,
        )

        out.update({
            f"W10_{name}": float(v)
            for name, v in zip(WINDOW_FEATURES, feat)
        })

    return out


def style_card(ax, title: str, subtitle: str | None = None):
    ax.set_facecolor(C["panel"])
    for spine in ax.spines.values():
        spine.set_color(C["border"]); spine.set_linewidth(1.1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=C["text"], pad=7, fontname=KOREAN_FONT or "DejaVu Sans")
    if subtitle:
        # NOTE: StageBoard 가 (x=1.0, y=1.015) 위치로 이 부제목 텍스트를 찾아 바꿉니다. 위치를 바꾸지 마세요.
        ax.text(1.0, 1.015, subtitle, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=C["muted"], fontname=KOREAN_FONT or "DejaVu Sans")


def style_plot(ax, title: str):
    ax.set_facecolor(C["panel"])
    for spine in ax.spines.values():
        spine.set_color(C["border"]); spine.set_linewidth(1.1)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=C["text"], pad=7, fontname=KOREAN_FONT or "DejaVu Sans")
    ax.grid(True, alpha=0.6, linewidth=0.7); ax.tick_params(labelsize=8)


def mono_text(ax, x, y, size=9.0):
    return ax.text(x, y, "", transform=ax.transAxes, va="top", ha="left", fontsize=size, family=MONO_FONT, color=C["text"], linespacing=1.45)


def make_pill(ax, x, w, label):
    patch = FancyBboxPatch((x, 0.07), w, 0.84, boxstyle="round,pad=0,rounding_size=0.008", transform=ax.transAxes, facecolor=C["dim"], edgecolor="none")
    ax.add_patch(patch)
    ax.text(x+w/2, 0.70, label, transform=ax.transAxes, ha="center", va="center", fontsize=7.5, fontweight="bold", color=C["text"], fontname=KOREAN_FONT or "DejaVu Sans")
    value = ax.text(x+w/2, 0.40, "-", transform=ax.transAxes, ha="center", va="center", fontsize=15, fontweight="bold", color=C["text"], fontname=KOREAN_FONT or "DejaVu Sans")
    return patch, value


def set_pill(pill, text, color):
    from matplotlib.colors import to_rgba
    patch, value = pill
    rgb = to_rgba(color)[:3]
    patch.set_facecolor((*rgb, 0.30)); value.set_text(text); value.set_color(C["text"])


def create_feature_panel(ax):
    font = MONO_FONT
    X = {"label": 0.05, "col1": 0.39, "col2": 0.64}
    artists = {
        "sample_title": ax.text(X["label"], .945, "샘플 (원본 + 차분)", transform=ax.transAxes, ha="left", va="top", fontsize=8.3, fontname=font, color=C["text"]),
        "sample_header_orig": ax.text(X["col1"], .895, "원본", transform=ax.transAxes, ha="left", va="top", fontsize=8, fontname=font, color=C["muted"]),
        "sample_header_diff": ax.text(X["col2"], .895, "차분", transform=ax.transAxes, ha="left", va="top", fontsize=8, fontname=font, color=C["muted"]),
        "window_title": ax.text(X["label"], .690, "윈도우 특징", transform=ax.transAxes, ha="left", va="top", fontsize=8.3, fontname=font, color=C["text"]),
        "window_header_05": ax.text(X["col1"], .650, "0.5초", transform=ax.transAxes, ha="left", va="top", fontsize=8, fontname=font, color=C["muted"]),
        "window_header_10": ax.text(X["col2"], .650, "1.0초", transform=ax.transAxes, ha="left", va="top", fontsize=8, fontname=font, color=C["muted"]),
        "waiting": ax.text(X["label"], .605, "", transform=ax.transAxes, ha="left", va="top", fontsize=7.8, fontname=font, color=C["muted"]),
    }
    for key, label, y in zip(("ai0","ai1","ai2"),("AI0","AI1","AI2"),(.845,.800,.755)):
        artists[f"sample_label_{key}"] = ax.text(X["label"], y, label, transform=ax.transAxes, ha="left", va="top", fontsize=8.1, fontname=font, color=C["text"])
        artists[f"sample_orig_{key}"] = ax.text(X["col1"], y, "-", transform=ax.transAxes, ha="left", va="top", fontsize=8.1, fontname=font, color=C["text"])
        artists[f"sample_diff_{key}"] = ax.text(X["col2"], y, "-", transform=ax.transAxes, ha="left", va="top", fontsize=8.1, fontname=font, color=C["text"])
    artists["divider"] = ax.plot([X["label"], .91], [.620,.620], transform=ax.transAxes, color=C["dim"], lw=.8, ls="--")[0]
    for i, name in enumerate(WINDOW_FEATURES):
        y=.585-i*.034
        artists[f"win_label_{i}"] = ax.text(X["label"], y, short_feature_name(name), transform=ax.transAxes, ha="left", va="top", fontsize=7.8, fontname=font, color=C["text"])
        artists[f"win05_{i}"] = ax.text(X["col1"], y, "-", transform=ax.transAxes, ha="left", va="top", fontsize=7.8, fontname=font, color=C["text"])
        artists[f"win10_{i}"] = ax.text(X["col2"], y, "-", transform=ax.transAxes, ha="left", va="top", fontsize=7.8, fontname=font, color=C["text"])
    return artists


def update_feature_panel(artists, detector, snapshot):
    current = detector.raw_buffer[-1]
    previous = detector.raw_buffer[-2] if len(detector.raw_buffer) >= 2 else None
    diffs = np.zeros(3) if previous is None else current - previous

    def fixed_num(value, decimals=4, integer_width=5):
        if value is None or not np.isfinite(value): return "-".ljust(11)
        value=float(value); sign="-" if value<0 else " "; av=abs(value)
        ip=int(av); frac=int(round((av-ip)*10**decimals))
        if frac >= 10**decimals: ip += 1; frac = 0
        return f"{sign}{ip:>{integer_width}}.{frac:0{decimals}d}"

    for key, idx in (("ai0",0),("ai1",1),("ai2",2)):
        artists[f"sample_orig_{key}"].set_text(fixed_num(current[idx]))
        artists[f"sample_diff_{key}"].set_text(fixed_num(diffs[idx]))
    if not snapshot:
        artists["waiting"].set_text(f"버퍼 {len(detector.raw_buffer)}/{detector.win1_size} 샘플 | 인과 윈도우 준비 중...")
    else:
        artists["waiting"].set_text("")
        for i, name in enumerate(WINDOW_FEATURES):
            artists[f"win05_{i}"].set_text(fixed_num(snapshot.get(f"W05_{name}")))
            artists[f"win10_{i}"].set_text(fixed_num(snapshot.get(f"W10_{name}")))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def execute_idle_classification(root: Path) -> pd.DataFrame:
    """Exact Step-2 Idle labeling logic, but save to the project's data/ path."""
    src = root / "data" / "press_data_normal.csv"
    if not src.exists():
        src = root / "data" / "press_data_normal_with_idle.csv"
    if not src.exists():
        raise FileNotFoundError("data/press_data_normal.csv 또는 press_data_normal_with_idle.csv가 없습니다.")

    df = pd.read_csv(src)
    if "TimeStamp" not in df.columns:
        raise ValueError("normal CSV에 TimeStamp가 없습니다.")

    ts = pd.to_datetime(df["TimeStamp"], errors="coerce")
    if ts.isna().any():
        raise ValueError("normal CSV에 잘못된 TimeStamp가 있습니다.")

    # Same interval used in src/2_Classification_idle_sections.py.
    df["Idle"] = 0
    idle_mask = (
        (ts >= pd.Timestamp("2022-07-12 00:59:53.992"))
        & (ts <= pd.Timestamp("2022-07-12 01:11:32.588"))
    )
    df.loc[idle_mask, "Idle"] = 1

    out = root / "data" / "press_data_normal_with_idle.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def make_stage_summary(root: Path, stage_no: int) -> str:
    """Read the actual artifact produced by each project stage. ('|' 로 나눠 현황판 결과 요약에 표시)"""
    if stage_no == 1:
        return "원본 시계열 분석 완료  |  정상/고장 센서 변동 확인"
    if stage_no == 2:
        p = root / "data" / "press_data_normal_with_idle.csv"
        df = pd.read_csv(p)
        idle = int(pd.to_numeric(df["Idle"], errors="coerce").fillna(0).eq(1).sum())
        return f"Idle 라벨 생성  |  전체 {len(df):,}행  |  Idle {idle:,}행"
    if stage_no == 3:
        p = root / "result" / "time_structure_analysis"
        files = list(p.glob("*.csv")) if p.exists() else []
        return f"timestamp gap / segment 분석 완료  |  산출물 {len(files)}개"
    if stage_no == 4:
        p = root / "result/modeling_dataset_1.0_0.1/model_windows.csv"
        df = read_csv_if_exists(p)
        return "1.0초 Window dataset 완료" if df is None else f"1.0초 Window dataset 완료  |  {len(df):,}개 window"
    if stage_no == 5:
        p = root / "result/modeling_dataset_0.5_0.1/model_windows.csv"
        df = read_csv_if_exists(p)
        return "0.5초 Window dataset 완료" if df is None else f"0.5초 Window dataset 완료  |  {len(df):,}개 window"
    if stage_no == 6:
        p = root / "outputs/5_2_mahalanobis_cv/cv_comparison.csv"
        df = read_csv_if_exists(p)
        return "Window Mahalanobis CV 완료" if df is None else f"Window Mahalanobis CV 완료  |  설정 {len(df):,}개"
    if stage_no == 7:
        p = root / "outputs/5_3_sample_level_cv/sample_cv_comparison.csv"
        df = read_csv_if_exists(p)
        return "Sample-level Mahalanobis CV 완료" if df is None else f"Sample-level Mahalanobis CV 완료  |  설정 {len(df):,}개"
    if stage_no == 8:
        p = root / "outputs/6_three_detector_ensemble/ensemble_summary.csv"
        df = read_csv_if_exists(p)
        if df is not None and not df.empty and "combo" in df.columns:
            rows = df[df["combo"].astype(str).eq("OR_3_P2")]
            if not rows.empty:
                row = rows.iloc[0]
                return (
                    f"3-Detector Ensemble 완료  |  OR_3_P2  "
                    f"event {float(row['event_detection_rate_mean'])*100:.1f}%  |  "
                    f"delay {float(row['delay_mean_sec_mean']):.3f}s"
                )
        return "3-Detector Ensemble 완료"
    if stage_no == 9:
        p = root / "outputs/7_isolation_forest_baseline/summary.csv"
        df = read_csv_if_exists(p)
        if df is not None and not df.empty:
            row = df.iloc[0]
            return f"Isolation Forest baseline 완료  |  F1 {float(row['f1_mean']):.4f}"
        return "Isolation Forest baseline 완료"
    if stage_no == 10:
        p = root / "outputs/8_fp_fn_analysis/fn_event_summary.csv"
        df = read_csv_if_exists(p)
        n = 0 if df is None else len(df)
        return f"FP/FN 분석 완료  |  FN event {n:,}건"
    if stage_no == 11:
        p = root / "outputs/10_variable_effect_analysis/variable_effect_summary.csv"
        df = read_csv_if_exists(p)
        n = 0 if df is None else len(df)
        return f"FN 시각화 + 변수 영향 분석 완료  |  effect {n:,}행"
    return "완료"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "KAMPact full project pipeline dashboard: "
            "1~11 actual analysis/modeling -> 12 causal streaming inference"
        )
    )
    parser.add_argument("--normal-path", default="data/press_data_normal.csv")
    parser.add_argument("--fault-path", default="data/outlier_data.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=float, default=None,
                        help="재생 속도(행/초). 기본: 데이터의 샘플링 간격에 맞춘 실시간 속도 x --replay-speed")
    parser.add_argument("--auto-start", type=float, default=0.0,
                        help="파이프라인 완료 후 N초 뒤 자동으로 실시간 추론 창으로 이동 (0 = 버튼/Enter 로 직접 이동)")
    parser.add_argument("--replay-speed", type=float, default=1.0,
                        help="실시간 대비 재생 배속 (1.0 = 데이터 timestamp 속도 그대로)")
    parser.add_argument("--feature-update-every", type=int, default=3)
    parser.add_argument("--axis-update-every", type=int, default=5)
    parser.add_argument("--plot-window", type=int, default=150)
    parser.add_argument("--demo-normal-samples", type=int, default=800)
    parser.add_argument("--demo-idle-samples", type=int, default=200)
    parser.add_argument("--demo-join-gap", type=float, default=2.0,
                        help="--demo-join-mode gap 일 때 블록 사이에 넣는 인위적 간격(초, 0.5 초과)")
    parser.add_argument("--demo-join-mode", choices=("continuous", "gap"), default="continuous",
                        help="continuous: 블록을 샘플 간격으로 이어 붙여 detector reset 없이 연속 재생 (기본). "
                             "gap: 블록 사이에 --demo-join-gap 간격을 넣어 매번 reset")
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument("--seed", type=int, default=PIPELINE_CONFIG["seed"])
    parser.add_argument(
        "--reuse-results", action="store_true",
        help="1~11 산출물이 이미 있으면 해당 단계를 다시 실행하지 않고 재사용합니다.",
    )
    parser.add_argument(
        "--skip-heavy-analysis", action="store_true",
        help="9~11의 무거운 분석을 생략합니다. 기본값은 전체 파이프라인 실행입니다.",
    )
    args = parser.parse_args()

    if (args.fps is not None and args.fps <= 0) or args.replay_speed <= 0 or args.feature_update_every < 1 or args.axis_update_every < 1:
        raise ValueError("fps / feature-update-every / axis-update-every must be positive.")
    if args.plot_window < 20:
        raise ValueError("plot-window must be >= 20.")
    if args.demo_normal_samples < 1 or args.demo_idle_samples < 1:
        raise ValueError("Demo sample counts must be >= 1.")
    if args.demo_join_mode == "gap" and args.demo_join_gap <= GAP_THRESHOLD_SEC:
        raise ValueError("demo-join-gap must be > 0.5 sec.")
    if not 0.05 <= args.calibration_fraction < 0.5:
        raise ValueError("calibration-fraction must be in [0.05, 0.5).")
    if not 0.95 <= args.quantile < 1.0:
        raise ValueError("quantile must be in [0.95, 1.0).")

    root = Path(__file__).resolve().parent.parent
    normal_path = root / args.normal_path
    fault_path = root / args.fault_path
    if not normal_path.exists():
        # The project may already have the Step-2 output only.
        normal_path = root / "data/press_data_normal_with_idle.csv"
    if not normal_path.exists() or not fault_path.exists():
        raise FileNotFoundError("normal/fault 입력 CSV 경로를 확인해주세요.")

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("KAMPact - FULL PIPELINE DASHBOARD")
    print("1~11 ACTUAL PIPELINE -> FINAL DEPLOYABLE MODEL -> CAUSAL STREAMING")
    print("=" * 110)

    # ------------------------------------------------------------
    # Figure is created FIRST so the user can watch stage execution.
    # ------------------------------------------------------------
    stage_labels = [
        "1  원본 시계열 분석",
        "2  Idle 라벨 생성",
        "3  시간 구조 / Segment",
        "4  1.0초 Window Dataset",
        "5  0.5초 Window Dataset",
        "6  Window Mahalanobis",
        "7  Sample Mahalanobis",
        "8  3-Detector Ensemble",
        "9  Isolation Forest",
        "10 FP / FN 분석",
        "11 FN 시각화 / 변수 영향",
        "12 최종 모델 학습 / 보정",
        "13 스트리밍 추론 (CSV replay)",
    ]
    PREPROCESS_STEPS[:] = stage_labels

    normal_raw_preview = pd.read_csv(normal_path, nrows=10)
    fault_raw_preview = pd.read_csv(fault_path, nrows=10)

    # ------------------------------------------------------------
    # 1번 창: 파이프라인 실행 전용 (실시간 추론 위젯과 분리)
    #   왼쪽  : 단계 실행 현황판 (상태 / 진행률 / 경과 / 결과 요약)
    #   가운데: 현재 단계 + 실행 로그
    #   오른쪽: 시각화 결과 갤러리 + 마지막 완료 단계 결과 상세
    #   아래  : 그래프 이동 버튼, '실시간 추론 시작' 버튼
    # 실시간 추론은 모든 단계가 끝난 뒤 2번 창으로 열린다.
    # ------------------------------------------------------------
    fig_p = plt.figure(figsize=(20, 11), facecolor=C["bg"])
    try:
        fig_p.canvas.manager.set_window_title("KAMPact - 1) 파이프라인 실행")
    except Exception:
        pass
    outer_p = fig_p.add_gridspec(2, 1, height_ratios=[0.70, 10], hspace=0.10,
                                 left=0.02, right=0.985, top=0.975, bottom=0.085)
    ax_hdr_p = fig_p.add_subplot(outer_p[0])
    body_p = outer_p[1].subgridspec(1, 3, width_ratios=[1.45, 1.0, 1.4], wspace=0.07)
    ax_board = fig_p.add_subplot(body_p[0])
    mid_p = body_p[1].subgridspec(2, 1, height_ratios=[1.0, 2.3], hspace=0.14)
    right_p = body_p[2].subgridspec(2, 1, height_ratios=[2.3, 1.0], hspace=0.14)
    ax_cur_p = fig_p.add_subplot(mid_p[0])
    ax_log_p = fig_p.add_subplot(mid_p[1])
    ax_gal = fig_p.add_subplot(right_p[0])
    ax_detail_p = fig_p.add_subplot(right_p[1])

    ax_hdr_p.axis("off")
    ax_hdr_p.set_xlim(0, 1)
    ax_hdr_p.set_ylim(0, 1)
    ax_hdr_p.text(0.002, 0.68, "KAMPact", transform=ax_hdr_p.transAxes, fontsize=23, fontweight="bold",
                  color=C["text"], va="center")
    ax_hdr_p.text(0.002, 0.24, "1) 파이프라인 실행 (1~12)   →   2) 실시간 추론 (13)",
                  transform=ax_hdr_p.transAxes, fontsize=9.5, color=C["muted"], va="center",
                  fontname=KOREAN_FONT or "DejaVu Sans")
    p_overall = ax_hdr_p.text(0.36, 0.50, "", transform=ax_hdr_p.transAxes, fontsize=11, color=C["text"],
                              va="center", ha="left", fontname=KOREAN_FONT or "DejaVu Sans")
    ax_hdr_p.add_patch(Rectangle((0, 0.0), 1, 0.05, transform=ax_hdr_p.transAxes,
                                 facecolor=C["panel"], edgecolor="none"))
    p_bar = Rectangle((0, 0.0), 0, 0.05, transform=ax_hdr_p.transAxes, facecolor=C["blue"], edgecolor="none")
    ax_hdr_p.add_patch(p_bar)

    gp = ax_gal.get_position()
    ax_prev = fig_p.add_axes([gp.x0, 0.020, 0.085, 0.042])
    ax_next = fig_p.add_axes([gp.x0 + 0.092, 0.020, 0.085, 0.042])
    lp = ax_log_p.get_position()   # 시작 버튼은 가운데 열 아래 (그래프 이동 버튼과 겹치지 않게)
    ax_start = fig_p.add_axes([lp.x0, 0.016, lp.width, 0.050])

    timing_store = TimingStore(output_dir / "stage_timing.json")
    gallery = FigureGallery(fig_p, ax_gal, C, KOREAN_FONT, ax_prev, ax_next)
    board = StageBoard(
        fig_p, ax_board,
        panels={"cur": ax_cur_p, "log": ax_log_p, "detail": ax_detail_p},
        labels=stage_labels[:12], colors=C, font=KOREAN_FONT, mono=MONO_FONT,
        timing=timing_store, overall_bar=p_bar, overall_text=p_overall,
        gallery=gallery, n_progress=12,
    )

    # '실시간 추론 시작' 버튼 (모든 단계가 끝나면 활성화)
    start_ui = {"ready": False, "go": False}
    start_btn = Button(ax_start, "파이프라인 실행 중 ...", color=C["panel"], hovercolor=C["panel"])
    start_btn.label.set_color(C["muted"])
    start_btn.label.set_fontname(KOREAN_FONT or "DejaVu Sans")
    start_btn.label.set_fontsize(10.5)

    def _on_start(_event=None):
        if start_ui["ready"]:
            start_ui["go"] = True

    def _on_key_press(event):
        key = str(getattr(event, "key", "")).lower()

        print(f"[KEY] {key}")

        if key in ("enter", "return", " ", "space"):
            if start_ui["ready"]:
                start_ui["go"] = True
                print("[READY] Enter 입력 → 실시간 추론으로 이동")

    start_btn.on_clicked(_on_start)
    fig_p.canvas.mpl_connect(
        "key_press_event",
        lambda ev: _on_start() if str(getattr(ev, "key", "")).lower() in ("enter", "return", " ", "space") else None)

    stage_status_idx = {n: n - 1 for n in range(1, 13)}

    def set_stage_state(index: int, state: str, detail: str = ""):
        """기존 호출부 호환용 래퍼 (12단계 최종 모델 표시에 사용)."""
        if state == "running":
            board.begin(index, note=detail)
        elif state == "done":
            board.finish(index, parts=[detail] if detail else [])
        else:
            board.fail(index, detail)

    # Show empty dashboard before executing actual stages.
    plt.ion()
    plt.show(block=False)
    fig_p.canvas.draw_idle()
    fig_p.canvas.flush_events()

    # ------------------------------------------------------------
    # ACTUAL PROJECT PIPELINE: 1~11
    # ------------------------------------------------------------
    stage_commands = {
        1: ["python", "src/1_visualize_normal_outlier.py"],
        3: ["python", "src/3_time_structure_analysis.py"],
        4: ["python", "src/4_make_window_dataset.py", "--window-sec", "1.0", "--step-sec", "0.1",
            "--output-dir", "result/modeling_dataset_1.0_0.1"],
        5: ["python", "src/4_make_window_dataset.py", "--window-sec", "0.5", "--step-sec", "0.1",
            "--output-dir", "result/modeling_dataset_0.5_0.1"],
    }

    # 단계 완료 마커: seed/quantile 이 바뀌면 이전 산출물을 재사용하지 않도록 비교한다.
    marker_dir = output_dir / "stage_markers"
    run_cfg = {"seed": args.seed, "quantile": args.quantile}

    def _apply_cfg(cmd: list[str]) -> list[str]:
        """하위 스크립트 명령의 seed / quantile 을 대시보드 인자(args)와 일치시킨다."""
        out = list(cmd)
        for i, tok in enumerate(out[:-1]):
            if tok == "--seed":
                out[i + 1] = str(args.seed)
            elif tok in ("--normal-quantile", "--window-quantile", "--sample-quantile"):
                out[i + 1] = str(args.quantile)
        return out

    def _marker_state(no: int):
        """None: 마커 없음 / True: 같은 설정으로 실행됨 / False: 설정이 다름"""
        p = marker_dir / f"stage_{no}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8")) == run_cfg
        except Exception:
            return False

    def _add_figs(display_no: int, idx: int, since):
        """단계가 만든 그래프를 갤러리에 추가 (실행했으면 이번에 만든 것만, 재사용이면 기존 것)."""
        try:
            paths = find_stage_images(root, display_no, since=since)
            if paths:
                board.add_figures(idx, display_no, paths)
        except Exception as exc:  # noqa: BLE001
            board.note(f"시각화 검색 실패: {exc}")

    def exec_stage(display_no: int, commands=None, reuse_check: list[Path] | None = None,
                   extra_env: dict[str, str] | None = None, func=None,
                   require_marker: bool = False, verify: list[Path] | None = None, fallback=None):
        """한 단계를 실행하고 현황판에 실시간 로그/진행률/결과를 표시.

        commands       : 단일 명령(list[str]) 또는 여러 명령(list[list[str]])
        func           : 외부 스크립트 대신 파이썬 함수를 실행할 때
        require_marker : 산출물 위치를 알 수 없는 단계(1, 3, 11)는 대시보드가 같은 설정으로 실행한 기록이 있어야 재사용
        verify/fallback: 스크립트 실행 후 verify 파일이 새로 만들어졌는지 확인, 아니면 fallback 으로 보완
        """
        idx = stage_status_idx[display_no]
        if commands and isinstance(commands[0], str):
            commands = [commands]

        if args.reuse_results and reuse_check and all(p.exists() for p in reuse_check):
            ms = _marker_state(display_no)
            if ms is True or (ms is None and not require_marker):
                parts = [p.strip() for p in make_stage_summary(root, display_no).split("|")]
                board.finish(idx, parts=parts, details=stage_details(root, display_no), status="reused")
                _add_figs(display_no, idx, None)
                return

        script = " + ".join(Path(c[1]).name for c in commands) if commands else "(내장 로직)"
        board.begin(idx, script=script)
        t_start = time.time() - 1.0
        try:
            if func is not None:
                board.note("내장 로직 실행 중 ...")
                board.run_with_pump(func)
            else:
                for cmd in commands:
                    cmd = _apply_cfg(cmd)
                    if cmd and cmd[0] == "python":
                        cmd[0] = str(Path(sys.executable))
                    rc, lines = run_command_live(root, cmd, extra_env,
                                                 on_update=board.feed, wait=plt.pause)
                    if rc != 0:
                        raise RuntimeError(f"종료 코드 {rc}: {' '.join(Path(c).name for c in cmd[:2])}")
                if verify and not all(p.exists() and p.stat().st_mtime >= t_start for p in verify):
                    if fallback is None:
                        raise RuntimeError("예상 산출물이 새로 생성되지 않았습니다: "
                                           + ", ".join(p.name for p in verify))
                    board.note("스크립트가 예상 위치에 결과를 만들지 않아 내장 로직으로 보완")
                    board.run_with_pump(fallback)
        except Exception as exc:
            board.fail(idx, str(exc).splitlines()[0][:80], tail=list(board.log_tail)[-22:])
            raise
        try:
            marker_dir.mkdir(parents=True, exist_ok=True)
            (marker_dir / f"stage_{display_no}.json").write_text(json.dumps(run_cfg), encoding="utf-8")
        except Exception:
            pass
        parts = [p.strip() for p in make_stage_summary(root, display_no).split("|")]
        board.finish(idx, parts=parts, details=stage_details(root, display_no))
        _add_figs(display_no, idx, t_start)

    try:
        # 1
        exec_stage(1, stage_commands[1],
                   reuse_check=[root / "data/press_data_normal.csv", root / "data/outlier_data.csv"],
                   extra_env={"MPLBACKEND": "Agg"}, require_marker=True)

        # 2 (actual same logic, saved under data/)
        idle_csv = root / "data/press_data_normal_with_idle.csv"
        if (root / "src/2_Classification_idle_sections.py").exists():
            # 실제 2번 스크립트를 실행. 결과가 data/ 에 새로 만들어지지 않으면 내장 로직으로 보완.
            exec_stage(2, ["python", "src/2_Classification_idle_sections.py"], reuse_check=[idle_csv],
                       verify=[idle_csv], fallback=lambda: execute_idle_classification(root))
        else:
            exec_stage(2, func=lambda: execute_idle_classification(root), reuse_check=[idle_csv])

        # 3
        exec_stage(3, stage_commands[3],
                   reuse_check=[root / "result/time_structure_analysis"], require_marker=True)

        # 4 and 5 are the two actual calls to script 4.
        exec_stage(4, stage_commands[4],
                   reuse_check=[root / "result/modeling_dataset_1.0_0.1/model_windows.csv"])
        exec_stage(5, stage_commands[5],
                   reuse_check=[root / "result/modeling_dataset_0.5_0.1/model_windows.csv"])

        # 6: 5_2 Window Mahalanobis CV
        exec_stage(6, ["python", "src/5_2_run_mahalanobis.py",
                       "--multiscale", "result/modeling_dataset_1.0_0.1", "result/modeling_dataset_0.5_0.1",
                       "--output-dir", "outputs/5_2_mahalanobis_cv",
                       "--n-splits", "5", "--repeats", "5", "--seed", "0",
                       "--covariance", "ledoitwolf",
                       "--threshold-modes", "normal_quantile", "--normal-quantile", "0.9999",
                       "--k-consecutive", "1"],
                   reuse_check=[root / "outputs/5_2_mahalanobis_cv/cv_comparison.csv",
                                root / "outputs/5_2_mahalanobis_cv/cv_event_details.csv"])

        # 7: 5_3 Sample Mahalanobis CV
        exec_stage(7, ["python", "src/5_3_run_sample_level_mahalanobis.py",
                       "--normal-path", "data/press_data_normal_with_idle.csv",
                       "--fault-path", "data/outlier_data.csv",
                       "--output-dir", "outputs/5_3_sample_level_cv",
                       "--feature-set", "raw_diff", "--agg", "mean", "--agg-k", "3",
                       "--n-splits", "5", "--repeats", "5", "--seed", "0",
                       "--covariance", "ledoitwolf",
                       "--threshold-modes", "normal_quantile", "--normal-quantile", "0.9999"],
                   reuse_check=[root / "outputs/5_3_sample_level_cv/sample_cv_comparison.csv",
                                root / "outputs/5_3_sample_level_cv/sample_cv_event_details.csv"])

        heavy_ok = not args.skip_heavy_analysis

        # 8: 6 ensemble
        exec_stage(8, ["python", "src/6_compare_three_detectors.py",
                       "--normal-path", "data/press_data_normal_with_idle.csv",
                       "--fault-path", "data/outlier_data.csv",
                       "--window-1.0", "result/modeling_dataset_1.0_0.1",
                       "--window-0.5", "result/modeling_dataset_0.5_0.1",
                       "--output-dir", "outputs/6_three_detector_ensemble",
                       "--n-splits", "5", "--repeats", "5", "--seed", "0",
                       "--window-threshold-mode", "normal_quantile", "--window-quantile", "0.9999",
                       "--window-k", "2", "--window-covariance", "ledoitwolf",
                       "--sample-threshold-mode", "normal_quantile", "--sample-quantile", "0.9999",
                       "--sample-agg-k", "3", "--sample-agg", "mean",
                       "--sample-feature-set", "raw_diff", "--sample-covariance", "ledoitwolf"],
                   reuse_check=[root / "outputs/6_three_detector_ensemble/ensemble_summary.csv",
                                root / "outputs/6_three_detector_ensemble/ensemble_event_details.csv"])

        if heavy_ok:
            # 9: 7 Isolation Forest
            exec_stage(9, ["python", "src/7_isolation_forest_baseline.py",
                           "--input", "result/modeling_dataset_1.0_0.1/model_windows.csv",
                           "--output-dir", "outputs/7_isolation_forest_baseline",
                           "--n-estimators", "300", "--threshold-mode", "f1",
                           "--k-consecutive", "1", "--seeds", "1", "--seed-start", "42"],
                       reuse_check=[root / "outputs/7_isolation_forest_baseline/summary.csv",
                                    root / "outputs/7_isolation_forest_baseline/test_scores.csv"])

            # 10: 8 FP/FN
            exec_stage(10, ["python", "src/8_fp_fn_analysis.py",
                            "--normal-path", "data/press_data_normal_with_idle.csv",
                            "--fault-path", "data/outlier_data.csv",
                            "--window-1.0", "result/modeling_dataset_1.0_0.1",
                            "--window-0.5", "result/modeling_dataset_0.5_0.1",
                            "--output-dir", "outputs/8_fp_fn_analysis",
                            "--n-splits", "5", "--repeats", "5", "--seed", "0",
                            "--window-threshold-mode", "normal_quantile", "--window-quantile", "0.9999",
                            "--window-k", "2", "--window-covariance", "ledoitwolf",
                            "--sample-threshold-mode", "normal_quantile", "--sample-quantile", "0.9999",
                            "--sample-agg-k", "3", "--sample-agg", "mean",
                            "--sample-feature-set", "raw_diff", "--sample-covariance", "ledoitwolf"],
                       reuse_check=[root / "outputs/8_fp_fn_analysis/fn_event_summary.csv",
                                    root / "outputs/8_fp_fn_analysis/fp_windows_1.0s.csv"])

            # 11: FN 시각화(9_) + 변수 영향 분석(10_) — 보조 분석이며 스트리밍 detector 입력은 아님.
            # 두 스크립트를 한 단계로 묶어 같은 진행 카드에서 순서대로 실행한다.
            exec_stage(11, [
                ["python", "src/9_visualize_fn_events.py",
                 "--normal-path", "data/press_data_normal_with_idle.csv",
                 "--fault-path", "data/outlier_data.csv",
                 "--window-1.0", "result/modeling_dataset_1.0_0.1",
                 "--window-0.5", "result/modeling_dataset_0.5_0.1",
                 "--output-dir", "outputs/9_fn_visualization",
                 "--n-splits", "5",
                 "--window-threshold-mode", "normal_quantile", "--window-quantile", "0.9999",
                 "--window-k", "2", "--window-covariance", "ledoitwolf",
                 "--sample-threshold-mode", "normal_quantile", "--sample-quantile", "0.9999",
                 "--sample-agg-k", "3", "--sample-agg", "mean",
                 "--sample-feature-set", "raw_diff", "--sample-covariance", "ledoitwolf"],
                ["python", "src/10_variable_effect_analysis.py",
                 "--normal-path", "data/press_data_normal_with_idle.csv",
                 "--fault-path", "data/outlier_data.csv",
                 "--window-0.5", "result/modeling_dataset_0.5_0.1/model_windows.csv",
                 "--window-1.0", "result/modeling_dataset_1.0_0.1/model_windows.csv",
                 "--fp-output-dir", "outputs/8_fp_fn_analysis",
                 "--output-dir", "outputs/10_variable_effect_analysis"],
            ], reuse_check=[root / "outputs/10_variable_effect_analysis/variable_effect_summary.csv",
                            root / "outputs/9_fn_visualization"], require_marker=True)
        else:
            for n in (9, 10, 11):
                board.skip(stage_status_idx[n], "heavy analysis 생략 (--skip-heavy-analysis)")

    except Exception as exc:
        print(f"[ERROR] Pipeline failed: {exc}")
        board.save_report(output_dir / "pipeline_stage_report.csv")
        # Keep the window open so the failed stage / log is visible.
        plt.ioff()
        plt.show()
        raise

    board.save_report(output_dir / "pipeline_stage_report.csv")

    # ------------------------------------------------------------
    # FINAL DEPLOYABLE MODEL: train/calibrate after 1~11 completed.
    # This is the model used by the causal row-by-row replay.
    # (오래 걸리는 계산도 별도 스레드로 돌리고 현황판은 계속 갱신)
    # ------------------------------------------------------------
    set_stage_state(stage_status_idx[12], "running", "최종 모델 준비: 정상 데이터 전처리 중 ...")

    normal_full = board.run_with_pump(
        lambda: preprocess_for_calibration(
            pd.read_csv(root / "data/press_data_normal_with_idle.csv"), "normal"
        )
    )
    fault_full = pd.read_csv(fault_path)
    fault_raw = fault_full.copy()

    demo_normal = select_demo_normal(normal_full, args.demo_normal_samples)
    demo_idle = select_demo_idle(normal_full, args.demo_idle_samples)
    excluded_demo_groups = set(demo_normal["group_id"].astype(str)) | set(demo_idle["group_id"].astype(str))
    train_df, calibration_df = build_calibration_split(
        normal_full, excluded_demo_groups, args.calibration_fraction, args.seed
    )
    board.note(f"학습 {len(train_df):,}행 / 보정 {len(calibration_df):,}행 분리 완료")

    detector = StreamingDetector(
        win1_size=PIPELINE_CONFIG["win1_size"],
        win05_size=PIPELINE_CONFIG["win05_size"],
        sample_agg_k=PIPELINE_CONFIG["sample_agg_k"],
        persistence_p=PIPELINE_CONFIG["persistence_p"],
        gap_threshold_sec=GAP_THRESHOLD_SEC,
        quantile_threshold=args.quantile,
        random_state=args.seed,
    )
    pipeline = StreamingPipeline(detector, GAP_THRESHOLD_SEC)

    # Fit model before streaming; this is the explicit deployment-model step.
    board.note("최종 모델 학습 · 기준값 보정 중 ...")
    fit_info = board.run_with_pump(
        lambda: detector.fit(normal_df=train_df, calibration_df=calibration_df)
    )

    print(f"[DEPLOY] train={fit_info['train_rows']:,} calibration={fit_info['calibration_rows']:,}")
    print(f"[DEPLOY] thresholds={fit_info['thresholds']}")
    board.note(f"최종 모델 READY  thresholds={fit_info['thresholds']}")

    # Create demo stream ONLY as an input source. Each row still goes through
    # StreamingPipeline.process(), so runtime preprocessing is not pre-applied.
    # normal -> idle -> normal -> fault 를 이어 붙인다.
    #  - continuous(기본): 블록 사이를 공칭 샘플 간격으로만 띄워 gap(0.5s) 규칙에 걸리지 않게 함
    #    -> 실제로 timestamp 가 끊긴 곳에서만 reset, 정상 이력이 이어진 채 fault 로 전환되는 모습을 시연
    #  - gap: 예전 방식 (블록마다 인위적 2초 gap -> 매번 reset)
    nominal_dt = nominal_interval(demo_normal["TimeStamp"])
    join_gap = nominal_dt if args.demo_join_mode == "continuous" else args.demo_join_gap
    half = len(demo_normal) // 2
    if half >= 1:
        blocks = [("normal", demo_normal.iloc[:half]), ("idle", demo_idle),
                  ("normal", demo_normal.iloc[half:]), ("fault", fault_raw)]
    else:
        blocks = [("normal", demo_normal), ("idle", demo_idle), ("fault", fault_raw)]
    demo_df = build_demo_stream(blocks, join_gap_sec=join_gap)
    # 재생 속도: 기본은 데이터 timestamp 속도 그대로(예: 0.1s 간격 -> 10 Hz) x --replay-speed
    fps_eff = float(args.fps) if args.fps else args.replay_speed / nominal_dt
    replay_x = fps_eff * nominal_dt

    # ------------------------------------------------------------
    # Prepare stage 12 visual state.
    # ------------------------------------------------------------
    model_files = save_final_model(root / "outputs/final_model", detector, fit_info, PIPELINE_CONFIG, args,
                                  root=root, inputs=[root / "data/press_data_normal_with_idle.csv", fault_path])
    th = fit_info["thresholds"]
    board.finish(
        stage_status_idx[12],
        parts=["최종 모델 학습 · 보정 완료", f"학습 {fit_info['train_rows']:,}행 / 보정 {fit_info['calibration_rows']:,}행",
               f"q={detector.quantile:.4f}  OR_3 + P{PIPELINE_CONFIG['persistence_p']}"],
        details=[f"threshold  sample {th['sample']:.2f}", f"           0.5s   {th['win05']:.2f}",
                 f"           1.0s   {th['win1']:.2f}", "",
                 "저장: outputs/final_model/"] + [f"  {x}" for x in model_files],
    )
    # ------------------------------------------------------------
    # 파이프라인 창 마무리: 시각화/결과를 확인한 뒤 2번 창(실시간 추론)으로 이동
    # ------------------------------------------------------------
    board.set_ready()
    start_ui["ready"] = True
    start_btn.label.set_text("실시간 추론 시작 →  (클릭 / Enter)")
    start_btn.label.set_color(C["bg"])
    start_btn.color = C["green"]
    start_btn.hovercolor = "#27ae60"
    start_btn.ax.set_facecolor(C["green"])
    fig_p.canvas.draw_idle()

    # 키 입력은 창에 포커스가 있어야 받으므로 포커스를 가져오고(Tk), 터미널에서 Enter 를 눌러도 이동되게 한다.
    try:
        window = fig_p.canvas.manager.window
        window.lift()
        window.focus_force()

        # Tk backend일 경우 캔버스를 다시 포커스 가능하게 설정
        if hasattr(window, "attributes"):
            window.attributes("-topmost", True)
            window.after(100, lambda: window.attributes("-topmost", False))

    except Exception:
        pass
    if sys.stdin is not None and sys.stdin.isatty():
        def _wait_terminal_enter():
            try:
                if sys.stdin.readline() != "":      # EOF 면 무시
                    start_ui["go"] = True
            except Exception:
                pass
        threading.Thread(target=_wait_terminal_enter, daemon=True).start()
        print("[READY] 파이프라인 완료 - 창의 버튼 클릭 / 창에서 Enter / 이 터미널에서 Enter 중 하나로 실시간 추론 창으로 이동합니다.")
    t_ready = time.time()
    while not start_ui["go"] and plt.fignum_exists(fig_p.number):
        if args.auto_start > 0 and time.time() - t_ready >= args.auto_start:
            break
        plt.pause(0.1)
    start_btn.label.set_text("실시간 추론 창 여는 중 ...")
    try:
        fig_p.canvas.draw()
        fig_p.canvas.flush_events()
    except Exception:
        pass
    try:
        fig_p.savefig(output_dir / "pipeline_dashboard.png", dpi=110, facecolor=C["bg"])
        print(f"[SAVED] {output_dir / 'pipeline_dashboard.png'}")
    except Exception:
        pass
    plt.close(fig_p)

    # ------------------------------------------------------------
    # 2번 창: 실시간 추론 (CSV replay)
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(20, 11), facecolor=C["bg"])
    try:
        fig.canvas.manager.set_window_title("KAMPact - 2) Real-time Inference (CSV replay)")
    except Exception:
        pass

    outer = fig.add_gridspec(
        2, 1, height_ratios=[0.70, 10], hspace=0.11,
        left=0.02, right=0.985, top=0.975, bottom=0.105,
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

    # Header
    ax_hdr.axis("off")
    ax_hdr.set_xlim(0, 1)
    ax_hdr.set_ylim(0, 1)
    ax_hdr.text(0.002, 0.68, "KAMPact", transform=ax_hdr.transAxes,
                fontsize=23, fontweight="bold", color=C["text"], va="center")
    ax_hdr.text(
        0.002, 0.24,
        "프레스 이상 탐지  |  13 인과 스트리밍 추론 (CSV replay)  ·  파이프라인 1~12 완료",
        transform=ax_hdr.transAxes, fontsize=9.5, color=C["muted"],
        va="center", fontname=KOREAN_FONT or "DejaVu Sans",
    )
    # 전체 진행 요약 (단계 N/M · 경과 · 잔여 예상) — StageBoard 가 갱신
    hdr_overall = ax_hdr.text(
        0.30, 0.50, "", transform=ax_hdr.transAxes, fontsize=9.5, color=C["text"],
        va="center", ha="left", fontname=KOREAN_FONT or "DejaVu Sans",
    )

    hdr_pos = ax_hdr.get_position()
    pipe_pos = ax_pipe.get_position()
    det_pos = ax_det.get_position()

    def fig_x_to_hdr(x_fig):
        return (x_fig - hdr_pos.x0) / hdr_pos.width

    def fig_w_to_hdr(w_fig):
        return w_fig / hdr_pos.width

    status_gap_fig = 0.008
    left_card_w = max(0.01, (pipe_pos.width - status_gap_fig) / 2.0)
    pill_phase = make_pill(ax_hdr, fig_x_to_hdr(pipe_pos.x0), fig_w_to_hdr(left_card_w), "단계")
    pill_truth = make_pill(ax_hdr, fig_x_to_hdr(pipe_pos.x0 + left_card_w + status_gap_fig), fig_w_to_hdr(left_card_w), "실제 상태")
    pill_det = make_pill(ax_hdr, fig_x_to_hdr(det_pos.x0), fig_w_to_hdr(det_pos.width), "탐지 결과")
    ax_hdr.add_patch(Rectangle((0, 0.0), 1, 0.05, transform=ax_hdr.transAxes,
                               facecolor=C["panel"], edgecolor="none"))
    progress_bar = Rectangle((0, 0.0), 0, 0.05, transform=ax_hdr.transAxes,
                             facecolor=C["blue"], edgecolor="none")
    ax_hdr.add_patch(progress_bar)

    # Plots
    plot_window = args.plot_window
    x_data = np.arange(plot_window)

    def nan_deque():
        return deque([np.nan] * plot_window, maxlen=plot_window)

    def zero_deque():
        return deque([CODE_EMPTY] * plot_window, maxlen=plot_window)

    bufs = {
        "ai0": nan_deque(), "ai1": nan_deque(), "ai2": nan_deque(),
        "r_sample": nan_deque(), "r_w05": nan_deque(), "r_w1": nan_deque(),
        "truth": zero_deque(), "det": zero_deque(),
    }

    def push(**values):
        for key, buf in bufs.items():
            buf.append(values[key])

    GAP_ENTRY = dict(
        ai0=np.nan, ai1=np.nan, ai2=np.nan,
        r_sample=np.nan, r_w05=np.nan, r_w1=np.nan,
        truth=CODE_EMPTY, det=CODE_EMPTY,
    )

    style_plot(ax_vib, "진동   AI0 / AI1")
    (line_ai0,) = ax_vib.plot(x_data, list(bufs["ai0"]), color=C["cyan"], lw=1.5, label="AI0")
    (line_ai1,) = ax_vib.plot(x_data, list(bufs["ai1"]), color=C["purple"], lw=1.5, label="AI1")
    ax_vib.legend(loc="upper left", ncol=2, fontsize=8, frameon=True, labelcolor=C["text"],
                  prop={"family": KOREAN_FONT or "DejaVu Sans"})
    ax_vib.set_xlim(-0.5, plot_window - 0.5)
    ax_vib.set_autoscale_on(False)
    ax_vib.tick_params(labelbottom=False)

    style_plot(ax_cur, "전류   AI2")
    (line_ai2,) = ax_cur.plot(x_data, list(bufs["ai2"]), color=C["orange"], lw=1.5, label="AI2")
    ax_cur.set_autoscale_on(False)
    ax_cur.tick_params(labelbottom=False)

    style_plot(ax_ratio, "이상 점수 / 기준값   (1.0 초과 = 탐지 알람)")
    ax_ratio.set_yscale("log")
    (line_r_sample,) = ax_ratio.plot(x_data, list(bufs["r_sample"]), color=C["cyan"], lw=1.3, label="샘플")
    (line_r_w05,) = ax_ratio.plot(x_data, list(bufs["r_w05"]), color=C["purple"], lw=1.3, label="0.5초 윈도우")
    (line_r_w1,) = ax_ratio.plot(x_data, list(bufs["r_w1"]), color=C["orange"], lw=1.3, label="1.0초 윈도우")
    ax_ratio.axhline(1.0, color=C["red"], lw=1.2, ls="--")
    ax_ratio.axhspan(1.0, 1e9, color=C["red"], alpha=0.06, lw=0)
    ax_ratio.set_ylim(0.05, 10)
    ax_ratio.set_autoscale_on(False)
    ax_ratio.legend(loc="upper left", ncol=3, fontsize=8, labelcolor=C["text"],
                    prop={"family": KOREAN_FONT or "DejaVu Sans"})
    ax_ratio.tick_params(labelbottom=False)

    style_plot(ax_strip, "상태 타임라인   실제 상태 vs 탐지 결과")
    ax_strip.grid(False)
    strip_img = ax_strip.imshow(
        np.zeros((2, plot_window)), cmap=STRIP_CMAP, vmin=0, vmax=4,
        aspect="auto", interpolation="nearest",
        extent=(-0.5, plot_window - 0.5, 2, 0),
    )
    ax_strip.set_xlim(-0.5, plot_window - 0.5)
    ax_strip.set_yticks([0.5, 1.5])
    ax_strip.set_yticklabels(["실제 상태", "탐지 결과"], fontsize=8,
                             fontname=KOREAN_FONT or "DejaVu Sans")
    ax_strip.set_xlabel("스트리밍 샘플  (최신 값이 오른쪽)", fontsize=8.5)
    ax_strip.axhline(1.0, color=C["bg"], lw=2)
    ax_strip.legend(
        handles=[Patch(color=C["green"], label="정상"), Patch(color=C["blue"], label="유휴"),
                 Patch(color=C["amber"], label="경고 (OR_3)"), Patch(color=C["red"], label="고장 / 알람")],
        loc="upper right", bbox_to_anchor=(1.0, -0.48), ncol=4, fontsize=8,
        frameon=False, labelcolor=C["muted"], prop={"family": KOREAN_FONT or "DejaVu Sans"},
    )

    shading_store = {ax_vib: [], ax_cur: [], ax_ratio: []}

    def draw_shading(truth_codes):
        arr = np.asarray(truth_codes)
        for ax, store in shading_store.items():
            for coll in store:
                coll.remove()
            store.clear()
            for code, color in ((CODE_IDLE, C["blue"]), (CODE_FAULT, C["red"])):
                mask = arr == code
                if mask.any():
                    store.append(ax.fill_between(
                        x_data, 0, 1, where=mask,
                        transform=ax.get_xaxis_transform(), color=color,
                        alpha=0.10, linewidth=0, step="mid"
                    ))

    # --------------------------------------------------------
    # 최종 모델 / Offline 성능
    # 왼쪽: Offline CV 성능
    # 오른쪽: 최종 모델 설정 / 기준값 / 입력
    # --------------------------------------------------------

    style_card(
        ax_pipe,
        "최종 모델 / Offline 성능",
        "최종 배포 detector · Stage 8 OR_3_P2 CV",
    )

    _th = fit_info["thresholds"]
    offline_perf = load_offline_performance(root)

    # 왼쪽: Offline 성능
    offline_text = ax_pipe.text(
        0.055,
        0.91,
        "",
        transform=ax_pipe.transAxes,
        va="top",
        ha="left",
        fontsize=7.2,
        family=MONO_FONT,
        color=C["text"],
        linespacing=1.45,
    )

    # 가운데 구분선
    ax_pipe.plot(
        [0.48, 0.48],
        [0.08, 0.94],
        transform=ax_pipe.transAxes,
        color=C["dim"],
        lw=0.8,
        ls="--",
    )

    # 오른쪽: 모델 설정 / 기준값
    model_text = ax_pipe.text(
        0.52,
        0.91,
        "",
        transform=ax_pipe.transAxes,
        va="top",
        ha="left",
        fontsize=7.2,
        family=MONO_FONT,
        color=C["text"],
        linespacing=1.45,
    )

    if offline_perf:
        offline_text.set_text(
            "OFFLINE CV\n"
            + "-" * 19 + "\n"
            f"CV            {offline_perf['repeats']}회 반복\n"
            f"Fault event   {offline_perf['n_events']}개\n"
            "\n"
            f"Event 탐지율  "
            f"{offline_perf['event_detection_rate'] * 100:.1f}%\n"
            f"평균 지연      "
            f"{offline_perf['delay_mean_sec']:.3f}s\n"
            f"Sample F1      "
            f"{offline_perf['sample_f1']:.3f}\n"
            f"Normal FA      "
            f"{offline_perf['normal_fa_cycle_rate'] * 100:.2f}%\n"
            f"Idle FA        "
            f"{offline_perf['idle_fa_cycle_rate'] * 100:.2f}%"
        )
    else:
        offline_text.set_text(
            "OFFLINE CV\n"
            + "-" * 19 + "\n"
            "성능 산출물 없음"
        )


    model_text.set_text(
        "MODEL / THRESHOLD\n"
        + "-" * 19 + "\n"
        f"앙상블        OR_3 + P{PIPELINE_CONFIG['persistence_p']}\n"
        f"분위수        q={detector.quantile:.4f}\n"
        f"공분산        Ledoit-Wolf\n"
        "\n"
        f"학습 / 보정   "
        f"{fit_info['train_rows']:,} / "
        f"{fit_info['calibration_rows']:,}\n"
        "\n"
        "기준값\n"
        f"  Sample     {_th['sample']:.2f}\n"
        f"  0.5초       {_th['win05']:.2f}\n"
        f"  1.0초       {_th['win1']:.2f}\n"
        "\n"
        f"입력          CSV replay\n"
        f"처리          causal / row-by-row\n"
        "look-ahead    없음"
    )

    style_card(ax_live, "스트림 입력", "CSV replay · 현재 행")
    live_text = mono_text(ax_live, 0.05, 0.94, size=7.8)

    style_card(ax_event, "이벤트 / 시스템", "파이프라인 상태")
    event_text = mono_text(ax_event, 0.05, 0.94, size=7.8)
    delay_label = ax_event.text(0.5, 0.20, "탐지 지연", transform=ax_event.transAxes,
                                ha="center", va="center", fontsize=8, fontweight="bold",
                                color=C["muted"], fontname=KOREAN_FONT or "DejaVu Sans")
    delay_text = ax_event.text(0.5, 0.09, "-", transform=ax_event.transAxes,
                               ha="center", va="center", fontsize=22, fontweight="bold",
                               color=C["muted"], fontname=KOREAN_FONT or "DejaVu Sans")

    style_card(ax_det, "탐지", "점수 / 기준값")
    ax_det.set_xlim(0, GAUGE_CAP)
    ax_det.set_ylim(-1.7, 4.4)
    gauge_y = [3.6, 2.4, 1.2]
    ax_det.set_yticks(gauge_y)
    ax_det.set_yticklabels(["샘플", "0.5초", "1.0초"], fontsize=8.5,
                            color=C["text"], fontname=KOREAN_FONT or "DejaVu Sans")
    ax_det.tick_params(axis="y", length=0)
    ax_det.barh(gauge_y, [GAUGE_CAP] * 3, height=0.46, color=C["bg"], zorder=1)
    gauge_bars = ax_det.barh(gauge_y, [0, 0, 0], height=0.46, color=C["green"], zorder=2)
    threshold_lines = []
    for y in gauge_y:
        (line,) = ax_det.plot([1.0, 1.0], [y - 0.23, y + 0.23], color=C["red"],
                              lw=1.5, ls="--", zorder=4, solid_capstyle="butt")
        line.set_visible(False)
        threshold_lines.append(line)
    threshold_text = ax_det.text(1.0, 0.82, "기준값", ha="center", va="top", fontsize=7.8,
                                 fontweight="bold", color=C["red"],
                                 fontname=KOREAN_FONT or "DejaVu Sans")
    threshold_text.set_visible(False)
    gauge_labels = [ax_det.text(0.03, y + 0.46, "-", ha="left", va="center", fontsize=7.8,
                                color=C["muted"], fontname=KOREAN_FONT or "DejaVu Sans") for y in gauge_y]
    or3_text = ax_det.text(0.04, 0.15, "OR_3  대기", ha="left", va="center", fontsize=9.5,
                           fontweight="bold", fontname=KOREAN_FONT or "DejaVu Sans",
                           color=C["muted"], bbox=dict(boxstyle="round,pad=0.35",
                                                        facecolor=C["dim"], edgecolor="none"))
    ax_det.text(1.55, 0.15, "지속 P2", ha="left", va="center", fontsize=9.5,
                fontweight="bold", fontname=KOREAN_FONT or "DejaVu Sans", color=C["muted"])
    p_boxes = []
    for i in range(2):
        box = Rectangle((2.05 + i * 0.40, -0.08), 0.30, 0.46,
                        facecolor=C["dim"], edgecolor="none", zorder=3)
        ax_det.add_patch(box)
        p_boxes.append(box)
    active_text = ax_det.text(0.04, -0.75, "활성 탐지기: 없음", ha="left", va="center",
                              fontsize=8.5, fontname=KOREAN_FONT or "DejaVu Sans", color=C["muted"])
    final_text = ax_det.text(GAUGE_CAP / 2, -1.3, "모델 대기", ha="center", va="center",
                             fontsize=13, fontweight="bold", fontname=KOREAN_FONT or "DejaVu Sans",
                             color=C["muted"])

    style_card(ax_feat, "실시간 처리", "원본 > 특징 추출 > 탐지")
    feat_intro_text = mono_text(ax_feat, 0.05, 0.96, size=8.2)
    feature_artists = create_feature_panel(ax_feat)
    for artist in feature_artists.values():
        artist.set_visible(False)

    for line in threshold_lines:
        line.set_visible(True)
    threshold_text.set_visible(True)
    set_pill(pill_phase, "리플레이", C["blue"])
    set_pill(pill_truth, "-", C["muted"])
    set_pill(pill_det, "대기", C["muted"])
    progress_bar.set_facecolor(C["blue"])
    hdr_overall.set_text("PIPELINE 12/12 완료  ·  STREAMING 대기")

    # Shared streaming state.
    first_fault_time = None
    first_alarm_time = None
    false_alarm_frames = 0
    last_feature_frame = -1
    last_axis_frame = -1
    log_rows: list[dict[str, object]] = []
    # Gap-separated Fault event history
    fault_event_counter = 0
    current_fault_event_id = None
    current_fault_event_start = None
    current_fault_event_gap = None
    current_fault_event_alarm = None

    event_history = []
    previous_kind = None

    fault_original_first = pd.to_datetime(fault_raw["TimeStamp"], errors="coerce").dropna().iloc[0]

    feat_intro_text.set_text(
        "최종 모델 READY\n" + "-" * 30 + "\n"
        f"학습 데이터   {fit_info['train_rows']:>10,}\n"
        f"보정 데이터   {fit_info['calibration_rows']:>10,}\n"
        f"기준 분위수   {detector.quantile:.4f}\n"
        "공분산        Ledoit-Wolf\n"
        "앙상블        OR_3 + P2\n\n"
        "입력 방식     CSV streaming replay\n"
        "처리 방식     causal / row-by-row\n"
        "look-ahead    없음\n\n"
        "현재부터 입력 1행마다\n"
        "검증 → segment → 특징 → 탐지"
    )
    feat_intro_text.set_visible(True)
    for artist in feature_artists.values():
        artist.set_visible(False)

    # Clear the feature table state only after the first stream row arrives.
    total_stream_frames = len(demo_df)
    infer_ms_list: list[float] = []
    replay_t = {"t0": None}
    stats = dict(normal_fp=0, idle_fp=0, fault_rows=0, fault_alarm=0, onset_idx=None, first_alarm_idx=None)

    def stream_update(frame_idx: int):
        nonlocal first_fault_time, first_alarm_time, false_alarm_frames
        nonlocal last_feature_frame, last_axis_frame

        nonlocal fault_event_counter
        nonlocal current_fault_event_id
        nonlocal current_fault_event_start
        nonlocal current_fault_event_gap
        nonlocal current_fault_event_alarm
        nonlocal previous_kind

        row = demo_df.iloc[frame_idx]
        source = str(row["block"])
        original_timestamp = row["orig_TimeStamp"]

        if replay_t["t0"] is None:
            replay_t["t0"] = time.time()
        _t0 = time.perf_counter()
        record, result = pipeline.process(
            row,
            source=source,
            block=source,
            original_timestamp=original_timestamp,
        )
        infer_ms = (time.perf_counter() - _t0) * 1000.0   # 행당 전처리+특징+탐지 처리 지연
        infer_ms_list.append(infer_ms)

        timestamp = record["timestamp"]
        kind = str(record["kind"])
        ai0 = float(record["AI0_Vibration"])
        ai1 = float(record["AI1_Vibration"])
        ai2 = float(record["AI2_Current"])


        # --------------------------------------------------------
        # Gap-separated Fault event tracking
        # --------------------------------------------------------

        new_fault_event = (
            kind == "fault"
            and (
                previous_kind != "fault"
                or bool(record["segment_reset"])
            )
        )

        if new_fault_event:
            fault_event_counter += 1

            current_fault_event_id = fault_event_counter
            current_fault_event_start = timestamp
            current_fault_event_gap = float(record["gap_seconds"])
            current_fault_event_alarm = None

            event_history.append({
                "event_id": fault_event_counter,
                "fault_start": timestamp,
                "gap_sec": current_fault_event_gap,
                "first_alarm": None,
                "delay_sec": None,
                "detected": False,
            })


        # 현재 Fault event에서 최초 P2 탐지
        if kind == "fault" and event_history:
            current_event = event_history[-1]

            if (
                result["final_alarm"]
                and current_fault_event_alarm is None
            ):
                current_fault_event_alarm = timestamp

                current_event["first_alarm"] = timestamp
                current_event["delay_sec"] = (
                    timestamp - current_fault_event_start
                ).total_seconds()
                current_event["detected"] = True


        previous_kind = kind


        if kind == "fault" and first_fault_time is None:
            first_fault_time = timestamp

        if first_fault_time is not None and timestamp >= first_fault_time and result["final_alarm"] and first_alarm_time is None:
            first_alarm_time = timestamp

        if result["final_alarm"] and kind != "fault":
            false_alarm_frames += 1
            stats["idle_fp" if kind == "idle" else "normal_fp"] += 1
        if kind == "fault":
            stats["fault_rows"] += 1
            if stats["onset_idx"] is None:
                stats["onset_idx"] = frame_idx
            if result["final_alarm"]:
                stats["fault_alarm"] += 1
                if stats["first_alarm_idx"] is None:
                    stats["first_alarm_idx"] = frame_idx

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

        if result["segment_reset"]:
            push(**GAP_ENTRY)
        push(ai0=ai0, ai1=ai1, ai2=ai2,
             r_sample=ratios[0], r_w05=ratios[1], r_w1=ratios[2],
             truth=truth_code, det=det_code)
        draw_shading(bufs["truth"])

        line_ai0.set_ydata(np.asarray(bufs["ai0"], dtype=float))
        line_ai1.set_ydata(np.asarray(bufs["ai1"], dtype=float))
        line_ai2.set_ydata(np.asarray(bufs["ai2"], dtype=float))
        line_r_sample.set_ydata(np.asarray(bufs["r_sample"], dtype=float))
        line_r_w05.set_ydata(np.asarray(bufs["r_w05"], dtype=float))
        line_r_w1.set_ydata(np.asarray(bufs["r_w1"], dtype=float))
        strip_img.set_data(np.vstack([
            np.asarray(bufs["truth"], dtype=float),
            np.asarray(bufs["det"], dtype=float),
        ]))

        if frame_idx - last_axis_frame >= args.axis_update_every:
            last_axis_frame = frame_idx
            vib = np.concatenate([
                np.asarray(bufs["ai0"], dtype=float),
                np.asarray(bufs["ai1"], dtype=float),
            ])
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
            ratio_all = np.concatenate([
                np.asarray(bufs["r_sample"], dtype=float),
                np.asarray(bufs["r_w05"], dtype=float),
                np.asarray(bufs["r_w1"], dtype=float),
            ])
            ratio_all = ratio_all[np.isfinite(ratio_all)]
            if len(ratio_all):
                ax_ratio.set_ylim(
                    min(0.05, max(1e-4, float(ratio_all.min()) / 2.0)),
                    max(10.0, float(ratio_all.max()) * 2.5),
                )

        set_pill(pill_phase, "완료" if frame_idx == total_stream_frames - 1 else "리플레이",
                 C["green"] if frame_idx == total_stream_frames - 1 else C["blue"])
        set_pill(pill_truth, {"normal": "정상", "idle": "유휴", "fault": "고장"}.get(kind, "-"),
                 {"normal": C["green"], "idle": C["blue"], "fault": C["red"]}.get(kind, C["muted"]))
        det_label = "오탐 경보" if result["final_alarm"] and kind != "fault" else final_state
        set_pill(pill_det, det_label, det_color)

        for bar, label, ratio, score, threshold, alarm in zip(
            gauge_bars,
            gauge_labels,
            ratios,
            [float(result["sample_score"]), float(result["win05_score"]), float(result["win1_score"])],
            [float(thr["sample"]), float(thr["win05"]), float(thr["win1"])],
            [bool(result["sample_alarm"]), bool(result["win05_alarm"]), bool(result["win1_alarm"])],
        ):
            bar.set_width(0.0 if not np.isfinite(ratio) else min(ratio, GAUGE_CAP))
            bar.set_facecolor(C["red"] if alarm else C["green"])
            label.set_text(f"{fmt(score, 2)} / {fmt(threshold, 2)}   x{fmt(ratio, 2)}")
            label.set_color(C["red"] if alarm else C["muted"])

        or3_on = bool(result["or3_base"])
        or3_text.set_text("OR_3  켜짐" if or3_on else "OR_3  꺼짐")
        or3_text.get_bbox_patch().set_facecolor(C["amber"] if or3_on else C["dim"])
        or3_text.set_color(C["bg"] if or3_on else C["text"])
        for i, box in enumerate(p_boxes):
            count = int(result["persistence_count"])
            box.set_facecolor(C["red"] if i < count and result["final_alarm"] else C["amber"] if i < count else C["dim"])

        active = result["active_detectors"]
        active_labels = {"sample": "샘플", "0.5s": "0.5초", "1.0s": "1.0초"}
        active_text.set_text(f"활성 탐지기: {', '.join(active_labels.get(x, x) for x in active) if active else '없음'}")
        active_text.set_color(C["text"] if active else C["muted"])
        final_text.set_text(final_state)
        final_text.set_color(det_color)
        for spine in ax_det.spines.values():
            spine.set_color(det_color if result["final_alarm"] else C["border"])

        live_text.set_text(
            f"입력 블록     {source.upper()}\n"
            f"원본 시간     {ts_str(record['orig_timestamp'])}\n"
            f"스트림 시간   {ts_str(timestamp)}\n"
            + "-" * 30 + "\n"
            f"AI0           {ai0: .6f}\n"
            f"AI1           {ai1: .6f}\n"
            f"AI2           {ai2: .3f}\n"
            + "-" * 30 + "\n"
            f"유휴 {record['Idle']}   상태 {record['Equipment_state']}\n"
            f"세그먼트      {record['group_id']}\n"
            f"Gap           {record['gap_seconds']:.3f}초\n"
            f"Segment reset {'예' if record['segment_reset'] else '아니오'}\n"
            f"진행          {frame_idx + 1}/{total_stream_frames}"
        )

        # --------------------------------------------------------
        # 가장 최근 Fault event의 Gap / 탐지 지연만 표시
        # --------------------------------------------------------

        latest_event_text = "-"

        if event_history:
            ev = event_history[-1]

            if ev["detected"]:
                latest_event_text = (
                    f"E{int(ev['event_id']):02d}  "
                    f"gap {float(ev['gap_sec']):.2f}s  "
                    f"delay {float(ev['delay_sec']):.2f}s ✓"
                )

            elif (
                ev["event_id"] == current_fault_event_id
                and current_fault_event_start is not None
            ):
                elapsed = (
                    timestamp - current_fault_event_start
                ).total_seconds()

                latest_event_text = (
                    f"E{int(ev['event_id']):02d}  "
                    f"gap {float(ev['gap_sec']):.2f}s  "
                    f"진행 +{elapsed:.2f}s"
                )

            else:
                latest_event_text = (
                    f"E{int(ev['event_id']):02d}  "
                    f"gap {float(ev['gap_sec']):.2f}s  "
                    f"MISS ✕"
                )


        detected_count = sum(
            1
            for ev in event_history
            if ev["detected"]
        )


        # 현재 처리 중인 Fault event
        current_event_line = "현재 Fault event  -"

        if (
            current_fault_event_id is not None
            and current_fault_event_start is not None
        ):

            # 이미 현재 event가 탐지된 경우
            if current_fault_event_alarm is not None:

                current_delay = (
                    current_fault_event_alarm
                    - current_fault_event_start
                ).total_seconds()

                current_event_line = (
                    f"현재 Event   E{current_fault_event_id:02d}  "
                    f"gap {float(current_fault_event_gap):.2f}s  "
                    f"delay {current_delay:.2f}s"
                )

            # 마지막 프레임까지 탐지되지 않은 경우
            elif frame_idx == total_stream_frames - 1:

                current_event_line = (
                    f"현재 Event   E{current_fault_event_id:02d}  "
                    f"gap {float(current_fault_event_gap):.2f}s  "
                    f"MISS"
                )

            # 현재 event가 아직 진행 중인 경우
            else:

                elapsed = (
                    timestamp
                    - current_fault_event_start
                ).total_seconds()

                current_event_line = (
                    f"현재 Event   E{current_fault_event_id:02d}  "
                    f"gap {float(current_fault_event_gap):.2f}s  "
                    f"경과 +{elapsed:.2f}s"
                )


        event_text.set_text(
            f"q={detector.quantile:.4f}  "
            f"OR_3+P{detector.p}  "
            f"mean{detector.sample_agg_k}\n"

            f"CSV replay {fps_eff:.0f}Hz "
            f"(x{replay_x:.1f}) | causal\n"

            f"gap > {detector.gap_threshold_sec:.1f}s "
            f"→ 다음 Fault event로 분리\n"

            f"{current_event_line}\n"

            f"Event 탐지      "
            f"{detected_count}/{fault_event_counter}\n"

            f"정상 오탐        "
            f"{stats['normal_fp']}  |  "
            f"Idle 오탐 {stats['idle_fp']}\n"

            f"최근 Gap 탐지    "
            f"{latest_event_text}\n"

            f"현재 상태        "
            f"{final_state}\n"

            f"추론 지연        "
            f"{float(np.mean(infer_ms_list)):.1f} / "
            f"p95 {float(np.percentile(infer_ms_list, 95)):.1f} ms"
        )

        if first_fault_time is not None and first_alarm_time is not None:
            delay = max(0.0, (first_alarm_time - first_fault_time).total_seconds())
            d_samples = int(stats["first_alarm_idx"] - stats["onset_idx"]) if stats["first_alarm_idx"] is not None else 0
            delay_label.set_text(f"탐지 지연  ({d_samples} samples · 간격 {nominal_dt:.2f}s)")
            delay_text.set_text(f"{delay:.2f} s")
            delay_text.set_color(C["green"])
        elif first_fault_time is not None and timestamp >= first_fault_time:
            delay_text.set_text("detecting...")
            delay_text.set_color(C["amber"])
        else:
            delay_text.set_text("-")
            delay_text.set_color(C["muted"])

        feature_snapshot = get_feature_snapshot(detector)
        if frame_idx - last_feature_frame >= args.feature_update_every or frame_idx == 0:
            last_feature_frame = frame_idx
            feat_intro_text.set_visible(False)
            for artist in feature_artists.values():
                artist.set_visible(True)
            update_feature_panel(feature_artists, detector, feature_snapshot)

        log_rows.append({
            "stream_index": int(frame_idx),
            "infer_ms": float(infer_ms),
            "source": record["source"], "block": record["block"], "kind": record["kind"],
            "TimeStamp": record["timestamp"], "orig_TimeStamp": record["orig_timestamp"],
            "ground_truth": int(record["ground_truth"]), "Idle": int(record["Idle"]),
            "Equipment_state": int(record["Equipment_state"]),
            "segment_id": int(record["segment_id"]), "group_id": record["group_id"],
            "segment_reset": bool(record["segment_reset"]), "gap_seconds": float(record["gap_seconds"]),
            "AI0_Vibration": ai0, "AI1_Vibration": ai1, "AI2_Current": ai2,
            "score_0.5s": float(result["win05_score"]), "threshold_0.5s": float(thr["win05"]),
            "alarm_0.5s": bool(result["win05_alarm"]),
            "score_1.0s": float(result["win1_score"]), "threshold_1.0s": float(thr["win1"]),
            "alarm_1.0s": bool(result["win1_alarm"]),
            "sample_score": float(result["sample_score"]), "sample_threshold": float(thr["sample"]),
            "sample_alarm": bool(result["sample_alarm"]), "OR_3": bool(result["or3_base"]),
            "P2": bool(result["final_alarm"]),
            "state": "ALARM" if result["final_alarm"] else "WARNING" if result["or3_base"] else "정상",
            "active_detectors": ",".join(result["active_detectors"]),
        })
        hdr_overall.set_text(
            f"PIPELINE 12/12 완료  ·  STREAMING {(frame_idx + 1) / total_stream_frames * 100:.0f}%"
            f"  ({frame_idx + 1:,}/{total_stream_frames:,}행)")
        progress_bar.set_width((frame_idx + 1) / total_stream_frames)
        if frame_idx == total_stream_frames - 1:
            hdr_overall.set_text(
                f"PIPELINE 12/12 완료  ·  STREAMING 완료  ·  replay 경과 {time.time() - replay_t['t0']:.0f}s"
                "  (추론 지연 아님)")
        return []

    plt.ioff()
    ani = animation.FuncAnimation(
        fig,
        stream_update,
        frames=len(demo_df),
        init_func=lambda: [],
        interval=max(1, int(1000 / fps_eff)),
        blit=False,
        repeat=False,
    )

    try:
        plt.show()
    finally:
        output_path = output_dir / "realtime_log.csv"
        if log_rows:
            pd.DataFrame(log_rows).to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"[SAVED] {output_path}")
        else:
            print("[WARN] No realtime log rows were produced.")

        # 12단계 완료 처리 후 단계별 리포트 갱신
        report_rows = board.report_rows() + [dict(
            stage=13, label=stage_labels[12],
            status="done" if len(log_rows) == total_stream_frames else "stopped",
            seconds=None if replay_t["t0"] is None else round(time.time() - replay_t["t0"], 2),
            summary=f"스트리밍 {len(log_rows):,}행 · 소요시간 = replay 경과 (추론 지연 아님)", figures=0)]
        pd.DataFrame(report_rows).to_csv(output_dir / "pipeline_stage_report.csv", index=False, encoding="utf-8-sig")
        metrics_df = events_df = None
        if log_rows:
            metrics_df, events_df = summarize_realtime(pd.DataFrame(log_rows), nominal_dt)
            metrics_df.to_csv(output_dir / "realtime_metrics.csv", index=False, encoding="utf-8-sig")
            events_df.to_csv(output_dir / "realtime_event_summary.csv", index=False, encoding="utf-8-sig")
            print(f"[SAVED] {output_dir / 'realtime_metrics.csv'}")
            print(f"[SAVED] {output_dir / 'realtime_event_summary.csv'}")

        # Final pipeline report for portfolio/demo use.
        report = output_dir / "pipeline_run_summary.txt"
        lines = ["KAMPact Full Pipeline Run", "=" * 80]
        for row in report_rows:
            sec = "-" if row["seconds"] is None else f"{row['seconds']:.1f}s"
            lines.append(f"{row['label']:<28} {row['status']:<8} {sec:>8}   {row['summary']}")
        lines += [
            "",
            f"Deployment train rows      : {fit_info['train_rows']:,}",
            f"Deployment calibration rows: {fit_info['calibration_rows']:,}",
            f"Deployment quantile         : {detector.quantile:.4f}",
            f"Deployment thresholds       : {fit_info['thresholds']}",
            f"Streaming rows              : {len(log_rows):,}",
            f"Fault onset                 : {first_fault_time}",
            f"First alarm                 : {first_alarm_time}",
        ]
        if first_fault_time is not None and first_alarm_time is not None:
            lines.append(
                f"Detection delay             : {(first_alarm_time - first_fault_time).total_seconds():.3f}s"
            )
        if metrics_df is not None:
            lines += ["", "Realtime metrics", "-" * 40]
            lines += [f"{m:<32}{v}" for m, v in zip(metrics_df["metric"], metrics_df["value"])]
        report.write_text("\n".join(lines), encoding="utf-8")
        print(f"[SAVED] {report}")

    _ = ani


if __name__ == "__main__":
    main()