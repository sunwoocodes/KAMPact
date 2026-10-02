"""
KAMPact - Real-Time Monitoring Dashboard
========================================

5단계 + 6단계 통합

목적
----
정상 운전 -> Fault 이벤트가 하나의 스트림으로 순차적으로 들어온다고 가정하고

    Raw Sensor
        ↓
    Buffer
        ↓
    0.5s Window Detector
    1.0s Window Detector
    Sample-level Detector
        ↓
    OR_3
        ↓
    P2
        ↓
    Real-Time Alarm

을 시각적으로 시연한다.

중요
----
- 정상/고장 데이터를 별도 실행하지 않는다.
- 기본 데모는:
      Normal operation segment
          ->
      fault_19
  순서로 이어서 재생한다.
- Normal CSV와 Fault CSV 사이의 실제 timestamp gap은
  새로운 segment 경계로 취급한다.
- 실제 운영을 가정하여 detector는 한 번만 fitting하고
  이후 입력 sample에 대해 순차적으로 추론한다.
- Equipment_state / Idle은 detector 입력에 사용하지 않는다.
  오직 Ground Truth 표시 및 demo 평가용으로만 사용한다.

출력
----
outputs/11_realtime_dashboard/
    realtime_log.csv

실행
----
python src/11_realtime_dashboard.py

빠른 발표용
----
python src/11_realtime_dashboard.py --speed 20

다른 fault event
----
python src/11_realtime_dashboard.py --fault-event fault_5
python src/11_realtime_dashboard.py --fault-event fault_19

Normal 재생 길이 조정
----
python src/11_realtime_dashboard.py --normal-seconds 30

모든 fault event 재생
----
python src/11_realtime_dashboard.py --fault-event all
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from sklearn.covariance import LedoitWolf


# ============================================================
# 기본 설정
# ============================================================

TIME_COL = "TimeStamp"
STATE_COL = "Equipment_state"
IDLE_COL = "Idle"

SENSORS = [
    "AI0_Vibration",
    "AI1_Vibration",
    "AI2_Current",
]

FEATURES = [
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

WINDOWS = {
    "0.5s": 0.5,
    "1.0s": 1.0,
}

DEFAULT_NORMAL_SECONDS = 20.0
DEFAULT_SPEED = 20.0
DEFAULT_FAULT_EVENT = "fault_19"

DISPLAY_SECONDS = 12.0

SAMPLE_AGG_K = 3
QUANTILE = 0.9999


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

    # 5_2 @dataclass 문제 방지
    sys.modules[name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise

    return module


# ============================================================
# Basic transforms
# ============================================================

def signed_log1p(
    x: np.ndarray,
) -> np.ndarray:

    return (
        np.sign(x)
        * np.log1p(np.abs(x))
    )


# ============================================================
# Data loading
# ============================================================

def load_csv(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [
        TIME_COL,
        *SENSORS,
        STATE_COL,
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path}\n"
            f"필수 컬럼 누락: {missing}"
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
        *SENSORS,
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

    return df


def add_segments(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    gap = (
        out[TIME_COL]
        .diff()
        .dt.total_seconds()
    )

    break_mask = gap > 0.5
    break_mask.iloc[0] = True

    out["segment_id"] = (
        break_mask.cumsum() - 1
    )

    return out


# ============================================================
# Demo stream construction
# ============================================================

def get_normal_demo_segment(
    normal: pd.DataFrame,
    seconds: float,
) -> pd.DataFrame:

    normal = normal.copy()

    # Idle 제외
    if IDLE_COL in normal.columns:
        normal = normal[
            normal[IDLE_COL].eq(0)
        ].copy()

    normal = add_segments(
        normal
    )

    segments = []

    for _, group in normal.groupby(
        "segment_id",
        sort=True,
    ):

        if len(group) < 5:
            continue

        duration = (
            group[TIME_COL].iloc[-1]
            - group[TIME_COL].iloc[0]
        ).total_seconds()

        if duration >= seconds:

            result = group[
                group[TIME_COL]
                <= (
                    group[TIME_COL].iloc[0]
                    + pd.Timedelta(
                        seconds=seconds
                    )
                )
            ].copy()

            if len(result) >= 5:
                return result.reset_index(
                    drop=True
                )

        segments.append(group)

    if not segments:
        raise ValueError(
            "사용 가능한 Normal operation segment가 없습니다."
        )

    # fallback
    result = segments[0].copy()

    return result.reset_index(
        drop=True
    )


def get_fault_event(
    fault: pd.DataFrame,
    event_id: str,
) -> pd.DataFrame:

    fault = fault.copy()

    fault = add_segments(
        fault
    )

    fault["group_id"] = (
        "fault_"
        + fault["segment_id"].astype(str)
    )

    if event_id == "all":

        parts = []

        for gid, group in fault.groupby(
            "group_id",
            sort=True,
        ):

            parts.append(
                group.reset_index(
                    drop=True
                )
            )

        if not parts:
            raise ValueError(
                "Fault event가 없습니다."
            )

        return pd.concat(
            parts,
            ignore_index=True,
        )

    if event_id not in set(
        fault["group_id"].astype(str)
    ):
        available = sorted(
            fault["group_id"]
            .astype(str)
            .unique()
        )

        raise ValueError(
            f"{event_id}를 찾을 수 없습니다.\n"
            f"가능한 event: {available}"
        )

    return (
        fault[
            fault["group_id"]
            .astype(str)
            .eq(event_id)
        ]
        .reset_index(
            drop=True
        )
    )


def build_demo_stream(
    normal: pd.DataFrame,
    fault: pd.DataFrame,
    normal_seconds: float,
    fault_event: str,
) -> pd.DataFrame:

    normal_demo = (
        get_normal_demo_segment(
            normal,
            normal_seconds,
        )
    )

    fault_demo = (
        get_fault_event(
            fault,
            fault_event,
        )
    )

    normal_demo = normal_demo.copy()
    fault_demo = fault_demo.copy()

    normal_demo["stream_source"] = "NORMAL"
    fault_demo["stream_source"] = "FAULT"

    # Demo timeline
    normal_demo["demo_time_sec"] = (
        (
            normal_demo[TIME_COL]
            - normal_demo[TIME_COL].iloc[0]
        )
        .dt.total_seconds()
    )

    normal_end = (
        normal_demo["demo_time_sec"].iloc[-1]
    )

    # 아주 짧은 transition
    # 실제 timestamp gap을 연결하지 않는다.
    fault_demo["demo_time_sec"] = (
        normal_end
        + 0.1
        + (
            (
                fault_demo[TIME_COL]
                - fault_demo[TIME_COL].iloc[0]
            )
            .dt.total_seconds()
        )
    )

    normal_demo["ground_truth"] = 0
    fault_demo["ground_truth"] = 1

    stream = pd.concat(
        [
            normal_demo,
            fault_demo,
        ],
        ignore_index=True,
    )

    stream["stream_index"] = np.arange(
        len(stream)
    )

    return stream


# ============================================================
# Window Feature
# 4_make_window_dataset.py와 동일한 정의
# ============================================================

def safe_rms(
    x: np.ndarray,
) -> float:

    return float(
        np.sqrt(
            np.mean(
                np.asarray(x, dtype=float) ** 2
            )
        )
    )


def safe_slope(
    x: np.ndarray,
    time_sec: np.ndarray,
) -> float:

    x = np.asarray(
        x,
        dtype=float,
    )

    time_sec = np.asarray(
        time_sec,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(time_sec)
    )

    x = x[valid]
    time_sec = time_sec[valid]

    if len(x) < 2:
        return np.nan

    if np.ptp(time_sec) == 0:
        return np.nan

    return float(
        np.polyfit(
            time_sec,
            x,
            1,
        )[0]
    )


def safe_corr(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0:
        return 0.0

    if np.std(y) == 0:
        return 0.0

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def extract_window_features(
    window: pd.DataFrame,
) -> np.ndarray:

    if window.empty:
        return np.full(
            len(FEATURES),
            np.nan,
        )

    feature_dict = {}

    time_sec = (
        (
            window[TIME_COL]
            - window[TIME_COL].iloc[0]
        )
        .dt.total_seconds()
        .to_numpy(
            dtype=float
        )
    )

    for sensor in SENSORS:

        x = pd.to_numeric(
            window[sensor],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        valid = np.isfinite(x)

        x_valid = x[valid]

        if len(x_valid) == 0:

            feature_dict[
                f"{sensor}_mean"
            ] = np.nan

            feature_dict[
                f"{sensor}_std"
            ] = np.nan

            feature_dict[
                f"{sensor}_rms"
            ] = np.nan

            feature_dict[
                f"{sensor}_ptp"
            ] = np.nan

            feature_dict[
                f"{sensor}_slope"
            ] = np.nan

            continue

        feature_dict[
            f"{sensor}_mean"
        ] = float(
            np.mean(
                x_valid
            )
        )

        feature_dict[
            f"{sensor}_std"
        ] = float(
            np.std(
                x_valid,
                ddof=1,
            )
        ) if len(x_valid) > 1 else 0.0

        feature_dict[
            f"{sensor}_rms"
        ] = safe_rms(
            x_valid
        )

        feature_dict[
            f"{sensor}_ptp"
        ] = float(
            np.ptp(
                x_valid
            )
        )

        feature_dict[
            f"{sensor}_slope"
        ] = safe_slope(
            x,
            time_sec,
        )

    feature_dict[
        "AI0_AI1_corr"
    ] = safe_corr(
        window[
            "AI0_Vibration"
        ].to_numpy(
            dtype=float
        ),
        window[
            "AI1_Vibration"
        ].to_numpy(
            dtype=float
        ),
    )

    return np.asarray(
        [
            feature_dict[c]
            for c in FEATURES
        ],
        dtype=float,
    )


# ============================================================
# Window Detector
# ============================================================

class WindowDetector:

    def __init__(
        self,
        train_windows: pd.DataFrame,
        window_sec: float,
        quantile: float,
    ):

        self.window_sec = window_sec
        self.quantile = quantile

        normal_train = (
            train_windows[
                (
                    train_windows[
                        "label"
                    ]
                    .astype(int)
                    .eq(0)
                )
                &
                (
                    train_windows[
                        "source"
                    ]
                    .astype(str)
                    .eq("normal")
                )
                &
                (
                    train_windows[
                        "split"
                    ]
                    .astype(str)
                    .eq("train")
                )
            ]
            .copy()
        )

        if normal_train.empty:
            raise ValueError(
                f"{window_sec}s normal train window가 없습니다."
            )

        X = (
            normal_train[
                FEATURES
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        # Train median imputation
        self.median = X.median()

        X = X.fillna(
            self.median
        ).to_numpy(
            dtype=float
        )

        X = signed_log1p(
            X
        )

        self.model = LedoitWolf(
            assume_centered=False
        )

        self.model.fit(
            X
        )

        train_scores = self.score(
            X
        )

        self.threshold = float(
            np.quantile(
                train_scores,
                quantile,
            )
        )

    def score(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        diff = (
            X
            - self.model.location_
        )

        return np.einsum(
            "ij,jk,ik->i",
            diff,
            self.model.precision_,
            diff,
        )

    def predict(
        self,
        window_df: pd.DataFrame,
    ) -> tuple[float, bool]:

        required_samples = int(
            np.ceil(
                self.window_sec / 0.1
            )
        )

        if len(window_df) < max(
            2,
            required_samples,
        ):
            return np.nan, False

        X = extract_window_features(
            window_df
        )

        X = pd.Series(
            X,
            index=FEATURES,
        )

        X = (
            pd.to_numeric(
                X,
                errors="coerce",
            )
            .fillna(
                self.median
            )
            .to_numpy(
                dtype=float
            )
        )

        X = signed_log1p(
            X.reshape(
                1,
                -1,
            )
        )

        score = float(
            self.score(X)[0]
        )

        alarm = (
            score
            >= self.threshold
        )

        return score, bool(
            alarm
        )


# ============================================================
# Sample-level detector
# ============================================================

class SampleDetector:

    def __init__(
        self,
        normal_raw: pd.DataFrame,
        quantile: float,
    ):

        self.quantile = quantile

        data = (
            normal_raw
            .copy()
        )

        # Idle 제외
        if IDLE_COL in data.columns:

            data = data[
                data[IDLE_COL].eq(0)
            ].copy()

        # segment
        data = add_segments(
            data
        )

        data["group_id"] = (
            "normal_"
            + data["segment_id"].astype(str)
        )

        values = (
            data[SENSORS]
            .to_numpy(
                dtype=float
            )
        )

        diff = (
            data
            .groupby("group_id")[
                SENSORS
            ]
            .diff()
            .fillna(0.0)
            .to_numpy(
                dtype=float
            )
        )

        X = np.hstack(
            [
                values,
                diff,
            ]
        )

        X = signed_log1p(
            X
        )

        self.model = LedoitWolf(
            assume_centered=False
        )

        self.model.fit(
            X
        )

        raw_scores = self.score(
            X
        )

        # causal mean3
        smooth = (
            pd.Series(
                raw_scores
            )
            .rolling(
                SAMPLE_AGG_K,
                min_periods=SAMPLE_AGG_K,
            )
            .mean()
            .to_numpy()
        )

        finite = np.isfinite(
            smooth
        )

        self.threshold = float(
            np.quantile(
                smooth[finite],
                quantile,
            )
        )

        # 마지막 raw sample
        self.prev_values = None

        self.score_history = deque(
            maxlen=SAMPLE_AGG_K
        )

    def score(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        diff = (
            X
            - self.model.location_
        )

        return np.einsum(
            "ij,jk,ik->i",
            diff,
            self.model.precision_,
            diff,
        )

    def reset(
        self,
    ) -> None:

        self.prev_values = None

        self.score_history.clear()

    def predict(
        self,
        row: pd.Series,
    ) -> tuple[float, bool]:

        values = row[
            SENSORS
        ].to_numpy(
            dtype=float
        )

        if (
            self.prev_values is None
        ):

            diff = np.zeros(
                3,
                dtype=float,
            )

        else:

            diff = (
                values
                - self.prev_values
            )

        self.prev_values = (
            values.copy()
        )

        X = np.hstack(
            [
                values,
                diff,
            ]
        ).reshape(
            1,
            -1,
        )

        X = signed_log1p(
            X
        )

        raw_score = float(
            self.score(X)[0]
        )

        self.score_history.append(
            raw_score
        )

        if len(
            self.score_history
        ) < SAMPLE_AGG_K:

            return raw_score, False

        smooth_score = float(
            np.mean(
                self.score_history
            )
        )

        alarm = (
            smooth_score
            >= self.threshold
        )

        return (
            smooth_score,
            bool(alarm),
        )


# ============================================================
# Dashboard
# ============================================================

class RealtimeDashboard:

    def __init__(
        self,
        stream: pd.DataFrame,
        detector_05: WindowDetector,
        detector_10: WindowDetector,
        sample_detector: SampleDetector,
        speed: float,
        output_dir: Path,
    ):

        self.stream = stream

        self.detector_05 = (
            detector_05
        )

        self.detector_10 = (
            detector_10
        )

        self.sample_detector = (
            sample_detector
        )

        self.speed = max(
            0.1,
            speed,
        )

        self.output_dir = (
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.buffer = []

        self.current_index = 0

        self.or3_history = deque(
            maxlen=2
        )

        self.p2_alarm = False

        self.logs = []

        self.start_wall_time = None

        self.fig = None

        self.axes = {}

        self.lines = {}

        self.text_status = None

        self.text_detectors = None

        self.text_workflow = None

        self.text_info = None

        self.animation = None

        self.last_demo_time = None

    # ========================================================
    # Buffer reset
    # ========================================================

    def reset_buffer(
        self,
    ) -> None:

        self.buffer = []

        self.or3_history.clear()

        self.p2_alarm = False

        self.sample_detector.reset()

    # ========================================================
    # Current trailing window
    # ========================================================

    def get_window(
        self,
        window_sec: float,
    ) -> pd.DataFrame:

        if not self.buffer:
            return pd.DataFrame()

        current_time = (
            self.buffer[-1][TIME_COL]
        )

        start_time = (
            current_time
            - pd.Timedelta(
                seconds=window_sec
            )
        )

        df = pd.DataFrame(
            self.buffer
        )

        return df[
            (
                df[TIME_COL]
                >= start_time
            )
        ].copy()

    # ========================================================
    # Update detection
    # ========================================================

    def process_row(
        self,
        row: pd.Series,
    ) -> dict:

        # ----------------------------------------------------
        # Source transition
        # ----------------------------------------------------

        source = str(
            row["stream_source"]
        )

        if (
            self.last_demo_time is not None
            and source
            != self.last_source
        ):
            self.reset_buffer()

        self.last_source = source

        self.buffer.append(
            row.to_dict()
        )

        # ----------------------------------------------------
        # Window detectors
        # ----------------------------------------------------

        window_05 = self.get_window(
            0.5
        )

        window_10 = self.get_window(
            1.0
        )

        score_05, alarm_05 = (
            self.detector_05.predict(
                window_05
            )
        )

        score_10, alarm_10 = (
            self.detector_10.predict(
                window_10
            )
        )

        # ----------------------------------------------------
        # Sample detector
        # ----------------------------------------------------

        score_sample, alarm_sample = (
            self.sample_detector.predict(
                row
            )
        )

        # ----------------------------------------------------
        # OR_3
        # ----------------------------------------------------

        or3 = (
            alarm_05
            or alarm_10
            or alarm_sample
        )

        # ----------------------------------------------------
        # P2
        # ----------------------------------------------------

        self.or3_history.append(
            bool(or3)
        )

        if len(
            self.or3_history
        ) == 2:

            self.p2_alarm = all(
                self.or3_history
            )

        else:

            self.p2_alarm = False

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        if self.p2_alarm:

            state = "ALARM"

        elif or3:

            state = "WARNING"

        else:

            state = "NORMAL"

        detector_names = []

        if alarm_10:
            detector_names.append(
                "1.0s"
            )

        if alarm_05:
            detector_names.append(
                "0.5s"
            )

        if alarm_sample:
            detector_names.append(
                "Sample"
            )

        if detector_names:

            detector_text = (
                " + ".join(
                    detector_names
                )
            )

        else:

            detector_text = "None"

        # ----------------------------------------------------
        # Log
        # ----------------------------------------------------

        current_demo_time = float(
            row["demo_time_sec"]
        )

        log_row = {
            "stream_index": int(
                row["stream_index"]
            ),

            "source": source,

            "TimeStamp": row[
                TIME_COL
            ],

            "demo_time_sec": (
                current_demo_time
            ),

            "ground_truth": int(
                row["ground_truth"]
            ),

            "AI0_Vibration": float(
                row["AI0_Vibration"]
            ),

            "AI1_Vibration": float(
                row["AI1_Vibration"]
            ),

            "AI2_Current": float(
                row["AI2_Current"]
            ),

            "score_0.5s": score_05,
            "threshold_0.5s": (
                self.detector_05.threshold
            ),
            "alarm_0.5s": alarm_05,

            "score_1.0s": score_10,
            "threshold_1.0s": (
                self.detector_10.threshold
            ),
            "alarm_1.0s": alarm_10,

            "sample_score": score_sample,
            "sample_threshold": (
                self.sample_detector.threshold
            ),
            "sample_alarm": alarm_sample,

            "OR_3": bool(or3),

            "P2": bool(
                self.p2_alarm
            ),

            "state": state,
            "active_detectors": (
                detector_text
            ),
        }

        self.logs.append(
            log_row
        )

        self.last_demo_time = (
            current_demo_time
        )

        return log_row

    # ========================================================
    # Figure setup
    # ========================================================

    def setup_figure(
        self,
    ):

        self.fig = plt.figure(
            figsize=(
                16,
                10,
            )
        )

        gs = GridSpec(
            6,
            4,
            figure=self.fig,
            width_ratios=[
                2.2,
                2.2,
                2.2,
                1.5,
            ],
            hspace=0.65,
            wspace=0.35,
        )

        # ----------------------------------------------------
        # Sensor graphs
        # ----------------------------------------------------

        self.axes["ai0"] = self.fig.add_subplot(
            gs[0, :3]
        )

        self.axes["ai1"] = self.fig.add_subplot(
            gs[1, :3]
        )

        self.axes["ai2"] = self.fig.add_subplot(
            gs[2, :3]
        )

        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        self.axes["score"] = self.fig.add_subplot(
            gs[3, :3]
        )

        # ----------------------------------------------------
        # Detector bars
        # ----------------------------------------------------

        self.axes["detectors"] = self.fig.add_subplot(
            gs[4, :3]
        )

        # ----------------------------------------------------
        # Log panel
        # ----------------------------------------------------

        self.axes["log"] = self.fig.add_subplot(
            gs[5, :3]
        )

        # ----------------------------------------------------
        # Right information panel
        # ----------------------------------------------------

        self.axes["info"] = self.fig.add_subplot(
            gs[:, 3]
        )

        self.axes[
            "info"
        ].axis("off")

        # ----------------------------------------------------
        # Sensor labels
        # ----------------------------------------------------

        self.axes["ai0"].set_ylabel(
            "AI0 Vibration"
        )

        self.axes["ai1"].set_ylabel(
            "AI1 Vibration"
        )

        self.axes["ai2"].set_ylabel(
            "AI2 Current"
        )

        self.axes["score"].set_ylabel(
            "Normalized Score"
        )

        self.axes["score"].set_xlabel(
            "Demo Time (s)"
        )

        for name in [
            "ai0",
            "ai1",
            "ai2",
            "score",
        ]:

            self.axes[name].grid(
                True,
                alpha=0.25,
            )

        # ----------------------------------------------------
        # Detector axis
        # ----------------------------------------------------

        self.axes[
            "detectors"
        ].set_ylim(
            -0.05,
            1.1,
        )

        self.axes[
            "detectors"
        ].set_yticks(
            [0, 1]
        )

        self.axes[
            "detectors"
        ].set_yticklabels(
            [
                "OFF",
                "ALARM",
            ]
        )

        self.axes[
            "detectors"
        ].set_title(
            "Detector Status"
        )

        # ----------------------------------------------------
        # Log panel
        # ----------------------------------------------------

        self.axes[
            "log"
        ].axis("off")

        # ----------------------------------------------------
        # Workflow
        # ----------------------------------------------------

        self.text_status = (
            self.axes[
                "info"
            ].text(
                0.03,
                0.96,
                "SYSTEM: STARTING",
                transform=self.axes[
                    "info"
                ].transAxes,
                va="top",
                fontsize=18,
                fontweight="bold",
            )
        )

        self.text_detectors = (
            self.axes[
                "info"
            ].text(
                0.03,
                0.82,
                "",
                transform=self.axes[
                    "info"
                ].transAxes,
                va="top",
                fontsize=11,
                family="monospace",
            )
        )

        self.text_info = (
            self.axes[
                "info"
            ].text(
                0.03,
                0.58,
                "",
                transform=self.axes[
                    "info"
                ].transAxes,
                va="top",
                fontsize=10,
                family="monospace",
            )
        )

        self.text_workflow = (
            self.axes[
                "info"
            ].text(
                0.03,
                0.27,
                (
                    "FIELD WORKFLOW\n\n"
                    "Sensor Input\n"
                    "      ↓\n"
                    "Anomaly Detection\n"
                    "      ↓\n"
                    "OR_3\n"
                    "      ↓\n"
                    "P2 Confirmed\n"
                    "      ↓\n"
                    "Operator Check\n"
                    "      ↓\n"
                    "Pump Inspection\n"
                    "      ↓\n"
                    "Preventive Maintenance"
                ),
                transform=self.axes[
                    "info"
                ].transAxes,
                va="top",
                fontsize=10,
                family="monospace",
            )
        )

        self.fig.suptitle(
            "KAMPact — Real-Time Hydraulic Pump Monitoring",
            fontsize=18,
            fontweight="bold",
        )

    # ========================================================
    # Plot update
    # ========================================================

    def update_plot(
        self,
        frame,
    ):

        # ----------------------------------------------------
        # replay speed
        # ----------------------------------------------------

        if self.current_index >= len(
            self.stream
        ):

            self.finish()

            return []

        row = self.stream.iloc[
            self.current_index
        ]

        # ----------------------------------------------------
        # 실제 시간 기준 pacing
        # ----------------------------------------------------

        if self.current_index > 0:

            prev = self.stream.iloc[
                self.current_index - 1
            ]

            delta = (
                float(
                    row["demo_time_sec"]
                )
                - float(
                    prev["demo_time_sec"]
                )
            )

            if delta > 0:

                time.sleep(
                    delta / self.speed
                )

        # ----------------------------------------------------
        # Detector
        # ----------------------------------------------------

        result = self.process_row(
            row
        )

        self.current_index += 1

        # ----------------------------------------------------
        # Plot history
        # ----------------------------------------------------

        logs = pd.DataFrame(
            self.logs
        )

        current_time = (
            result["demo_time_sec"]
        )

        left_time = max(
            0.0,
            current_time
            - DISPLAY_SECONDS,
        )

        view = logs[
            logs[
                "demo_time_sec"
            ]
            >= left_time
        ].copy()

        # ----------------------------------------------------
        # Sensor plots
        # ----------------------------------------------------

        sensor_mapping = {
            "ai0": "AI0_Vibration",
            "ai1": "AI1_Vibration",
            "ai2": "AI2_Current",
        }

        for axis_name, col in (
            sensor_mapping.items()
        ):

            ax = self.axes[
                axis_name
            ]

            ax.clear()

            ax.plot(
                view[
                    "demo_time_sec"
                ],
                view[col],
                linewidth=1.5,
            )

            ax.set_xlim(
                left_time,
                max(
                    DISPLAY_SECONDS,
                    current_time,
                ),
            )

            ax.set_ylabel(
                col
            )

            ax.grid(
                True,
                alpha=0.25,
            )

            # Fault onset indicator
            fault_view = view[
                view["ground_truth"].eq(1)
            ]

            if not fault_view.empty:

                fault_start = (
                    fault_view[
                        "demo_time_sec"
                    ].iloc[0]
                )

                ax.axvline(
                    fault_start,
                    linestyle="--",
                    linewidth=1.2,
                )

        # ----------------------------------------------------
        # Score plot
        # ----------------------------------------------------

        ax = self.axes[
            "score"
        ]

        ax.clear()

        if not view.empty:

            ax.plot(
                view[
                    "demo_time_sec"
                ],
                (
                    view[
                        "score_0.5s"
                    ]
                    / view[
                        "threshold_0.5s"
                    ]
                ),
                linewidth=1.5,
                label="0.5s Window",
            )

            ax.plot(
                view[
                    "demo_time_sec"
                ],
                (
                    view[
                        "score_1.0s"
                    ]
                    / view[
                        "threshold_1.0s"
                    ]
                ),
                linewidth=1.5,
                label="1.0s Window",
            )

            ax.plot(
                view[
                    "demo_time_sec"
                ],
                (
                    view[
                        "sample_score"
                    ]
                    / view[
                        "sample_threshold"
                    ]
                ),
                linewidth=1.5,
                label="Sample",
            )

        ax.axhline(
            1.0,
            linestyle="--",
            linewidth=1.2,
            label="Threshold",
        )

        ax.set_ylim(
            0,
            max(
                2.5,
                float(
                    np.nanmax(
                        view[
                            [
                                "score_0.5s",
                                "score_1.0s",
                            ]
                        ]
                    )
                    /
                    min(
                        view[
                            "threshold_0.5s"
                        ].iloc[-1],
                        view[
                            "threshold_1.0s"
                        ].iloc[-1],
                    )
                )
                if not view.empty
                else 2.5,
            ),
        )

        ax.set_xlim(
            left_time,
            max(
                DISPLAY_SECONDS,
                current_time,
            ),
        )

        ax.set_ylabel(
            "Score / Threshold"
        )

        ax.set_xlabel(
            "Demo Time (s)"
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        ax.legend(
            loc="upper left",
            ncol=4,
            fontsize=8,
        )

        # ----------------------------------------------------
        # Detector status
        # ----------------------------------------------------

        ax = self.axes[
            "detectors"
        ]

        ax.clear()

        names = [
            "1.0s Window",
            "0.5s Window",
            "Sample",
            "OR_3",
            "P2",
        ]

        statuses = [
            int(
                result[
                    "alarm_1.0s"
                ]
            ),
            int(
                result[
                    "alarm_0.5s"
                ]
            ),
            int(
                result[
                    "sample_alarm"
                ]
            ),
            int(
                result["OR_3"]
            ),
            int(
                result["P2"]
            ),
        ]

        x = np.arange(
            len(names)
        )

        ax.bar(
            x,
            statuses,
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            names,
            rotation=15,
        )

        ax.set_ylim(
            -0.05,
            1.1,
        )

        ax.set_yticks(
            [0, 1]
        )

        ax.set_yticklabels(
            [
                "OFF",
                "ALARM",
            ]
        )

        ax.set_title(
            (
                "Current Detector State"
                f"  |  "
                f"{result['active_detectors']}"
            )
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        # ----------------------------------------------------
        # Log panel
        # ----------------------------------------------------

        ax = self.axes[
            "log"
        ]

        ax.clear()
        ax.axis("off")

        recent = logs.tail(
            5
        )

        lines = [
            (
                "Time       Source   "
                "Detectors        State"
            ),
            "-" * 60,
        ]

        for _, r in recent.iterrows():

            source = str(
                r["source"]
            )

            state = str(
                r["state"]
            )

            detectors = str(
                r["active_detectors"]
            )

            t = float(
                r["demo_time_sec"]
            )

            lines.append(
                f"{t:7.2f}s   "
                f"{source:<7} "
                f"{detectors:<15} "
                f"{state}"
            )

        ax.text(
            0.01,
            0.95,
            "\n".join(
                lines
            ),
            transform=ax.transAxes,
            va="top",
            family="monospace",
            fontsize=9,
        )

        # ----------------------------------------------------
        # Info panel
        # ----------------------------------------------------

        state = result[
            "state"
        ]

        ground_truth = (
            "FAULT"
            if result[
                "ground_truth"
            ]
            else "NORMAL"
        )

        active = result[
            "active_detectors"
        ]

        delay = ""

        fault_rows = logs[
            logs[
                "ground_truth"
            ].eq(1)
        ]

        if (
            result["P2"]
            and not fault_rows.empty
        ):

            fault_start = (
                fault_rows[
                    "demo_time_sec"
                ].iloc[0]
            )

            delay_sec = (
                current_time
                - fault_start
            )

            delay = (
                f"Detection delay: "
                f"{delay_sec:.2f}s"
            )

        self.text_status.set_text(
            (
                f"SYSTEM: {state}\n"
                f"Ground Truth: {ground_truth}"
            )
        )

        self.text_detectors.set_text(
            (
                "DETECTORS\n\n"
                f"1.0s Window : "
                f"{'ALARM' if result['alarm_1.0s'] else 'normal'}\n"
                f"0.5s Window : "
                f"{'ALARM' if result['alarm_0.5s'] else 'normal'}\n"
                f"Sample      : "
                f"{'ALARM' if result['sample_alarm'] else 'normal'}\n\n"
                f"OR_3        : "
                f"{'ALARM' if result['OR_3'] else 'normal'}\n"
                f"P2           : "
                f"{'CONFIRMED' if result['P2'] else 'waiting'}"
            )
        )

        self.text_info.set_text(
            (
                "CURRENT EVENT\n\n"
                f"Time       : "
                f"{result['TimeStamp']}\n"
                f"Demo time  : "
                f"{current_time:.2f}s\n"
                f"AI0        : "
                f"{result['AI0_Vibration']:.4f}\n"
                f"AI1        : "
                f"{result['AI1_Vibration']:.4f}\n"
                f"AI2        : "
                f"{result['AI2_Current']:.2f}\n\n"
                f"Score 0.5s : "
                f"{result['score_0.5s']:.2f}\n"
                f"Score 1.0s : "
                f"{result['score_1.0s']:.2f}\n"
                f"Score Sample: "
                f"{result['sample_score']:.2f}\n\n"
                f"{delay}"
            )
        )

        # ----------------------------------------------------
        # Figure title
        # ----------------------------------------------------

        self.fig.suptitle(
            (
                "KAMPact — Real-Time Hydraulic Pump Monitoring"
                f"  |  "
                f"{state}"
            ),
            fontsize=18,
            fontweight="bold",
        )

        return []

    # ========================================================
    # Finish
    # ========================================================

    def finish(
        self,
    ):

        log_path = (
            self.output_dir
            / "realtime_log.csv"
        )

        pd.DataFrame(
            self.logs
        ).to_csv(
            log_path,
            index=False,
            encoding="utf-8-sig",
        )

        print()
        print(
            "=" * 90
        )
        print(
            "REAL-TIME DEMO COMPLETE"
        )
        print(
            "=" * 90
        )

        if not self.logs:
            print(
                "로그가 없습니다."
            )
            return

        logs = pd.DataFrame(
            self.logs
        )

        fault_logs = logs[
            logs["ground_truth"].eq(1)
        ]

        alarms = logs[
            logs["P2"].eq(True)
        ]

        print(
            f"Total samples : "
            f"{len(logs):,}"
        )

        print(
            f"Final alarms  : "
            f"{len(alarms):,}"
        )

        if not fault_logs.empty:

            onset = (
                fault_logs[
                    "demo_time_sec"
                ].iloc[0]
            )

            detections = (
                fault_logs[
                    fault_logs["P2"].eq(True)
                ]
            )

            if not detections.empty:

                first_alarm = (
                    detections[
                        "demo_time_sec"
                    ].iloc[0]
                )

                print(
                    f"Fault onset   : "
                    f"{onset:.2f}s"
                )

                print(
                    f"First P2 alarm: "
                    f"{first_alarm:.2f}s"
                )

                print(
                    f"Demo delay    : "
                    f"{first_alarm - onset:.2f}s"
                )

            else:

                print(
                    "Fault detected: NO"
                )

        print()
        print(
            f"Saved log: {log_path}"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "KAMPact real-time monitoring demo"
        )
    )

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
        "--window-0.5",
        dest="window_05_path",
        default=(
            "result/"
            "modeling_dataset_0.5_0.1/"
            "model_windows.csv"
        ),
    )

    parser.add_argument(
        "--window-1.0",
        dest="window_10_path",
        default=(
            "result/"
            "modeling_dataset_1.0_0.1/"
            "model_windows.csv"
        ),
    )

    parser.add_argument(
        "--normal-seconds",
        type=float,
        default=DEFAULT_NORMAL_SECONDS,
    )

    parser.add_argument(
        "--fault-event",
        default=DEFAULT_FAULT_EVENT,
        help=(
            "예: fault_5, fault_19, all"
        ),
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help=(
            "1 = 실제 0.1초 간격, "
            "20 = 20배 빠른 발표용 재생"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/"
            "11_realtime_dashboard"
        ),
    )

    args = parser.parse_args()

    root = Path.cwd()

    # ========================================================
    # Paths
    # ========================================================

    normal_path = (
        root / args.normal_path
    )

    fault_path = (
        root / args.fault_path
    )

    window_05_path = (
        root / args.window_05_path
    )

    window_10_path = (
        root / args.window_10_path
    )

    # ========================================================
    # Load
    # ========================================================

    print()
    print("=" * 90)
    print(
        "KAMPact - Real-Time Monitoring Dashboard"
    )
    print("=" * 90)

    print()
    print(
        "[1] Loading raw data..."
    )

    normal = load_csv(
        normal_path
    )

    fault = load_csv(
        fault_path
    )

    print(
        f"Normal rows: {len(normal):,}"
    )

    print(
        f"Fault rows : {len(fault):,}"
    )

    # ========================================================
    # Window training data
    # ========================================================

    print()
    print(
        "[2] Fitting deployment-style detectors..."
    )

    windows_05 = safe_read_window(
        window_05_path
    )

    windows_10 = safe_read_window(
        window_10_path
    )

    detector_05 = WindowDetector(
        windows_05,
        0.5,
        QUANTILE,
    )

    detector_10 = WindowDetector(
        windows_10,
        1.0,
        QUANTILE,
    )

    sample_detector = SampleDetector(
        normal,
        QUANTILE,
    )

    print(
        f"0.5s threshold : "
        f"{detector_05.threshold:.4f}"
    )

    print(
        f"1.0s threshold : "
        f"{detector_10.threshold:.4f}"
    )

    print(
        f"Sample threshold: "
        f"{sample_detector.threshold:.4f}"
    )

    # ========================================================
    # Build stream
    # ========================================================

    print()
    print(
        "[3] Building unified Normal → Fault stream..."
    )

    stream = build_demo_stream(
        normal=normal,
        fault=fault,
        normal_seconds=args.normal_seconds,
        fault_event=args.fault_event,
    )

    print(
        f"Normal demo duration: "
        f"{stream.loc[stream['stream_source'].eq('NORMAL'), 'demo_time_sec'].max():.2f}s"
    )

    print(
        f"Fault event: "
        f"{args.fault_event}"
    )

    fault_part = stream[
        stream[
            "stream_source"
        ].eq("FAULT")
    ]

    if not fault_part.empty:

        print(
            f"Fault duration: "
            f"{fault_part['demo_time_sec'].iloc[-1] - fault_part['demo_time_sec'].iloc[0]:.2f}s"
        )

    print(
        f"Total stream samples: "
        f"{len(stream):,}"
    )

    # ========================================================
    # Dashboard
    # ========================================================

    print()
    print(
        "[4] Starting dashboard..."
    )

    dashboard = RealtimeDashboard(
        stream=stream,
        detector_05=detector_05,
        detector_10=detector_10,
        sample_detector=sample_detector,
        speed=args.speed,
        output_dir=(
            root / args.output_dir
        ),
    )

    dashboard.setup_figure()

    dashboard.animation = FuncAnimation(
        dashboard.fig,
        dashboard.update_plot,
        interval=1,
        cache_frame_data=False,
    )

    plt.show()

    # 창을 닫아도 로그 저장
    dashboard.finish()


# ============================================================
# Safe window reader
# ============================================================

def safe_read_window(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Window dataset을 찾을 수 없습니다: {path}"
        )

    df = pd.read_csv(
        path
    )

    missing = [
        c
        for c in [
            *FEATURES,
            "source",
            "label",
            "split",
        ]
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path}\n"
            f"필수 컬럼 누락: {missing}"
        )

    return df


if __name__ == "__main__":
    main()