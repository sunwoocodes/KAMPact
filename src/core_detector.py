
"""
KAMPact - Real-time Streaming Detector Engine
----------------------------------------------
Final configuration:
    1.0s Window Mahalanobis
    0.5s Window Mahalanobis
    Sample-level raw+diff Mahalanobis
    -> OR_3
    -> P2 persistence

Design goals
------------
- Train/calibrate only on non-Idle normal data.
- Keep train and threshold-calibration data separate.
- Use the same 16 window features and signed-log transform as the
  offline Mahalanobis baseline.
- Use causal state only: current sample + past samples.
- Reset all temporal buffers at segment/time-gap boundaries.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


SENSOR_COLS = [
    "AI0_Vibration",
    "AI1_Vibration",
    "AI2_Current",
]

WINDOW_FEATURES = [
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


class StreamingDetector:
    """Causal online detector matching KAMPact's final OR_3_P2 design."""

    def __init__(
        self,
        win1_size: int = 10,
        win05_size: int = 5,
        sample_agg_k: int = 3,
        persistence_p: int = 2,
        gap_threshold_sec: float = 0.5,
        quantile_threshold: float = 0.9999,
        random_state: int = 42,
    ) -> None:
        if win05_size < 2 or win1_size < win05_size:
            raise ValueError("Window sizes must satisfy 2 <= win05_size <= win1_size.")
        if sample_agg_k < 1:
            raise ValueError("sample_agg_k must be >= 1.")
        if persistence_p < 1:
            raise ValueError("persistence_p must be >= 1.")
        if not 0.0 < quantile_threshold < 1.0:
            raise ValueError("quantile_threshold must be between 0 and 1.")

        self.win1_size = int(win1_size)
        self.win05_size = int(win05_size)
        self.sample_agg_k = int(sample_agg_k)
        self.p = int(persistence_p)
        self.gap_threshold_sec = float(gap_threshold_sec)
        self.quantile = float(quantile_threshold)
        self.random_state = int(random_state)

        self.models: dict[str, LedoitWolf | None] = {
            "sample": None,
            "win05": None,
            "win1": None,
        }
        self.thresholds: dict[str, float] = {
            "sample": np.inf,
            "win05": np.inf,
            "win1": np.inf,
        }

        self.feature_medians: dict[str, pd.Series | None] = {
            "win05": None,
            "win1": None,
        }

        self.last_time: pd.Timestamp | None = None
        self.raw_buffer: deque[np.ndarray] = deque(
            maxlen=max(self.win1_size, self.win05_size)
        )
        self.sample_score_buffer: deque[float] = deque(maxlen=self.sample_agg_k)
        self.or3_history: deque[bool] = deque(maxlen=self.p)

        self.fitted = False
        self.calibration_info: dict[str, object] = {}

        self.reset_state()

    # ------------------------------------------------------------------
    # Common transforms
    # ------------------------------------------------------------------

    @staticmethod
    def signed_log1p(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x))

    @staticmethod
    def _mahalanobis_score(
        x: np.ndarray,
        model: LedoitWolf,
    ) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)

        diff = x - model.location_
        return np.einsum(
            "ij,jk,ik->i",
            diff,
            model.precision_,
            diff,
        )

    # ------------------------------------------------------------------
    # Segment state
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Clear all causal history used by sample/window/persistence logic."""
        self.raw_buffer.clear()
        self.sample_score_buffer.clear()
        self.or3_history.clear()

        for _ in range(self.p):
            self.or3_history.append(False)

    def _check_gap(self, timestamp: pd.Timestamp) -> None:
        if self.last_time is not None:
            gap = (timestamp - self.last_time).total_seconds()
            if gap < 0:
                raise ValueError("Streaming timestamps must be non-decreasing.")
            if gap > self.gap_threshold_sec:
                self.reset_state()

        self.last_time = timestamp

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sample_feature(
        current: np.ndarray,
        previous: np.ndarray | None,
    ) -> np.ndarray:
        diff = current - previous if previous is not None else np.zeros_like(current)
        return np.hstack([current, diff])

    @staticmethod
    def _extract_window_features(buffer: np.ndarray) -> np.ndarray:
        """Exactly 16 window features used by KAMPact."""
        if buffer.ndim != 2 or buffer.shape[1] != 3:
            raise ValueError("Window buffer must have shape (N, 3).")

        n = len(buffer)
        if n < 2:
            raise ValueError("Window feature extraction requires at least 2 samples.")

        x_idx = np.arange(n, dtype=float)
        features: list[float] = []

        for col in range(3):
            data = buffer[:, col].astype(float)

            mean = float(np.mean(data))
            std = float(np.std(data, ddof=1))
            rms = float(np.sqrt(np.mean(data**2)))
            ptp = float(np.max(data) - np.min(data))
            slope = float(np.polyfit(x_idx, data, 1)[0])

            features.extend([mean, std, rms, ptp, slope])

        std_ai0 = float(np.std(buffer[:, 0]))
        std_ai1 = float(np.std(buffer[:, 1]))

        if std_ai0 > 0.0 and std_ai1 > 0.0:
            corr = float(np.corrcoef(buffer[:, 0], buffer[:, 1])[0, 1])
        else:
            corr = 0.0

        features.append(corr)

        out = np.asarray(features, dtype=float)

        if len(out) != 16 or not np.isfinite(out).all():
            raise ValueError("Window feature extraction produced invalid values.")

        return out

    def _build_sample_matrix(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        parts: list[np.ndarray] = []

        for _, group in df.groupby("group_id", sort=False):
            vals = group[SENSOR_COLS].to_numpy(dtype=float)

            diffs = np.vstack(
                [
                    np.zeros((1, 3), dtype=float),
                    np.diff(vals, axis=0),
                ]
            )

            parts.append(np.hstack([vals, diffs]))

        if not parts:
            raise ValueError("No sample features could be created.")

        x = np.vstack(parts)

        if not np.isfinite(x).all():
            raise ValueError("Sample features contain NaN/Inf.")

        return x

    def _build_window_matrix(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        features: list[np.ndarray] = []

        for _, group in df.groupby("group_id", sort=False):
            vals = group[SENSOR_COLS].to_numpy(dtype=float)

            for i in range(len(vals)):
                if i >= self.win05_size - 1:
                    features.append(
                        self._extract_window_features(
                            vals[i - self.win05_size + 1 : i + 1]
                        )
                    )

        if not features:
            raise ValueError("No 0.5s window features could be created.")

        out = np.vstack(features)

        if not np.isfinite(out).all():
            raise ValueError("Window features contain NaN/Inf.")

        return out

    def _build_window_matrix_for_size(
        self,
        df: pd.DataFrame,
        size: int,
    ) -> np.ndarray:
        features: list[np.ndarray] = []

        for _, group in df.groupby("group_id", sort=False):
            vals = group[SENSOR_COLS].to_numpy(dtype=float)

            for i in range(size - 1, len(vals)):
                features.append(
                    self._extract_window_features(
                        vals[i - size + 1 : i + 1]
                    )
                )

        if not features:
            raise ValueError(
                f"No window features could be created for size={size}."
            )

        out = np.vstack(features)

        if not np.isfinite(out).all():
            raise ValueError("Window features contain NaN/Inf.")

        return out

    # ------------------------------------------------------------------
    # Fit / calibration
    # ------------------------------------------------------------------

    def _prepare_normal(
        self,
        df: pd.DataFrame,
        exclude_group_ids: set[str] | None = None,
    ) -> pd.DataFrame:
        work = df.copy()

        if "TimeStamp" not in work.columns:
            raise ValueError("normal_df requires TimeStamp.")
        if not set(SENSOR_COLS).issubset(work.columns):
            missing = [c for c in SENSOR_COLS if c not in work.columns]
            raise ValueError(f"Missing sensor columns: {missing}")

        work["TimeStamp"] = pd.to_datetime(
            work["TimeStamp"],
            errors="coerce",
        )

        work = work.dropna(subset=["TimeStamp"]).sort_values("TimeStamp").copy()

        for c in SENSOR_COLS:
            work[c] = pd.to_numeric(work[c], errors="coerce")

        if work[SENSOR_COLS].isna().any().any():
            raise ValueError("Normal sensor data contain NaN.")

        if "Idle" in work.columns:
            idle = pd.to_numeric(
                work["Idle"],
                errors="coerce",
            ).fillna(0)
            work = work[idle.ne(1)].copy()

        if "kind" in work.columns:
            work = work[work["kind"].eq("normal")].copy()

        if "group_id" not in work.columns:
            gap = work["TimeStamp"].diff().dt.total_seconds()
            break_mask = gap > self.gap_threshold_sec
            break_mask.iloc[0] = True
            segment = break_mask.cumsum() - 1
            work["group_id"] = "normal_" + segment.astype(str)

        work["group_id"] = work["group_id"].astype(str)

        if exclude_group_ids:
            work = work[~work["group_id"].isin(exclude_group_ids)].copy()

        if work.empty:
            raise ValueError("No non-Idle normal rows remain for fitting.")

        return work.reset_index(drop=True)

    def _split_fit_calibration(
        self,
        normal_df: pd.DataFrame,
        calibration_fraction: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not 0.05 <= calibration_fraction < 0.5:
            raise ValueError("calibration_fraction must be in [0.05, 0.5).")

        groups = np.asarray(
            sorted(normal_df["group_id"].astype(str).unique()),
            dtype=str,
        )

        if len(groups) < 2:
            raise ValueError("At least two normal segments are required.")

        rng = np.random.default_rng(self.random_state)
        rng.shuffle(groups)

        n_cal = max(1, int(np.ceil(len(groups) * calibration_fraction)))
        n_cal = min(n_cal, len(groups) - 1)

        cal_groups = set(groups[:n_cal])

        train = normal_df[~normal_df["group_id"].isin(cal_groups)].copy()
        calibration = normal_df[normal_df["group_id"].isin(cal_groups)].copy()

        return train.reset_index(drop=True), calibration.reset_index(drop=True)

    @staticmethod
    def _fit_ledoitwolf(x: np.ndarray) -> LedoitWolf:
        if x.ndim != 2 or len(x) < 2:
            raise ValueError("Need at least two training samples.")

        model = LedoitWolf(assume_centered=False)
        model.fit(x)

        return model

    def fit(
        self,
        normal_df: pd.DataFrame,
        calibration_df: pd.DataFrame | None = None,
        calibration_fraction: float = 0.2,
        exclude_group_ids: set[str] | None = None,
    ) -> dict[str, object]:
        """
        Fit only on non-Idle normal data.

        If calibration_df is not supplied, segments are split deterministically
        into train and calibration groups. The calibration set is used only to
        set q=0.9999 thresholds.
        """
        base = self._prepare_normal(
            normal_df,
            exclude_group_ids=exclude_group_ids,
        )

        if calibration_df is None:
            train_df, cal_df = self._split_fit_calibration(
                base,
                calibration_fraction,
            )
        else:
            train_df = base
            cal_df = self._prepare_normal(
                calibration_df,
                exclude_group_ids=exclude_group_ids,
            )

            train_groups = set(train_df["group_id"].astype(str))
            cal_df = cal_df[
                ~cal_df["group_id"].isin(train_groups)
            ].copy()

            if cal_df.empty:
                raise ValueError(
                    "calibration_df must contain groups not used for training."
                )

        # --------------------------------------------------------------
        # 1) Sample-level model: raw + first difference
        # --------------------------------------------------------------
        x_train_sample = self._build_sample_matrix(train_df)
        x_cal_sample = self._build_sample_matrix(cal_df)

        x_train_sample_log = self.signed_log1p(x_train_sample)
        x_cal_sample_log = self.signed_log1p(x_cal_sample)

        sample_model = self._fit_ledoitwolf(x_train_sample_log)
        self.models["sample"] = sample_model

        cal_raw_sample_score = self._mahalanobis_score(
            x_cal_sample_log,
            sample_model,
        )

        # Causal mean-3 calibration, exactly like offline pipeline.
        cal_smooth_scores = []
        offset = 0

        for _, group in cal_df.groupby("group_id", sort=False):
            n = len(group)
            raw_scores = cal_raw_sample_score[offset : offset + n]
            offset += n

            smooth = (
                pd.Series(raw_scores)
                .rolling(
                    self.sample_agg_k,
                    min_periods=self.sample_agg_k,
                )
                .mean()
                .dropna()
                .to_numpy(dtype=float)
            )
            cal_smooth_scores.append(smooth)

        if not cal_smooth_scores:
            raise ValueError("No sample calibration scores were produced.")

        sample_cal_scores = np.concatenate(cal_smooth_scores)
        if not len(sample_cal_scores):
            raise ValueError("Sample calibration has no finite aggregated scores.")

        self.thresholds["sample"] = float(
            np.quantile(sample_cal_scores, self.quantile)
        )

        # --------------------------------------------------------------
        # 2) Window models: exact 16 features, train medians, signed log
        # --------------------------------------------------------------
        x_train_w05 = self._build_window_matrix_for_size(
            train_df,
            self.win05_size,
        )
        x_cal_w05 = self._build_window_matrix_for_size(
            cal_df,
            self.win05_size,
        )

        x_train_w1 = self._build_window_matrix_for_size(
            train_df,
            self.win1_size,
        )
        x_cal_w1 = self._build_window_matrix_for_size(
            cal_df,
            self.win1_size,
        )

        med_w05 = pd.DataFrame(
            x_train_w05,
            columns=WINDOW_FEATURES,
        ).median()
        med_w1 = pd.DataFrame(
            x_train_w1,
            columns=WINDOW_FEATURES,
        ).median()

        self.feature_medians["win05"] = med_w05
        self.feature_medians["win1"] = med_w1

        train_w05 = self.signed_log1p(
            pd.DataFrame(x_train_w05, columns=WINDOW_FEATURES)
            .fillna(med_w05)
            .to_numpy(dtype=float)
        )
        cal_w05 = self.signed_log1p(
            pd.DataFrame(x_cal_w05, columns=WINDOW_FEATURES)
            .fillna(med_w05)
            .to_numpy(dtype=float)
        )

        train_w1 = self.signed_log1p(
            pd.DataFrame(x_train_w1, columns=WINDOW_FEATURES)
            .fillna(med_w1)
            .to_numpy(dtype=float)
        )
        cal_w1 = self.signed_log1p(
            pd.DataFrame(x_cal_w1, columns=WINDOW_FEATURES)
            .fillna(med_w1)
            .to_numpy(dtype=float)
        )

        model_w05 = self._fit_ledoitwolf(train_w05)
        model_w1 = self._fit_ledoitwolf(train_w1)

        self.models["win05"] = model_w05
        self.models["win1"] = model_w1

        cal_w05_score = self._mahalanobis_score(cal_w05, model_w05)
        cal_w1_score = self._mahalanobis_score(cal_w1, model_w1)

        self.thresholds["win05"] = float(
            np.quantile(cal_w05_score, self.quantile)
        )
        self.thresholds["win1"] = float(
            np.quantile(cal_w1_score, self.quantile)
        )

        self.fitted = True
        self.reset_stream()

        self.calibration_info = {
            "train_rows": int(len(train_df)),
            "calibration_rows": int(len(cal_df)),
            "train_segments": int(train_df["group_id"].nunique()),
            "calibration_segments": int(cal_df["group_id"].nunique()),
            "quantile": float(self.quantile),
            "thresholds": {
                k: float(v) for k, v in self.thresholds.items()
            },
        }

        return self.calibration_info

    # ------------------------------------------------------------------
    # Streaming reset
    # ------------------------------------------------------------------

    def reset_stream(self) -> None:
        self.last_time = None
        self.reset_state()

    # ------------------------------------------------------------------
    # Causal inference
    # ------------------------------------------------------------------

    def _score_window(
        self,
        buffer_arr: np.ndarray,
        size: int,
        key: str,
    ) -> float:
        feature = self._extract_window_features(
            buffer_arr[-size:],
        )

        median = self.feature_medians[key]
        if median is None:
            raise RuntimeError("Detector has not been fitted.")

        x = pd.DataFrame(
            feature.reshape(1, -1),
            columns=WINDOW_FEATURES,
        ).fillna(median).to_numpy(dtype=float)

        x = self.signed_log1p(x)

        model = self.models[key]
        assert model is not None

        return float(self._mahalanobis_score(x, model)[0])

    def step(
        self,
        timestamp: pd.Timestamp,
        ai0: float,
        ai1: float,
        ai2: float,
    ) -> dict[str, object]:
        """Process exactly one raw sample using only current/past information."""
        if not self.fitted:
            raise RuntimeError("Call fit() before step().")

        timestamp = pd.Timestamp(timestamp)
        self._check_gap(timestamp)

        current = np.asarray(
            [ai0, ai1, ai2],
            dtype=float,
        )

        if not np.isfinite(current).all():
            raise ValueError("Streaming sensor values must be finite.")

        previous = self.raw_buffer[-1] if self.raw_buffer else None

        self.raw_buffer.append(current)
        buffer_arr = np.asarray(self.raw_buffer, dtype=float)

        # --------------------------------------------------------------
        # Sample-level causal detector
        # --------------------------------------------------------------
        sample_feature = self._extract_sample_feature(
            current,
            previous,
        )
        sample_feature_log = self.signed_log1p(
            sample_feature.reshape(1, -1),
        )

        sample_model = self.models["sample"]
        assert sample_model is not None

        sample_raw_score = float(
            self._mahalanobis_score(
                sample_feature_log,
                sample_model,
            )[0]
        )

        self.sample_score_buffer.append(sample_raw_score)

        if len(self.sample_score_buffer) == self.sample_agg_k:
            sample_score = float(
                np.mean(self.sample_score_buffer)
            )
        else:
            sample_score = 0.0

        sample_alarm = (
            len(self.sample_score_buffer) == self.sample_agg_k
            and sample_score >= self.thresholds["sample"]
        )

        # --------------------------------------------------------------
        # Window-level causal detector
        # --------------------------------------------------------------
        win05_score = 0.0
        win1_score = 0.0
        win05_alarm = False
        win1_alarm = False

        if len(buffer_arr) >= self.win05_size:
            win05_score = self._score_window(
                buffer_arr,
                self.win05_size,
                "win05",
            )
            win05_alarm = (
                win05_score >= self.thresholds["win05"]
            )

        if len(buffer_arr) >= self.win1_size:
            win1_score = self._score_window(
                buffer_arr,
                self.win1_size,
                "win1",
            )
            win1_alarm = (
                win1_score >= self.thresholds["win1"]
            )

        # --------------------------------------------------------------
        # OR_3 + P2
        # --------------------------------------------------------------
        or3_base = bool(
            sample_alarm
            or win05_alarm
            or win1_alarm
        )

        self.or3_history.append(or3_base)

        final_alarm = (
            len(self.or3_history) == self.p
            and all(self.or3_history)
        )

        active = []
        if sample_alarm:
            active.append("sample")
        if win05_alarm:
            active.append("0.5s")
        if win1_alarm:
            active.append("1.0s")

        return {
            "timestamp": timestamp,
            "sample_score": sample_score,
            "win05_score": win05_score,
            "win1_score": win1_score,
            "sample_alarm": bool(sample_alarm),
            "win05_alarm": bool(win05_alarm),
            "win1_alarm": bool(win1_alarm),
            "or3_base": or3_base,
            "final_alarm": bool(final_alarm),
            "active_detectors": active,
            "persistence_count": int(sum(self.or3_history)),
        }
