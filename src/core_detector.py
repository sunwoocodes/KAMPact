"""
KAMPact - Real-time Streaming Detector Engine
------------------------------------------------
이 모듈은 대시보드(Online)와 평가 스크립트(Offline) 간의 논리적 일치를 보장합니다.
- O(1) 고정 길이 큐(Deque) 기반 상태 관리 (무한 버퍼 증가 문제 해결)
- Segment 경계(Gap) 탐지 시 과거 이력 즉시 초기화 (Data Leakage 방지)
- Script 7과 완벽히 동일한 OR_3_P2 앙상블 및 Causal Feature 추출
"""

from collections import deque
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

class StreamingDetector:
    def __init__(
        self, 
        win1_size: int = 10, 
        win05_size: int = 5, 
        sample_agg_k: int = 3, 
        persistence_p: int = 2,
        gap_threshold_sec: float = 0.5,
        quantile_threshold: float = 0.9999
    ):
        """
        초기 파라미터 설정 (Script 7 최적 파라미터 기본값 적용)
        """
        self.win1_size = win1_size
        self.win05_size = win05_size
        self.sample_agg_k = sample_agg_k
        self.p = persistence_p
        self.gap_threshold_sec = gap_threshold_sec
        self.quantile = quantile_threshold
        
        # Models & Thresholds
        self.models = {'sample': None, 'win05': None, 'win1': None}
        self.thresholds = {'sample': np.inf, 'win05': np.inf, 'win1': np.inf}
        
        # State Buffers
        self.last_time = None
        self.raw_buffer = deque(maxlen=max(win1_size, win05_size))
        self.sample_score_buffer = deque(maxlen=sample_agg_k)
        self.or3_history = deque(maxlen=persistence_p)
        
        self.reset_state()

    def reset_state(self):
        """세그먼트 경계(Time Gap)를 넘을 때 과거 상태 잔재 초기화"""
        self.raw_buffer.clear()
        self.sample_score_buffer.clear()
        self.or3_history.clear()
        for _ in range(self.p):
            self.or3_history.append(False)

    def _signed_log1p(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x))

    def _extract_sample_feature(self, current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        """단일 샘플 피처 추출 (raw_diff)"""
        diff = current - previous if previous is not None else np.zeros_like(current)
        feats = np.hstack([current, diff])
        return self._signed_log1p(feats)

    def _extract_window_features(self, buffer: np.ndarray) -> np.ndarray:
        """16개 Window 피처 추출 (Numpy 벡터화로 O(1) 연산 최적화)"""
        # buffer shape: (window_size, 3) -> AI0, AI1, AI2
        features = []
        n = len(buffer)
        x_idx = np.arange(n)
        
        for col in range(3):
            data = buffer[:, col]
            mean = np.mean(data)
            std = np.std(data, ddof=1) if n > 1 else 0.0
            rms = np.sqrt(np.mean(data**2))
            ptp = np.max(data) - np.min(data)
            # Slope (Linear Regression)
            if n > 1:
                slope = np.polyfit(x_idx, data, 1)[0]
            else:
                slope = 0.0
            features.extend([mean, std, rms, ptp, slope])
            
        # AI0, AI1 Correlation
        if n > 1:
            std_ai0, std_ai1 = np.std(buffer[:,0]), np.std(buffer[:,1])
            if std_ai0 > 0 and std_ai1 > 0:
                corr = np.corrcoef(buffer[:,0], buffer[:,1])[0, 1]
            else:
                corr = 0.0
        else:
            corr = 0.0
            
        features.append(corr)
        return np.array(features)

    def fit(self, normal_df: pd.DataFrame):
        """
        정상(Normal) 데이터베이스만 사용하여 Threshold와 Covariance Matrix를 계산합니다.
        대시보드 시작 시 단 1회만 호출됩니다.
        """
        sensor_cols = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
        time_col = "TimeStamp"
        
        # 1. Sample Model Fitting
        sample_feats = []
        for gid, group in normal_df.groupby("group_id"):
            vals = group[sensor_cols].to_numpy()
            diffs = np.vstack([np.zeros((1, 3)), np.diff(vals, axis=0)])
            feats = self._signed_log1p(np.hstack([vals, diffs]))
            sample_feats.append(feats)
            
        x_sample = np.vstack(sample_feats)
        model_sample = LedoitWolf(assume_centered=False).fit(x_sample)
        self.models['sample'] = model_sample
        
        # Calculate causal smoothed scores for thresholding
        raw_scores = self._score(x_sample, model_sample)
        smooth_scores = pd.Series(raw_scores).rolling(self.sample_agg_k, min_periods=self.sample_agg_k).mean().fillna(0).to_numpy()
        self.thresholds['sample'] = np.quantile(smooth_scores, self.quantile)

        # 2. Window Model Fitting (0.5s & 1.0s)
        win05_feats, win1_feats = [], []
        for gid, group in normal_df.groupby("group_id"):
            vals = group[sensor_cols].to_numpy()
            for i in range(len(vals)):
                if i >= self.win05_size - 1:
                    win05_feats.append(self._extract_window_features(vals[i - self.win05_size + 1 : i + 1]))
                if i >= self.win1_size - 1:
                    win1_feats.append(self._extract_window_features(vals[i - self.win1_size + 1 : i + 1]))
                    
        x_w05 = np.vstack(win05_feats)
        model_w05 = LedoitWolf(assume_centered=False).fit(x_w05)
        self.models['win05'] = model_w05
        self.thresholds['win05'] = np.quantile(self._score(x_w05, model_w05), self.quantile)

        x_w1 = np.vstack(win1_feats)
        model_w1 = LedoitWolf(assume_centered=False).fit(x_w1)
        self.models['win1'] = model_w1
        self.thresholds['win1'] = np.quantile(self._score(x_w1, model_w1), self.quantile)
        
        print(f"✅ 모델 훈련 완료 | 임계값 세팅 -> Sample: {self.thresholds['sample']:.2f}, Win0.5: {self.thresholds['win05']:.2f}, Win1.0: {self.thresholds['win1']:.2f}")

    def _score(self, x: np.ndarray, model: LedoitWolf) -> np.ndarray:
        """Mahalanobis 거리 계산"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        d = x - model.location_
        return np.einsum("ij,jk,ik->i", d, model.precision_, d)

    def step(self, timestamp: pd.Timestamp, ai0: float, ai1: float, ai2: float) -> dict:
        """
        매 스트리밍 샘플마다 호출되는 추론 함수. 
        대시보드 애니메이션 루프 내에서 O(1) 속도로 결과를 반환합니다.
        """
        # 1. Segment Gap Checking (Data Leakage 방지)
        if self.last_time is not None:
            gap = (timestamp - self.last_time).total_seconds()
            if gap > self.gap_threshold_sec:
                self.reset_state()
        self.last_time = timestamp

        current_vals = np.array([ai0, ai1, ai2])
        prev_vals = self.raw_buffer[-1] if len(self.raw_buffer) > 0 else None
        
        self.raw_buffer.append(current_vals)
        buffer_arr = np.array(self.raw_buffer)

        # 2. Sample-level Causal Detection
        sample_f = self._extract_sample_feature(current_vals, prev_vals)
        sample_raw_score = self._score(sample_f, self.models['sample'])[0]
        self.sample_score_buffer.append(sample_raw_score)
        
        sample_smooth_score = np.mean(self.sample_score_buffer) if len(self.sample_score_buffer) == self.sample_agg_k else 0.0
        sample_alarm = sample_smooth_score >= self.thresholds['sample']

        # 3. Window-level Causal Detection (정확한 10샘플 / 5샘플만 사용)
        win05_score, win1_score = 0.0, 0.0
        win05_alarm, win1_alarm = False, False

        if len(self.raw_buffer) >= self.win05_size:
            f05 = self._extract_window_features(buffer_arr[-self.win05_size:])
            win05_score = self._score(f05, self.models['win05'])[0]
            win05_alarm = win05_score >= self.thresholds['win05']

        if len(self.raw_buffer) == self.win1_size:
            f1 = self._extract_window_features(buffer_arr[-self.win1_size:])
            win1_score = self._score(f1, self.models['win1'])[0]
            win1_alarm = win1_score >= self.thresholds['win1']

        # 4. OR_3 앙상블 및 P2 (Persistence) 로직 적용
        or3_base = sample_alarm or win05_alarm or win1_alarm
        self.or3_history.append(or3_base)
        
        # or3_history에 P개(2개)의 True가 연속으로 쌓여야 최종 알람 승인
        final_or3_p2_alarm = sum(self.or3_history) == self.p

        # 대시보드 플로팅용 결과 딕셔너리 반환
        return {
            "timestamp": timestamp,
            "sample_score": sample_smooth_score,
            "win05_score": win05_score,
            "win1_score": win1_score,
            "sample_alarm": sample_alarm,
            "win05_alarm": win05_alarm,
            "win1_alarm": win1_alarm,
            "or3_base": or3_base,
            "final_alarm": final_or3_p2_alarm
        }