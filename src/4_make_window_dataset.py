import os
import numpy as np
import pandas as pd

from tqdm import tqdm


# ============================================================
# 설정
# ============================================================

NORMAL_PATH = "data/press_data_normal_with_idle.csv"
FAULT_PATH = "data/outlier_data.csv"

OUTPUT_DIR = "result/modeling_dataset_0.8_0.1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 컬럼
TIME_COL = "TimeStamp"
IDLE_COL = "Idle"
STATE_COL = "Equipment_state"

SENSOR_COLS = [
    "AI0_Vibration",
    "AI1_Vibration",
    "AI2_Current"
]

# ------------------------------------------------------------
# 시계열 기준
# ------------------------------------------------------------

# 0.5초보다 큰 gap이 발생하면 새로운 segment
GAP_THRESHOLD_SEC = 0.5

# Window 설정
WINDOW_SEC = 0.8
STEP_SEC = 0.1  

# 최소 샘플 수
# 0.1초 sampling 기준 window에 들어가는 이상적 샘플 수의 80%
# (W=1.0 -> 8, W=0.9 -> 7, W=0.8 -> 6)
# W가 바뀌어도 "샘플 1개 누락"에 대한 여유가 동일하도록 W에 비례시킴
SAMPLING_SEC = 0.1
MIN_SAMPLE_RATIO = 0.8
MIN_SAMPLES = max(2, int(WINDOW_SEC / SAMPLING_SEC * MIN_SAMPLE_RATIO))

# 공통 평가 이벤트 기준 (비교하려는 W 중 최댓값)
# fault segment 길이가 이 값보다 짧은 이벤트는
# 어떤 W 설정에서도 안정적으로 평가할 수 없으므로 split에서 제외
COMMON_EVENT_MIN_DURATION = 1.0

# ------------------------------------------------------------
# Segment split
# ------------------------------------------------------------

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

RANDOM_SEED = 42


# ============================================================
# 데이터 로드
# ============================================================

def load_data(path, source_name):
    print()
    print("=" * 80)
    print(f"[{source_name}] 데이터 로드")
    print("=" * 80)

    df = pd.read_csv(path)

    if TIME_COL not in df.columns:
        raise ValueError(
            f"{TIME_COL} 컬럼이 없습니다.\n"
            f"현재 컬럼: {list(df.columns)}"
        )

    for col in SENSOR_COLS:
        if col not in df.columns:
            raise ValueError(
                f"{col} 컬럼이 없습니다."
            )

    if STATE_COL not in df.columns:
        raise ValueError(
            f"{STATE_COL} 컬럼이 없습니다."
        )

    # timestamp
    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL],
        errors="coerce"
    )

    invalid_time = df[TIME_COL].isna().sum()

    if invalid_time > 0:
        print(
            f"timestamp 변환 실패 {invalid_time:,}개 제거"
        )

        df = df.dropna(
            subset=[TIME_COL]
        )

    # 정렬
    df = (
        df.sort_values(TIME_COL)
        .reset_index(drop=True)
    )

    # 숫자 변환
    for col in SENSOR_COLS + [STATE_COL]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    if IDLE_COL in df.columns:
        df[IDLE_COL] = pd.to_numeric(
            df[IDLE_COL],
            errors="coerce"
        )

    # source
    df["source"] = source_name

    print(f"행 수 : {len(df):,}")
    print(
        f"시간 : {df[TIME_COL].min()} "
        f"~ {df[TIME_COL].max()}"
    )

    return df


# ============================================================
# Segment 생성
# ============================================================

def create_segments(df):
    df = df.copy()

    # timestamp gap
    df["gap_sec"] = (
        df[TIME_COL]
        .diff()
        .dt.total_seconds()
    )

    # 0.5초 초과면 segment 분리
    segment_break = (
        df["gap_sec"] > GAP_THRESHOLD_SEC
    )

    # 첫 행은 첫 segment
    segment_break.iloc[0] = True

    df["segment_id"] = (
        segment_break.cumsum() - 1
    )

    # source + segment를 합친 전역 ID
    df["group_id"] = (
        df["source"]
        + "_"
        + df["segment_id"].astype(str)
    )

    return df


# ============================================================
# Segment 통계
# ============================================================

def create_segment_manifest(df):

    group_cols = [
        "source",
        "segment_id",
        "group_id"
    ]

    segments = []

    for group_id, group in df.groupby(
        "group_id",
        sort=True
    ):

        start_time = group[TIME_COL].min()
        end_time = group[TIME_COL].max()

        duration_sec = (
            end_time - start_time
        ).total_seconds()

        rows = len(group)

        max_gap = group["gap_sec"].max()

        # Idle 정보
        if IDLE_COL in group.columns:

            idle_ratio = (
                group[IDLE_COL] == 1
            ).mean()

            idle_majority = int(
                idle_ratio >= 0.5
            )

        else:

            idle_ratio = 0.0
            idle_majority = 0

        # Label
        state_ratio = (
            group[STATE_COL] >= 1
        ).mean()

        label = int(
            state_ratio >= 0.5
        )

        # 실제 fault 상태 진입 시점
        # 이 데이터에서 Equipment_state >= 1이 처음 나타나는 timestamp
        fault_rows = group[group[STATE_COL] >= 1]

        if not fault_rows.empty:
            fault_onset_time = fault_rows[TIME_COL].min()
        else:
            fault_onset_time = pd.NaT

        segments.append({
            "source": group["source"].iloc[0],
            "segment_id": group["segment_id"].iloc[0],
            "group_id": group_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration_sec": duration_sec,
            "rows": rows,
            "max_gap_sec": max_gap,
            "idle_ratio": idle_ratio,
            "idle_majority": idle_majority,
            "label": label,
            "fault_onset_time": fault_onset_time,
        })

    segments = pd.DataFrame(
        segments
    )

    return segments


# ============================================================
# Segment Train / Val / Test 배정
# ============================================================

def assign_segment_splits(
    segment_manifest
):

    manifest = segment_manifest.copy()

    manifest["split"] = "UNASSIGNED"

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Idle segment 제외
    # --------------------------------------------------------

    idle_mask = (
        manifest["idle_majority"] == 1
    )

    manifest.loc[
        idle_mask,
        "split"
    ] = "EXCLUDED_IDLE"

    # --------------------------------------------------------
    # 너무 짧은 fault 이벤트 제외
    #
    # segment 길이 < COMMON_EVENT_MIN_DURATION 이면
    # W가 커질수록 window가 0개가 되어 이벤트가 조용히 사라짐.
    # 설정 간 비교가 가능하도록 split 단계에서 미리 제외.
    # --------------------------------------------------------

    short_fault_mask = (
        (manifest["source"] == "fault")
        & manifest["fault_onset_time"].notna()
        & (
            manifest["duration_sec"]
            < COMMON_EVENT_MIN_DURATION
        )
        & ~idle_mask
    )

    manifest.loc[
        short_fault_mask,
        "split"
    ] = "EXCLUDED_SHORT"

    eligible = manifest.loc[
        ~idle_mask & ~short_fault_mask
    ].copy()

    # --------------------------------------------------------
    # label별로 독립적으로 split
    # --------------------------------------------------------
    #
    # label=0 : Normal
    # label=1 : Fault
    #
    # 따라서 각 split에 정상/고장 segment가
    # 모두 들어가도록 구성
    # --------------------------------------------------------

    for label in sorted(
        eligible["label"].unique()
    ):

        idx = eligible.index[
            eligible["label"] == label
        ].to_numpy() .copy()

        rng.shuffle(idx)

        n = len(idx)

        n_train = int(
            round(n * TRAIN_RATIO)
        )

        n_val = int(
            round(n * VAL_RATIO)
        )

        # 최소 1개씩 확보
        if n >= 3:
            n_train = max(
                1,
                min(n_train, n - 2)
            )

            n_val = max(
                1,
                min(n_val, n - n_train - 1)
            )

        train_idx = idx[
            :n_train
        ]

        val_idx = idx[
            n_train:
            n_train + n_val
        ]

        test_idx = idx[
            n_train + n_val:
        ]

        manifest.loc[
            train_idx,
            "split"
        ] = "train"

        manifest.loc[
            val_idx,
            "split"
        ] = "val"

        manifest.loc[
            test_idx,
            "split"
        ] = "test"

    return manifest


# ============================================================
# Feature 함수
# ============================================================

def safe_rms(x):

    x = np.asarray(
        x,
        dtype=float
    )

    return np.sqrt(
        np.mean(x ** 2)
    )


def safe_slope(x, time_sec):

    x = np.asarray(
        x,
        dtype=float
    )

    time_sec = np.asarray(
        time_sec,
        dtype=float
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(time_sec)
    )

    x = x[valid]
    time_sec = time_sec[valid]

    if len(x) < 2:
        return np.nan

    # 시간이 모두 동일한 경우
    if np.ptp(time_sec) == 0:
        return np.nan

    slope = np.polyfit(
        time_sec,
        x,
        1
    )[0]

    return slope


def safe_corr(x, y):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
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

    return np.corrcoef(
        x,
        y
    )[0, 1]


# ============================================================
# Window Feature 생성
# ============================================================

def extract_features(
    window,
    window_start,
    window_end,
    fault_onset_time
):

    features = {}

    # --------------------------------------------------------
    # 기본 Window 정보
    # --------------------------------------------------------

    features["window_start"] = window_start
    features["window_end"] = window_end
    features["fault_onset_time"] = fault_onset_time
    features["sample_count"] = len(window)

    actual_duration = (
        window[TIME_COL].iloc[-1]
        - window[TIME_COL].iloc[0]
    ).total_seconds()

    features["actual_duration_sec"] = (
        actual_duration
    )

    # 내부 gap
    time_diff = (
        window[TIME_COL]
        .diff()
        .dt.total_seconds()
    )

    features["max_gap_sec"] = (
        time_diff.max()
        if len(time_diff) > 1
        else 0.0
    )

    # --------------------------------------------------------
    # Idle
    # --------------------------------------------------------

    if IDLE_COL in window.columns:

        idle_ratio = (
            window[IDLE_COL] == 1
        ).mean()

    else:

        idle_ratio = 0.0

    features["idle_ratio"] = idle_ratio

    features["idle_majority"] = int(
        idle_ratio >= 0.5
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    fault_ratio = (
        window[STATE_COL] >= 1
    ).mean()

    features["fault_ratio"] = (
        fault_ratio
    )

    features["label"] = int(
        fault_ratio >= 0.5
    )

    # --------------------------------------------------------
    # 시간축
    # --------------------------------------------------------

    time_sec = (
        window[TIME_COL]
        - window[TIME_COL].iloc[0]
    ).dt.total_seconds().to_numpy()

    # --------------------------------------------------------
    # Sensor Feature
    # --------------------------------------------------------

    for col in SENSOR_COLS:

        x = pd.to_numeric(
            window[col],
            errors="coerce"
        ).to_numpy(
            dtype=float
        )

        valid = np.isfinite(x)

        x_valid = x[valid]

        if len(x_valid) == 0:

            features[f"{col}_mean"] = np.nan
            features[f"{col}_std"] = np.nan
            features[f"{col}_rms"] = np.nan
            features[f"{col}_ptp"] = np.nan
            features[f"{col}_slope"] = np.nan

            continue

        features[f"{col}_mean"] = (
            np.mean(x_valid)
        )

        features[f"{col}_std"] = (
            np.std(
                x_valid,
                ddof=1
            )
            if len(x_valid) > 1
            else 0.0
        )

        features[f"{col}_rms"] = (
            safe_rms(x_valid)
        )

        features[f"{col}_ptp"] = (
            np.ptp(x_valid)
        )

        features[f"{col}_slope"] = (
            safe_slope(
                x,
                time_sec
            )
        )

    # --------------------------------------------------------
    # AI0 - AI1 상관관계
    # --------------------------------------------------------

    features["AI0_AI1_corr"] = (
        safe_corr(
            window["AI0_Vibration"].to_numpy(
                dtype=float
            ),
            window["AI1_Vibration"].to_numpy(
                dtype=float
            )
        )
    )

    return features


# ============================================================
# Segment 하나에서 Window 생성
# ============================================================

def generate_segment_windows(
    segment,
    split_name
):

    segment = (
        segment
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )

    if len(segment) < MIN_SAMPLES:
        return []

    # nan timestamp 제거
    segment = segment.dropna(
        subset=[TIME_COL]
    ).reset_index(drop=True)

    if len(segment) < MIN_SAMPLES:
        return []

    # --------------------------------------------------------
    # 실제 fault onset 계산
    # --------------------------------------------------------
    state_numeric = pd.to_numeric(
        segment[STATE_COL],
        errors="coerce"
    )

    fault_rows = segment[
        state_numeric >= 1
    ]

    if not fault_rows.empty:
        fault_onset_time = fault_rows[TIME_COL].min()
    else:
        fault_onset_time = pd.NaT

    # numpy timestamp
    timestamps_ns = (
        segment[TIME_COL]
        .values
        .astype("datetime64[ns]")
        .astype(np.int64)
    )

    window_ns = int(
        WINDOW_SEC * 1_000_000_000
    )

    step_ns = int(
        STEP_SEC * 1_000_000_000
    )

    results = []

    n = len(segment)

    start_idx = 0

    while start_idx < n:

        start_ns = timestamps_ns[
            start_idx
        ]

        end_ns = (
            start_ns
            + window_ns
        )

        # Window 종료시간이 segment 종료를
        # 넘어가면 종료
        if end_ns > timestamps_ns[-1]:
            break

        end_idx = np.searchsorted(
            timestamps_ns,
            end_ns,
            side="left"
        )

        if end_idx <= start_idx:
            start_idx += 1
            continue

        window = segment.iloc[
            start_idx:end_idx
        ].copy()

        # 최소 샘플
        if len(window) < MIN_SAMPLES:
            start_idx += 1
            continue

        window_start = (
            pd.to_datetime(
                start_ns
            )
        )

        window_end = (
            pd.to_datetime(
                end_ns
            )
        )

        features = extract_features(
            window,
            window_start,
            window_end,
            fault_onset_time
        )

        # Metadata
        features["source"] = (
            segment["source"].iloc[0]
        )

        features["segment_id"] = (
            segment["segment_id"].iloc[0]
        )

        features["group_id"] = (
            segment["group_id"].iloc[0]
        )

        features["split"] = (
            split_name
        )

        results.append(
            features
        )

        # 다음 window
        next_start_ns = (
            start_ns
            + step_ns
        )

        start_idx = np.searchsorted(
            timestamps_ns,
            next_start_ns,
            side="left"
        )

    return results


# ============================================================
# 전체 Window 생성
# ============================================================

def create_windows(
    df,
    segment_manifest
):

    print()
    print("=" * 80)
    print("Window 생성")
    print("=" * 80)

    # segment별 split mapping
    split_map = (
        segment_manifest
        .set_index("group_id")["split"]
        .to_dict()
    )

    all_windows = []

    group_iterator = df.groupby(
        "group_id",
        sort=True
    )

    for group_id, segment in tqdm(
        group_iterator,
        total=df["group_id"].nunique(),
        desc="Window 생성"
    ):

        split_name = split_map[
            group_id
        ]

        # Idle / 짧은 fault segment는 모델 데이터에서 제외
        if str(split_name).startswith("EXCLUDED"):
            continue

        windows = generate_segment_windows(
            segment,
            split_name
        )

        all_windows.extend(
            windows
        )

    windows_df = pd.DataFrame(
        all_windows
    )

    return windows_df


# ============================================================
# 결과 요약
# ============================================================

def print_segment_summary(
    manifest
):

    print()
    print("=" * 80)
    print("SEGMENT SPLIT SUMMARY")
    print("=" * 80)

    summary = (
        manifest
        .groupby(
            ["split", "label"],
            dropna=False
        )
        .agg(
            segments=("group_id", "count"),
            total_rows=("rows", "sum"),
            total_duration_sec=(
                "duration_sec",
                "sum"
            )
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("전체 split별 segment")

    print(
        manifest["split"]
        .value_counts()
        .to_string()
    )


def print_window_summary(
    windows_df
):

    print()
    print("=" * 80)
    print("WINDOW SUMMARY")
    print("=" * 80)

    if len(windows_df) == 0:
        print("생성된 window가 없습니다.")
        return

    summary = (
        windows_df
        .groupby(
            ["split", "label"]
        )
        .agg(
            windows=("label", "size"),
            mean_samples=(
                "sample_count",
                "mean"
            ),
            min_samples=(
                "sample_count",
                "min"
            ),
            max_samples=(
                "sample_count",
                "max"
            )
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("Split별 Window")

    print(
        windows_df["split"]
        .value_counts()
        .to_string()
    )


# ============================================================
# 데이터 품질 검증
# ============================================================

def validate_windows(
    windows_df
):

    print()
    print("=" * 80)
    print("WINDOW 데이터 품질 검증")
    print("=" * 80)

    feature_cols = [
        c
        for c in windows_df.columns
        if c not in [
            "window_start",
            "window_end",
            "fault_onset_time",
            "source",
            "segment_id",
            "group_id",
            "split",
            "label",
            "fault_ratio",
            "idle_ratio",
            "idle_majority"
        ]
    ]

    nan_counts = (
        windows_df[feature_cols]
        .isna()
        .sum()
    )

    nan_counts = nan_counts[
        nan_counts > 0
    ].sort_values(
        ascending=False
    )

    print()
    print("NaN Feature")

    if len(nan_counts) == 0:
        print("NaN 없음")
    else:
        print(
            nan_counts.to_string()
        )

    # --------------------------------------------------------
    # Fault onset 확인
    # --------------------------------------------------------

    fault_windows = windows_df[
        windows_df["label"] == 1
    ]

    print()
    print("Fault onset timestamp 확인")

    if len(fault_windows) == 0:
        print("Fault window 없음")
    else:
        onset_missing = fault_windows["fault_onset_time"].isna().sum()
        onset_events = fault_windows.loc[
            fault_windows["fault_onset_time"].notna(),
            "group_id"
        ].nunique()
        total_events = fault_windows["group_id"].nunique()

        print(f"Fault window : {len(fault_windows):,}")
        print(f"Fault event  : {total_events:,}")
        print(f"Onset event  : {onset_events:,}")
        print(f"Onset 누락   : {onset_missing:,}")

    # --------------------------------------------------------
    # 실제 segment leakage 확인
    # --------------------------------------------------------

    segment_split_count = (
        windows_df
        .groupby("group_id")["split"]
        .nunique()
    )

    leaked_segments = (
        segment_split_count[
            segment_split_count > 1
        ]
    )

    print()
    print(
        f"여러 split에 중복된 segment : "
        f"{len(leaked_segments):,}"
    )

    if len(leaked_segments) == 0:
        print("✓ Segment leakage 없음")
    else:
        print(
            leaked_segments.to_string()
        )

    # --------------------------------------------------------
    # Idle 확인
    # --------------------------------------------------------

    idle_windows = (
        windows_df["idle_majority"] == 1
    ).sum()

    print()
    print(
        f"Idle majority=1 window : "
        f"{idle_windows:,}"
    )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # 1. 데이터 로드
    # ========================================================

    normal = load_data(
        NORMAL_PATH,
        "normal"
    )

    fault = load_data(
        FAULT_PATH,
        "fault"
    )

    # ========================================================
    # 2. Segment 생성
    # ========================================================

    normal = create_segments(
        normal
    )

    fault = create_segments(
        fault
    )

    print()
    print("=" * 80)
    print("SEGMENT 생성 결과")
    print("=" * 80)

    print(
        f"Normal segment : "
        f"{normal['group_id'].nunique():,}"
    )

    print(
        f"Fault segment  : "
        f"{fault['group_id'].nunique():,}"
    )

    # ========================================================
    # 3. 데이터 결합
    # ========================================================

    df = pd.concat(
        [
            normal,
            fault
        ],
        ignore_index=True
    )

    # ========================================================
    # 4. Segment manifest 생성
    # ========================================================

    manifest = create_segment_manifest(
        df
    )

    # ========================================================
    # 5. Train / Val / Test 배정
    # ========================================================

    manifest = assign_segment_splits(
        manifest
    )

    # 저장
    manifest_path = os.path.join(
        OUTPUT_DIR,
        "segment_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig"
    )

    # 출력
    print_segment_summary(
        manifest
    )

    # ========================================================
    # 5-1. Event manifest (W와 무관한 정답 이벤트 목록)
    # ========================================================

    event_manifest = manifest[
        (manifest["source"] == "fault")
        & manifest["fault_onset_time"].notna()
        & (manifest["split"] != "EXCLUDED_IDLE")
    ].copy()

    event_manifest["evaluable_common"] = (
        event_manifest["duration_sec"]
        >= COMMON_EVENT_MIN_DURATION
    )

    event_manifest["evaluable_this_W"] = (
        event_manifest["duration_sec"]
        >= WINDOW_SEC
    )

    event_manifest_path = os.path.join(
        OUTPUT_DIR,
        "event_manifest.csv"
    )

    event_manifest.to_csv(
        event_manifest_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 80)
    print("EVENT MANIFEST")
    print("=" * 80)
    print(
        f"전체 이벤트            : {len(event_manifest)}"
    )
    print(
        f"공통 평가 가능 (>= {COMMON_EVENT_MIN_DURATION:.1f}s) : "
        f"{int(event_manifest['evaluable_common'].sum())}"
    )
    print(
        f"이 W에서 평가 가능     : "
        f"{int(event_manifest['evaluable_this_W'].sum())}"
    )

    excluded_short = event_manifest[
        ~event_manifest["evaluable_common"]
    ]

    if len(excluded_short) > 0:
        print()
        print("제외된 짧은 이벤트 (EXCLUDED_SHORT)")
        print(
            excluded_short[
                [
                    "group_id",
                    "duration_sec",
                    "fault_onset_time"
                ]
            ].to_string(index=False)
        )

    print()
    print("공통 이벤트의 split 분포")
    print(
        event_manifest[
            event_manifest["evaluable_common"]
        ]
        .groupby("split")
        .size()
        .to_string()
    )

    # ========================================================
    # 6. 각 Segment에 split 붙이기
    # ========================================================

    split_map = (
        manifest
        .set_index("group_id")["split"]
        .to_dict()
    )

    df["split"] = (
        df["group_id"]
        .map(split_map)
    )

    # ========================================================
    # 7. Split 내 Window 생성
    # ========================================================

    windows_df = create_windows(
        df,
        manifest
    )

    # ========================================================
    # 8. Window 전체 저장
    # ========================================================

    all_window_path = os.path.join(
        OUTPUT_DIR,
        "window_features_all.csv"
    )

    windows_df.to_csv(
        all_window_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 9. Idle window 제거
    # ========================================================

    model_windows = (
        windows_df[
            windows_df["idle_majority"] == 0
        ]
        .copy()
        .reset_index(drop=True)
    )

    # ========================================================
    # 10. Feature NaN 제거
    # ========================================================

    model_windows = (
        model_windows
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # 이후 모델에서 사용할 feature만
    metadata_cols = [
        "window_start",
        "window_end",
        "fault_onset_time",
        "source",
        "segment_id",
        "group_id",
        "split",
        "label",
        "fault_ratio",
        "idle_ratio",
        "idle_majority"
    ]

    feature_cols = [
        c
        for c in model_windows.columns
        if c not in metadata_cols
    ]

    before_rows = len(
        model_windows
    )

    model_windows = (
        model_windows
        .dropna(
            subset=feature_cols
        )
        .reset_index(drop=True)
    )

    removed_rows = (
        before_rows
        - len(model_windows)
    )

    print()
    print(
        f"Feature NaN/Inf 제거 : "
        f"{removed_rows:,} windows"
    )

    # ========================================================
    # 11. 최종 모델 데이터 저장
    # ========================================================

    model_path = os.path.join(
        OUTPUT_DIR,
        "model_windows.csv"
    )

    model_windows.to_csv(
        model_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 12. Window Summary
    # ========================================================

    print_window_summary(
        model_windows
    )

    # ========================================================
    # 13. 품질 검증
    # ========================================================

    validate_windows(
        model_windows
    )

    # ========================================================
    # 13-1. 공통 이벤트 소실 검증
    # ========================================================

    common_groups = set(
        event_manifest.loc[
            event_manifest["evaluable_common"],
            "group_id"
        ]
    )

    pos_groups = set(
        model_windows.loc[
            model_windows["label"] == 1,
            "group_id"
        ]
    )

    lost_common = sorted(
        common_groups - pos_groups
    )

    print()
    print("=" * 80)
    print("공통 이벤트 소실 검증")
    print("=" * 80)

    if len(lost_common) == 0:
        print(
            f"✓ 공통 이벤트 {len(common_groups)}개 모두 "
            f"label=1 window 보유"
        )
    else:
        print(
            f"[WARNING] label=1 window가 없는 공통 이벤트: "
            f"{lost_common}"
        )

    # ========================================================
    # 14. Split별 최종 데이터 수
    # ========================================================

    split_class_summary = (
        model_windows
        .groupby(
            ["split", "label"]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    split_class_summary = (
        split_class_summary
        .rename(
            columns={
                0: "normal",
                1: "fault"
            }
        )
        .reset_index()
    )

    split_class_summary_path = os.path.join(
        OUTPUT_DIR,
        "split_class_summary.csv"
    )

    split_class_summary.to_csv(
        split_class_summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 80)
    print("최종 Split별 Label 분포")
    print("=" * 80)

    print(
        split_class_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # 완료
    # ========================================================

    # ========================================================
    # Fault onset 최종 요약
    # ========================================================

    fault_model = model_windows[
        model_windows["label"] == 1
    ]

    print()
    print("=" * 80)
    print("FAULT ONSET SUMMARY")
    print("=" * 80)

    if len(fault_model) > 0:

        onset_summary = (
            fault_model[[
                "group_id",
                "fault_onset_time"
            ]]
            .drop_duplicates("group_id")
            .sort_values("fault_onset_time")
        )

        print(
            f"Fault event : {len(onset_summary):,}"
        )

        print(
            onset_summary.to_string(index=False)
        )

    else:

        print("Fault window가 없습니다.")

    # ========================================================
    # 완료
    # ========================================================

    print()
    print("=" * 80)
    print("전처리 완료")
    print("=" * 80)

    print(
        f"Window 설정      : {WINDOW_SEC:.1f}s window / {STEP_SEC:.1f}s step"
    )

    print(
        f"MIN_SAMPLES      : {MIN_SAMPLES}"
    )

    print(
        f"Event manifest   : {event_manifest_path}"
    )

    print(
        f"Segment manifest : {manifest_path}"
    )

    print(
        f"전체 Window      : {all_window_path}"
    )

    print(
        f"모델용 Window    : {model_path}"
    )

    print(
        f"Split Summary    : "
        f"{split_class_summary_path}"
    )


if __name__ == "__main__":
    main()