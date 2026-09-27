import os
import numpy as np
import pandas as pd

# ============================================================
# 설정
# ============================================================

NORMAL_PATH = "data/press_data_normal_with_idle.csv"
FAULT_PATH = "data/outlier_data.csv"

OUTPUT_DIR = "result/time_structure_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 컬럼명
TIME_COL = "TimeStamp"
IDLE_COL = "Idle"

# 정상적인 sampling 간격
EXPECTED_INTERVAL = 0.1  # sec

# segment 분리 후보 기준
GAP_THRESHOLDS = [0.2, 0.3, 0.5, 1.0, 2.0, 5.0]

# ============================================================
# 기본 함수
# ============================================================

def load_data(path):
    print("=" * 80)
    print(f"파일 로드: {path}")
    print("=" * 80)

    df = pd.read_csv(path)

    print(f"행 수: {len(df):,}")
    print(f"컬럼: {list(df.columns)}")

    if TIME_COL not in df.columns:
        raise ValueError(
            f"'{TIME_COL}' 컬럼이 없습니다.\n"
            f"현재 컬럼: {list(df.columns)}"
        )

    # timestamp 변환
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

    invalid_time = df[TIME_COL].isna().sum()

    print(f"timestamp 변환 실패: {invalid_time:,}")

    if invalid_time > 0:
        df = df.dropna(subset=[TIME_COL]).copy()

    # 시간순 정렬
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    return df


def analyze_gap(df, name):
    """
    timestamp 사이의 gap을 계산하고 통계를 출력
    """

    print()
    print("=" * 80)
    print(f"[{name}] TIMESTAMP GAP 분석")
    print("=" * 80)

    df = df.copy()

    # 초 단위 간격
    df["time_diff_sec"] = (
        df[TIME_COL]
        .diff()
        .dt.total_seconds()
    )

    # 첫 행 제외
    gaps = df["time_diff_sec"].dropna()

    if len(gaps) == 0:
        print("분석 가능한 gap이 없습니다.")
        return df, None

    # 기본 통계
    stats = {
        "count": len(gaps),
        "mean": gaps.mean(),
        "median": gaps.median(),
        "std": gaps.std(),
        "min": gaps.min(),
        "max": gaps.max(),
        "q01": gaps.quantile(0.01),
        "q05": gaps.quantile(0.05),
        "q25": gaps.quantile(0.25),
        "q50": gaps.quantile(0.50),
        "q75": gaps.quantile(0.75),
        "q95": gaps.quantile(0.95),
        "q99": gaps.quantile(0.99),
    }

    print(f"전체 gap 수       : {stats['count']:,}")
    print(f"평균               : {stats['mean']:.4f} sec")
    print(f"중앙값             : {stats['median']:.4f} sec")
    print(f"표준편차           : {stats['std']:.4f} sec")
    print(f"최소               : {stats['min']:.4f} sec")
    print(f"25%                : {stats['q25']:.4f} sec")
    print(f"50%                : {stats['q50']:.4f} sec")
    print(f"75%                : {stats['q75']:.4f} sec")
    print(f"95%                : {stats['q95']:.4f} sec")
    print(f"99%                : {stats['q99']:.4f} sec")
    print(f"최대               : {stats['max']:.4f} sec")

    # 0.1초 기준 비율
    tolerance = 0.01

    normal_interval = gaps.between(
        EXPECTED_INTERVAL - tolerance,
        EXPECTED_INTERVAL + tolerance
    )

    print()
    print("-" * 80)
    print("정상 sampling 간격 확인")
    print("-" * 80)

    print(
        f"{EXPECTED_INTERVAL:.1f} sec ± {tolerance:.2f} sec "
        f": {normal_interval.sum():,}개 "
        f"({normal_interval.mean() * 100:.2f}%)"
    )

    # 주요 gap 범위
    ranges = [
        ("0.1초 이하", 0, 0.15),
        ("0.15~0.2초", 0.15, 0.2),
        ("0.2~0.5초", 0.2, 0.5),
        ("0.5~1초", 0.5, 1.0),
        ("1~2초", 1.0, 2.0),
        ("2~5초", 2.0, 5.0),
        ("5초 초과", 5.0, np.inf),
    ]

    print()
    print("-" * 80)
    print("Gap 구간별 분포")
    print("-" * 80)

    range_results = []

    for label, lower, upper in ranges:

        if np.isinf(upper):
            mask = gaps > lower
        else:
            mask = (gaps > lower) & (gaps <= upper)

        count = mask.sum()
        ratio = count / len(gaps) * 100

        print(
            f"{label:15s} : "
            f"{count:8,}개 ({ratio:7.3f}%)"
        )

        range_results.append({
            "range": label,
            "count": count,
            "ratio_percent": ratio
        })

    range_df = pd.DataFrame(range_results)

    # ========================================================
    # 후보 threshold별 segment 분석
    # ========================================================

    print()
    print("=" * 80)
    print("SEGMENT GAP THRESHOLD 비교")
    print("=" * 80)

    threshold_results = []

    for threshold in GAP_THRESHOLDS:

        # threshold보다 큰 gap이면 segment 분리
        split_mask = df["time_diff_sec"] > threshold

        split_count = split_mask.sum()

        # segment 번호
        segment_count = split_count + 1

        large_gap_count = split_mask.sum()
        large_gap_ratio = large_gap_count / len(gaps) * 100

        threshold_results.append({
            "threshold_sec": threshold,
            "large_gap_count": large_gap_count,
            "large_gap_ratio_percent": large_gap_ratio,
            "segment_count": segment_count
        })

        print(
            f"threshold > {threshold:>4.1f} sec | "
            f"큰 gap {large_gap_count:>8,}개 "
            f"({large_gap_ratio:7.3f}%) | "
            f"segment {segment_count:>8,}개"
        )

    threshold_df = pd.DataFrame(threshold_results)

    # ========================================================
    # 큰 gap 발생 지점 저장
    # ========================================================

    gap_events = df.loc[
        df["time_diff_sec"] > min(GAP_THRESHOLDS),
        [TIME_COL, "time_diff_sec"]
    ].copy()

    if len(gap_events) > 0:
        gap_events["gap_end_time"] = gap_events[TIME_COL]
        gap_events["gap_start_time"] = (
            gap_events[TIME_COL]
            - pd.to_timedelta(gap_events["time_diff_sec"], unit="s")
        )

        gap_events = gap_events[
            [
                "gap_start_time",
                "gap_end_time",
                "time_diff_sec"
            ]
        ].sort_values(
            "time_diff_sec",
            ascending=False
        )

    return df, {
        "stats": stats,
        "range_df": range_df,
        "threshold_df": threshold_df,
        "gap_events": gap_events
    }


# ============================================================
# Idle 분석
# ============================================================

def analyze_idle(df, name):

    print()
    print("=" * 80)
    print(f"[{name}] IDLE 분석")
    print("=" * 80)

    if IDLE_COL not in df.columns:

        print(f"'{IDLE_COL}' 컬럼이 없습니다.")
        print("→ Idle 분석을 건너뜁니다.")

        return None

    df = df.copy()

    # 숫자형 변환
    df[IDLE_COL] = pd.to_numeric(
        df[IDLE_COL],
        errors="coerce"
    )

    print()
    print("Idle 값 분포")
    print("-" * 80)

    value_counts = (
        df[IDLE_COL]
        .value_counts(dropna=False)
        .sort_index()
    )

    for value, count in value_counts.items():

        ratio = count / len(df) * 100

        print(
            f"Idle={value} : "
            f"{count:,} rows "
            f"({ratio:.2f}%)"
        )

    # --------------------------------------------------------
    # 시간 차이
    # --------------------------------------------------------

    df["time_diff_sec"] = (
        df[TIME_COL]
        .diff()
        .dt.total_seconds()
    )

    # 첫 행 제거
    valid = df["time_diff_sec"].notna()

    idle_duration = (
        df.loc[valid]
        .groupby(IDLE_COL)["time_diff_sec"]
        .sum()
    )

    print()
    print("-" * 80)
    print("Idle 상태별 추정 지속시간")
    print("-" * 80)

    for idle_value, duration in idle_duration.items():

        hours = duration / 3600
        minutes = duration / 60

        if idle_value == 0:
            label = "운전"
        elif idle_value == 1:
            label = "유휴"
        else:
            label = f"기타({idle_value})"

        print(
            f"{label:8s} : "
            f"{duration:,.2f} sec "
            f"/ {minutes:,.2f} min "
            f"/ {hours:,.2f} h"
        )

    # --------------------------------------------------------
    # Idle 상태 전환 횟수
    # --------------------------------------------------------

    transitions = (
        df[IDLE_COL]
        .ne(df[IDLE_COL].shift())
        .sum()
        - 1
    )

    print()
    print(f"Idle 상태 전환 횟수 : {transitions:,}회")

    # --------------------------------------------------------
    # Idle 연속구간 확인
    # --------------------------------------------------------

    df["idle_group"] = (
        df[IDLE_COL]
        .ne(df[IDLE_COL].shift())
        .cumsum()
    )

    idle_segments = (
        df.groupby(
            ["idle_group", IDLE_COL],
            dropna=False
        )
        .agg(
            start_time=(TIME_COL, "min"),
            end_time=(TIME_COL, "max"),
            rows=(TIME_COL, "size")
        )
        .reset_index()
    )

    idle_segments["duration_sec"] = (
        idle_segments["end_time"]
        - idle_segments["start_time"]
    ).dt.total_seconds()

    # 길이가 긴 순
    idle_segments = idle_segments.sort_values(
        "duration_sec",
        ascending=False
    )

    print()
    print("-" * 80)
    print("가장 긴 Idle 구간")
    print("-" * 80)

    print(
        idle_segments[
            [
                IDLE_COL,
                "start_time",
                "end_time",
                "rows",
                "duration_sec"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    return idle_segments


# ============================================================
# Segment 생성 기준 추천용
# ============================================================

def recommend_threshold(normal_gap_result):

    threshold_df = normal_gap_result["threshold_df"]

    print()
    print("=" * 80)
    print("SEGMENT 기준 판단 참고")
    print("=" * 80)

    print(
        """
판단 원칙

1. 정상 sampling 간격인 약 0.1초는 하나의 연속 시계열로 간주
2. 작은 일시적 gap까지 segment를 잘라버리면 window 생성에 불필요한 단절 발생
3. 너무 큰 threshold를 사용하면 실제 데이터 단절을 하나의 시계열로 연결할 위험
4. 따라서 여러 threshold에서 segment 수와 큰 gap 비율을 비교한 후 결정
"""
    )

    print(
        threshold_df.to_string(index=False)
    )

    print()
    print(
        "초기 분석에서는 1초를 후보 기준으로 두고, "
        "0.5초 / 1초 / 2초 결과를 비교하는 것을 권장합니다."
    )


# ============================================================
# 메인
# ============================================================

def main():

    # ========================================================
    # NORMAL
    # ========================================================

    normal_df = load_data(NORMAL_PATH)

    normal_df, normal_gap = analyze_gap(
        normal_df,
        "NORMAL"
    )

    normal_idle = analyze_idle(
        normal_df,
        "NORMAL"
    )

    # ========================================================
    # FAULT
    # ========================================================

    fault_df = load_data(FAULT_PATH)

    fault_df, fault_gap = analyze_gap(
        fault_df,
        "FAULT"
    )

    fault_idle = analyze_idle(
        fault_df,
        "FAULT"
    )

    # ========================================================
    # 저장
    # ========================================================

    print()
    print("=" * 80)
    print("분석 결과 저장")
    print("=" * 80)

    # Gap threshold 비교
    if normal_gap is not None:

        normal_gap["threshold_df"].to_csv(
            os.path.join(
                OUTPUT_DIR,
                "normal_gap_threshold_comparison.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

        normal_gap["range_df"].to_csv(
            os.path.join(
                OUTPUT_DIR,
                "normal_gap_distribution.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

        normal_gap["gap_events"].to_csv(
            os.path.join(
                OUTPUT_DIR,
                "normal_large_gap_events.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

    if fault_gap is not None:

        fault_gap["threshold_df"].to_csv(
            os.path.join(
                OUTPUT_DIR,
                "fault_gap_threshold_comparison.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

        fault_gap["range_df"].to_csv(
            os.path.join(
                OUTPUT_DIR,
                "fault_gap_distribution.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

        fault_gap["gap_events"].to_csv(
            os.path.join(
                OUTPUT_DIR,
                "fault_large_gap_events.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

    if normal_idle is not None:

        normal_idle.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "normal_idle_segments.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

    if fault_idle is not None:

        fault_idle.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "fault_idle_segments.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

    print()
    print(f"결과 저장 위치: {OUTPUT_DIR}")

    # ========================================================
    # 추천
    # ========================================================

    if normal_gap is not None:
        recommend_threshold(normal_gap)


if __name__ == "__main__":
    main()