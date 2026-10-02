"""
KAMPact - Variable Effect & Interaction Analysis
================================================

4단계:
    주요 영향변수 + 변수 간 상호작용 분석

비교
----
1) Normal vs Fault
   - 0.5s window feature
   - 16개 feature

2) Normal vs FP
   - 1.0s window feature
   - 8_fp_fn_analysis 결과의 FP context 사용

3) Stable Detected vs Any-Miss Fault
   - 0.5s fault window
   - event별 feature median을 먼저 계산
   - event 단위 비교
   - Stable Detected:
         missed_count == 0
   - Any-Miss:
         missed_count > 0

Sensor Interaction
------------------
Raw sensor correlation:

    Normal Operation
    Fault
    FP Context
    Any-Miss Fault

Sensors:
    AI0_Vibration
    AI1_Vibration
    AI2_Current

Output
------
outputs/10_variable_effect_analysis/

    variable_effect_summary.csv
    event_feature_medians.csv
    sensor_correlation_matrix.csv
    feature_coverage_report.csv

    feature_effect_plot.png
    sensor_correlation_heatmap.png
    fn_duration_detection.png

    analysis_report.txt

실행
----
python src/10_variable_effect_analysis.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 기본 설정
# ============================================================

TIME_COL = "TimeStamp"

SENSORS = [
    "AI0_Vibration",
    "AI1_Vibration",
    "AI2_Current",
]

DEFAULT_FEATURES = [
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


# ============================================================
# Utility
# ============================================================

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {path}"
        )

    return pd.read_csv(path)


def numeric_series(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:

    if column not in df.columns:
        return pd.Series(
            dtype=float
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna()


# ============================================================
# Feature 자동 추론
# ============================================================

def infer_features(
    df: pd.DataFrame,
) -> list[str]:

    available = [
        c
        for c in DEFAULT_FEATURES
        if c in df.columns
    ]

    if len(available) == 16:
        return available

    # fallback
    numeric = []

    for c in df.columns:

        if pd.api.types.is_numeric_dtype(
            df[c]
        ):
            cl = c.lower()

            if (
                cl.startswith("ai0")
                or cl.startswith("ai1")
                or cl.startswith("ai2")
                or "corr" in cl
            ):
                numeric.append(c)

    if len(numeric) >= 16:
        return numeric[:16]

    raise ValueError(
        "16개 feature를 찾지 못했습니다.\n"
        f"현재 feature 후보: {numeric}"
    )


# ============================================================
# Robust effect
# ============================================================

def robust_effect(
    baseline: pd.Series,
    target: pd.Series,
) -> dict:

    baseline = pd.to_numeric(
        baseline,
        errors="coerce",
    ).dropna()

    target = pd.to_numeric(
        target,
        errors="coerce",
    ).dropna()

    if baseline.empty:

        return {
            "baseline_count": 0,
            "target_count": int(
                len(target)
            ),
            "baseline_median": np.nan,
            "target_median": np.nan,
            "median_delta": np.nan,
            "median_delta_pct": np.nan,
            "robust_effect": np.nan,
            "direction": "",
        }

    b_med = float(
        baseline.median()
    )

    b_q25 = float(
        baseline.quantile(0.25)
    )

    b_q75 = float(
        baseline.quantile(0.75)
    )

    b_iqr = (
        b_q75 - b_q25
    )

    if target.empty:

        t_med = np.nan

    else:

        t_med = float(
            target.median()
        )

    delta = (
        t_med - b_med
        if np.isfinite(t_med)
        else np.nan
    )

    # 정상 중앙값이 0 근처일 때 퍼센트 폭발을 방지하기 위해
    # 퍼센트는 참고용으로만 사용한다.
    pct = (
        100.0
        * delta
        / abs(b_med)
        if (
            np.isfinite(delta)
            and abs(b_med) > 1e-12
        )
        else np.nan
    )

    robust_sd = (
        b_iqr / 1.349
        if b_iqr > 0
        else np.nan
    )

    effect = (
        delta / robust_sd
        if (
            np.isfinite(delta)
            and np.isfinite(robust_sd)
            and robust_sd > 0
        )
        else np.nan
    )

    if not np.isfinite(delta):

        direction = ""

    elif abs(delta) < 1e-12:

        direction = "≈"

    elif delta > 0:

        direction = "↑"

    else:

        direction = "↓"

    return {
        "baseline_count": int(
            len(baseline)
        ),

        "target_count": int(
            len(target)
        ),

        "baseline_median": b_med,

        "target_median": t_med,

        "median_delta": delta,

        "median_delta_pct": pct,

        "robust_effect": effect,

        "direction": direction,
    }


# ============================================================
# Feature Effect Table
# ============================================================

def build_variable_effect_table(
    normal_fault_df: pd.DataFrame,
    normal_fp_df: pd.DataFrame,
    detected_fault_df: pd.DataFrame,
    missed_fault_df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    rows = []

    for feature in features:

        nf = robust_effect(
            normal_fault_df[feature],
            normal_fault_df.loc[
                normal_fault_df["label"].eq(1),
                feature,
            ],
        )

        nfp = robust_effect(
            normal_fp_df[feature],
            normal_fp_df.loc[
                normal_fp_df["is_fp"].eq(True),
                feature,
            ],
        )

        dm = robust_effect(
            detected_fault_df[feature],
            missed_fault_df[feature],
        )

        rows.append({
            "feature": feature,

            # Normal vs Fault
            "normal_fault_normal_median": (
                nf["baseline_median"]
            ),
            "normal_fault_fault_median": (
                nf["target_median"]
            ),
            "normal_fault_delta": (
                nf["median_delta"]
            ),
            "normal_fault_effect": (
                nf["robust_effect"]
            ),
            "normal_fault_direction": (
                nf["direction"]
            ),

            # Normal vs FP
            "normal_fp_normal_median": (
                nfp["baseline_median"]
            ),
            "normal_fp_fp_median": (
                nfp["target_median"]
            ),
            "normal_fp_delta": (
                nfp["median_delta"]
            ),
            "normal_fp_effect": (
                nfp["robust_effect"]
            ),
            "normal_fp_direction": (
                nfp["direction"]
            ),

            # Detected vs Missed
            "detected_missed_detected_median": (
                dm["baseline_median"]
            ),
            "detected_missed_missed_median": (
                dm["target_median"]
            ),
            "detected_missed_delta": (
                dm["median_delta"]
            ),
            "detected_missed_effect": (
                dm["robust_effect"]
            ),
            "detected_missed_direction": (
                dm["direction"]
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# Event-level feature median
# ============================================================

def build_event_feature_medians(
    fault_windows: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    rows = []

    for group_id, group in (
        fault_windows
        .groupby("group_id")
    ):

        row = {
            "group_id": group_id,
            "n_windows": int(
                len(group)
            ),
        }

        for feature in features:

            values = numeric_series(
                group,
                feature,
            )

            row[feature] = (
                float(values.median())
                if len(values)
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Raw sensor correlation
# ============================================================

def correlation_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:

    available = [
        c
        for c in SENSORS
        if c in df.columns
    ]

    if len(available) < 2:
        raise ValueError(
            "Sensor correlation을 계산할 sensor가 부족합니다."
        )

    return (
        df[available]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .corr()
    )


# ============================================================
# FP context raw rows
# ============================================================

def build_fp_raw_mask(
    raw: pd.DataFrame,
    fp_windows: pd.DataFrame,
) -> np.ndarray:

    mask = np.zeros(
        len(raw),
        dtype=bool,
    )

    if "group_id" not in raw.columns:
        raise ValueError(
            "raw 데이터에 group_id가 없습니다. "
            "add_raw_metadata()를 먼저 적용하세요."
        )

    if fp_windows.empty:
        return mask

    raw_time = pd.to_datetime(
        raw[TIME_COL],
        errors="coerce",
    ).to_numpy(
        dtype="datetime64[ns]"
    )

    raw_group = (
        raw["group_id"]
        .astype(str)
        .to_numpy()
    )

    work = fp_windows.copy()

    work["window_start"] = (
        pd.to_datetime(
            work["window_start"],
            errors="coerce",
        )
    )

    work["window_end"] = (
        pd.to_datetime(
            work["window_end"],
            errors="coerce",
        )
    )

    for _, row in work.iterrows():

        gid = str(
            row["group_id"]
        )

        start = np.datetime64(
            row["window_start"]
        )

        end = np.datetime64(
            row["window_end"]
        )

        duration = (
            end - start
        )

        # 6번과 동일한 causal mapping
        shifted_start = (
            start + duration
        )

        shifted_end = (
            end + duration
        )

        local_mask = (
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

        mask |= local_mask

    return mask


# ============================================================
# Feature effect plot
# ============================================================

def plot_feature_effects(
    summary: pd.DataFrame,
    output_path: Path,
):

    effect_cols = {
        "Normal→Fault":
            "normal_fault_effect",

        "Normal→FP":
            "normal_fp_effect",

        "Detected→Any-Miss":
            "detected_missed_effect",
    }

    # 전체 비교에서 가장 큰 effect를 가진 feature 선정
    ranking = (
        summary[
            list(
                effect_cols.values()
            )
        ]
        .abs()
        .max(axis=1)
    )

    top_indices = (
        ranking
        .sort_values(
            ascending=False
        )
        .head(10)
        .index
    )

    plot_df = (
        summary.loc[
            top_indices
        ]
        .copy()
    )

    plot_df = plot_df.sort_values(
        "feature"
    )

    x = np.arange(
        len(plot_df)
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    for i, (
        label,
        column,
    ) in enumerate(
        effect_cols.items()
    ):

        values = pd.to_numeric(
            plot_df[column],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        ax.bar(
            x
            + (
                i - 1
            )
            * width,
            values,
            width,
            label=label,
        )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        plot_df["feature"],
        rotation=55,
        ha="right",
    )

    ax.set_ylabel(
        "Robust effect"
    )

    ax.set_title(
        "Top Feature Effects"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Sensor correlation heatmap
# ============================================================

def plot_correlation_heatmap(
    matrices: dict[str, pd.DataFrame],
    output_path: Path,
):

    labels = list(
        matrices.keys()
    )

    n = len(labels)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(
            4.2 * n,
            4.2,
        ),
    )

    if n == 1:
        axes = [axes]

    last_image = None

    for ax, label in zip(
        axes,
        labels,
    ):

        matrix = matrices[
            label
        ]

        last_image = ax.imshow(
            matrix.to_numpy(
                dtype=float
            ),
            vmin=-1,
            vmax=1,
        )

        ax.set_xticks(
            range(len(matrix))
        )

        ax.set_xticklabels(
            matrix.columns,
            rotation=45,
            ha="right",
        )

        ax.set_yticks(
            range(len(matrix))
        )

        ax.set_yticklabels(
            matrix.index
        )

        ax.set_title(
            label
        )

        for i in range(
            len(matrix)
        ):

            for j in range(
                len(matrix)
            ):

                value = matrix.iloc[
                    i,
                    j,
                ]

                if np.isfinite(value):

                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )

    if last_image is not None:

        fig.colorbar(
            last_image,
            ax=axes,
            shrink=0.75,
            label="Pearson correlation",
        )

    fig.suptitle(
        "Sensor Interaction Correlation"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# FN duration plot
# ============================================================

def plot_fn_duration(
    fn_duration: pd.DataFrame,
    output_path: Path,
):

    if fn_duration.empty:
        return

    df = fn_duration.copy()

    x = np.arange(
        len(df)
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        x,
        df[
            "detection_rate_pct"
        ],
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        df[
            "duration_bin"
        ].astype(str)
    )

    ax.set_ylabel(
        "Detection rate (%)"
    )

    ax.set_ylim(
        0,
        105,
    )

    ax.set_title(
        "Fault Detection Rate by Event Duration"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for i, value in enumerate(
        df[
            "detection_rate_pct"
        ]
    ):

        if np.isfinite(value):

            ax.text(
                i,
                value + 2,
                f"{value:.1f}%",
                ha="center",
            )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 자동 해석
# ============================================================

def top_effect_text(
    summary: pd.DataFrame,
    column: str,
    top_n: int = 5,
) -> str:

    work = summary.copy()

    work = work.dropna(
        subset=[column]
    )

    if work.empty:
        return "유효한 feature effect가 없습니다."

    work["abs_effect"] = (
        work[column].abs()
    )

    work = (
        work
        .sort_values(
            "abs_effect",
            ascending=False,
        )
        .head(top_n)
    )

    parts = []

    for _, row in work.iterrows():

        effect = row[column]

        if effect > 0:
            direction = "↑"
        elif effect < 0:
            direction = "↓"
        else:
            direction = "≈"

        parts.append(
            f"{row['feature']} "
            f"{direction} "
            f"(effect={effect:+.2f})"
        )

    return ", ".join(parts)

def add_raw_metadata(
    df: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    """
    4번/6번과 동일한 규칙으로 raw 데이터에
    segment_id / group_id / kind / label을 생성한다.
    """

    out = df.copy()

    out[TIME_COL] = pd.to_datetime(
        out[TIME_COL],
        errors="coerce",
    )

    out = (
        out
        .dropna(subset=[TIME_COL])
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )

    # Timestamp gap
    gap = (
        out[TIME_COL]
        .diff()
        .dt.total_seconds()
    )

    segment_break = (
        gap > 0.5
    )

    segment_break.iloc[0] = True

    out["segment_id"] = (
        segment_break.cumsum() - 1
    )

    out["source"] = source

    # 6번 코드와 동일한 group_id
    out["group_id"] = (
        source
        + "_"
        + out["segment_id"].astype(str)
    )

    # Idle
    if "Idle" in out.columns:
        idle = (
            pd.to_numeric(
                out["Idle"],
                errors="coerce",
            )
            .fillna(0)
        )
    else:
        idle = pd.Series(
            0,
            index=out.index,
        )

    idle_ratio = (
        (idle == 1)
        .groupby(out["group_id"])
        .transform("mean")
    )

    # kind
    if source == "fault":
        out["kind"] = "fault"
    else:
        out["kind"] = np.where(
            idle_ratio >= 0.5,
            "idle",
            "normal",
        )

    # label
    out["label"] = (
        pd.to_numeric(
            out["Equipment_state"],
            errors="coerce",
        )
        .fillna(0)
        .ge(1)
        .astype(int)
    )

    return out


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "KAMPact variable effect "
            "and sensor interaction analysis"
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
        dest="window_05",
        default=(
            "result/"
            "modeling_dataset_0.5_0.1/"
            "model_windows.csv"
        ),
    )

    parser.add_argument(
        "--window-1.0",
        dest="window_10",
        default=(
            "result/"
            "modeling_dataset_1.0_0.1/"
            "model_windows.csv"
        ),
    )

    parser.add_argument(
        "--fp-output-dir",
        default=(
            "outputs/"
            "8_fp_fn_analysis"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/"
            "10_variable_effect_analysis"
        ),
    )

    args = parser.parse_args()

    root = Path.cwd()

    normal_path = (
        root / args.normal_path
    )

    fault_path = (
        root / args.fault_path
    )

    window_05_path = (
        root / args.window_05
    )

    window_10_path = (
        root / args.window_10
    )

    fp_dir = (
        root / args.fp_output_dir
    )

    output_dir = (
        root / args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load data
    # ========================================================

    normal_raw = add_raw_metadata(
        safe_read_csv(normal_path),
        "normal",
    )

    fault_raw = add_raw_metadata(
        safe_read_csv(fault_path),
        "fault",
    )

    normal_05 = safe_read_csv(
        window_05_path
    )

    normal_10 = safe_read_csv(
        window_10_path
    )

    fp_windows_path = (
        fp_dir
        / "fp_windows_1.0s.csv"
    )

    fn_event_path = (
        fp_dir
        / "fn_event_summary.csv"
    )

    fn_duration_path = (
        fp_dir
        / "fn_duration_summary.csv"
    )

    fp_windows = safe_read_csv(
        fp_windows_path
    )

    fn_events = safe_read_csv(
        fn_event_path
    )

    fn_duration = safe_read_csv(
        fn_duration_path
    )

    # ========================================================
    # Feature columns
    # ========================================================

    features = infer_features(
        normal_05
    )

    print()
    print("=" * 100)
    print(
        f"Features ({len(features)})"
    )
    print("=" * 100)

    for feature in features:
        print(
            f"  - {feature}"
        )

    # ========================================================
    # 1. Normal vs Fault
    #
    # 0.5s window 사용
    # ========================================================

    normal_05_operation = (
        normal_05[
            (
                normal_05["source"]
                .astype(str)
                .eq("normal")
            )
            &
            (
                pd.to_numeric(
                    normal_05["label"],
                    errors="coerce",
                ).eq(0)
            )
        ]
        .copy()
    )

    fault_05 = (
        normal_05[
            (
                normal_05["source"]
                .astype(str)
                .eq("fault")
            )
            &
            (
                pd.to_numeric(
                    normal_05["label"],
                    errors="coerce",
                ).eq(1)
            )
        ]
        .copy()
    )

    normal_fault_combined = pd.concat(
        [
            normal_05_operation,
            fault_05,
        ],
        ignore_index=True,
    )

    # ========================================================
    # 2. Normal vs FP
    #
    # 1.0s window 사용
    # ========================================================

    normal_10_operation = (
        normal_10[
            (
                normal_10["source"]
                .astype(str)
                .eq("normal")
            )
            &
            (
                pd.to_numeric(
                    normal_10["label"],
                    errors="coerce",
                ).eq(0)
            )
        ]
        .copy()
    )

    fp_windows = fp_windows.copy()

    fp_windows["is_fp"] = True

    normal_10_operation[
        "is_fp"
    ] = False

    normal_fp_combined = pd.concat(
        [
            normal_10_operation,
            fp_windows,
        ],
        ignore_index=True,
        sort=False,
    )

    # ========================================================
    # 3. Event-level Detected vs Any-Miss
    #
    # 0.5s fault window
    # ========================================================

    fn_events["missed_count"] = (
        pd.to_numeric(
            fn_events["missed_count"],
            errors="coerce",
        )
        .fillna(0)
    )

    stable_detected_ids = set(
        fn_events.loc[
            fn_events["missed_count"].eq(0),
            "group_id",
        ]
        .astype(str)
    )

    any_miss_ids = set(
        fn_events.loc[
            fn_events["missed_count"].gt(0),
            "group_id",
        ]
        .astype(str)
    )

    fault_feature_windows = (
        normal_05[
            (
                normal_05["source"]
                .astype(str)
                .eq("fault")
            )
            &
            (
                pd.to_numeric(
                    normal_05["label"],
                    errors="coerce",
                ).eq(1)
            )
        ]
        .copy()
    )

    event_medians = (
        build_event_feature_medians(
            fault_feature_windows,
            features,
        )
    )

    event_medians[
        "event_status"
    ] = "unclassified"

    event_medians.loc[
        event_medians["group_id"]
        .astype(str)
        .isin(stable_detected_ids),
        "event_status",
    ] = "stable_detected"

    event_medians.loc[
        event_medians["group_id"]
        .astype(str)
        .isin(any_miss_ids),
        "event_status",
    ] = "any_miss"

    detected_event_features = (
        event_medians[
            event_medians[
                "event_status"
            ].eq("stable_detected")
        ]
        .copy()
    )

    missed_event_features = (
        event_medians[
            event_medians[
                "event_status"
            ].eq("any_miss")
        ]
        .copy()
    )

    # ========================================================
    # Coverage report
    # ========================================================

    feature_coverage = pd.DataFrame([
        {
            "group": "stable_detected",
            "event_count_in_fn_summary": len(
                stable_detected_ids
            ),
            "event_count_with_0.5s_features": len(
                detected_event_features
            ),
        },
        {
            "group": "any_miss",
            "event_count_in_fn_summary": len(
                any_miss_ids
            ),
            "event_count_with_0.5s_features": len(
                missed_event_features
            ),
        },
    ])

    feature_coverage.to_csv(
        output_dir
        / "feature_coverage_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Feature effect table
    # ========================================================

    effect_table = pd.DataFrame(
        {
            "feature": features
        }
    )

    # --------------------------------------------------------
    # Normal vs Fault
    # --------------------------------------------------------

    nf_rows = []

    for feature in features:

        result = robust_effect(
            normal_05_operation[feature],
            fault_05[feature],
        )

        nf_rows.append({
            "feature": feature,
            "normal_fault_normal_median":
                result["baseline_median"],
            "normal_fault_fault_median":
                result["target_median"],
            "normal_fault_delta":
                result["median_delta"],
            "normal_fault_effect":
                result["robust_effect"],
            "normal_fault_direction":
                result["direction"],
        })

    nf_df = pd.DataFrame(
        nf_rows
    )

    # --------------------------------------------------------
    # Normal vs FP
    # --------------------------------------------------------

    nfp_rows = []

    for feature in features:

        result = robust_effect(
            normal_10_operation[feature],
            fp_windows[feature],
        )

        nfp_rows.append({
            "feature": feature,
            "normal_fp_normal_median":
                result["baseline_median"],
            "normal_fp_fp_median":
                result["target_median"],
            "normal_fp_delta":
                result["median_delta"],
            "normal_fp_effect":
                result["robust_effect"],
            "normal_fp_direction":
                result["direction"],
        })

    nfp_df = pd.DataFrame(
        nfp_rows
    )

    # --------------------------------------------------------
    # Detected vs Any-Miss
    # --------------------------------------------------------

    dm_rows = []

    for feature in features:

        result = robust_effect(
            detected_event_features[feature],
            missed_event_features[feature],
        )

        dm_rows.append({
            "feature": feature,
            "detected_missed_detected_median":
                result["baseline_median"],
            "detected_missed_missed_median":
                result["target_median"],
            "detected_missed_delta":
                result["median_delta"],
            "detected_missed_effect":
                result["robust_effect"],
            "detected_missed_direction":
                result["direction"],
        })

    dm_df = pd.DataFrame(
        dm_rows
    )

    effect_table = (
        nf_df
        .merge(
            nfp_df,
            on="feature",
            how="outer",
        )
        .merge(
            dm_df,
            on="feature",
            how="outer",
        )
    )

    # 전체 effect 기준 참고 순위
    effect_cols = [
        "normal_fault_effect",
        "normal_fp_effect",
        "detected_missed_effect",
    ]

    effect_table[
        "max_abs_effect"
    ] = (
        effect_table[
            effect_cols
        ]
        .abs()
        .max(
            axis=1
        )
    )

    effect_table = (
        effect_table
        .sort_values(
            "max_abs_effect",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    effect_table.to_csv(
        output_dir
        / "variable_effect_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Event feature medians
    # ========================================================

    event_medians.to_csv(
        output_dir
        / "event_feature_medians.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Sensor interaction
    # ========================================================

    fp_raw_mask = (
        build_fp_raw_mask(
            normal_raw,
            fp_windows,
        )
    )

    normal_raw_sensor = (
        normal_raw[
            pd.to_numeric(
                normal_raw["Idle"],
                errors="coerce",
            )
            .fillna(0)
            .eq(0)
        ]
        .copy()
    )

    fault_raw_sensor = (
        fault_raw[
            fault_raw[
                "Equipment_state"
            ].ge(1)
        ]
        .copy()
    )

    fp_raw_sensor = (
        normal_raw[
            fp_raw_mask
        ]
        .copy()
    )

    any_miss_raw_sensor = (
        fault_raw[
            fault_raw[
                "group_id"
            ]
            .astype(str)
            .isin(any_miss_ids)
        ]
        .copy()
    )

    sensor_groups = {
        "Normal": normal_raw_sensor,
        "Fault": fault_raw_sensor,
        "FP Context": fp_raw_sensor,
        "Any-Miss Fault": any_miss_raw_sensor,
    }

    matrices = {}

    correlation_rows = []

    for group_name, group_df in (
        sensor_groups.items()
    ):

        matrix = correlation_matrix(
            group_df
        )

        matrices[
            group_name
        ] = matrix

        for sensor_a in matrix.index:

            for sensor_b in matrix.columns:

                correlation_rows.append({
                    "group": group_name,
                    "sensor_a": sensor_a,
                    "sensor_b": sensor_b,
                    "correlation": matrix.loc[
                        sensor_a,
                        sensor_b,
                    ],
                })

    correlation_df = pd.DataFrame(
        correlation_rows
    )

    correlation_df.to_csv(
        output_dir
        / "sensor_correlation_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Plots
    # ========================================================

    plot_feature_effects(
        effect_table,
        output_dir
        / "feature_effect_plot.png",
    )

    plot_correlation_heatmap(
        matrices,
        output_dir
        / "sensor_correlation_heatmap.png",
    )

    plot_fn_duration(
        fn_duration,
        output_dir
        / "fn_duration_detection.png",
    )

    # ========================================================
    # Report
    # ========================================================

    report = []

    report.append(
        "KAMPact - Variable Effect & Interaction Analysis"
    )

    report.append(
        "=" * 80
    )

    report.append("")

    report.append(
        "[1] Normal vs Fault"
    )

    report.append(
        "-" * 80
    )

    report.append(
        "Window scale: 0.5s"
    )

    report.append(
        f"Normal windows: {len(normal_05_operation):,}"
    )

    report.append(
        f"Fault windows : {len(fault_05):,}"
    )

    report.append("")

    report.append(
        "Top effects:"
    )

    report.append(
        top_effect_text(
            effect_table,
            "normal_fault_effect",
            top_n=5,
        )
    )

    report.append("")

    report.append(
        "[2] Normal vs FP"
    )

    report.append(
        "-" * 80
    )

    report.append(
        "Window scale: 1.0s"
    )

    report.append(
        f"Normal windows: {len(normal_10_operation):,}"
    )

    report.append(
        f"FP windows    : {len(fp_windows):,}"
    )

    report.append("")

    report.append(
        "Top effects:"
    )

    report.append(
        top_effect_text(
            effect_table,
            "normal_fp_effect",
            top_n=5,
        )
    )

    report.append("")

    report.append(
        "[3] Stable Detected vs Any-Miss Fault"
    )

    report.append(
        "-" * 80
    )

    report.append(
        "Window scale: 0.5s"
    )

    report.append(
        "Comparison unit: event-level feature median"
    )

    report.append(
        f"Stable detected events in FN summary: "
        f"{len(stable_detected_ids)}"
    )

    report.append(
        f"Stable detected events with feature rows: "
        f"{len(detected_event_features)}"
    )

    report.append(
        f"Any-miss events in FN summary: "
        f"{len(any_miss_ids)}"
    )

    report.append(
        f"Any-miss events with feature rows: "
        f"{len(missed_event_features)}"
    )

    report.append("")

    report.append(
        "Top effects:"
    )

    report.append(
        top_effect_text(
            effect_table,
            "detected_missed_effect",
            top_n=5,
        )
    )

    report.append("")

    report.append(
        "[4] Sensor Interaction"
    )

    report.append(
        "-" * 80
    )

    for group_name, matrix in (
        matrices.items()
    ):

        report.append(
            f"\n{group_name}"
        )

        report.append(
            matrix.to_string(
                float_format=lambda x: f"{x:.3f}"
            )
        )

    report.append("")

    report.append(
        "[5] Interpretation"
    )

    report.append(
        "-" * 80
    )

    report.append(
        "Feature effect is descriptive and represents "
        "the difference in group medians normalized by "
        "the baseline IQR-derived robust scale."
    )

    report.append(
        "A large effect does not prove that the feature "
        "caused the alarm or fault."
    )

    report.append(
        "Detected vs Any-Miss comparison is performed at "
        "event level to prevent long fault events from "
        "dominating the analysis."
    )

    report.append(
        "Very short events without a generated 0.5s window "
        "cannot contribute to the 16-feature event comparison."
    )

    report.append("")

    report.append(
        "[6] Output"
    )

    report.append(
        "-" * 80
    )

    for filename in [
        "variable_effect_summary.csv",
        "event_feature_medians.csv",
        "sensor_correlation_matrix.csv",
        "feature_coverage_report.csv",
        "feature_effect_plot.png",
        "sensor_correlation_heatmap.png",
        "fn_duration_detection.png",
        "analysis_report.txt",
    ]:

        report.append(
            str(
                output_dir
                / filename
            )
        )

    report_path = (
        output_dir
        / "analysis_report.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 100)
    print(
        "KAMPact - Variable Effect & Interaction Analysis"
    )
    print("=" * 100)

    print()
    print(
        "[Normal vs Fault - 0.5s]"
    )

    print(
        top_effect_text(
            effect_table,
            "normal_fault_effect",
            5,
        )
    )

    print()
    print(
        "[Normal vs FP - 1.0s]"
    )

    print(
        top_effect_text(
            effect_table,
            "normal_fp_effect",
            5,
        )
    )

    print()
    print(
        "[Stable Detected vs Any-Miss - event level]"
    )

    print(
        top_effect_text(
            effect_table,
            "detected_missed_effect",
            5,
        )
    )

    print()
    print(
        "[Event feature coverage]"
    )

    print(
        feature_coverage.to_string(
            index=False
        )
    )

    print()
    print(
        "[Sensor correlations]"
    )

    for group_name, matrix in (
        matrices.items()
    ):

        print()
        print(
            group_name
        )

        print(
            matrix.to_string(
                float_format=lambda x: f"{x:.3f}"
            )
        )

    print()
    print(
        "=" * 100
    )

    print(
        "Analysis complete"
    )

    print(
        f"Output: {output_dir}"
    )


if __name__ == "__main__":
    main()