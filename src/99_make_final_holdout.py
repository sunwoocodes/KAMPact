from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# 설정
# ============================================================

INPUT_DIR = Path(
    "result/modeling_dataset_1.0_0.1"
)

OUTPUT_DIR = Path(
    "result/final_evaluation_1.0_0.1"
)

SEED = 20260930

# 최종 hold-out 비율
HOLDOUT_RATIO = 0.20

# hold-out을 제외한 Development 영역에서
# train / val 비율
DEV_VAL_RATIO = 0.25
# 전체 기준으로 보면
# train 60% / val 20% / final test 20%


# ============================================================
# 유틸
# ============================================================

def choose_holdout_groups(
    groups,
    ratio,
    rng,
):
    groups = np.array(
        sorted(groups),
        dtype=object,
    )

    if len(groups) < 2:
        raise ValueError(
            f"그룹 수가 너무 적습니다: {len(groups)}"
        )

    n_holdout = max(
        1,
        math.ceil(
            len(groups) * ratio
        ),
    )

    if n_holdout >= len(groups):
        n_holdout = len(groups) - 1

    selected = rng.choice(
        groups,
        size=n_holdout,
        replace=False,
    )

    return set(selected.tolist())


def split_dev_groups(
    groups,
    val_ratio,
    rng,
):
    groups = np.array(
        sorted(groups),
        dtype=object,
    )

    rng.shuffle(groups)

    n_val = max(
        1,
        math.ceil(
            len(groups) * val_ratio
        ),
    )

    if n_val >= len(groups):
        n_val = len(groups) - 1

    val_groups = set(
        groups[:n_val].tolist()
    )

    train_groups = set(
        groups[n_val:].tolist()
    )

    return train_groups, val_groups


# ============================================================
# Main
# ============================================================

def main():

    model_path = (
        INPUT_DIR / "model_windows.csv"
    )

    idle_path = (
        INPUT_DIR / "idle_windows.csv"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"파일이 없습니다: {model_path}"
        )

    model_df = pd.read_csv(
        model_path
    )

    idle_df = None

    if idle_path.exists():
        idle_df = pd.read_csv(
            idle_path
        )

    required = [
        "group_id",
        "source",
        "label",
    ]

    missing = [
        c for c in required
        if c not in model_df.columns
    ]

    if missing:
        raise ValueError(
            f"필수 컬럼 누락: {missing}"
        )

    rng = np.random.default_rng(SEED)

    # ========================================================
    # 1. Group 목록
    # ========================================================

    group_info = (
        model_df
        .groupby(
            "group_id",
            as_index=False,
        )
        .agg(
            source=("source", "first"),
            label=("label", "first"),
        )
    )

    normal_groups = set(
        group_info.loc[
            group_info["source"].eq("normal"),
            "group_id",
        ]
    )

    fault_groups = set(
        group_info.loc[
            group_info["source"].eq("fault"),
            "group_id",
        ]
    )

    print("=" * 80)
    print("GROUP SUMMARY")
    print("=" * 80)

    print(
        f"normal groups : {len(normal_groups)}"
    )
    print(
        f"fault groups  : {len(fault_groups)}"
    )

    # ========================================================
    # 2. Final Hold-out group 선택
    # ========================================================

    holdout_normal = choose_holdout_groups(
        normal_groups,
        HOLDOUT_RATIO,
        rng,
    )

    holdout_fault = choose_holdout_groups(
        fault_groups,
        HOLDOUT_RATIO,
        rng,
    )

    holdout_groups = (
        holdout_normal
        | holdout_fault
    )

    # ========================================================
    # 3. Development group
    # ========================================================

    dev_normal = (
        normal_groups
        - holdout_normal
    )

    dev_fault = (
        fault_groups
        - holdout_fault
    )

    normal_train, normal_val = (
        split_dev_groups(
            dev_normal,
            DEV_VAL_RATIO,
            rng,
        )
    )

    fault_train, fault_val = (
        split_dev_groups(
            dev_fault,
            DEV_VAL_RATIO,
            rng,
        )
    )

    train_groups = (
        normal_train
        | fault_train
    )

    val_groups = (
        normal_val
        | fault_val
    )

    # ========================================================
    # 4. 그룹별 최종 split
    # ========================================================

    split_map = {}

    for g in train_groups:
        split_map[g] = "train"

    for g in val_groups:
        split_map[g] = "val"

    for g in holdout_groups:
        split_map[g] = "test"

    # 모든 모델 그룹이 정확히 배정됐는지 확인
    missing_groups = (
        set(group_info["group_id"])
        - set(split_map)
    )

    if missing_groups:
        raise RuntimeError(
            f"split 미배정 group 존재: {missing_groups}"
        )

    if (
        train_groups & val_groups
        or train_groups & holdout_groups
        or val_groups & holdout_groups
    ):
        raise RuntimeError(
            "Group overlap 발생"
        )

    # ========================================================
    # 5. 모델 데이터 생성
    # ========================================================

    final_df = model_df.copy()

    # 기존 split은 감사/비교용으로 보존
    if "split" in final_df.columns:
        final_df["original_split"] = (
            final_df["split"]
        )

    final_df["split"] = (
        final_df["group_id"]
        .map(split_map)
    )

    # ========================================================
    # 6. Development 데이터
    # ========================================================

    dev_df = (
        final_df[
            final_df["split"].isin(
                ["train", "val"]
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    # ========================================================
    # 7. Hold-out 데이터
    # ========================================================

    test_df = (
        final_df[
            final_df["split"].eq("test")
        ]
        .copy()
        .reset_index(drop=True)
    )

    # ========================================================
    # 8. Idle도 별도로 분리
    # ========================================================

    final_idle = None
    dev_idle = None

    if idle_df is not None:

        idle_group_ids = set(
            idle_df["group_id"]
        )

        holdout_idle = choose_holdout_groups(
            idle_group_ids,
            HOLDOUT_RATIO,
            rng,
        )

        final_idle = (
            idle_df[
                idle_df["group_id"]
                .isin(holdout_idle)
            ]
            .copy()
            .reset_index(drop=True)
        )

        dev_idle = (
            idle_df[
                ~idle_df["group_id"]
                .isin(holdout_idle)
            ]
            .copy()
            .reset_index(drop=True)
        )

    # ========================================================
    # 9. 저장
    # ========================================================

    dev_dir = (
        OUTPUT_DIR / "dev"
    )

    final_dir = (
        OUTPUT_DIR / "final"
    )

    dev_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dev_df.to_csv(
        dev_dir / "model_windows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_df.to_csv(
        final_dir / "model_windows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_df.to_csv(
        final_dir / "model_windows_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if dev_idle is not None:
        dev_idle.to_csv(
            dev_dir / "idle_windows.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if final_idle is not None:
        final_idle.to_csv(
            final_dir / "idle_windows.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # ========================================================
    # 10. Holdout manifest
    # ========================================================

    manifest_rows = []

    for g in train_groups:
        manifest_rows.append({
            "group_id": g,
            "final_split": "train",
        })

    for g in val_groups:
        manifest_rows.append({
            "group_id": g,
            "final_split": "val",
        })

    for g in holdout_groups:
        manifest_rows.append({
            "group_id": g,
            "final_split": "test",
        })

    holdout_manifest = pd.DataFrame(
        manifest_rows
    )

    holdout_manifest = (
        holdout_manifest
        .merge(
            group_info,
            on="group_id",
            how="left",
        )
        .sort_values(
            ["final_split", "source", "group_id"]
        )
    )

    holdout_manifest.to_csv(
        OUTPUT_DIR / "final_holdout_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 11. 결과 출력
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL SPLIT")
    print("=" * 80)

    print(
        holdout_manifest
        .groupby(
            ["final_split", "source"]
        )
        .size()
        .to_string()
    )

    print()
    print(
        f"train groups : {len(train_groups)}"
    )
    print(
        f"val groups   : {len(val_groups)}"
    )
    print(
        f"test groups  : {len(holdout_groups)}"
    )

    print()
    print("저장 완료")
    print(
        OUTPUT_DIR / "dev/model_windows.csv"
    )
    print(
        OUTPUT_DIR / "final/model_windows_all.csv"
    )
    print(
        OUTPUT_DIR / "final_holdout_manifest.csv"
    )

    print()
    print(
        "중요: final_holdout 데이터의 성능을 "
        "파라미터 선택에 사용하지 마세요."
    )


if __name__ == "__main__":
    main()