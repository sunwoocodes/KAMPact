"""
KAMPact - 3개 이상탐지기 통합 비교 (Persistence & 상세 Fault 추적)
-------------------------------------------------------------------------
비교 대상
  1) 1.0s window Mahalanobis (Causal Shifted)
  2) 0.5s window Mahalanobis (Causal Shifted)
  3) sample-level causal Mahalanobis

추가 분석:
  - segment별 탐지율 및 delay 상세 로깅
  - OR_3_P2 모델의 Missed Fault 집중 분석
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# 공통 설정
# ============================================================

TIME_COL = "TimeStamp"
STATE_COL = "Equipment_state"
IDLE_COL = "Idle"

SENSOR_COLS = [
    "AI0_Vibration",
    "AI1_Vibration",
    "AI2_Current",
]

GAP_THRESHOLD_SEC = 0.5
FOLD_SOURCE_ORDER = ["fault", "normal", "idle"]


# ============================================================
# 모듈 로드
# ============================================================

def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"스크립트를 찾을 수 없습니다: {path}")

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈 로드 실패: {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


# ============================================================
# window feature column 자동 탐색
# ============================================================

def infer_window_feature_cols(paths: list[Path]) -> list[str]:
    if not paths:
        raise ValueError("model_windows.csv 경로가 없습니다.")

    frames = [pd.read_csv(path, nrows=200) for path in paths]
    common = [c for c in frames[0].columns if all(c in df.columns for df in frames[1:])]

    meta_exact = {
        "Unnamed: 0", "TimeStamp", "window_start", "window_end",
        "group_id", "segment_id", "source", "kind", "split", "label",
        "fault_ratio", "normal_ratio", "idle_ratio", "n_samples",
        "start_idx", "end_idx", "window_id", "window_index",
        "duration_sec", "Equipment_state", "Idle",
    }

    numeric_common = []
    for c in common:
        if c in meta_exact:
            continue
        if all(pd.api.types.is_numeric_dtype(df[c]) for df in frames):
            numeric_common.append(c)

    preferred = []
    for c in numeric_common:
        cl = c.lower()
        if (cl.startswith("ai0") or cl.startswith("ai1") or cl.startswith("ai2")
                or "corr" in cl or "correlation" in cl):
            preferred.append(c)

    if len(preferred) == 16:
        return preferred
    if len(numeric_common) == 16:
        return numeric_common
    if len(preferred) >= 16:
        return preferred[:16]

    raise ValueError(f"16개 feature 컬럼을 자동 추론할 수 없습니다. 후보: {numeric_common}")


# ============================================================
# raw 데이터 로드
# ============================================================

def load_raw(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    need = [TIME_COL, STATE_COL] + SENSOR_COLS
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: 필수 컬럼 누락: {missing}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)

    for c in SENSOR_COLS + [STATE_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if IDLE_COL in df.columns:
        df[IDLE_COL] = pd.to_numeric(df[IDLE_COL], errors="coerce").fillna(0)
    else:
        df[IDLE_COL] = 0

    gap = df[TIME_COL].diff().dt.total_seconds()
    brk = gap > GAP_THRESHOLD_SEC
    brk.iloc[0] = True
    seg = brk.cumsum() - 1

    df["source"] = source
    df["segment_id"] = seg
    df["group_id"] = source + "_" + seg.astype(str)

    idle_ratio = (df[IDLE_COL] == 1).groupby(df["group_id"]).transform("mean")
    df["kind"] = np.where(source == "fault", "fault", np.where(idle_ratio >= 0.5, "idle", "normal"))
    df["label"] = pd.to_numeric(df[STATE_COL], errors="coerce").fillna(0).ge(1).astype(int)

    return df


# ============================================================
# 공통 fold 배정
# ============================================================

def build_fold_map(raw_df: pd.DataFrame, n_splits: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    fold_of: dict[str, int] = {}
    groups_by_kind: dict[str, set[str]] = {"fault": set(), "normal": set(), "idle": set()}

    for kind, g in raw_df.groupby("kind"):
        groups_by_kind.setdefault(kind, set()).update(g["group_id"].astype(str).unique())

    for kind in FOLD_SOURCE_ORDER:
        groups = np.array(sorted(groups_by_kind.get(kind, set())))
        rng.shuffle(groups)
        for i, gid in enumerate(groups):
            fold_of[str(gid)] = i % n_splits

    return fold_of


# ============================================================
# Sample-level Detector 관련
# ============================================================

def build_sample_features(df: pd.DataFrame, feature_set: str) -> tuple[np.ndarray, list[str]]:
    parts = [df[SENSOR_COLS].to_numpy(dtype=float)]
    names = list(SENSOR_COLS)

    if feature_set in ("raw_diff", "raw_diff_roll3"):
        diff = df.groupby("group_id")[SENSOR_COLS].diff().fillna(0.0).to_numpy(dtype=float)
        parts.append(diff)
        names += [f"d_{c}" for c in SENSOR_COLS]

    if feature_set == "raw_diff_roll3":
        for c in SENSOR_COLS:
            roll = (
                df.groupby("group_id")[c]
                .transform(lambda s: s.rolling(3, min_periods=2).std())
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            parts.append(roll.to_numpy().reshape(-1, 1))
            names.append(f"std3_{c}")

    return np.hstack(parts), names


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def fit_sample_mahalanobis(x_train: np.ndarray, covariance: str):
    from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS

    if covariance == "empirical":
        est = EmpiricalCovariance(assume_centered=False)
    elif covariance == "oas":
        est = OAS(assume_centered=False)
    else:
        est = LedoitWolf(assume_centered=False)

    est.fit(x_train)
    mean = est.location_
    precision = est.precision_

    def score(x: np.ndarray) -> np.ndarray:
        d = x - mean
        return np.einsum("ij,jk,ik->i", d, precision, d)

    return score


def causal_smooth(scores: np.ndarray, group_positions: list[np.ndarray], k: int, how: str = "mean") -> np.ndarray:
    out = np.full(len(scores), np.nan)
    for idx in group_positions:
        s = pd.Series(scores[idx]).rolling(k, min_periods=k)
        out[idx] = s.max().to_numpy() if how == "max" else s.mean().to_numpy()
    return out


def choose_sample_threshold_f1(y: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import f1_score, precision_score

    finite = np.isfinite(scores)
    y = np.asarray(y)[finite].astype(int)
    s = np.asarray(scores)[finite].astype(float)

    if len(s) == 0 or y.sum() == 0:
        raise ValueError("sample-level validation에 데이터가 부족합니다.")

    candidates = np.unique(s)
    best = None
    for t in candidates:
        pred = (s >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        precision = precision_score(y, pred, zero_division=0)
        key = (float(f1), float(precision), -float(t))
        if best is None or key > best[0]:
            best = (key, float(t))
    return best[1]


def choose_sample_threshold_quantile(y: np.ndarray, scores: np.ndarray, q: float) -> float:
    normal = scores[(np.asarray(y).astype(int) == 0) & np.isfinite(scores)]
    if len(normal) == 0:
        raise ValueError("sample-level normal score가 없습니다.")
    return float(np.quantile(normal, q))


# ============================================================
# Causal Window alarm -> raw sample 매핑 (미래 참조 제거)
# ============================================================

def apply_window_alarm_to_raw(raw: pd.DataFrame, test_work: pd.DataFrame, alarm_col: str) -> np.ndarray:
    alarm_raw = np.zeros(len(raw), dtype=bool)
    if test_work.empty:
        return alarm_raw

    tw = test_work.copy()
    tw["window_start"] = pd.to_datetime(tw["window_start"])
    tw["window_end"] = pd.to_datetime(tw["window_end"])

    raw_time = raw[TIME_COL].to_numpy(dtype="datetime64[ns]")
    raw_group = raw["group_id"].astype(str).to_numpy()

    for _, row in tw.iterrows():
        if not bool(row[alarm_col]):
            continue

        gid = str(row["group_id"])
        start = np.datetime64(row["window_start"])
        end = np.datetime64(row["window_end"])

        window_duration = end - start
        shifted_start = start + window_duration
        shifted_end = end + window_duration

        mask = (raw_group == gid) & (raw_time >= shifted_start) & (raw_time < shifted_end)
        alarm_raw[mask] = True

    return alarm_raw


# ============================================================
# Segment-aware Persistence & Temporal Filter
# ============================================================

def apply_group_persistence(alarm: np.ndarray, group_positions: list[np.ndarray], p: int) -> np.ndarray:
    if p <= 1:
        return alarm.copy()

    out = np.zeros_like(alarm, dtype=bool)
    for idx in group_positions:
        s = pd.Series(alarm[idx].astype(int))
        roll_sum = s.rolling(p, min_periods=p).sum().fillna(0).to_numpy()
        out[idx] = (roll_sum == p)
    return out


def apply_temporal_tolerance(alarm: np.ndarray, group_positions: list[np.ndarray], window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return alarm.copy()

    out = np.zeros_like(alarm, dtype=bool)
    for idx in group_positions:
        s = pd.Series(alarm[idx].astype(int))
        roll_max = s.rolling(window_samples, min_periods=1).max().fillna(0).to_numpy()
        out[idx] = (roll_max > 0)
    return out


# ============================================================
# Metrics
# ============================================================

def count_episodes(alarm: np.ndarray) -> int:
    alarm = np.asarray(alarm, dtype=bool)
    prev = np.r_[False, alarm[:-1]]
    return int((alarm & ~prev).sum())


def evaluate_raw_alarm(raw: pd.DataFrame, alarm: np.ndarray, fault_groups: set[str], normal_groups: set[str], idle_groups: set[str]) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score

    alarm = np.asarray(alarm, dtype=bool)
    y = raw["label"].to_numpy(dtype=int)

    non_idle = ~raw["kind"].eq("idle").to_numpy()
    yp = alarm[non_idle].astype(int)
    yt = y[non_idle]

    f1 = f1_score(yt, yp, zero_division=0)
    precision = precision_score(yt, yp, zero_division=0)
    recall = recall_score(yt, yp, zero_division=0)

    # Fault Event
    event_rows = []
    for gid in sorted(fault_groups):
        sub = raw[raw["group_id"].eq(gid)]
        idx = sub.index.to_numpy()
        hit = np.flatnonzero(alarm[idx])

        if len(hit):
            first_idx = idx[hit[0]]
            start_time = raw.loc[idx[0], TIME_COL]
            alarm_time = raw.loc[first_idx, TIME_COL]
            delay = (alarm_time - start_time).total_seconds()
            detected = True
        else:
            delay = np.nan
            detected = False

        event_rows.append({"group_id": gid, "detected": detected, "delay_sec": delay, "n_samples": len(idx)})

    events = pd.DataFrame(event_rows)
    detected_events = int(events["detected"].sum())
    n_events = len(events)
    mean_delay = float(events.loc[events["detected"], "delay_sec"].mean()) if detected_events else np.nan

    # Normal Cycle FA
    normal_cycle_rows = []
    for gid in sorted(normal_groups):
        idx = raw.index[raw["group_id"].eq(gid)].to_numpy()
        a = alarm[idx]
        normal_cycle_rows.append({"group_id": gid, "fa": bool(a.any()), "episodes": count_episodes(a)})

    normal_cycles_df = pd.DataFrame(normal_cycle_rows)
    fa_cycles = int(normal_cycles_df["fa"].sum()) if len(normal_cycles_df) else 0
    n_normal = len(normal_cycles_df)
    fa_cycle_rate = (fa_cycles / n_normal) if n_normal else np.nan
    fa_episodes = int(normal_cycles_df["episodes"].sum()) if len(normal_cycles_df) else 0

    # Idle Cycle FA
    idle_cycle_rows = []
    for gid in sorted(idle_groups):
        idx = raw.index[raw["group_id"].eq(gid)].to_numpy()
        a = alarm[idx]
        idle_cycle_rows.append({"group_id": gid, "fa": bool(a.any()), "episodes": count_episodes(a)})

    idle_df = pd.DataFrame(idle_cycle_rows)
    idle_fa_cycles = int(idle_df["fa"].sum()) if len(idle_df) else 0
    n_idle = len(idle_df)
    idle_fa_cycle_rate = (idle_fa_cycles / n_idle) if n_idle else np.nan

    return {
        "event_detection_rate": (detected_events / n_events) if n_events else np.nan,
        "detected_events": detected_events,
        "n_events": n_events,
        "delay_mean_sec": mean_delay,
        "sample_f1": f1,
        "sample_precision": precision,
        "sample_recall": recall,
        "normal_fa_cycle_rate": fa_cycle_rate,
        "normal_fa_cycles": fa_cycles,
        "normal_cycles": n_normal,
        "normal_fa_episodes": fa_episodes,
        "idle_fa_cycle_rate": idle_fa_cycle_rate,
        "idle_fa_cycles": idle_fa_cycles,
        "idle_cycles": n_idle,
        "idle_fa_episodes": int(idle_df["episodes"].sum()) if len(idle_df) else 0,
        "event_details": events, # 상세 기록 추가 반환
    }


# ============================================================
# 1회 fold에서 sample-level detector
# ============================================================

def make_sample_alarm_for_fold(
    raw: pd.DataFrame,
    x_all: np.ndarray,
    group_index: dict[str, np.ndarray],
    fold_map: dict[str, int],
    test_fold: int,
    n_splits: int,
    covariance: str,
    feature_set: str,
    agg_k: int,
    agg_how: str,
    threshold_mode: str,
    normal_quantile: float,
) -> tuple[np.ndarray, float]:

    kinds = raw["kind"].to_numpy()
    gids = raw["group_id"].astype(str).to_numpy()
    labels = raw["label"].to_numpy(dtype=int)

    fold_arr = np.array([fold_map[g] for g in gids])
    val_fold = (test_fold + 1) % n_splits

    train_mask = (kinds == "normal") & (fold_arr != test_fold) & (fold_arr != val_fold)
    val_mask = (kinds != "idle") & (fold_arr == val_fold)

    score_fn = fit_sample_mahalanobis(x_all[train_mask], covariance)
    raw_scores = score_fn(x_all)
    smooth = causal_smooth(raw_scores, list(group_index.values()), agg_k, agg_how)

    if threshold_mode == "normal_quantile":
        thr = choose_sample_threshold_quantile(labels[val_mask], smooth[val_mask], normal_quantile)
    else:
        thr = choose_sample_threshold_f1(labels[val_mask], smooth[val_mask])

    alarm = np.nan_to_num(smooth, nan=-np.inf) >= thr
    test_mask = fold_arr == test_fold
    out = np.zeros(len(raw), dtype=bool)
    out[test_mask] = alarm[test_mask]

    return out, thr


# ============================================================
# 1회 repeat
# ============================================================

def run_repeat(
    raw: pd.DataFrame,
    x_sample: np.ndarray,
    feature_cols: list[str],
    mod5: object,
    window_paths: list[Path],
    n_splits: int,
    seed: int,
    window_threshold_mode: str,
    window_quantile: float,
    window_k: int,
    sample_threshold_mode: str,
    sample_quantile: float,
    sample_agg_k: int,
    sample_agg_how: str,
    sample_feature_set: str,
):
    scales = [mod5.load_scale(mod5_base, p, feature_cols) for p in window_paths]
    scales.sort(key=lambda s: -s.window_sec)

    fold_map = build_fold_map(raw, n_splits=n_splits, seed=seed)
    group_index = raw.groupby("group_id").indices

    all_one = np.zeros(len(raw), dtype=bool)
    all_half = np.zeros(len(raw), dtype=bool)
    all_sample = np.zeros(len(raw), dtype=bool)
    threshold_log = []

    for fold in tqdm(range(n_splits), desc=f"Seed {seed} Folds", leave=False):
        fold_one = np.zeros(len(raw), dtype=bool)
        fold_half = np.zeros(len(raw), dtype=bool)

        for sc in scales:
            work = mod5.make_fold_frame(sc, fold_map, fold, n_splits)
            metrics, test_work, _ = mod5_base.run_once(
                df=work,
                feature_cols=feature_cols,
                covariance=window_covariance,
                threshold_mode=window_threshold_mode,
                normal_quantile=window_quantile,
                k_consecutive=window_k,
                train_seed=None,
                train_fraction=1.0,
            )
            alarm_raw = apply_window_alarm_to_raw(raw, test_work, mod5.ALARM_COL)

            if sc.window_sec >= 0.99:
                fold_one |= alarm_raw
            else:
                fold_half |= alarm_raw

            threshold_log.append({"fold": fold, "scale": sc.name, "threshold": metrics["threshold"]})

        all_one |= fold_one
        all_half |= fold_half

        sample_alarm, sample_thr = make_sample_alarm_for_fold(
            raw=raw,
            x_all=x_sample,
            group_index=group_index,
            fold_map=fold_map,
            test_fold=fold,
            n_splits=n_splits,
            covariance=sample_covariance,
            feature_set=sample_feature_set,
            agg_k=sample_agg_k,
            agg_how=sample_agg_how,
            threshold_mode=sample_threshold_mode,
            normal_quantile=sample_quantile,
        )
        all_sample |= sample_alarm
        threshold_log.append({"fold": fold, "scale": "sample", "threshold": sample_thr})

    # ========================================================
    # 앙상블 조합 구성 (Persistence & Temporal Tolerant)
    # ========================================================
    group_positions = list(group_index.values())

    or3_base = all_one | all_half | all_sample
    or3_p2 = apply_group_persistence(or3_base, group_positions, p=2)
    or3_p3 = apply_group_persistence(or3_base, group_positions, p=3)

    tol_one = apply_temporal_tolerance(all_one, group_positions, window_samples=3)
    tol_half = apply_temporal_tolerance(all_half, group_positions, window_samples=3)
    tol_sample = apply_temporal_tolerance(all_sample, group_positions, window_samples=3)
    two_of_three_tol3 = (tol_one.astype(int) + tol_half.astype(int) + tol_sample.astype(int)) >= 2

    window_any = all_one | all_half
    window_tol = apply_temporal_tolerance(window_any, group_positions, window_samples=5)
    sample_corroborated = all_sample & window_tol
    or3_sample_corroborated = window_any | sample_corroborated

    combos = {
        "window_1.0_only": all_one,
        "window_0.5_only": all_half,
        "sample_only": all_sample,
        "OR_3": or3_base,
        "OR_3_P2": or3_p2,
        "OR_3_P3": or3_p3,
        "2of3_exact": (all_one.astype(int) + all_half.astype(int) + all_sample.astype(int)) >= 2,
        "2of3_tol3": two_of_three_tol3,
        "OR_3_sample_corrob": or3_sample_corroborated,
    }

    rows = []
    events_list = []
    fault_groups = set(raw.loc[raw["kind"].eq("fault"), "group_id"].astype(str))
    normal_groups = set(raw.loc[raw["kind"].eq("normal"), "group_id"].astype(str))
    idle_groups = set(raw.loc[raw["kind"].eq("idle"), "group_id"].astype(str))

    for name, alarm in combos.items():
        m = evaluate_raw_alarm(raw, alarm, fault_groups, normal_groups, idle_groups)
        # 상세 기록 추출
        ev_df = m.pop("event_details")
        ev_df["combo"] = name
        ev_df["repeat_seed"] = seed
        events_list.append(ev_df)

        m.update({"combo": name, "repeat_seed": seed})
        rows.append(m)

    return pd.DataFrame(rows), pd.DataFrame(threshold_log), combos, pd.concat(events_list, ignore_index=True)


# ============================================================
# Main
# ============================================================

def main():
    global mod5_base, window_covariance, sample_covariance

    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-path", default="data/press_data_normal_with_idle.csv")
    parser.add_argument("--fault-path", default="data/outlier_data.csv")
    parser.add_argument("--window-1.0", dest="window_10", default="result/modeling_dataset_1.0_0.1")
    parser.add_argument("--window-0.5", dest="window_05", default="result/modeling_dataset_0.5_0.1")
    parser.add_argument("--base-script", default="src/5_run_mahalanobis.py")
    parser.add_argument("--script-5-2", default="src/5_2_run_mahalanobis.py")
    parser.add_argument("--script-5-3", default="src/5_3_run_sample_level_mahalanobis.py")
    parser.add_argument("--output-dir", default="outputs/6_three_detector_ensemble")

    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    # 극한 안정형 추천 파라미터 기본 세팅
    parser.add_argument("--window-threshold-mode", choices=["f1", "normal_quantile"], default="normal_quantile")
    parser.add_argument("--window-quantile", type=float, default=0.9999)
    parser.add_argument("--window-k", type=int, default=2)
    parser.add_argument("--window-covariance", choices=["empirical", "ledoitwolf", "oas"], default="ledoitwolf")

    parser.add_argument("--sample-threshold-mode", choices=["f1", "normal_quantile"], default="normal_quantile")
    parser.add_argument("--sample-quantile", type=float, default=0.9999)
    parser.add_argument("--sample-agg-k", type=int, default=3)
    parser.add_argument("--sample-agg", choices=["mean", "max"], default="mean")
    parser.add_argument("--sample-feature-set", choices=["raw", "raw_diff", "raw_diff_roll3"], default="raw_diff")
    parser.add_argument("--sample-covariance", choices=["empirical", "ledoitwolf", "oas"], default="ledoitwolf")

    args = parser.parse_args()
    root = Path.cwd()

    normal_path = Path(args.normal_path)
    fault_path = Path(args.fault_path)
    win1_path = Path(args.window_10)
    win05_path = Path(args.window_05)

    if not normal_path.exists() or not fault_path.exists():
        raise FileNotFoundError("입력 데이터 파일 경로를 확인해주세요.")

    mod5_base = load_module(root / args.base_script, "kamp_5_base_for_ensemble")
    mod5 = load_module(root / args.script_5_2, "kamp_5_2_for_ensemble")
    _mod53 = load_module(root / args.script_5_3, "kamp_5_3_for_ensemble")

    window_covariance = args.window_covariance
    sample_covariance = args.sample_covariance

    raw_normal = load_raw(normal_path, "normal")
    raw_fault = load_raw(fault_path, "fault")
    raw = pd.concat([raw_normal, raw_fault], ignore_index=True).reset_index(drop=True)

    x_sample, _ = build_sample_features(raw, args.sample_feature_set)
    x_sample = signed_log1p(x_sample)

    feature_cols = infer_window_feature_cols([win1_path / "model_windows.csv", win05_path / "model_windows.csv"])
    print(f"Window features ({len(feature_cols)}): {feature_cols}")

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("KAMPact - 3 Detector Ensemble (Detailed Fault Tracking)")
    print("=" * 90)
    print(f"Window threshold : {args.window_threshold_mode} ({args.window_quantile}), k={args.window_k}")
    print(f"Sample threshold : {args.sample_threshold_mode} ({args.sample_quantile}), agg={args.sample_agg}{args.sample_agg_k}")
    print(f"Folds / repeats  : {args.n_splits} / {args.repeats}\n")

    all_rows = []
    all_thresholds = []
    all_event_details = []

    for r in tqdm(range(args.repeats), desc="Total Progress (Repeats)"):
        seed = args.seed + r

        repeat_df, threshold_df, _, event_df = run_repeat(
            raw=raw,
            x_sample=x_sample,
            feature_cols=feature_cols,
            mod5=mod5,
            window_paths=[win1_path / "model_windows.csv", win05_path / "model_windows.csv"],
            n_splits=args.n_splits,
            seed=seed,
            window_threshold_mode=args.window_threshold_mode,
            window_quantile=args.window_quantile,
            window_k=args.window_k,
            sample_threshold_mode=args.sample_threshold_mode,
            sample_quantile=args.sample_quantile,
            sample_agg_k=args.sample_agg_k,
            sample_agg_how=args.sample_agg,
            sample_feature_set=args.sample_feature_set,
        )

        all_rows.append(repeat_df)
        threshold_df["repeat_seed"] = seed
        all_thresholds.append(threshold_df)
        all_event_details.append(event_df)

        tqdm.write(f"[repeat {r + 1}/{args.repeats}] seed={seed}")
        tqdm.write(
            repeat_df[
                [
                    "combo", "event_detection_rate", "delay_mean_sec",
                    "sample_f1", "normal_fa_cycle_rate", "normal_fa_episodes",
                ]
            ].to_string(index=False)
        )
        tqdm.write("")

    results = pd.concat(all_rows, ignore_index=True)
    events_all = pd.concat(all_event_details, ignore_index=True)
    summary_rows = []

    for combo, g in results.groupby("combo", sort=False):
        row = {"combo": combo, "repeats": len(g), "n_events": int(g["n_events"].iloc[0])}
        for col in [
            "event_detection_rate", "delay_mean_sec", "sample_f1",
            "normal_fa_cycle_rate", "normal_fa_episodes",
            "idle_fa_cycle_rate", "idle_fa_episodes",
        ]:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    print("=" * 105)
    print("SUMMARY (mean ± std)")
    print("=" * 105)

    show = summary.copy()
    show["event%"] = (show["event_detection_rate_mean"] * 100).round(2).astype(str) + " ± " + (show["event_detection_rate_std"] * 100).round(2).astype(str)
    show["delay(s)"] = show["delay_mean_sec_mean"].round(3).astype(str) + " ± " + show["delay_mean_sec_std"].round(3).astype(str)
    show["sample_F1"] = show["sample_f1_mean"].round(4).astype(str) + " ± " + show["sample_f1_std"].round(4).astype(str)
    show["normal_FA_cycle%"] = (show["normal_fa_cycle_rate_mean"] * 100).round(2).astype(str) + " ± " + (show["normal_fa_cycle_rate_std"] * 100).round(2).astype(str)
    show["normal_episodes"] = show["normal_fa_episodes_mean"].round(1).astype(str) + " ± " + show["normal_fa_episodes_std"].round(1).astype(str)

    print(
        show[
            [
                "combo", "event%", "delay(s)", "sample_F1",
                "normal_FA_cycle%", "normal_episodes",
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------------------
    # OR_3_P2 상세 분석 블록 추가
    # -------------------------------------------------------------
    print("\n" + "=" * 105)
    print("🔍 [OR_3_P2] Fault Segment Detailed Analysis (Across 5 repeats)")
    print("=" * 105)
    
    or3p2_events = events_all[events_all["combo"] == "OR_3_P2"]
    
    # 각 group_id(고장 세그먼트)별 통계 집계
    fault_stats = or3p2_events.groupby("group_id").agg(
        total_repeats=("detected", "count"),
        detected_count=("detected", "sum"),
        avg_delay_sec=("delay_sec", "mean"),
        n_samples=("n_samples", "mean") # n_samples는 매 반복마다 동일하므로 mean 사용
    ).reset_index()
    
    fault_stats["missed_count"] = fault_stats["total_repeats"] - fault_stats["detected_count"]
    fault_stats["detection_rate_%"] = (fault_stats["detected_count"] / fault_stats["total_repeats"]) * 100

    # 놓친 횟수(missed_count) 기준 내림차순, 그다음 group_id 오름차순 정렬
    fault_stats = fault_stats.sort_values(by=["missed_count", "group_id"], ascending=[False, True])

    # 출력 포맷 맞추기 (float 소수점 3자리)
    print(fault_stats.to_string(index=False, float_format="%.3f"))

    results.to_csv(output_dir / "ensemble_repeat_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "ensemble_summary.csv", index=False, encoding="utf-8-sig")
    events_all.to_csv(output_dir / "ensemble_event_details.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {output_dir / 'ensemble_event_details.csv'}")

if __name__ == "__main__":
    main()