"""
KAMPact - Isolation Forest baseline under the final streaming-CV protocol
--------------------------------------------------------------------------
7_isolation_forest_baseline.py 는 별도 protocol(1.0s window, F1 기준 threshold,
단일 split)이라 최종 모델과 직접 비교할 수 없었다. 이 스크립트는 그 문제를 없앤다.

동일하게 맞추는 것
-----------------
- 데이터 전처리 / segment / Idle 분류 : 12_streaming_cv.preprocess_raw 그대로
- fold 분할 (5 folds x 5 repeats, seed = base + repeat) : 12_streaming_cv 함수 그대로
- train / calibration : normal-operation segment only (Idle 제외)
- threshold : calibration normal score 의 q-quantile (여러 q 를 한 번에 평가)
- causal streaming : StreamingDetector.step() 을 row 단위로 실행, segment 경계에서 reset
- 구조 : 1.0s window + 0.5s window + sample-level -> OR_3 -> P2
- 지표 : 12_streaming_cv.evaluate_metrics 그대로

달라지는 것은 anomaly score 모델 하나뿐이다.
    Mahalanobis(Ledoit-Wolf)  ->  Isolation Forest

여러 quantile 을 빠르게 평가하는 방법
------------------------------------
quantile 은 threshold 만 바꾸고 모델/점수는 바꾸지 않는다. 가장 오래 걸리는 것은
row 단위 streaming 점수 계산이므로 fold 마다 다음과 같이 한다.

1. quantile 마다 detector.fit() 을 호출해 threshold 만 얻는다 (fit 은 상대적으로 빠르다).
2. 가장 큰 quantile 의 detector 로 streaming 을 *한 번만* 실행하며 row 별 score 를 저장한다.
3. 나머지 quantile 의 alarm 은 저장된 score 와 threshold 로 계산한다
   (sample mean-3 유효 구간, window 유효 구간, OR_3, P2 규칙은 step() 과 동일).
4. 2번에서 detector 가 직접 낸 alarm 과 3번 방식으로 계산한 alarm 이 모든 row 에서
   일치하는지 매 fold 검사한다. 하나라도 다르면 즉시 중단한다.

Usage
-----
# 기본: q = 0.99, 0.995, 0.999, 0.9995, 0.9999 전부, repeat 5회
python src/7_2_isolation_forest_streaming_cv.py

# 빠른 동작 확인
python src/7_2_isolation_forest_streaming_cv.py --repeats 1

# IF + Mahalanobis 를 같은 실행에서 비교
python src/7_2_isolation_forest_streaming_cv.py --models both

Outputs  (repeat 하나가 끝날 때마다 갱신되므로 중간에 멈춰도 완료된 repeat 결과가 남는다)
-------
outputs/7_2_isolation_forest_streaming_cv/
    if_streaming_cv_quantile_sweep.csv    # OR_3+P2 만, quantile x model (보고서용)
    if_streaming_cv_comparison.csv        # 모든 quantile x 구성 (문자열 표)
    if_streaming_cv_summary.csv           # 숫자 평균/표준편차
    if_streaming_cv_repeat_metrics.csv    # repeat 별 원자료
    if_streaming_cv_event_details.csv     # event 별 탐지 여부 / delay
    if_streaming_cv_thresholds.csv        # fold 별 threshold
    config.json                           # 인자 + 라이브러리 버전
    q_0.99/ ... q_0.9999/                 # quantile 별 동일 구성 파일
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

try:
    from tqdm import tqdm
except ImportError:  # tqdm 이 없어도 실행되도록 최소 대체 클래스 제공
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            pass

        def update(self, n: int = 1) -> None:
            pass

        def set_postfix_str(self, *args, **kwargs) -> None:
            pass

        def close(self) -> None:
            pass

        @staticmethod
        def write(msg: str) -> None:
            print(msg)

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 파일명이 숫자로 시작하므로 importlib 로 가져온다.
cv = importlib.import_module("12_streaming_cv")
from core_detector import StreamingDetector  # noqa: E402


# 최종 배포 구성 (12_streaming_cv.py 와 동일)
WIN1_SIZE = 10
WIN05_SIZE = 5
SAMPLE_AGG_K = 3
PERSISTENCE_P = 2

DEFAULT_QUANTILES = [0.99, 0.995, 0.999, 0.9995, 0.9999]

MODEL_LABEL = {
    "mahalanobis": "Ledoit-Wolf Mahalanobis",
    "iforest": "Isolation Forest",
}

# 구성 요소별 최종 alarm 으로 사용할 alarm 컬럼
VARIANTS = {
    "1.0s Window": "win1_alarm",
    "0.5s Window": "win05_alarm",
    "Sample-level": "sample_alarm",
    "OR_3": "or3_base",
    "OR_3 + P2": "final_alarm",
}
HEADLINE_VARIANT = "OR_3 + P2"

METRIC_COLS = [
    "event_detection_rate",
    "delay_mean_sec",
    "sample_f1",
    "sample_precision",
    "sample_recall",
    "normal_fa_cycle_rate",
    "normal_fa_episodes",
    "idle_fa_cycle_rate",
    "idle_fa_episodes",
]

ALARM_COLS = ["sample_alarm", "win05_alarm", "win1_alarm", "or3_base", "final_alarm"]


# ============================================================
# Isolation Forest 를 StreamingDetector 에 끼워 넣기
# ============================================================

class StreamingIFDetector(StreamingDetector):
    """StreamingDetector 와 동일한 전처리/버퍼/threshold/OR_3/P2 로직을 쓰되
    anomaly score 만 Isolation Forest 로 바꾼 detector.

    부모 클래스의 fit()/step()은 모델 생성을 self._fit_ledoitwolf(),
    점수 계산을 self._mahalanobis_score() 로 호출한다. 두 메서드만 덮어쓰면
    나머지(signed log1p, 16 feature, mean-3, gap reset, P2)는 그대로 재사용된다.
    메서드 이름은 부모와 맞추기 위해 유지했으며 내용은 Isolation Forest 이다.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_samples: str | int | float = "auto",
        max_features: float = 1.0,
        **kwargs,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.max_features = float(max_features)
        super().__init__(**kwargs)

    def _fit_ledoitwolf(self, x: np.ndarray) -> IsolationForest:  # type: ignore[override]
        if x.ndim != 2 or len(x) < 2:
            raise ValueError("Need at least two training samples.")

        model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            contamination="auto",  # threshold 는 calibration quantile 로 따로 정한다.
            random_state=self.random_state,
            n_jobs=1,  # row 단위 호출이므로 병렬화 오버헤드를 피한다.
        )
        model.fit(x)
        return model

    def _mahalanobis_score(  # type: ignore[override]
        self,
        x: np.ndarray,
        model: IsolationForest,
    ) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        # score_samples: 클수록 정상. 부호를 뒤집어 "클수록 이상"으로 맞춘다.
        return -model.score_samples(x)


def make_detector(
    model_name: str,
    args: argparse.Namespace,
    seed: int,
    quantile: float,
) -> StreamingDetector:
    common = dict(
        win1_size=WIN1_SIZE,
        win05_size=WIN05_SIZE,
        sample_agg_k=SAMPLE_AGG_K,
        persistence_p=PERSISTENCE_P,
        gap_threshold_sec=cv.GAP_THRESHOLD_SEC,
        quantile_threshold=quantile,
        random_state=seed,
    )

    if model_name == "mahalanobis":
        return StreamingDetector(**common)

    return StreamingIFDetector(
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        max_features=args.max_features,
        **common,
    )


# ============================================================
# Streaming (1회) + quantile 별 alarm 계산
# ============================================================

def stream_scores(detector: StreamingDetector, test_df: pd.DataFrame) -> pd.DataFrame:
    """12_streaming_cv.run_streaming_test 와 같은 순서로 row 단위 streaming 을 실행하되
    row 별 score 와 segment 내 위치(pos)를 저장한다.

    반환 행 순서는 (group_id 정렬, segment 내 시간순) 이다. derive_alarms 가 이 순서에
    의존하므로 정렬을 바꾸지 않는다.
    """
    rows: list[dict[str, object]] = []

    for gid, group in test_df.groupby("group_id", sort=True):
        group = group.sort_values(cv.TIME_COL)
        detector.reset_stream()

        for pos, (idx, row) in enumerate(group.iterrows()):
            ts = pd.Timestamp(row[cv.TIME_COL])
            res = detector.step(
                ts,
                float(row[cv.SENSORS[0]]),
                float(row[cv.SENSORS[1]]),
                float(row[cv.SENSORS[2]]),
            )
            rows.append(
                {
                    "raw_index": int(idx),
                    "group_id": str(gid),
                    "kind": str(row["kind"]),
                    "label": int(row["label"]),
                    "TimeStamp": ts,
                    "pos": int(pos),
                    "sample_score": float(res["sample_score"]),
                    "win05_score": float(res["win05_score"]),
                    "win1_score": float(res["win1_score"]),
                    "step_sample_alarm": bool(res["sample_alarm"]),
                    "step_win05_alarm": bool(res["win05_alarm"]),
                    "step_win1_alarm": bool(res["win1_alarm"]),
                    "step_or3_base": bool(res["or3_base"]),
                    "step_final_alarm": bool(res["final_alarm"]),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No streaming test results were produced.")
    return out.reset_index(drop=True)


def derive_alarms(
    base: pd.DataFrame,
    thresholds: dict[str, float],
    k: int = SAMPLE_AGG_K,
    p: int = PERSISTENCE_P,
    win05: int = WIN05_SIZE,
    win1: int = WIN1_SIZE,
) -> pd.DataFrame:
    """저장된 score 와 threshold 로 StreamingDetector.step() 과 같은 alarm 을 계산한다.

    step() 의 규칙:
      - sample alarm : 최근 k 개 score 가 모였을 때(segment 내 pos >= k-1) mean >= thr
      - window alarm : buffer 길이 >= window 크기일 때(pos >= size-1) score >= thr
      - OR_3         : 세 alarm 의 OR
      - P2           : OR_3 이력(초기값 False)의 최근 p 개가 모두 True
    base 는 (segment, 시간순) 으로 정렬되어 있어야 한다.
    """
    pos = base["pos"].to_numpy()

    sample = (pos >= k - 1) & (base["sample_score"].to_numpy() >= thresholds["sample"])
    w05 = (pos >= win05 - 1) & (base["win05_score"].to_numpy() >= thresholds["win05"])
    w1 = (pos >= win1 - 1) & (base["win1_score"].to_numpy() >= thresholds["win1"])
    or3 = sample | w05 | w1

    final = np.ones(len(or3), dtype=bool)
    for j in range(p):
        if j == 0:
            shifted = or3
        else:
            shifted = np.r_[np.zeros(j, dtype=bool), or3[:-j]]
        # pos < j 이면 해당 이력은 reset 직후의 초기값(False)이다.
        final &= shifted & (pos >= j)

    out = base[["raw_index", "group_id", "kind", "label", "TimeStamp"]].copy()
    out["sample_alarm"] = sample
    out["win05_alarm"] = w05
    out["win1_alarm"] = w1
    out["or3_base"] = or3
    out["final_alarm"] = final
    return out


def verify_against_step(derived: pd.DataFrame, base: pd.DataFrame, context: str) -> None:
    """detector 가 직접 낸 alarm 과 score 기반 계산이 모든 row 에서 같은지 확인한다."""
    for col in ALARM_COLS:
        a = derived[col].to_numpy(dtype=bool)
        b = base["step_" + col].to_numpy(dtype=bool)
        if not np.array_equal(a, b):
            n_diff = int((a != b).sum())
            raise RuntimeError(
                f"[{context}] '{col}' mismatch between step() and score-derived alarms "
                f"({n_diff} rows). Quantile shortcut is not equivalent; aborting."
            )


def check_threshold_monotone(
    thresholds_by_q: dict[float, dict[str, float]],
    quantiles: list[float],
    context: str,
) -> None:
    """같은 모델이라면 quantile 이 커질수록 threshold 는 줄어들지 않는다."""
    for key in ("sample", "win05", "win1"):
        vals = [thresholds_by_q[q][key] for q in quantiles]
        for a, b in zip(vals, vals[1:]):
            if b < a - 1e-9 * max(1.0, abs(a)):
                raise RuntimeError(
                    f"[{context}] '{key}' threshold decreases as quantile increases "
                    f"({a} -> {b}). The per-quantile fits are not consistent."
                )


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate_variants(
    model_name: str,
    quantile: float,
    repeat: int,
    seed: int,
    fold_results: list[pd.DataFrame],
    raw: pd.DataFrame,
) -> tuple[list[dict], list[pd.DataFrame]]:
    """한 repeat 의 모든 fold 결과를 합쳐 구성별 지표를 계산한다.

    12_streaming_cv.main() 의 repeat 집계 방식과 동일하다
    (한 repeat 에서 모든 segment 는 정확히 한 번 test 로 쓰인다).
    """
    res_all = (
        pd.concat(fold_results, ignore_index=True)
        .drop_duplicates("raw_index")
        .sort_values("raw_index")
    )
    test_rows = raw.loc[
        raw["group_id"].astype(str).isin(res_all["group_id"].astype(str))
    ].copy()

    metric_rows: list[dict] = []
    event_frames: list[pd.DataFrame] = []

    for variant, alarm_col in VARIANTS.items():
        r = res_all.copy()
        # evaluate_metrics 는 'final_alarm' 컬럼을 최종 alarm 으로 본다.
        r["final_alarm"] = r[alarm_col].astype(bool)

        metrics, events = cv.evaluate_metrics(r, test_rows)
        metrics.update(
            {
                "model": model_name,
                "quantile": quantile,
                "variant": variant,
                "repeat": repeat,
                "seed": seed,
            }
        )
        metric_rows.append(metrics)

        if not events.empty:
            events = events.copy()
            events["model"] = model_name
            events["quantile"] = quantile
            events["variant"] = variant
            events["repeat"] = repeat
            events["seed"] = seed
            event_frames.append(events)

    return metric_rows, event_frames


def summarize(repeat_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, quantile, variant), g in repeat_df.groupby(
        ["model", "quantile", "variant"], sort=False
    ):
        row: dict[str, object] = {
            "model": model,
            "quantile": quantile,
            "variant": variant,
            "repeats": int(len(g)),
            "n_events": int(g["n_events"].iloc[0]),
        }
        for c in METRIC_COLS:
            row[c + "_mean"] = float(g[c].mean())
            row[c + "_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(mean: float, std: float, scale: float = 1.0, nd: int = 2, suffix: str = "") -> str:
    if pd.isna(mean):
        return "n/a"
    if pd.isna(std):
        return f"{mean * scale:.{nd}f}{suffix}"
    return f"{mean * scale:.{nd}f} ± {std * scale:.{nd}f}{suffix}"


def comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            {
                "Model": MODEL_LABEL.get(r["model"], r["model"]),
                "Quantile": f"{r['quantile']:g}",
                "Configuration": r["variant"],
                "Event Detection": _fmt(
                    r["event_detection_rate_mean"], r["event_detection_rate_std"], 100, 2, "%"
                ),
                "Delay (s)": _fmt(r["delay_mean_sec_mean"], r["delay_mean_sec_std"], 1, 3),
                "Sample F1": _fmt(r["sample_f1_mean"], r["sample_f1_std"], 1, 4),
                "Normal FA Cycle": _fmt(
                    r["normal_fa_cycle_rate_mean"], r["normal_fa_cycle_rate_std"], 100, 2, "%"
                ),
                "Idle FA Cycle": _fmt(
                    r["idle_fa_cycle_rate_mean"], r["idle_fa_cycle_rate_std"], 100, 2, "%"
                ),
                "Normal FA Episodes/repeat": f"{r['normal_fa_episodes_mean']:.1f}",
            }
        )
    return pd.DataFrame(rows)


def _to_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_outputs(
    out_dir: Path,
    quantiles: list[float],
    metric_rows: list[dict],
    event_frames: list[pd.DataFrame],
    threshold_rows: list[dict],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """현재까지 완료된 repeat 의 결과를 모두 저장한다 (repeat 마다 호출)."""
    repeat_df = pd.DataFrame(metric_rows)
    summary_df = summarize(repeat_df)
    table_df = comparison_table(summary_df)
    event_df = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    threshold_df = pd.DataFrame(threshold_rows)

    sweep_df = table_df[table_df["Configuration"].eq(HEADLINE_VARIANT)].drop(
        columns=["Configuration"]
    )

    _to_csv(repeat_df, out_dir / "if_streaming_cv_repeat_metrics.csv")
    _to_csv(summary_df, out_dir / "if_streaming_cv_summary.csv")
    _to_csv(table_df, out_dir / "if_streaming_cv_comparison.csv")
    _to_csv(sweep_df, out_dir / "if_streaming_cv_quantile_sweep.csv")
    _to_csv(threshold_df, out_dir / "if_streaming_cv_thresholds.csv")
    if not event_df.empty:
        _to_csv(event_df, out_dir / "if_streaming_cv_event_details.csv")

    # quantile 별 폴더 (outputs/12_streaming_cv/q_* 구조와 맞춤)
    for q in quantiles:
        qdir = out_dir / f"q_{q:g}"
        qdir.mkdir(parents=True, exist_ok=True)
        q_repeat = repeat_df[repeat_df["quantile"].eq(q)]
        q_summary = summary_df[summary_df["quantile"].eq(q)]
        _to_csv(q_repeat, qdir / "if_streaming_cv_repeat_metrics.csv")
        _to_csv(q_summary, qdir / "if_streaming_cv_summary.csv")
        _to_csv(comparison_table(q_summary), qdir / "if_streaming_cv_comparison.csv")
        _to_csv(threshold_df[threshold_df["quantile"].eq(q)], qdir / "if_streaming_cv_thresholds.csv")
        if not event_df.empty:
            _to_csv(event_df[event_df["quantile"].eq(q)], qdir / "if_streaming_cv_event_details.csv")

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)

    return table_df, sweep_df


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolation Forest baseline under the KAMPact streaming-CV protocol."
    )
    parser.add_argument("--normal-path", default="data/press_data_normal_with_idle.csv")
    parser.add_argument("--fault-path", default="data/outlier_data.csv")
    parser.add_argument("--output-dir", default="outputs/7_2_isolation_forest_streaming_cv")
    parser.add_argument(
        "--models",
        choices=["iforest", "mahalanobis", "both"],
        default="iforest",
        help="'both' 는 같은 fold/seed 에서 두 모델을 함께 평가한다.",
    )
    # 12_streaming_cv.py 와 동일한 기본값 -> 기존 Mahalanobis 결과와 직접 비교 가능
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=DEFAULT_QUANTILES,
        help="평가할 threshold quantile 목록 (한 번의 streaming 으로 모두 평가).",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=None,
        help="(이전 버전 호환) 지정하면 --quantiles 대신 이 값 하나만 평가한다.",
    )
    # Isolation Forest hyper-parameters
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-samples", default="auto")
    parser.add_argument("--max-features", type=float, default=1.0)
    args = parser.parse_args()

    if args.n_splits < 3:
        raise ValueError("--n-splits must be >= 3")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    quantiles = [args.quantile] if args.quantile is not None else list(args.quantiles)
    quantiles = sorted(set(float(q) for q in quantiles))
    if not quantiles or any(not 0.0 < q < 1.0 for q in quantiles):
        raise ValueError("--quantiles must be between 0 and 1")
    args.quantiles = quantiles

    # max_samples 는 'auto' | 정수 | 실수 를 받는다.
    ms = args.max_samples
    if isinstance(ms, str) and ms != "auto":
        ms = float(ms) if "." in ms else int(ms)
    args.max_samples = ms

    models = ["mahalanobis", "iforest"] if args.models == "both" else [args.models]
    primary_q = quantiles[-1]  # streaming 을 실제로 돌릴 detector 의 quantile

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normal = cv.preprocess_raw(pd.read_csv(args.normal_path), "normal")
    fault = cv.preprocess_raw(pd.read_csv(args.fault_path), "fault")
    raw = pd.concat([normal, fault], ignore_index=True)
    raw = raw.sort_values(cv.TIME_COL).reset_index(drop=True)

    print("=" * 88)
    print("KAMPact - Isolation Forest baseline (streaming CV protocol)")
    print("=" * 88)
    print(f"Rows          : {len(raw):,}")
    print(f"Fault events  : {raw.loc[raw['kind'].eq('fault'), 'group_id'].nunique():,}")
    print(f"Folds/repeats : {args.n_splits}/{args.repeats}")
    print(f"Quantiles     : {', '.join(f'{q:g}' for q in quantiles)}")
    print(f"Models        : {', '.join(models)}")
    if "iforest" in models:
        print(
            f"IForest       : n_estimators={args.n_estimators}, "
            f"max_samples={args.max_samples}, max_features={args.max_features}"
        )
    print()

    metric_rows: list[dict] = []
    event_frames: list[pd.DataFrame] = []
    threshold_rows: list[dict] = []
    t_start = time.perf_counter()

    def make_config(done_repeats: int) -> dict:
        return vars(args) | {
            "models_evaluated": models,
            "repeats_completed": done_repeats,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "elapsed_sec": round(time.perf_counter() - t_start, 1),
            "protocol": (
                "StreamingDetector structure (OR_3 + P2); normal-only train/calibration; "
                "held-out segment test; shared fold map with 12_streaming_cv.py; "
                "one streaming pass per fold, per-quantile alarms derived from stored scores "
                "and verified against step() at the largest quantile"
            ),
        }

    pbar = tqdm(
        total=args.repeats * args.n_splits * len(models),
        desc="Streaming CV",
        unit="fold",
        dynamic_ncols=True,
    )

    table_df = sweep_df = pd.DataFrame()

    for repeat in range(args.repeats):
        seed = args.seed + repeat
        fold_map = cv.build_fold_map(raw, args.n_splits, seed)
        fold_results: dict[str, dict[float, list[pd.DataFrame]]] = {
            m: {q: [] for q in quantiles} for m in models
        }

        for fold in range(args.n_splits):
            train_df, cal_df, test_df = cv.split_by_groups(
                raw, fold_map, fold, args.n_splits
            )

            for m in models:
                t0 = time.perf_counter()
                tag = f"{m} | repeat {repeat + 1}/{args.repeats} fold {fold + 1}/{args.n_splits}"

                # 1) quantile 마다 fit -> threshold 만 사용
                pbar.set_postfix_str(f"{tag} | fitting")
                thresholds_by_q: dict[float, dict[str, float]] = {}
                primary_detector: StreamingDetector | None = None
                for q in quantiles:
                    detector = make_detector(m, args, seed, q)
                    fit_info = detector.fit(train_df, calibration_df=cal_df)
                    thresholds_by_q[q] = {k: float(v) for k, v in fit_info["thresholds"].items()}
                    if q == primary_q:
                        primary_detector = detector

                check_threshold_monotone(thresholds_by_q, quantiles, tag)
                assert primary_detector is not None

                # 2) 가장 큰 quantile 의 detector 로 streaming 1회
                pbar.set_postfix_str(f"{tag} | streaming")
                base = stream_scores(primary_detector, test_df)

                # 3) quantile 별 alarm 계산 (+ 4) step() 과 일치 검증)
                for q in quantiles:
                    derived = derive_alarms(base, thresholds_by_q[q])
                    if q == primary_q:
                        verify_against_step(derived, base, tag)
                    fold_results[m][q].append(derived)

                    for scale, thr in thresholds_by_q[q].items():
                        threshold_rows.append(
                            {
                                "model": m,
                                "quantile": q,
                                "repeat": repeat,
                                "seed": seed,
                                "fold": fold,
                                "detector": scale,
                                "threshold": thr,
                            }
                        )

                pbar.set_postfix_str(f"{tag} | last {time.perf_counter() - t0:.0f}s")
                pbar.update(1)

        for m in models:
            for q in quantiles:
                rows, events = evaluate_variants(m, q, repeat, seed, fold_results[m][q], raw)
                metric_rows.extend(rows)
                event_frames.extend(events)

                head = next(r for r in rows if r["variant"] == HEADLINE_VARIANT)
                tqdm.write(
                    f"[repeat {repeat + 1}/{args.repeats}] {m:<11s} q={q:<7g} OR_3+P2 | "
                    f"event={head['event_detection_rate'] * 100:.2f}% "
                    f"delay={head['delay_mean_sec']:.3f}s "
                    f"normal_FA={head['normal_fa_cycle_rate'] * 100:.2f}% "
                    f"idle_FA={head['idle_fa_cycle_rate'] * 100:.2f}%"
                )

        # repeat 하나가 끝날 때마다 저장 -> 중간에 중단돼도 결과가 남는다.
        table_df, sweep_df = write_outputs(
            out_dir, quantiles, metric_rows, event_frames, threshold_rows,
            make_config(repeat + 1),
        )

    pbar.close()

    print("\n" + "=" * 88)
    print(f"QUANTILE SWEEP - {HEADLINE_VARIANT}  (mean ± std over {args.repeats} repeats)")
    print("=" * 88)
    print(sweep_df.to_string(index=False))
    print(f"\n[SAVED] {out_dir}")
    print("구성별(1.0s / 0.5s / sample / OR_3) 전체 표: if_streaming_cv_comparison.csv")


if __name__ == "__main__":
    main()