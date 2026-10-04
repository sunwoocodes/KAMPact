"""
KAMPact - 탐지 지연 분석 (어느 탐지기가, 언제 처음 반응했는가)
------------------------------------------------------------------
최종 모델(OR_3 + P2)의 지연이 왜 event마다 다른지 확인하기 위한 분석 스크립트.

7_2_isolation_forest_streaming_cv.py 의 fold 분할/모델 학습/streaming 실행 함수를 그대로 재사용한다
(12_streaming_cv 와 같은 5x5 group CV, 같은 StreamingDetector). 지표 계산 대신 event 마다 아래를 기록한다.

  first_{sample,win05,win1}_sec : 각 탐지기가 처음 경보를 낸 시점 (event 시작 기준, 초)
  first_or3_sec                 : OR_3 후보 경보가 처음 나온 시점
  delay_sec                     : 최종 경보(P2) 시점 = 12_streaming_cv 의 delay 와 같은 정의
  leading                       : 가장 먼저 반응한 탐지기 (동시면 'sample+win05' 처럼 표기)
  max_ratio_first1s_{...}       : event 시작 후 1.0초 안에서 (점수 / 임계값) 의 최댓값
                                  (1.0 이상이면 임계값을 넘은 것, 0.8 이면 80% 까지 접근)
  max_ratio_event_{...}         : event 전체 구간에서의 최댓값

P2 를 적용했을 때 각 탐지기가 최종 경보를 낼 수 있는 가장 이른 시점은 구조상 정해져 있다.
  Sample-level : 3번째 샘플부터 점수 계산 -> 연속 2회 -> 0.3초
  0.5s Window  : 5번째 샘플부터            -> 연속 2회 -> 0.5초
  1.0s Window  : 10번째 샘플부터           -> 연속 2회 -> 1.0초
(샘플 간격 0.1초 기준). 이 값보다 늦은 지연은 그 시점에 임계값을 넘지 못했다는 뜻이다.

Usage
-----
python src/7_3_delay_analysis.py --repeats 1      # 빠른 확인
python src/7_3_delay_analysis.py                  # 5회 (보고서용)
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

m72 = importlib.import_module("7_2_isolation_forest_streaming_cv")
cv = m72.cv
tqdm = m72.tqdm

NOMINAL_DT = 0.1

# 탐지기 이름 -> (step 출력의 alarm 컬럼, score 컬럼, 점수가 계산되기 시작하는 segment 내 위치)
DETECTORS = {
    "sample": ("step_sample_alarm", "sample_score", m72.SAMPLE_AGG_K - 1),
    "win05": ("step_win05_alarm", "win05_score", m72.WIN05_SIZE - 1),
    "win1": ("step_win1_alarm", "win1_score", m72.WIN1_SIZE - 1),
}


def earliest_p_time(valid_pos: int) -> float:
    """P 연속 조건을 만족해 최종 경보가 나올 수 있는 가장 이른 시점(초)."""
    return round((valid_pos + m72.PERSISTENCE_P - 1) * NOMINAL_DT, 3)


def analyze_event(g: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, object]:
    g = g.sort_values("pos")
    t = (g["TimeStamp"] - g["TimeStamp"].iloc[0]).dt.total_seconds().to_numpy()
    pos = g["pos"].to_numpy()

    final = g["step_final_alarm"].to_numpy(dtype=bool)
    or3 = g["step_or3_base"].to_numpy(dtype=bool)

    out: dict[str, object] = {
        "n_samples": int(len(g)),
        "duration_sec": float(t[-1]),
        "detected": bool(final.any()),
        "delay_sec": float(t[final.argmax()]) if final.any() else np.nan,
        "first_or3_sec": float(t[or3.argmax()]) if or3.any() else np.nan,
    }

    firsts: dict[str, float] = {}
    for name, (alarm_col, score_col, valid_pos) in DETECTORS.items():
        a = g[alarm_col].to_numpy(dtype=bool)
        valid = pos >= valid_pos
        ratio = np.where(valid, g[score_col].to_numpy() / thresholds[name], np.nan)
        early = valid & (t <= 1.0)

        firsts[name] = float(t[a.argmax()]) if a.any() else np.nan
        out[f"first_{name}_sec"] = firsts[name]
        out[f"alarm_rows_{name}"] = int(a.sum())
        out[f"max_ratio_first1s_{name}"] = float(np.nanmax(ratio[early])) if early.any() else np.nan
        out[f"max_ratio_event_{name}"] = float(np.nanmax(ratio)) if valid.any() else np.nan

    reached = {k: v for k, v in firsts.items() if not np.isnan(v)}
    if reached:
        first_t = min(reached.values())
        out["leading"] = "+".join(k for k, v in reached.items() if abs(v - first_t) < 1e-9)
    else:
        out["leading"] = "none"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-event detection delay decomposition.")
    parser.add_argument("--normal-path", default="data/press_data_normal_with_idle.csv")
    parser.add_argument("--fault-path", default="data/outlier_data.csv")
    parser.add_argument("--output-dir", default="outputs/7_3_delay_analysis")
    parser.add_argument("--model", choices=["mahalanobis", "iforest"], default="mahalanobis")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quantile", type=float, default=0.9999)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-samples", default="auto")
    parser.add_argument("--max-features", type=float, default=1.0)
    args = parser.parse_args()

    if args.n_splits < 3 or args.repeats < 1:
        raise ValueError("--n-splits must be >= 3 and --repeats must be >= 1")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normal = cv.preprocess_raw(pd.read_csv(args.normal_path), "normal")
    fault = cv.preprocess_raw(pd.read_csv(args.fault_path), "fault")
    raw = pd.concat([normal, fault], ignore_index=True)
    raw = raw.sort_values(cv.TIME_COL).reset_index(drop=True)

    print("=" * 88)
    print("KAMPact - per-event delay decomposition")
    print("=" * 88)
    print(f"Model/quantile : {args.model} / {args.quantile}")
    print(f"Folds/repeats  : {args.n_splits}/{args.repeats}")
    print("Earliest feasible final-alarm time with P2 (sec): " + ", ".join(
        f"{name}={earliest_p_time(vp):.1f}" for name, (_, _, vp) in DETECTORS.items()
    ))
    print()

    rows: list[dict[str, object]] = []
    pbar = tqdm(total=args.repeats * args.n_splits, desc="Delay analysis", unit="fold", dynamic_ncols=True)
    t_start = time.perf_counter()

    for repeat in range(args.repeats):
        seed = args.seed + repeat
        fold_map = cv.build_fold_map(raw, args.n_splits, seed)

        for fold in range(args.n_splits):
            train_df, cal_df, test_df = cv.split_by_groups(raw, fold_map, fold, args.n_splits)

            detector = m72.make_detector(args.model, args, seed, args.quantile)
            fit_info = detector.fit(train_df, calibration_df=cal_df)
            thresholds = {k: float(v) for k, v in fit_info["thresholds"].items()}

            base = m72.stream_scores(detector, test_df)
            fault_base = base[base["kind"].eq("fault")]

            for gid, g in fault_base.groupby("group_id", sort=True):
                rec = analyze_event(g, thresholds)
                rec.update({"group_id": gid, "repeat": repeat, "seed": seed, "fold": fold})
                rows.append(rec)

            pbar.update(1)

    pbar.close()

    ev = pd.DataFrame(rows)
    ev.to_csv(out_dir / "delay_event_details.csv", index=False, encoding="utf-8-sig")

    # event 별 요약
    g = ev.groupby("group_id")
    summary = g.agg(
        n_samples=("n_samples", "first"),
        duration_sec=("duration_sec", "first"),
        detected=("detected", "sum"),
        repeats=("detected", "size"),
        mean_delay=("delay_sec", "mean"),
        first_sample_med=("first_sample_sec", "median"),
        first_win05_med=("first_win05_sec", "median"),
        first_win1_med=("first_win1_sec", "median"),
        ratio1s_sample=("max_ratio_first1s_sample", "median"),
        ratio1s_win05=("max_ratio_first1s_win05", "median"),
        ratio1s_win1=("max_ratio_first1s_win1", "median"),
    )
    summary["leading_mode"] = g["leading"].agg(lambda s: s.mode().iloc[0])
    summary = summary.sort_values(["mean_delay", "duration_sec"], na_position="first")
    summary.round(3).to_csv(out_dir / "delay_event_summary.csv", encoding="utf-8-sig")

    # 최초 반응 탐지기별 지연 (탐지된 건만)
    det = ev[ev["detected"]].copy()
    lead = det.groupby("leading").agg(
        evaluations=("delay_sec", "size"),
        events=("group_id", "nunique"),
        mean_delay=("delay_sec", "mean"),
        min_delay=("delay_sec", "min"),
        max_delay=("delay_sec", "max"),
    )
    lead["share_pct"] = lead["evaluations"] / lead["evaluations"].sum() * 100
    lead.round(3).to_csv(out_dir / "delay_leading_detector_summary.csv", encoding="utf-8-sig")

    # 지연이 구조상 가장 이른 시점과 같은지
    feasible = sorted({earliest_p_time(vp) for _, _, vp in DETECTORS.values()})
    at_floor = det["delay_sec"].round(1).isin([round(x, 1) for x in feasible])
    print("\n" + "=" * 88)
    print("EVENT SUMMARY (정렬: 평균 지연)")
    print("=" * 88)
    pd.set_option("display.width", 220)
    print(summary.round(2).to_string())
    print("\n" + "=" * 88)
    print("가장 먼저 반응한 탐지기별 지연 (탐지된 건)")
    print("=" * 88)
    print(lead.round(3).to_string())
    print(
        f"\n탐지 {len(det)}건 중 지연이 구조상 최소 시점({', '.join(f'{x:.1f}' for x in feasible)}초)과 "
        f"같은 건: {int(at_floor.sum())}건 ({at_floor.mean() * 100:.1f}%)"
    )
    print(f"[SAVED] {out_dir}  (elapsed {time.perf_counter() - t_start:.0f}s)")


if __name__ == "__main__":
    main()