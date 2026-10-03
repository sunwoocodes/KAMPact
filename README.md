# KAMPact

**진동·전류 시계열 기반 프레스 유압펌프 이상 조기탐지 및 오경보 분석**

KAMPact는 프레스 유압펌프에서 수집된 **진동·전류 시계열**을 이용해 정상 운전 패턴을 학습하고, 정상 패턴에서 벗어나는 이상을 **가능한 한 빠르게 탐지**하면서 동시에 **정상 구간의 오탐(False Positive)** 과 **고장 구간의 미탐(False Negative)** 을 분석하는 예지보전 프로젝트입니다.

단순한 고장 분류 문제 대신 다음을 함께 봅니다.

```text
정상 패턴 학습
      ↓
다중 시간 스케일 이상탐지
      ↓
Fault Event Detection
      +
Detection Delay
      +
Normal / Idle False Alarm
      +
FP / FN 원인 분석
      ↓
인과적(Causal) 실시간 추론
```

> **Detection Delay 정의:** 원본 데이터에는 물리적 고장이 실제로 발생한 별도의 ground-truth timestamp가 없으므로, 본 프로젝트의 delay는 **데이터에서 정의한 fault segment의 첫 시점부터 최초 alarm까지의 시간**입니다. 실제 물리적 고장 발생 전 예지시간을 의미하지 않습니다.

---

## 1. Problem

대상은 프레스 유압펌프의 다음 3개 센서 시계열입니다.

| Sensor | 의미 |
|---|---|
| `AI0_Vibration` | 진동 센서 0 |
| `AI1_Vibration` | 진동 센서 1 |
| `AI2_Current` | 전류 센서 |

목표는 정상 운전 중의 작은 변화와 fault 패턴을 구분하면서, 현장에서 중요한 두 요소를 함께 확인하는 것입니다.

```text
탐지율을 높이면
→ 오탐이 늘어날 수 있음

오탐을 줄이면
→ 짧은 fault를 놓치거나 delay가 늘어날 수 있음
```

따라서 KAMPact는 **Event Detection + Detection Delay + False Alarm**을 함께 평가하고, 마지막에는 같은 detector를 **행 단위 causal streaming**으로 실행합니다.

---

## 2. Dataset

원본 CSV는 저장소에 포함하지 않습니다.

현재 분석에 사용한 데이터 규모는 다음과 같습니다.

```text
Normal : 20,000 rows
Fault  : 600 rows
Fault events : 21
Nominal sampling interval : 0.1 sec
```

### Input Schema

```text
TimeStamp
AI0_Vibration
AI1_Vibration
AI2_Current
Equipment_state
Idle
```

### Label 규칙

```text
Equipment_state >= 1 → Fault
Equipment_state = 0  → Normal candidate
Idle = 1              → Idle
```

정상 데이터에는 실제 프레스가 동작하지 않는 **Idle 구간**이 포함됩니다. Idle은 정상 운전과 다른 신호 분포를 가질 수 있기 때문에 **모델 학습 및 threshold calibration에서는 제외하고 평가 단계에서 false alarm만 측정**합니다.

---

## 3. Repository Structure

```text
KAMPact/
├─ data/
│  ├─ press_data_normal.csv                 # 사용자가 준비하는 원본 Normal CSV
│  ├─ outlier_data.csv                      # 사용자가 준비하는 원본 Fault CSV
│  └─ press_data_normal_with_idle.csv       # Step 2 생성
│
├─ result/
│  ├─ time_structure_analysis/
│  ├─ modeling_dataset_1.0_0.1/
│  └─ modeling_dataset_0.5_0.1/
│
├─ outputs/
│  ├─ 5_2_mahalanobis_cv/
│  ├─ 5_3_sample_level_cv/
│  ├─ 6_three_detector_ensemble/
│  ├─ 7_isolation_forest_baseline/
│  ├─ 8_fp_fn_analysis/
│  ├─ 9_fn_visualization/
│  ├─ 10_variable_effect_analysis/
│  ├─ 11_realtime_dashboard/
│  ├─ 12_streaming_cv/
│  └─ final_model/
│
├─ src/
│  ├─ 1_visualize_normal_outlier.py
│  ├─ 2_Classification_idle_sections.py
│  ├─ 3_time_structure_analysis.py
│  ├─ 4_make_window_dataset.py
│  ├─ 5_run_mahalanobis.py
│  ├─ 5_2_run_mahalanobis.py
│  ├─ 5_3_run_sample_level_mahalanobis.py
│  ├─ 6_compare_three_detectors.py
│  ├─ 7_isolation_forest_baseline.py
│  ├─ 8_fp_fn_analysis.py
│  ├─ 9_visualize_fn_events.py
│  ├─ 10_variable_effect_analysis.py
│  ├─ 11_realtime_dashboard.py
│  ├─ 12_streaming_cv.py
│  ├─ core_detector.py
│  └─ dashboard_stage_monitor.py
│
├─ requirements.txt
└─ README.md
```

`5_run_mahalanobis.py`는 초기/단일 스케일 실험 코드를 유지한 것이며, 현재 최종 파이프라인의 핵심 window CV에는 `5_2_run_mahalanobis.py`, sample-level CV에는 `5_3_run_sample_level_mahalanobis.py`가 사용됩니다.

---

## 4. End-to-End Pipeline

최종 대시보드는 아래 과정을 한 번에 실행합니다.

```text
01  원본 시계열 분석
02  Idle 라벨 생성
03  시간 구조 / Segment
04  1.0초 Window Dataset
05  0.5초 Window Dataset
06  Window Mahalanobis CV
07  Sample-level Mahalanobis CV
08  3-Detector Ensemble
09  Isolation Forest Baseline
10  FP / FN 분석
11  FN 시각화 + 변수 영향 분석
12  최종 모델 학습 / Calibration
13  Causal Streaming Inference (CSV Replay)
```

대시보드의 **1번 창**은 1~12단계의 실행 현황을 보여주고, 모든 학습·보정이 끝난 뒤 **2번 창**에서 13단계 streaming replay를 실행합니다.

---

## 5. Step 1~3 — Data Understanding

### Step 1. 원본 시계열 분석

`1_visualize_normal_outlier.py`

정상/고장 데이터의 세 센서 시계열을 시각화하고 변동 범위를 확인합니다.

### Step 2. Idle 라벨 생성

`2_Classification_idle_sections.py`

정상 CSV의 실제 Idle 구간을 `Idle=1`로 라벨링합니다.

현재 데이터에서 사용한 Idle 기준 구간은:

```text
2022-07-12 00:59:53.992
~
2022-07-12 01:11:32.588
```

대시보드에서는 동일 로직을 내부 fallback으로도 가지고 있어 `data/press_data_normal_with_idle.csv`가 아직 없더라도 Step 2를 완성할 수 있습니다.

### Step 3. Time Structure / Segment

`3_time_structure_analysis.py`

timestamp gap을 분석하고 다음 기준으로 segment를 나눕니다.

```text
gap > 0.5 sec
→ new segment
```

segment 단위로 분할하여 동일 cycle의 일부가 train/validation/test에 동시에 들어가는 **segment leakage**를 방지합니다.

---

## 6. Step 4~5 — Window Dataset

`4_make_window_dataset.py`가 공통 window feature를 생성합니다.

최종 파이프라인의 두 시간 스케일:

```text
Window = 1.0 sec
Window = 0.5 sec
Step   = 0.1 sec
```

각 센서에서 다음 5개 특징을 계산합니다.

```text
Mean
Std
RMS
Peak-to-Peak
Slope
```

따라서:

```text
3 sensors × 5 features = 15
AI0 ↔ AI1 correlation = 1
--------------------------------
Total = 16 features
```

### 16 Features

```text
AI0_Vibration_mean
AI0_Vibration_std
AI0_Vibration_rms
AI0_Vibration_ptp
AI0_Vibration_slope

AI1_Vibration_mean
AI1_Vibration_std
AI1_Vibration_rms
AI1_Vibration_ptp
AI1_Vibration_slope

AI2_Current_mean
AI2_Current_std
AI2_Current_rms
AI2_Current_ptp
AI2_Current_slope

AI0_AI1_corr
```

window feature의 `slope`는 실제 `TimeStamp`를 사용해 초당 변화율을 계산합니다.

---

## 7. Anomaly Score — Signed Log + Mahalanobis

정상 데이터의 다변량 분포를 학습하고 현재 feature가 정상 분포에서 얼마나 멀리 떨어져 있는지 Mahalanobis distance로 계산합니다.

$$
D^2(x)=(x-\mu)^T\Sigma^{-1}(x-\mu)
$$

feature scale 차이와 큰 값의 영향 완화를 위해 다음 변환을 사용합니다.

$$
x' = sign(x)\log(1+|x|)
$$

최종 covariance estimator는:

```text
Ledoit-Wolf
```

입니다.

지원하는 offline covariance estimator에는 `Empirical Covariance`, `Ledoit-Wolf`, `OAS`가 있으며, 최종 구성과 streaming detector는 Ledoit-Wolf를 사용합니다.

---

## 8. Window Detector

Window detector의 구조는 다음과 같습니다.

```text
Raw Time Series
      ↓
Trailing Window
      ↓
16 Features
      ↓
Signed Log1p
      ↓
Ledoit-Wolf Mahalanobis
      ↓
Normal-only Quantile Threshold
      ↓
Window Alarm
```

최종 배포 detector는:

```text
0.5s Window = 최근 5 samples
1.0s Window = 최근 10 samples
```

을 사용합니다.

각 window는 **현재 sample까지의 과거 데이터만 사용**하며, 실제 timestamp를 기반으로 slope를 계산합니다. 따라서 미래 sample로 과거의 alarm을 만드는 look-ahead 구조가 아닙니다.

---

## 9. Sample-level Detector

`5_3_run_sample_level_mahalanobis.py`와 `core_detector.py`에서는 window 통계 대신 원본 sample 수준에서 anomaly score를 계산합니다.

최종 sample feature는 6개입니다.

```text
AI0_Vibration
AI1_Vibration
AI2_Current

d_AI0_Vibration
d_AI1_Vibration
d_AI2_Current
```

즉:

```text
Raw sensor value
+
First difference
```

입니다.

그 다음 최근 3개 sample score의 **causal mean**을 사용합니다.

```text
t-2 ─┐
t-1 ─┼─→ mean(3) → sample alarm
t   ─┘
```

미래 sample은 사용하지 않습니다.

---

## 10. Three-Detector Ensemble

최종 detector는 세 개의 서로 다른 시간 관점을 결합합니다.

```text
1.0s Window Mahalanobis
        +
0.5s Window Mahalanobis
        +
Sample-level Mahalanobis
        ↓
      OR_3
        ↓
Persistence P2
        ↓
Final Alarm
```

### OR_3

세 detector 중 하나라도 alarm이면 후보 alarm을 생성합니다.

### P2 Persistence

OR_3 후보가 연속 2개 sample에서 유지될 때 최종 alarm으로 확정합니다.

```text
P2 = 2 consecutive OR_3 alarms
```

명목 sampling interval이 0.1초이므로 일반적인 연속 sample 기준 추가 확인 간격은 약 0.1초입니다. 실제 timestamp gap이 다른 경우에는 실제 timestamp 차이가 사용됩니다.

---

## 11. Threshold Calibration

최종 threshold는 fault 데이터가 아니라 **정상 데이터의 calibration score**로 결정합니다.

최종 configuration:

```text
Quantile = 0.9999
```

의미는 calibration 정상 score 분포의 상위 0.01% 지점을 threshold로 사용하는 것입니다.

### 학습 / calibration 분리

최종 배포 모델은 다음 순서입니다.

```text
Normal operation
      ↓
non-Idle only
      ↓
Train / Calibration group split
      ↓
Train Mahalanobis model
      ↓
Calibration normal scores
      ↓
q = 0.9999
      ↓
Threshold
```

Fault와 Idle은 threshold calibration에 사용하지 않습니다.

---

## 12. Final Deployment Configuration

현재 저장소의 `outputs/final_model/model_config.json` 기준 최종 배포 설정은 다음과 같습니다.

```text
Gap threshold        : 0.5 sec
0.5s window size     : 5 samples
1.0s window size     : 10 samples
Sample aggregation   : mean, k=3
Persistence          : P2
Quantile             : 0.9999
Covariance           : Ledoit-Wolf
Seed                 : 0
Calibration fraction : 0.20
```

현재 생성된 배포 모델의 train/calibration 규모:

```text
Train        : 12,985 rows
Calibration  : 3,140 rows
```

현재 `outputs/final_model/thresholds.json`에 저장된 threshold는 다음과 같습니다.

| Detector | Threshold |
|---|---:|
| Sample-level | 21.76348 |
| 0.5s Window | 67.50117 |
| 1.0s Window | 65.52940 |

이 값은 고정된 이론값이 아니라 **현재 데이터와 현재 deployment train/calibration split에서 생성된 calibration 결과**입니다.

---

## 13. Cross Validation Protocol

시계열 특성상 같은 press cycle이 train과 test 양쪽에 들어가면 성능이 부풀려질 수 있습니다. 따라서 `group_id` 단위로 fold를 구성합니다.

기본 protocol:

```text
5 folds × 5 repeats

repeat r
  ↓
  group shuffle(seed = base_seed + r)
  ↓
  fold i = test
  fold (i+1) mod K = calibration / validation
  remaining = train
```

학습은 **non-Idle normal**만 사용합니다.

Idle은 학습/threshold 산정에서 제외하고 test fold에서 false alarm을 평가합니다.

### Final Holdout 관련 주의

독립적인 미사용 final holdout 데이터가 별도로 제공된 것이 아니므로, 본 프로젝트의 최종 수치는 **독립 holdout 성능이 아니라 group-aware cross-validation 결과**입니다.

또한 최종 설정은 개발 과정에서 수행한 비교 결과를 참고해 결정했으므로, 그 CV 수치를 완전히 독립적인 confirmatory evaluation으로 해석해서는 안 됩니다.

---

## 14. Offline Ensemble Result

`outputs/6_three_detector_ensemble/ensemble_summary.csv`의 5-fold × 5-repeat 결과입니다.

| Configuration | Event Detection | Delay | Sample F1 | Normal FA Cycle | Idle FA Cycle |
|---|---:|---:|---:|---:|---:|
| 1.0s Window | 73.33 ± 2.61% | 1.156 ± 0.027s | 0.7422 ± 0.0146 | 0.47 ± 0.22% | 0.00% |
| 0.5s Window | 76.19 ± 0.00% | 0.776 ± 0.015s | 0.7894 ± 0.0099 | 0.47 ± 0.18% | 0.00% |
| Sample-level | 85.71 ± 0.00% | 0.571 ± 0.002s | 0.5817 ± 0.0040 | 0.98 ± 0.28% | 0.00% |
| OR_3 | 99.05 ± 2.13% | 0.470 ± 0.019s | 0.8482 ± 0.0134 | 1.17 ± 0.14% | 0.00% |
| **OR_3 + P2** | **91.43 ± 2.13%** | **0.571 ± 0.014s** | **0.8300 ± 0.0136** | **0.90 ± 0.11%** | **0.00%** |
| OR_3 + P3 | 79.05 ± 2.61% | 0.729 ± 0.022s | 0.8109 ± 0.0133 | 0.67 ± 0.18% | 0.00% |
| 2-of-3 Exact | 71.43 ± 3.37% | 0.897 ± 0.013s | 0.7633 ± 0.0088 | 0.39 ± 0.14% | 0.00% |
| 2-of-3 Tolerance | 72.38 ± 2.13% | 0.854 ± 0.010s | 0.7764 ± 0.0097 | 0.43 ± 0.16% | 0.00% |
| OR_3 Sample Corroboration | 81.90 ± 2.13% | 0.776 ± 0.025s | 0.8068 ± 0.0142 | 0.59 ± 0.14% | 0.00% |

`±`는 5개 repeat 결과의 표준편차입니다. confidence interval이나 event 자체의 독립 반복 수를 의미하지 않습니다.

---

## 15. Deployment-aligned Streaming CV

`src/12_streaming_cv.py`는 실제 대시보드의 `StreamingDetector`를 row-by-row로 실행하는 검증 코드입니다.

즉, offline window dataframe을 미리 계산해 놓고 평가하는 방식과 달리:

```text
raw row 1
  ↓
preprocess
  ↓
detector.step()
  ↓
raw row 2
  ↓
...
```

형태로 detector state를 유지하면서 평가합니다.

현재 `q=0.9999` 결과:

| Metric | Mean ± Std |
|---|---:|
| Event Detection Rate | **94.29 ± 2.13%** |
| Detection Delay | **0.524 ± 0.017s** |
| Sample F1 | 0.8521 |
| Precision | 0.9444 |
| Recall | 0.7763 |
| Normal FA Cycle Rate | **0.90 ± 0.11%** |
| Idle FA Cycle Rate | 0.00% |
| Normal FA Episodes | 9.4 / repeat |

Protocol:

```text
StreamingDetector
+ normal-only train/calibration
+ held-out group test
+ 5 folds × 5 repeats
+ q = 0.9999
```

### Threshold Sensitivity

동일한 streaming CV protocol에서 quantile을 변경한 결과입니다.

| Quantile | Event Detection | Delay | F1 | Normal FA Cycle |
|---:|---:|---:|---:|---:|
| 0.9900 | 95.24% | 0.435s | 0.7836 | 7.83% |
| 0.9950 | 95.24% | 0.449s | 0.8280 | 4.07% |
| 0.9990 | 95.24% | 0.495s | 0.8507 | 1.64% |
| 0.9995 | 94.29% | 0.506s | 0.8570 | 1.14% |
| 0.9999 | 94.29% | 0.524s | 0.8521 | 0.90% |

이 표는 quantile이 높아질수록 일반적으로 threshold가 올라가고 false alarm이 감소하는 대신 일부 탐지 기회를 잃거나 delay가 증가할 수 있음을 보여줍니다.

---

## 16. Isolation Forest Baseline

`src/7_isolation_forest_baseline.py`는 비교용 비지도 baseline입니다.

현재 저장된 baseline 결과:

```text
F1                    : 0.3898
Precision             : 1.0000
Recall                : 0.2421
Event Detection Rate : 100.0%
Mean Detection Delay : 1.267s
Normal FA Rate        : 0.0%
```

> 이 baseline은 1.0초 window dataset과 F1-based threshold를 사용하는 별도 protocol입니다. 최종 streaming CV와 동일한 평가 조건으로 직접 비교하는 수치가 아닙니다.

---

## 17. FP / FN Analysis

`src/8_fp_fn_analysis.py`에서는 최종 ensemble 구성 `OR_3 + P2`를 기준으로 다음을 분석합니다.

### FP

정상 운전에서 최종 alarm이 발생한 window를 수집하여 정상 전체와 비교합니다.

주요 산출물:

```text
outputs/8_fp_fn_analysis/
├─ fp_windows_1.0s.csv
├─ fp_feature_summary.csv
├─ source_detector_fp_windows_1.0s.csv
├─ fp_window_audit.csv
├─ fn_event_summary.csv
├─ fn_duration_summary.csv
├─ fn_event_repeat_details.csv
└─ analysis_report.txt
```

### FN

fault event별로:

```text
지속시간
sample 수
1.0s window 생성 수
0.5s window 생성 수
repeat별 탐지 / 미탐 횟수
```

를 확인합니다.

짧은 fault event는 window 길이 자체 때문에 평가 가능한 window가 충분히 생성되지 않을 수 있습니다. 특히:

```text
fault_5  : 3 samples ≈ 0.2 sec
fault_19 : 10 samples ≈ 0.9 sec
```

와 같이 짧은 event는 window detector와 sample-level detector의 차이를 보여주는 대표 사례입니다.

---

## 18. Variable Effect & Interaction Analysis

`src/10_variable_effect_analysis.py`는 탐지 성능에 영향을 주는 변수와 sensor interaction을 확인합니다.

비교 대상:

```text
Normal vs Fault
Normal vs False Positive context
Stable Detected vs Any-Miss Fault event
```

그리고 센서 상관 구조:

```text
Normal Operation
Fault
FP Context
Any-Miss Fault
```

를 비교합니다.

산출물:

```text
outputs/10_variable_effect_analysis/
├─ variable_effect_summary.csv
├─ event_feature_medians.csv
├─ sensor_correlation_matrix.csv
├─ feature_coverage_report.csv
├─ feature_effect_plot.png
├─ sensor_correlation_heatmap.png
├─ fn_duration_detection.png
└─ analysis_report.txt
```

---

## 19. Real-time / Streaming Design

최종 배포 detector는 `src/core_detector.py`의 `StreamingDetector`입니다.

핵심은 **미리 전처리된 window 결과를 읽는 것이 아니라, 들어오는 raw row를 한 행씩 처리한다는 점**입니다.

```text
Raw row
  ↓
StreamingPreprocessor.process()
  ↓
현재 row 검증
  ↓
현재 timestamp와 이전 timestamp 비교
  ↓
gap > 0.5 sec 이면 state reset
  ↓
StreamingDetector.step()
  ├─ sample-level raw + diff score
  ├─ causal mean-3
  ├─ trailing 0.5s window
  ├─ trailing 1.0s window
  ├─ OR_3
  └─ P2 persistence
  ↓
Final Alarm
```

### No Look-ahead

streaming preprocessing은 현재 row와 이전 timestamp만 사용합니다.

window feature는 detector buffer의 최근 samples로 계산되고, buffer는 현재 시점까지의 값만 보유합니다.

즉 다음 sample이나 미래 fault 정보를 미리 보고 현재 alarm을 만드는 구조가 아닙니다.

### Segment Reset

```text
gap <= 0.5 sec
→ 기존 state 유지

gap > 0.5 sec
→ raw/window/sample/persistence state reset
```

이는 서로 다른 press cycle의 과거 정보가 새 cycle의 초기 판단에 섞이는 것을 방지합니다.

---

## 20. Full Pipeline Dashboard

`src/11_realtime_dashboard.py`는 프로젝트 전체 과정을 시각적으로 실행하는 최종 entry point입니다.

실행:

```bash
python src/11_realtime_dashboard.py
```

대시보드 구성:

```text
┌───────────────────────────────────────────────────────────────┐
│ 1) Pipeline Execution                                        │
│                                                               │
│ [Stage Board] [Current Stage / Live Log] [Figures / Detail] │
│                                                               │
│ 1 → 12 단계가 실제 스크립트 실행과 함께 진행                 │
│                                                               │
│                 ↓                                             │
│       Final model fit + calibration                            │
│                 ↓                                             │
│       '실시간 추론 시작'                                      │
└───────────────────────────────────────────────────────────────┘

                         ↓

┌───────────────────────────────────────────────────────────────┐
│ 2) Causal Streaming Replay                                   │
│                                                               │
│ Raw Sensor Graph                                              │
│ Detector Score / Threshold                                    │
│ OR_3 / P2                                                     │
│ Current State / Segment / Gap                                │
│ Sample + Window Features                                      │
│ Detection Delay / Inference Latency                           │
└───────────────────────────────────────────────────────────────┘
```

### Dashboard 실행 순서

```text
1~3   데이터 이해 / 정리
4~5   Window dataset 생성
6~8   Mahalanobis / Ensemble
9~11  비교·FP/FN·변수 영향 분석
12    최종 모델 학습 + threshold calibration
13    raw CSV replay 기반 causal streaming inference
```

### 재생 데이터 구성

기본 demo stream은 다음 순서로 연결됩니다.

```text
normal
  → idle
  → normal
  → fault
```

기본 `continuous` 모드에서는 블록 사이 timestamp를 공칭 sample interval 수준으로 이어 붙입니다. 따라서 detector가 block 전환마다 인위적으로 reset되지 않고, 실제 gap 규칙이 적용되는 모습을 확인할 수 있습니다.

`gap` 모드를 사용하면 블록 사이에 0.5초보다 큰 간격을 넣어 reset 동작을 시연할 수 있습니다.

---

## 21. Dashboard Options

기본값으로 전체 pipeline을 실행합니다.

```bash
python src/11_realtime_dashboard.py \
  --normal-path data/press_data_normal.csv \
  --fault-path data/outlier_data.csv
```

주요 옵션:

| Option | Default | Description |
|---|---:|---|
| `--reuse-results` | off | 1~11단계의 기존 산출물을 재사용 |
| `--skip-heavy-analysis` | off | 9~11단계 생략 |
| `--quantile` | `0.9999` | 최종 normal-only threshold quantile |
| `--seed` | `0` | 배포 모델 seed |
| `--calibration-fraction` | `0.20` | 정상 segment 중 calibration 비율 |
| `--replay-speed` | `1.0` | 실제 timestamp 대비 replay 배속 |
| `--fps` | auto | replay 행/초를 직접 지정 |
| `--demo-join-mode` | `continuous` | `continuous` 또는 `gap` |
| `--demo-join-gap` | `2.0` | `gap` 모드의 블록 사이 간격 |
| `--auto-start` | `0` | pipeline 완료 후 streaming 창 자동 이동까지의 초 |
| `--plot-window` | `150` | 실시간 그래프에 표시할 최근 points |
| `--feature-update-every` | `3` | feature panel 갱신 주기 |
| `--axis-update-every` | `5` | 축 자동조정 주기 |

예:

```bash
# 기존 1~11 산출물을 활용하고 최종 모델/streaming만 빠르게 확인
python src/11_realtime_dashboard.py --reuse-results

# 무거운 9~11단계 생략
python src/11_realtime_dashboard.py --skip-heavy-analysis

# 데이터 시간의 5배 속도로 streaming replay
python src/11_realtime_dashboard.py --replay-speed 5

# block 사이를 큰 gap으로 만들고 reset 동작 확인
python src/11_realtime_dashboard.py --demo-join-mode gap --demo-join-gap 2
```

---

## 22. Dashboard Outputs

실행 결과는 `outputs/11_realtime_dashboard/`에 저장됩니다.

```text
outputs/11_realtime_dashboard/
├─ pipeline_stage_report.csv
├─ pipeline_run_summary.txt
├─ pipeline_dashboard.png
├─ stage_timing.json
├─ stage_markers/
│  ├─ stage_1.json ... stage_13.json
│
├─ realtime_log.csv
├─ realtime_metrics.csv
└─ realtime_event_summary.csv
```

### `realtime_log.csv`

한 row씩 streaming된 결과를 기록합니다.

대표 컬럼:

```text
TimeStamp
orig_TimeStamp
AI0_Vibration
AI1_Vibration
AI2_Current
segment_id
segment_reset
gap_seconds
score_0.5s
threshold_0.5s
alarm_0.5s
score_1.0s
threshold_1.0s
alarm_1.0s
sample_score
sample_threshold
sample_alarm
OR_3
P2
state
infer_ms
```

### `realtime_metrics.csv`

streaming 실행에서 다음을 요약합니다.

```text
fault event detection rate
first detection delay
normal false alarm
idle false alarm
fault frame recall
inference latency mean / p95 / max
```

### `pipeline_stage_report.csv`

각 단계의 상태·소요시간·결과 요약을 기록합니다.

따라서 대시보드가 단순 시각화 프로그램이 아니라:

```text
전처리
→ 모델링
→ 학습
→ calibration
→ 추론
→ 결과 생성
```

전체 과정을 보여주는 실행형 프로젝트 데모 역할을 합니다.

---

## 23. Example Streaming Run

현재 저장소에는 실제 dashboard replay에서 생성된 예시 산출물도 포함되어 있습니다.

예시 실행 로그에서는:

```text
Streaming rows           : 1,451
Fault events             : 1
Event detection          : 1 / 1
Detection delay          : 0.100 sec
Normal false alarm frame : 1
Idle false alarm frame   : 0
Mean inference latency   : 2.561 ms
P95 inference latency    : 3.157 ms
Max inference latency    : 3.649 ms
```

가 기록되었습니다.

> 이 수치는 **1개 fault event를 포함한 특정 dashboard replay 사례**이며, 최종 성능 benchmark로 사용하지 않습니다. 일반화된 성능은 `outputs/12_streaming_cv/`의 group-aware streaming CV 결과를 기준으로 봅니다.

또한 dashboard의 `replay 경과`는 UI에서 데이터를 재생하는 데 걸린 wall-clock 시간이고, `infer_ms`는 한 행의 전처리+특징+탐지 처리에 걸린 추론 측정값입니다. 두 시간은 같은 의미가 아닙니다.

---

## 24. Detection Delay

delay는 event별로 다음과 같이 계산합니다.

```text
fault segment first timestamp
          ↓
      first final alarm
          ↓
       delay_sec
```

### Sample-level

현재 sample에서 최종 alarm이 발생하면 해당 sample timestamp를 최초 alarm 시점으로 사용합니다.

### Window-level

trailing window는 현재 sample까지의 데이터를 사용하므로, 해당 window가 끝나는 현재 timestamp에서 alarm 여부를 계산합니다.

### Missed Event

탐지하지 못한 event는:

```text
delay = NaN
```

으로 처리하며 탐지된 event의 평균 delay에 포함하지 않습니다.

따라서 **탐지율이 다른 detector의 평균 delay는 동일한 event 집합을 기반으로 하지 않을 수 있습니다.**

---

## 25. Short Fault Events

원본 nominal sampling interval이 0.1초이므로:

```text
3 samples  → 첫 sample ~ 마지막 sample ≈ 0.2 sec
10 samples → 첫 sample ~ 마지막 sample ≈ 0.9 sec
```

입니다.

예를 들어:

```text
fault_5  = 3 samples ≈ 0.2 sec
fault_19 = 10 samples ≈ 0.9 sec
```

처럼 짧은 event는 1.0초 window가 충분히 만들어지지 않을 수 있습니다.

이 문제 때문에 KAMPact는:

```text
window detector
+
sample-level detector
```

를 함께 사용합니다. sample-level detector는 window 길이에 관계없이 raw sample에서 바로 score를 계산할 수 있습니다.

---

## 26. Reproducibility

최종 배포 모델에는 다음 정보가 함께 저장됩니다.

```text
outputs/final_model/model_config.json
outputs/final_model/thresholds.json
outputs/final_model/detector.pkl
```

`model_config.json`에는 다음 재현성 정보가 포함됩니다.

```text
Python version
NumPy version
Pandas version
scikit-learn version
git commit
input file hash
train rows
calibration rows
quantile
seed
```

현재 저장된 deployment metadata의 예:

```text
Python        : 3.11.9
NumPy         : 2.4.6
Pandas        : 3.0.6
scikit-learn : 1.9.1
Git commit    : a06913f
```

입력 CSV의 일부 hash도 함께 저장되므로 다른 파일로 모델이 생성되었는지 확인할 수 있습니다.

---

## 27. Installation

권장 Python 버전:

```text
Python 3.11.x
```

가상환경 생성:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

대시보드는 Matplotlib의 interactive GUI를 사용하므로 데스크톱 환경에서 실행하는 것을 전제로 합니다.

---

## 28. Data Setup

원본 데이터는 직접 `data/`에 배치합니다.

```text
data/
├─ press_data_normal.csv
└─ outlier_data.csv
```

그 뒤:

```bash
python src/11_realtime_dashboard.py
```

를 실행하면 Step 2에서:

```text
data/press_data_normal_with_idle.csv
```

가 생성되고 이후 단계가 이어집니다.

---

## 29. Individual Script Execution

전체 대시보드가 아니라 각 단계만 실행할 수도 있습니다.

```bash
# 1. Raw visualization
python src/1_visualize_normal_outlier.py

# 2. Idle labeling
python src/2_Classification_idle_sections.py

# 3. Time structure / segment analysis
python src/3_time_structure_analysis.py

# 4. 1.0s window dataset
python src/4_make_window_dataset.py --window-sec 1.0 --step-sec 0.1 --output-dir result/modeling_dataset_1.0_0.1

# 5. 0.5s window dataset
python src/4_make_window_dataset.py --window-sec 0.5 --step-sec 0.1 --output-dir result/modeling_dataset_0.5_0.1

# 6. Window Mahalanobis CV
python src/5_2_run_mahalanobis.py --multiscale result/modeling_dataset_1.0_0.1 result/modeling_dataset_0.5_0.1

# 7. Sample-level Mahalanobis CV
python src/5_3_run_sample_level_mahalanobis.py \
  --normal-path data/press_data_normal_with_idle.csv \
  --fault-path data/outlier_data.csv

# 8. Three-detector ensemble
python src/6_compare_three_detectors.py \
  --normal-path data/press_data_normal_with_idle.csv \
  --fault-path data/outlier_data.csv \
  --window-1.0 result/modeling_dataset_1.0_0.1 \
  --window-0.5 result/modeling_dataset_0.5_0.1

# 9. Isolation Forest baseline
python src/7_isolation_forest_baseline.py

# 10. FP / FN analysis
python src/8_fp_fn_analysis.py

# 11. FN visualization + variable effect analysis
python src/9_visualize_fn_events.py
python src/10_variable_effect_analysis.py

# 12. Deployment-aligned streaming CV
python src/12_streaming_cv.py \
  --normal-path data/press_data_normal_with_idle.csv \
  --fault-path data/outlier_data.csv \
  --quantile 0.9999
```

세부 인자는 각 스크립트의 `--help`를 확인합니다.

---

## 30. Current Final Flow

KAMPact의 최종 사용 흐름을 하나로 요약하면 다음과 같습니다.

```text
[Raw CSV]
    │
    ├─ Normal
    └─ Fault
    │
    ▼
[Data preprocessing]
    ├─ timestamp validation
    ├─ gap / segment
    └─ Idle classification
    │
    ▼
[Feature engineering]
    ├─ 1.0s window / 16 features
    ├─ 0.5s window / 16 features
    └─ sample raw + diff / 6 features
    │
    ▼
[Anomaly modeling]
    ├─ Ledoit-Wolf Mahalanobis
    ├─ normal-only calibration
    └─ q = 0.9999
    │
    ▼
[Ensemble]
    ├─ sample-level
    ├─ 0.5s window
    ├─ 1.0s window
    ├─ OR_3
    └─ P2
    │
    ▼
[Analysis]
    ├─ event detection
    ├─ detection delay
    ├─ normal / idle false alarm
    ├─ FP / FN
    └─ variable effect / interaction
    │
    ▼
[Deployment]
    ├─ StreamingPreprocessor
    ├─ StreamingDetector.step()
    ├─ causal state
    └─ row-by-row alarm
```

---

## 31. Important Limitations

1. **Physical fault onset timestamp 부재**
   - delay는 데이터상 fault segment 시작점 기준입니다.

2. **독립 final holdout 부재**
   - 최종 성능은 group-aware 5-fold × 5-repeat CV 기반입니다.

3. **Threshold 선택과 최종 구성 선택의 개발 데이터 의존성**
   - 동일 데이터의 CV 결과를 개발 과정에서 참고했기 때문에 완전한 독립 검증으로 볼 수 없습니다.

4. **Idle 구간의 의미**
   - Idle은 정상 운전의 한 형태로 볼 수도 있지만, 본 프로젝트에서는 학습에서 제외하고 false alarm 평가 대상으로 별도 취급합니다.

5. **짧은 fault event**
   - 매우 짧은 event는 긴 window detector에서 정보가 부족할 수 있으므로 sample-level detector를 함께 둡니다.

6. **입력 데이터의 시간 간격**
   - 명목 간격은 0.1초이지만 실제 timestamp는 완전히 균일하지 않을 수 있습니다. delay와 slope는 가능한 경우 실제 timestamp를 사용합니다.

---

## 32. Recommended Entry Point

프로젝트 전체 흐름을 한 번에 확인하려면 다음 하나만 실행하면 됩니다.

```bash
python src/11_realtime_dashboard.py
```

이 명령이:

```text
데이터 전처리
→ feature 생성
→ 모델 학습 / CV
→ threshold calibration
→ ensemble 분석
→ FP/FN 분석
→ 최종 모델 저장
→ causal streaming inference
→ 결과 CSV / 보고서 생성
```

전체 과정을 하나의 실행형 데모로 연결합니다.

---

## 33. Project Status

현재 저장소에는 다음 최종 흐름이 구현되어 있습니다.

```text
Analysis
  ✓
Window modeling
  ✓
Sample-level modeling
  ✓
Three-detector ensemble
  ✓
Isolation Forest baseline
  ✓
FP / FN analysis
  ✓
Variable effect analysis
  ✓
Final deployment model
  ✓
Causal streaming detector
  ✓
End-to-end dashboard
  ✓
Deployment-aligned streaming CV
  ✓
```

KAMPact는 **오프라인 분석 결과를 보여주는 프로젝트**에서 끝나지 않고, 최종 모델이 실제 입력 row를 한 개씩 받아 전처리·특징 추출·이상 점수 계산·알람 결정을 수행하는 **causal streaming 구조**까지 연결하는 것을 최종 목표로 합니다.
