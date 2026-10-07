# KAMPact

**프레스 유압펌프의 진동·전류 시계열을 이용한 이상 조기탐지 및 오경보 분석**

KAMPact는 프레스 유압펌프의 `AI0_Vibration`, `AI1_Vibration`, `AI2_Current` 시계열을 대상으로 정상 운전의 분포를 학습하고, 정상에서 벗어난 패턴을 **행 단위 causal streaming**으로 탐지하는 프로젝트입니다.

핵심 목표는 단순히 F1을 높이는 것이 아니라 다음 세 가지를 함께 만족하는 것입니다.

```text
빠른 이상 탐지
    +
낮은 정상 오탐
    +
짧은 Fault event에 대한 대응
```

이를 위해 `1.0s Window`, `0.5s Window`, `Sample-level`의 세 detector를 결합하고, 최종적으로 `OR_3 + P2` 구조를 사용합니다.

---

## 1. Problem

대상 센서는 다음 세 가지입니다.

| Sensor | 의미 |
|---|---|
| `AI0_Vibration` | 진동 센서 0 |
| `AI1_Vibration` | 진동 센서 1 |
| `AI2_Current` | 전류 센서 |

프레스 설비 이상탐지에서는 탐지 민감도를 높일수록 정상 운전의 일시적인 변동이 오탐으로 이어질 수 있고, 반대로 오탐을 강하게 억제하면 짧은 Fault를 놓치거나 탐지 지연이 증가할 수 있습니다.

KAMPact는 이를 다음 지표로 함께 평가합니다.

```text
Event Detection Rate
Detection Delay
Normal False Alarm Rate
Idle False Alarm Rate
FP / FN 조건
```

최종적으로 같은 detector를 실제 입력처럼 한 행씩 받아 처리하는 streaming 구조까지 연결합니다.

---

## 2. Dataset

원본 CSV는 저장소에 포함하지 않습니다. `.gitignore`에서 `data/*.csv`와 `data/**/*.csv`를 제외합니다.

현재 프로젝트 분석에 사용한 데이터 규모:

```text
Normal        : 20,000 rows
Fault         : 600 rows
Fault events  : 21
Nominal dt    : 0.1 sec
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

### Label

```text
Equipment_state >= 1 → Fault
Equipment_state = 0  → Normal candidate
Idle = 1              → Idle
```

정상 데이터의 Idle 구간은 정상 가동 분포와 다른 특성을 가질 수 있기 때문에 **모델 학습 및 threshold calibration에서는 제외**하고, 별도의 false-alarm 평가 대상으로 취급합니다.

### Data placement

클론 후 다음 파일을 직접 배치합니다.

```text
data/
├─ press_data_normal.csv
└─ outlier_data.csv
```

Step 2 실행 후 다음 파일이 생성됩니다.

```text
data/press_data_normal_with_idle.csv
```

---

## 3. System Overview

전체 구조는 다음과 같습니다.

```text
                         ┌─────────────────────────┐
                         │       Raw CSV           │
                         │  Normal + Fault data    │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ Data Preprocessing      │
                         │ - timestamp validation  │
                         │ - gap / segment         │
                         │ - Idle classification   │
                         └────────────┬────────────┘
                                      │
                                      ▼
                ┌─────────────────────┴─────────────────────┐
                │                                           │
                ▼                                           ▼
      ┌────────────────────┐                    ┌────────────────────┐
      │ Window Detector    │                    │ Sample Detector    │
      │                    │                    │                    │
      │ 1.0s / 16 features │                    │ raw + first diff   │
      │ 0.5s / 16 features │                    │ 6 features         │
      │                    │                    │ mean-3 smoothing   │
      └──────────┬─────────┘                    └──────────┬─────────┘
                 │                                         │
                 └──────────────────┬──────────────────────┘
                                    ▼
                              ┌────────────┐
                              │   OR_3     │
                              └─────┬──────┘
                                    ▼
                              ┌────────────┐
                              │    P2      │
                              └─────┬──────┘
                                    ▼
                              ┌────────────┐
                              │ Final Alarm│
                              └────────────┘
```

---

## 4. End-to-End Pipeline

`src/11_realtime_dashboard.py`가 프로젝트 전체 실행을 담당합니다.

### Dashboard 13단계

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
13  Causal Streaming Inference
```

중요하게 구분할 점:

- **Dashboard Stage 11**은 `9_visualize_fn_events.py`와 `10_variable_effect_analysis.py`를 묶어서 실행합니다.
- **Dashboard Stage 12**는 별도의 `12_*.py`가 아니라 `11_realtime_dashboard.py` 내부에서 최종 `StreamingDetector`를 학습/보정합니다.
- `src/12_streaming_cv.py`는 **검증용 별도 스크립트**이며 Dashboard Stage 12가 아닙니다.
- **Dashboard Stage 13**은 학습된 detector를 raw row 단위로 실행하는 streaming replay입니다.

---

## 5. Data Preprocessing

### 5.1 Timestamp validation

`TimeStamp`를 datetime으로 변환하고 잘못된 timestamp를 제거합니다.

### 5.2 Segment

실제 timestamp 차이가 다음 조건을 만족하면 새로운 segment로 분리합니다.

```text
gap > 0.5 sec
```

이 segment가 하나의 press cycle 또는 독립적인 시계열 구간의 단위가 됩니다.

이를 기준으로 train / calibration / test를 나누기 때문에 같은 segment의 정보가 여러 split에 섞이는 **segment leakage**를 방지합니다.

### 5.3 Idle

정상 CSV의 실제 Idle 시간대를 `Idle=1`로 라벨링합니다.

현재 샘플 데이터에서는:

```text
Normal rows : 20,000
Idle rows   : 3,049
```

가 생성되었습니다.

---

## 6. Feature Engineering

최종 window detector는 두 개의 시간 스케일을 사용합니다.

```text
1.0 second window
0.5 second window
step = 0.1 second
```

각 센서마다 5개 통계 특징을 계산합니다.

```text
Mean
Std
RMS
Peak-to-Peak
Slope
```

3개 센서에서 총 15개:

```text
3 sensors × 5 features = 15
```

추가로 두 진동 센서의 상관관계:

```text
AI0 ↔ AI1 correlation = 1
```

을 사용하여 최종 **16개 window feature**를 구성합니다.

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

`slope`는 가능한 경우 실제 `TimeStamp`를 사용해 초당 변화율을 계산합니다.

---

## 7. Anomaly Model

### 7.1 Signed Log Transform

feature scale 차이와 큰 값의 영향을 완화하기 위해 다음 변환을 사용합니다.

$$
x' = sign(x)\log(1+|x|)
$$

### 7.2 Ledoit-Wolf Mahalanobis Distance

정상 데이터의 다변량 분포를 학습한 뒤 현재 입력의 정상 분포로부터의 거리를 계산합니다.

$$
D^2(x)=(x-\mu)^T\Sigma^{-1}(x-\mu)
$$

최종 배포 모델의 covariance estimator는:

```text
Ledoit-Wolf
```

입니다.

Offline 실험 코드에서는 `Empirical Covariance`, `Ledoit-Wolf`, `OAS` 비교도 지원합니다.

---

## 8. Window Detector

각 시간 스케일의 구조는 다음과 같습니다.

```text
Raw samples
    ↓
Trailing window
    ↓
16 features
    ↓
Signed Log1p
    ↓
Ledoit-Wolf Mahalanobis
    ↓
Normal-only calibration threshold
    ↓
Window alarm
```

최종 detector:

```text
0.5s → 5 samples
1.0s → 10 samples
```

window는 항상 **현재 sample과 과거 sample만** 사용합니다.

---

## 9. Sample-level Detector

window 길이에 의존하지 않는 별도의 detector를 사용합니다.

입력 feature:

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
Raw sensor value + First difference = 6 features
```

raw sample에 Mahalanobis score를 계산한 뒤 최근 3개 score의 causal mean을 사용합니다.

```text
t-2 ─┐
t-1 ─┼─→ mean(3) → Sample Alarm
t   ─┘
```

미래 sample은 사용하지 않습니다.

---

## 10. Ensemble Logic

최종 detector는 세 가지 시간 관점을 결합합니다.

```text
1.0s Window
      +
0.5s Window
      +
Sample-level
      ↓
     OR_3
      ↓
      P2
      ↓
 Final Alarm
```

### OR_3

세 detector 중 하나라도 alarm이면 후보 alarm을 생성합니다.

```text
OR_3 = sample_alarm OR win05_alarm OR win1_alarm
```

### P2 Persistence

OR_3 후보가 연속 2개 sample에서 발생할 때 최종 alarm으로 확정합니다.

```text
P2 = 2 consecutive OR_3 alarms
```

nominal sampling이 0.1초이므로 정상적으로 연속 sample이 들어오는 경우 P2는 약 0.1초의 추가 확인을 요구합니다. 실제 timestamp gap은 detector가 직접 확인합니다.

---

## 11. Threshold Calibration

최종 threshold는 Fault 데이터로 결정하지 않습니다.

```text
Normal-only training
        ↓
Separate normal calibration groups
        ↓
Mahalanobis score
        ↓
q = 0.9999
        ↓
Threshold
```

최종 배포 설정:

```text
Quantile = 0.9999
```

즉 calibration 정상 score 분포의 상위 tail을 threshold로 사용합니다.

### Train / Calibration separation

최종 배포 모델은 다음 정보를 사용합니다.

```text
Train rows       : 12,985
Calibration rows : 3,140
```

둘은 group 단위로 분리되며, demo replay에 사용할 그룹도 배포 모델 학습/보정에서 제외합니다.

Fault와 Idle은 threshold calibration에 사용하지 않습니다.

---

## 12. Final Deployment Configuration

현재 저장된 `outputs/final_model/model_config.json` 기준:

| Parameter | Value |
|---|---:|
| Gap threshold | 0.5 sec |
| 0.5s window | 5 samples |
| 1.0s window | 10 samples |
| Sample aggregation | mean, k=3 |
| Persistence | P2 |
| Quantile | 0.9999 |
| Covariance | Ledoit-Wolf |
| Seed | 0 |
| Calibration fraction | 0.20 |

현재 저장된 threshold:

| Detector | Threshold |
|---|---:|
| Sample-level | 21.76348 |
| 0.5s Window | 67.50117 |
| 1.0s Window | 65.52940 |

threshold는 데이터와 calibration split에 의해 생성되는 값이므로 모든 데이터셋에서 고정되는 상수는 아닙니다.

---

## 13. Cross-Validation Protocol

시계열 데이터는 같은 cycle의 유사한 행이 train/test에 섞이면 성능이 과대평가될 수 있습니다.

KAMPact는 다음 **group-aware CV**를 사용합니다.

```text
5 folds × 5 repeats

repeat r
    ↓
group shuffle (seed = base + r)
    ↓
fold i          → test
fold (i + 1)%K → calibration / validation
remaining       → train
```

학습에는:

```text
non-Idle normal only
```

을 사용합니다.

Idle은 학습/threshold 산정에서 제외하고 test 단계에서 false alarm을 평가합니다.

### 평가 단위

주요 metric은 다음을 함께 봅니다.

```text
Event-level:
  Event Detection Rate
  Detection Delay

Cycle-level:
  Normal False Alarm Cycle Rate
  Idle False Alarm Cycle Rate

Sample-level:
  F1
  Precision
  Recall
```

---

## 14. Offline Ensemble Results

`outputs/6_three_detector_ensemble/ensemble_summary.csv`의 5-repeat 결과입니다.

| Configuration | Event Detection | Delay | Sample F1 | Normal FA Cycle | Idle FA Cycle |
|---|---:|---:|---:|---:|---:|
| 1.0s Window | 73.33 ± 2.61% | 1.156 ± 0.027s | 0.7422 | 0.47 ± 0.22% | 0% |
| 0.5s Window | 76.19 ± 0.00% | 0.776 ± 0.015s | 0.7894 | 0.47 ± 0.18% | 0% |
| Sample-level | 85.71 ± 0.00% | 0.571 ± 0.002s | 0.5817 | 0.98 ± 0.28% | 0% |
| OR_3 | 99.05 ± 2.13% | 0.470 ± 0.019s | 0.8482 | 1.17 ± 0.14% | 0% |
| **OR_3 + P2** | **91.43 ± 2.13%** | **0.571 ± 0.014s** | **0.8300** | **0.90 ± 0.11%** | **0%** |
| OR_3 + P3 | 79.05 ± 2.61% | 0.729 ± 0.022s | 0.8109 | 0.67 ± 0.18% | 0% |
| 2-of-3 Exact | 71.43 ± 3.37% | 0.897 ± 0.013s | 0.7633 | 0.39 ± 0.14% | 0% |

`±`는 5회 repeat의 표준편차입니다.

이 결과에서 `OR_3`는 빠른 탐지를 제공하는 대신 정상 cycle false alarm이 증가하고, `P2`를 추가하면 탐지율/지연/오탐 사이의 균형점이 형성됩니다.

---

## 15. Deployment-aligned Streaming CV

offline window 평가만으로는 실제 streaming 동작을 충분히 설명하기 어렵기 때문에 `src/12_streaming_cv.py`에서 **최종 `StreamingDetector` 자체를 row-by-row로 평가**합니다.

즉:

```text
Raw row 1
  ↓
detector.step()
  ↓
Raw row 2
  ↓
detector.step()
  ↓
...
```

형태로 detector의 causal state를 유지합니다.

### Final streaming result

`q = 0.9999`, 5-fold × 5-repeat 기준:

| Metric | Mean ± Std |
|---|---:|
| **Event Detection Rate** | **94.29 ± 2.13%** |
| **Detection Delay** | **0.524 ± 0.017s** |
| Sample F1 | 0.8521 |
| Precision | 0.9444 |
| Recall | 0.7763 |
| **Normal FA Cycle Rate** | **0.90 ± 0.11%** |
| **Idle FA Cycle Rate** | **0.00%** |
| Normal FA Episodes | 9.4 ± 2.4 / repeat |

### Threshold sensitivity

동일한 streaming protocol에서 threshold quantile을 변경한 결과:

| Quantile | Event Detection | Delay | F1 | Normal FA Cycle |
|---:|---:|---:|---:|---:|
| 0.9900 | 95.24% | 0.435s | 0.7836 | 7.83% |
| 0.9950 | 95.24% | 0.449s | 0.8280 | 4.07% |
| 0.9990 | 95.24% | 0.495s | 0.8507 | 1.64% |
| 0.9995 | 94.29% | 0.506s | 0.8570 | 1.14% |
| **0.9999** | **94.29%** | **0.524s** | **0.8521** | **0.90%** |

KAMPact는 현장 적용 관점에서 정상 오탐을 낮추는 것을 중요하게 보고 최종 configuration을 `q=0.9999`로 설정했습니다.

---

## 16. Isolation Forest Baseline

Mahalanobis가 특정 통계 구조에 의존한 결과인지 확인하기 위해 Isolation Forest baseline도 비교합니다.

### 동일한 streaming-CV protocol에서 `q=0.9999`, `OR_3 + P2`

| Model | Event Detection | Delay | F1 | Normal FA Cycle |
|---|---:|---:|---:|---:|
| **Mahalanobis** | **94.29%** | **0.524s** | **0.8521** | 0.90% |
| Isolation Forest | 76.19% | 0.816s | 0.6344 | **0.51%** |

Isolation Forest는 정상 오탐을 더 낮추는 대신 event detection과 F1이 크게 감소합니다.

별도의 historical baseline(`src/7_isolation_forest_baseline.py`)도 유지하고 있습니다. 해당 결과는 1.0초 window dataset + F1 threshold protocol이므로 streaming-CV 표와 직접 같은 조건은 아닙니다.

---

## 17. False Positive / False Negative Analysis

`src/8_fp_fn_analysis.py`는 최종 offline ensemble인 `OR_3 + P2`를 기준으로 오탐/미탐 조건을 분석합니다.

### False Positive

현재 분석에서는:

```text
Unique normal 1.0s windows : 14,341
FP context windows         : 132
```

오경보 구간은 특히 다음 특징에서 큰 차이를 보였습니다.

```text
AI0_Vibration_mean
AI0_Vibration_rms
AI0_Vibration_std
AI0_Vibration_ptp
AI1_Vibration_mean
AI0_AI1_corr
```

예를 들어 FP context의 중앙값은 일부 진동 feature에서 정상보다 크게 나타났습니다.

> 이 분석은 descriptive comparison입니다. 특정 feature가 오경보를 **인과적으로 발생시켰다는 의미는 아닙니다.**

### False Negative

21개 fault event 중 대표적인 짧은 event:

```text
fault_5  : 3 samples  ≈ 0.2 sec
fault_19 : 10 samples ≈ 0.9 sec
```

분석 결과:

```text
fault_5  → 5회 repeat 모두 미탐
fault_19 → 5회 중 1회 탐지
>= 1.0s event → 16개 event가 모든 repeat에서 탐지
```

지속시간 구간별 repeat 평가:

| Duration | Events | Detection |
|---|---:|---:|
| < 0.5s | 3 | 66.67% |
| 0.5–<1.0s | 2 | 60.00% |
| >= 1.0s | 16 | 100.00% |

1.0초 미만 event의 repeat 기준 탐지율은 64.00%입니다.

중요한 점은 짧은 duration만으로 FN의 원인을 단정하지 않는다는 것입니다. 실제 분석에서는 신호 편차와 detector persistence / window 구조를 함께 봅니다.

---

## 18. Variable Effect & Interaction Analysis

`src/10_variable_effect_analysis.py`는 다음 비교를 수행합니다.

```text
Normal vs Fault
Normal vs FP Context
Stable Detected vs Any-Miss Fault Event
```

또한:

```text
Normal sensor correlation
Fault sensor correlation
FP context correlation
Any-miss fault correlation
```

을 비교합니다.

대표적으로 event-level 비교에서 크게 나타난 변수는:

```text
AI1_Vibration_slope
AI2_Current_slope
AI0_AI1_corr
AI0_Vibration_rms
AI2_Current_rms / mean
```

등입니다.

이 분석의 목적은 모델 내부 feature importance를 의미하는 것이 아니라 **어떤 신호 변화와 탐지 성공/실패가 함께 나타나는지**를 확인하는 것입니다.

---

## 19. Causal Streaming Design

최종 배포 엔진은 `src/core_detector.py`의 `StreamingDetector`입니다.

핵심 원칙은 **look-ahead 금지**입니다.

```text
Current raw row
    ↓
현재 timestamp / 이전 timestamp 확인
    ↓
현재 row 기반 diff
    ↓
현재까지의 raw buffer
    ↓
현재 시점까지 만들 수 있는 feature
    ↓
Sample / 0.5s / 1.0s score
    ↓
OR_3
    ↓
P2
    ↓
Final Alarm
```

### Segment reset

```text
gap <= 0.5 sec
→ 기존 detector state 유지

gap > 0.5 sec
→ raw/window/sample/P2 history reset
```

이를 통해 이전 press cycle의 history가 새로운 cycle의 초기 판단에 섞이지 않도록 합니다.

### Runtime feature generation

streaming 단계에서는 미리 계산된 `model_windows.csv`를 읽어 alarm을 만드는 것이 아닙니다.

각 row가 다음 과정을 거칩니다.

```text
raw CSV row
→ runtime preprocessing
→ buffer update
→ causal feature extraction
→ anomaly score
→ alarm
```

---

## 20. Full Pipeline Dashboard

### 실행

```bash
python src/11_realtime_dashboard.py
```

대시보드는 크게 두 부분으로 동작합니다.

```text
[1] Pipeline Window

1~11 분석 단계
        ↓
12 최종 모델 학습 / calibration
        ↓
실시간 추론 시작


[2] Streaming Window

raw row
  ↓
preprocess
  ↓
sample / window score
  ↓
OR_3
  ↓
P2
  ↓
Final state
```

실시간 창에서는 다음 정보를 함께 확인할 수 있습니다.

```text
Raw sensor graph
Detector score
Threshold
Sample / 0.5s / 1.0s alarm
OR_3
P2
Current state
Segment
Timestamp gap
Segment reset
Feature panel
Detection delay
Inference latency
```

기본 내부 demo stream은:

```text
normal
  →
idle
  →
normal
  →
fault
```

순서로 이어집니다.

`continuous` 모드에서는 block 사이 timestamp를 공칭 sample interval 수준으로 이어 붙여 불필요한 reset을 만들지 않습니다.

`gap` 모드에서는 인위적인 `> 0.5 sec` gap을 넣어 reset 동작을 시연할 수 있습니다.

---

## 21. External Streaming

최종 detector는 저장된 모델을 이용하여 별도의 raw CSV를 직접 스트리밍할 수도 있습니다.

### One-shot external CSV

```bash
python src/11_realtime_dashboard.py \
  --stream-path path/to/live_stream.csv
```

이 모드에서는:

```text
1~11 pipeline 실행 생략
12 최종 모델 재학습 생략
저장된 outputs/final_model/detector.pkl 로드
외부 CSV를 row-by-row inference
```

합니다.

### Required columns

```text
TimeStamp
AI0_Vibration
AI1_Vibration
AI2_Current
```

선택 컬럼:

```text
Equipment_state
Idle
```

`Equipment_state`가 있으면 ground truth 기반의 event / false-alarm 지표를 계산할 수 있습니다.

실제 현장 raw stream처럼 `Equipment_state`가 없으면 detector는 **추론만 수행**합니다.

### CSV tail mode

파일에 행이 계속 append되는 상황을 흉내 내기 위해 tail 모드를 지원합니다.

```bash
python src/11_realtime_dashboard.py \
  --stream-path path/to/live_stream.csv \
  --tail
```

기본 polling interval:

```text
0.05 sec
```

완성된 CSV row만 순차적으로 읽으며, 아직 줄바꿈이 끝나지 않은 partial row는 다음 polling에서 이어서 처리합니다.

결과:

```text
outputs/11_realtime_dashboard/external_stream/
```

에 저장됩니다.

---

## 22. Dashboard Options

주요 옵션:

| Option | Default | Description |
|---|---:|---|
| `--normal-path` | `data/press_data_normal.csv` | Normal 원본 CSV |
| `--fault-path` | `data/outlier_data.csv` | Fault 원본 CSV |
| `--quantile` | `0.9999` | 최종 threshold quantile |
| `--seed` | `0` | 배포 모델 seed |
| `--calibration-fraction` | `0.20` | normal group 중 calibration 비율 |
| `--replay-speed` | `1.0` | demo replay 배속 |
| `--fps` | auto | replay 행/초 직접 지정 |
| `--demo-join-mode` | `continuous` | `continuous` / `gap` |
| `--demo-join-gap` | `2.0` | gap 모드의 block 간 간격 |
| `--plot-window` | `150` | 실시간 그래프 표시 points |
| `--feature-update-every` | `3` | feature panel 갱신 주기 |
| `--axis-update-every` | `5` | 축 자동 조정 주기 |
| `--reuse-results` | off | 기존 1~11 산출물 재사용 |
| `--skip-heavy-analysis` | off | Stage 9~11 생략 |
| `--stream-path` | none | 외부 raw CSV |
| `--tail` | off | 외부 CSV append 감시 |
| `--tail-poll-interval` | `0.05` | tail polling interval |

예:

```bash
# 전체 pipeline
python src/11_realtime_dashboard.py

# 기존 산출물 재사용
python src/11_realtime_dashboard.py --reuse-results

# 무거운 FP/FN·시각화 단계 생략
python src/11_realtime_dashboard.py --skip-heavy-analysis

# 5배 빠르게 demo replay
python src/11_realtime_dashboard.py --replay-speed 5

# gap reset 동작 시연
python src/11_realtime_dashboard.py \
  --demo-join-mode gap \
  --demo-join-gap 2

# 저장 모델로 외부 CSV 추론
python src/11_realtime_dashboard.py \
  --stream-path path/to/live_stream.csv

# 외부 CSV append 감시
python src/11_realtime_dashboard.py \
  --stream-path path/to/live_stream.csv \
  --tail
```

---

## 23. Output Artifacts

### Final model

```text
outputs/final_model/
├─ detector.pkl
├─ model_config.json
└─ thresholds.json
```

`model_config.json`에는 다음 재현성 정보가 기록됩니다.

```text
Python version
NumPy / Pandas / scikit-learn version
Git commit
input file hash
train rows
calibration rows
seed
quantile
```

### Dashboard

```text
outputs/11_realtime_dashboard/
├─ pipeline_stage_report.csv
├─ pipeline_run_summary.txt
├─ pipeline_dashboard.png
├─ stage_timing.json
├─ stage_markers/
├─ realtime_log.csv
├─ realtime_metrics.csv
└─ realtime_event_summary.csv
```

### Streaming CV

```text
outputs/12_streaming_cv/
├─ q_0.99/
├─ q_0.995/
├─ q_0.999/
├─ q_0.9995/
└─ q_0.9999/
```

각 quantile 폴더에서 주요 결과:

```text
streaming_cv_summary.csv
streaming_cv_repeat_metrics.csv
streaming_cv_fold_metrics.csv
streaming_cv_thresholds.csv
streaming_cv_event_details.csv
```

### Analysis outputs

```text
outputs/
├─ 5_2_mahalanobis_cv/
├─ 5_3_sample_level_cv/
├─ 6_three_detector_ensemble/
├─ 7_isolation_forest_baseline/
├─ 7_2_isolation_forest_streaming_cv/
├─ 7_3_delay_analysis/
├─ 8_fp_fn_analysis/
├─ 9_fn_visualization/
├─ 10_variable_effect_analysis/
├─ 11_realtime_dashboard/
└─ final_model/
```

---

## 24. Detection Delay Definition

delay는 다음 기준으로 계산합니다.

```text
Fault segment first timestamp
            ↓
First final alarm timestamp
            ↓
Detection delay
```

중요한 한계:

현재 Fault CSV는 Fault 구간 전체가 `Equipment_state=1`이므로 데이터만으로 실제 물리적 고장 발생 순간을 구분할 수 없습니다.

따라서 현재 delay는:

```text
"실제 물리 고장 발생 후 지연"
```

이 아니라:

```text
"fault segment 시작 시점 이후 첫 final alarm까지의 시간"
```

입니다.

### Missed event

탐지 실패 event는:

```text
delay = NaN
```

이며 탐지된 event의 평균 delay에는 포함하지 않습니다.

따라서 detector마다 탐지된 event 집합이 다르면 평균 delay를 동일한 event subset의 비교로 해석하면 안 됩니다.

---

## 25. Short Fault Events

nominal sampling interval이 0.1초이므로:

```text
3 samples  → 첫~마지막 timestamp 약 0.2 sec
10 samples → 첫~마지막 timestamp 약 0.9 sec
```

입니다.

따라서 매우 짧은 Fault는 긴 window detector가 충분한 sample을 확보하기 전에 끝날 수 있습니다.

이 때문에 KAMPact는:

```text
Window detector
+
Sample-level detector
```

를 함께 사용합니다.

sample-level detector는 window가 완전히 채워지지 않은 짧은 event에서도 raw sample 기준 score를 계산할 수 있습니다.

또한 최종 `P2`는 너무 짧은 이상 pulse가 바로 최종 alarm으로 확정되는 것을 일부 억제합니다.

---

## 26. Reproducibility

권장 환경:

```text
Python 3.11.x
```

현재 dependency 범위:

```text
numpy        >=2.0,<3
pandas       >=2.2,<4
scikit-learn >=1.4,<2
matplotlib   >=3.8,<4
tqdm         >=4.66,<5
```

최종 저장 모델이 생성된 환경 기록:

```text
Python        : 3.11.9
NumPy         : 2.4.6
Pandas        : 3.0.6
scikit-learn : 1.9.1
```

### Artifact note

현재 `outputs/final_model/model_config.json`에 저장된 모델 metadata는:

```text
code_commit : fe23345
created_at  : 2026-10-07T23:08:46
```

을 기록하고 있습니다.

현재 `main`의 HEAD는 이후 `0ab8fa3`으로 이동했으며, 이 최신 변경에는 **external CSV tail streaming** 기능이 추가되었습니다.

따라서 코드와 저장 모델의 metadata까지 현재 HEAD에 완전히 맞추려면 전체 pipeline을 다시 실행하여 `outputs/final_model/`을 재생성하는 것이 가장 재현성 높은 방법입니다.

---

## 27. Installation

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

대시보드는 Matplotlib interactive GUI를 사용하므로 일반적인 데스크톱 Python 환경에서 실행하는 것을 권장합니다.

---

## 28. Quick Start

가장 간단한 실행:

```bash
python src/11_realtime_dashboard.py
```

이 한 번의 실행으로:

```text
원본 분석
→ Idle labeling
→ Segment analysis
→ Window feature 생성
→ Window Mahalanobis CV
→ Sample-level Mahalanobis CV
→ Ensemble
→ Isolation Forest baseline
→ FP/FN 분석
→ Variable effect 분석
→ 최종 모델 학습 / calibration
→ 저장 model 생성
→ causal streaming replay
```

까지 연결됩니다.

---

## 29. Individual Scripts

### Step 1–3

```bash
python src/1_visualize_normal_outlier.py
python src/2_Classification_idle_sections.py
python src/3_time_structure_analysis.py
```

### Step 4–5: Window dataset

```bash
python src/4_make_window_dataset.py \
  --window-sec 1.0 \
  --step-sec 0.1 \
  --output-dir result/modeling_dataset_1.0_0.1

python src/4_make_window_dataset.py \
  --window-sec 0.5 \
  --step-sec 0.1 \
  --output-dir result/modeling_dataset_0.5_0.1
```

### Step 6: Window Mahalanobis CV

```bash
python src/5_2_run_mahalanobis.py \
  --multiscale \
  result/modeling_dataset_1.0_0.1 \
  result/modeling_dataset_0.5_0.1
```

### Step 7: Sample-level Mahalanobis CV

```bash
python src/5_3_run_sample_level_mahalanobis.py \
  --normal-path data/press_data_normal_with_idle.csv \
  --fault-path data/outlier_data.csv
```

### Step 8: Ensemble

```bash
python src/6_compare_three_detectors.py \
  --normal-path data/press_data_normal_with_idle.csv \
  --fault-path data/outlier_data.csv \
  --window-1.0 result/modeling_dataset_1.0_0.1 \
  --window-0.5 result/modeling_dataset_0.5_0.1
```

### Step 9: Isolation Forest baseline

```bash
python src/7_isolation_forest_baseline.py
```

### Step 10–11: FP/FN + variable analysis

```bash
python src/8_fp_fn_analysis.py

python src/9_visualize_fn_events.py
python src/10_variable_effect_analysis.py
```

### Streaming CV

```bash
python src/12_streaming_cv.py \
  --normal-path data/press_data_normal_with_idle.csv \
  --fault-path data/outlier_data.csv \
  --quantile 0.9999
```

모든 스크립트는 세부 인자를 `--help`로 확인할 수 있습니다.

---

## 30. Repository Structure

```text
KAMPact/
├─ data/
│  └─ .gitkeep
│
├─ result/
│  └─ visualize_normal_outlier/
│
├─ outputs/
│  ├─ 5_2_mahalanobis_cv/
│  ├─ 5_3_sample_level_cv/
│  ├─ 6_three_detector_ensemble/
│  ├─ 7_2_isolation_forest_streaming_cv/
│  ├─ 7_3_delay_analysis/
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
│  ├─ 7_2_isolation_forest_streaming_cv.py
│  ├─ 7_3_delay_analysis.py
│  ├─ 8_fp_fn_analysis.py
│  ├─ 9_visualize_fn_events.py
│  ├─ 10_variable_effect_analysis.py
│  ├─ 11_realtime_dashboard.py
│  ├─ 12_streaming_cv.py
│  ├─ core_detector.py
│  └─ dashboard_stage_monitor.py
│
├─ requirements.txt
├─ .gitignore
├─ LICENSE
└─ README.md
```

`result/`의 CSV와 `data/`의 원본 CSV는 `.gitignore`에 의해 기본적으로 추적하지 않습니다.

---

## 31. Script Roles

| File | Role |
|---|---|
| `1_visualize_normal_outlier.py` | 정상/고장 원본 시계열 시각화 |
| `2_Classification_idle_sections.py` | 정상 CSV의 Idle 라벨 생성 |
| `3_time_structure_analysis.py` | timestamp gap / segment 분석 |
| `4_make_window_dataset.py` | 0.5s / 1.0s window feature dataset 생성 |
| `5_run_mahalanobis.py` | 초기/단일 스케일 Mahalanobis 구현 |
| `5_2_run_mahalanobis.py` | 최종 window CV / multi-scale 평가 |
| `5_3_run_sample_level_mahalanobis.py` | sample-level Mahalanobis CV |
| `6_compare_three_detectors.py` | 3 detector ensemble 비교 |
| `7_isolation_forest_baseline.py` | historical Isolation Forest baseline |
| `7_2_isolation_forest_streaming_cv.py` | streaming protocol 기반 IF 비교 |
| `7_3_delay_analysis.py` | event별 delay 분해 분석 |
| `8_fp_fn_analysis.py` | FP/FN 조건 분석 |
| `9_visualize_fn_events.py` | FN event 시각화 |
| `10_variable_effect_analysis.py` | feature / interaction 분석 |
| `11_realtime_dashboard.py` | full pipeline + 최종 모델 + streaming UI |
| `12_streaming_cv.py` | deployment-aligned streaming CV |
| `core_detector.py` | 최종 causal detector engine |
| `dashboard_stage_monitor.py` | pipeline 실행 현황/로그/시각화 UI 지원 |

---

## 32. Limitations

### 32.1 Physical fault onset

현재 Fault 데이터의 `Equipment_state`는 Fault segment 전체에서 1이므로 실제 물리적인 고장 발생 순간은 별도로 알 수 없습니다.

따라서 delay는 데이터상 fault segment onset 기준입니다.

### 32.2 Independent final holdout

독립적으로 한 번도 사용하지 않은 final holdout 데이터가 별도로 제공된 상태가 아닙니다.

따라서 최종 성능은 **group-aware CV 기반의 개발 검증 결과**로 해석해야 합니다.

### 32.3 Model selection dependence

threshold와 최종 ensemble 구성은 개발 과정의 여러 CV 결과를 참고하여 결정되었습니다.

따라서 본 프로젝트의 CV 결과를 완전히 독립적인 confirmatory evaluation으로 해석하면 안 됩니다.

### 32.4 Idle interpretation

Idle을 정상 상태의 일부로 볼 수도 있지만, 본 프로젝트에서는 운전 정상과 분리된 분포로 보고 training/calibration에서 제외하며 false alarm을 별도 평가합니다.

### 32.5 Short fault events

매우 짧은 Fault event는 window detector가 충분한 sample을 확보하기 전에 종료될 수 있습니다.

### 32.6 Timestamp irregularity

nominal sampling interval은 0.1초이지만 실제 timestamp가 완전히 균일하지 않을 수 있습니다.

KAMPact는:

```text
gap
delay
slope
```

등에서 가능한 경우 실제 `TimeStamp`를 사용합니다.

---

## 33. Project Status

현재 저장소에 구현된 주요 항목:

```text
✓ Raw time-series analysis
✓ Idle classification
✓ Segment-aware preprocessing
✓ 1.0s / 0.5s window modeling
✓ 16-feature window detector
✓ Sample-level detector
✓ Ledoit-Wolf Mahalanobis
✓ OR_3 + P2 ensemble
✓ Isolation Forest baseline
✓ FP / FN analysis
✓ Variable effect / interaction analysis
✓ Final deployment model
✓ Causal row-by-row streaming detector
✓ End-to-end dashboard
✓ Deployment-aligned 5-fold × 5-repeat streaming CV
✓ External CSV replay
✓ External CSV tail streaming
```

KAMPact의 최종 형태는 단순한 오프라인 anomaly score 계산이 아니라,

```text
Raw Sensor Input
      ↓
Runtime Preprocessing
      ↓
Feature Extraction
      ↓
Anomaly Scoring
      ↓
Multi-scale Ensemble
      ↓
Persistence
      ↓
Final Alarm
```

을 실제 입력 흐름 그대로 수행하는 **causal streaming anomaly detection system**입니다.

---

## 34. License

This project is released under the [MIT License](LICENSE).
