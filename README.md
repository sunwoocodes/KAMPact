# KAMPact

### 진동·전류 시계열 기반 프레스 유압펌프 이상 조기탐지 및 오경보 분석

프레스 소성가공 공정에서 수집된 **진동 및 전류 시계열 데이터**를 기반으로 유압펌프의 이상 상태를 조기에 탐지하고, 정상 운전 구간과 Idle 구간에서 발생하는 오경보(False Alarm)를 분석하는 예지보전 AI 프로젝트입니다.

KAMPact는 단순한 정상/고장 분류가 아니라 다음 세 가지를 함께 고려합니다.

* **Fault Event Detection** — 실제 이상 이벤트를 놓치지 않는가
* **Early Detection** — 이상 이벤트를 얼마나 빠르게 탐지하는가
* **False Alarm Analysis** — 정상 운전 및 Idle 상태에서 불필요한 경보가 얼마나 발생하는가

---

## 1. Problem

프레스 유압펌프의 진동·전류 센서는 정상 운전 중에도 지속적인 변동을 보입니다.

또한 설비가 가동되지 않는 **Idle 상태**에서도 센서 신호가 존재하기 때문에 단순한 고정 임계값 기반 이상탐지는 다음과 같은 문제를 가질 수 있습니다.

```text
Normal Operation
      │
      ▼
Sensor Variation
      │
      ▼
False Alarm
```

KAMPact에서는 정상 운전 데이터의 **다변량 분포**를 학습하고, 현재 센서 상태가 정상 분포에서 얼마나 벗어났는지를 **Mahalanobis Distance**로 계산합니다.

이를 통해 여러 센서 간 상관관계를 고려하면서 이상 상태를 탐지합니다.

---

## 2. Project Goals

### Main Goals

1. 진동·전류 시계열 기반 이상 탐지
2. Fault Event의 조기 탐지
3. 정상 운전 Cycle의 False Alarm 분석
4. Idle 상태의 False Alarm 분석
5. Window Scale에 따른 탐지 성능 비교
6. Sample-level과 Window-level detector 비교
7. Group-based Cross Validation을 통한 시계열 데이터 누수 방지
8. 여러 detector를 결합한 최종 Ensemble 구성

---

## 3. Dataset

본 프로젝트는 프레스 유압펌프의 진동 및 전류 시계열 데이터를 사용합니다.

### Sensors

| Column            | Description |
| ----------------- | ----------- |
| `TimeStamp`       | 센서 측정 시각    |
| `AI0_Vibration`   | 진동 센서 0     |
| `AI1_Vibration`   | 진동 센서 1     |
| `AI2_Current`     | 전류 센서       |
| `Equipment_state` | 설비 상태       |
| `Idle`            | Idle 상태 여부  |

### Data Structure

```text
Normal Data
├── Normal Operation
└── Idle

Fault Data
└── Fault Events
```

사용되는 입력 파일:

```text
data/
├── press_data_normal_with_idle.csv
└── outlier_data.csv
```

원본 데이터는 저장소에 포함하지 않습니다.

---

## 4. Project Pipeline

```text
Raw Sensor Data
        │
        ▼
┌──────────────────────────┐
│ 1. Data Visualization    │
│    Normal / Fault        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 2. Idle Classification    │
│    Normal / Idle         │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 3. Time Structure        │
│    Gap / Segment         │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 4. Window Dataset        │
│    Feature Extraction   │
└────────────┬─────────────┘
             │
             ├──────────────────┐
             ▼                  ▼
┌────────────────────┐  ┌─────────────────────┐
│ 5_2 Window Model   │  │ 5_3 Sample Model    │
│                    │  │                     │
│ 1.0s / 0.5s        │  │ Raw + Difference    │
│ Mahalanobis        │  │ Causal Aggregation  │
└──────────┬─────────┘  └──────────┬──────────┘
           │                       │
           └───────────┬───────────┘
                       ▼
             ┌──────────────────────┐
             │ 6. Three Detector    │
             │    Ensemble          │
             └──────────┬───────────┘
                        ▼
                   OR_3 + P2
                        │
                        ▼
                 Final Alarm
```

---

# 5. Time-Series Preprocessing

## 5.1 Sampling Gap

Timestamp 간격을 분석하여 일정 수준 이상의 gap이 발생하면 서로 다른 시계열 segment로 분리합니다.

현재 기본 기준:

```text
Gap > 0.5 sec
→ New Segment
```

이를 통해 하나의 독립적인 운전 구간을 `group_id` 단위로 관리합니다.

---

## 5.2 Idle Classification

정상 데이터 내부의 Idle 구간은 정상 운전과 다른 신호 특성을 가질 수 있으므로 별도로 분류합니다.

```text
Idle = 0
→ Normal Operation

Idle = 1
→ Idle
```

Idle 데이터는 정상 운전 모델의 학습에 사용하지 않고 **독립적인 오경보 평가 대상**으로 사용합니다.

---

## 5.3 Segment-based Split

시계열 데이터를 window 단위로 무작위 분할하면 같은 원본 segment에서 생성된 유사한 window가 train/test에 동시에 존재할 수 있습니다.

이러한 데이터 누수를 방지하기 위해 다음 순서로 처리합니다.

```text
Raw Data
   ↓
Segment Creation
   ↓
Group-based Split
   ↓
Window Generation
```

따라서 동일한 `group_id`가 서로 다른 split에 동시에 포함되지 않도록 구성합니다.

---

# 6. Window Feature Engineering

각 window에서 센서별 통계 특징을 추출합니다.

### Feature Set

각 센서마다 다음 5개 feature를 계산합니다.

```text
Mean
Standard Deviation
RMS
Peak-to-Peak (PTP)
Slope
```

3개 센서 × 5개 feature:

```text
3 × 5 = 15
```

추가로 두 진동 센서의 상관관계를 계산합니다.

```text
AI0_AI1_corr
```

따라서 Window detector의 최종 입력은 **16개 feature**입니다.

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

---

# 7. Mahalanobis Anomaly Detection

## 7.1 Mahalanobis Distance

정상 운전 데이터의 평균과 공분산 구조를 이용하여 각 관측치가 정상 분포에서 얼마나 떨어져 있는지를 계산합니다.

개념적으로:

```text
Normal Distribution
        │
        ├── Close → Normal
        │
        └── Far   → Anomaly
```

Mahalanobis Distance는 feature 간 상관관계를 고려하기 때문에 단순한 Euclidean Distance보다 다변량 센서 데이터의 분포를 반영할 수 있습니다.

---

## 7.2 Covariance Estimation

다음 공분산 추정 방법을 지원합니다.

```text
Empirical Covariance
Ledoit-Wolf
OAS
```

현재 최종 ensemble 실험에서는 **Ledoit-Wolf**를 사용합니다.

---

## 7.3 Signed Log Transformation

feature scale 차이와 extreme value의 영향을 완화하기 위해 다음 변환을 적용할 수 있습니다.

```text
sign(x) × log(1 + |x|)
```

양수/음수 방향성을 유지하면서 큰 값의 영향을 완화합니다.

---

# 8. Threshold

Anomaly score를 정상/이상으로 구분하기 위한 threshold를 두 가지 방식으로 지원합니다.

## F1-based Threshold

Validation 데이터의 F1-score가 최대가 되는 threshold를 선택합니다.

```text
Validation Scores
      ↓
Candidate Thresholds
      ↓
F1 Evaluation
      ↓
Best F1 Threshold
```

## Normal Quantile Threshold

Validation의 정상 데이터 score 분포를 기준으로 quantile threshold를 계산합니다.

예:

```text
Normal Scores
     ↓
99.99 Percentile
     ↓
Threshold
```

최종 Three-Detector Ensemble에서는 False Alarm을 줄이기 위해 다음 설정을 사용했습니다.

```text
normal_quantile = 0.9999
```

---

# 9. Window-level Detector

Window-level Mahalanobis detector는 서로 다른 시간 길이를 사용하여 이상 패턴을 탐지합니다.

현재 사용되는 두 가지 scale:

```text
1.0 sec Window
0.5 sec Window
```

두 모델은 동일한 16개 feature를 사용하지만 서로 다른 시간 범위의 신호 패턴을 관찰합니다.

---

# 10. Sample-level Detector

`5_3_run_sample_level_mahalanobis.py`에서는 원본 sample 단위로 Mahalanobis Distance를 계산합니다.

지원되는 feature 구성:

```text
raw
raw_diff
raw_diff_roll3
```

현재 ensemble에서는:

```text
Feature = raw_diff
Aggregation = mean3
```

을 사용합니다.

즉 현재 sample score와 과거 2개 sample score를 포함한 **3-sample causal mean**을 사용합니다.

```text
t-2 ─┐
t-1 ─┼─→ Mean → Alarm
t   ─┘
```

미래 sample을 사용하지 않기 때문에 causal detection 구조를 유지합니다.

---

# 11. Three-Detector Ensemble

최종 ensemble은 다음 세 detector를 사용합니다.

```text
1.0s Window Mahalanobis
        +
0.5s Window Mahalanobis
        +
Sample-level Causal Mahalanobis
```

각 detector의 결과를 원본 sample timeline 기준으로 통합합니다.

### OR Ensemble

세 detector 중 하나라도 이상으로 판단하면 후보 alarm을 생성합니다.

```text
1.0s Window ─┐
0.5s Window ─┼── OR ──→ OR_3
Sample       ┘
```

이 방식은 서로 다른 detector가 놓치는 fault를 상호 보완하는 것을 목적으로 합니다.

---

# 12. Persistence Filter

OR 결합만 사용할 경우 단발성 score spike가 최종 alarm으로 이어질 수 있습니다.

이를 줄이기 위해 **2-sample persistence**를 적용합니다.

```text
OR_3

Sample 1 → Alarm
Sample 2 → Alarm
       ↓
Final Alarm
```

현재 최종 구조:

```text
Detector 1
Detector 2
Detector 3
     │
     ▼
   OR_3
     │
     ▼
Persistence P2
     │
     ▼
Final Alarm
```

최종 설정:

```text
P2 = 2 consecutive alarm samples
```

Persistence는 순간적인 정상 신호 변동에 의한 오경보를 줄이는 대신 매우 짧은 fault event를 놓칠 수 있는 trade-off를 갖습니다.

---

# 13. Evaluation Metrics

KAMPact에서는 일반적인 sample-level F1만 사용하지 않고 **event-level / cycle-level 평가**를 함께 수행합니다.

| Metric               | Description                          |
| -------------------- | ------------------------------------ |
| Event Detection Rate | Fault event 중 탐지된 event 비율           |
| Detection Delay      | 정의된 fault onset부터 최초 alarm까지의 시간     |
| F1                   | Sample-level alarm classification 성능 |
| Precision            | Alarm 중 실제 fault의 비율                 |
| Recall               | 실제 fault 중 탐지된 비율                    |
| Normal FA Cycle Rate | 정상 cycle 중 1회 이상 오탐 발생 비율            |
| FA Episodes          | 정상 구간에서 발생한 오탐 episode 수             |
| Idle FA Cycle Rate   | Idle cycle 중 오탐 발생 비율                |

### Evaluation Principle

이번 프로젝트에서는 특히 다음을 함께 봅니다.

```text
Detection
    +
Delay
    +
False Alarm
```

따라서 F1 하나만으로 모델을 판단하지 않습니다.

---

# 14. Detection Delay Definition

현재 데이터에는 실제 물리적 고장 발생 시각에 대한 별도 ground-truth timestamp가 없기 때문에 데이터에서 정의된 fault onset을 사용합니다.

## Window Detector

```text
Fault Onset
     ↓
First Alarm Window
     ↓
Window End
```

window detector의 alarm은 미래 sample을 참조하지 않도록 **causal shifted timeline**에 매핑합니다.

## Sample Detector

```text
Fault Segment Start
        ↓
First Alarm Sample
```

따라서 본 프로젝트의 Detection Delay는 실제 물리적 고장이 발생하기 이전에 예측한 시간을 의미하는 것이 아니라,

> **데이터에서 정의한 이상 onset 이후 최초 alarm까지의 탐지 지연**

입니다.

---

# 15. Cross Validation

시계열 데이터의 segment leakage를 방지하기 위해 `group_id` 단위의 Group-based K-Fold를 사용합니다.

### Window / Sample Individual Experiments

기존 `5_2`, `5_3` 실험은 **4-fold × 5-repeat** 기반으로 구성되어 있습니다.

### Final Ensemble

최종 Three-Detector Ensemble은:

```text
5-fold
×
5 repeats
```

로 평가했습니다.

각 repeat마다 fold assignment를 변경하고 결과를 `mean ± std`로 집계합니다.

---

# 16. Final Ensemble Result

현재 Three-Detector Ensemble의 주요 결과입니다.

| Detector / Ensemble |    Event Detection |              Delay |           Sample F1 |  Normal FA Cycle |
| ------------------- | -----------------: | -----------------: | ------------------: | ---------------: |
| 1.0s Window         |     73.33% ± 2.61% |     1.156 ± 0.027s |     0.7422 ± 0.0146 |     0.47 ± 0.22% |
| 0.5s Window         |     76.19% ± 0.00% |     0.776 ± 0.015s |     0.7894 ± 0.0099 |     0.47 ± 0.18% |
| Sample-level        |     85.71% ± 0.00% |     0.571 ± 0.002s |     0.5817 ± 0.0040 |     0.98 ± 0.28% |
| OR_3                | **99.05% ± 2.13%** | **0.470 ± 0.019s** | **0.8482 ± 0.0134** |     1.17 ± 0.14% |
| **OR_3 + P2**       | **91.43% ± 2.13%** | **0.571 ± 0.014s** | **0.8300 ± 0.0136** | **0.90 ± 0.11%** |
| OR_3 + P3           |     79.05% ± 2.61% |     0.729 ± 0.022s |     0.8109 ± 0.0133 |     0.67 ± 0.18% |
| 2-of-3 Exact        |     71.43% ± 3.37% |     0.897 ± 0.013s |     0.7633 ± 0.0088 |     0.39 ± 0.14% |
| 2-of-3 Tolerance    |     72.38% ± 2.13% |     0.854 ± 0.010s |     0.7764 ± 0.0097 |     0.43 ± 0.16% |

### Final Configuration

현재 프로젝트에서는 다음 조합을 최종 ensemble 구성으로 사용합니다.

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

### Selected Settings

```text
Window covariance : Ledoit-Wolf
Window threshold  : Normal Quantile
Window quantile   : 0.9999
Window k          : 2

Sample feature    : raw_diff
Sample covariance : Ledoit-Wolf
Sample threshold  : Normal Quantile
Sample quantile   : 0.9999
Sample aggregation: mean
Sample agg-k      : 3

Cross Validation  : 5-fold × 5 repeats
```

`OR_3`는 높은 fault-event 탐지율을 제공하지만 정상 운전 중 false alarm이 상대적으로 증가합니다.

반면 `OR_3_P2`는 2-sample persistence를 적용하여 정상 cycle false alarm을 줄이면서 높은 event detection을 유지하는 구조입니다.

---

# 17. Persistence Trade-off Analysis

총 21개의 fault segment를 5회 반복하여 총 105회의 event detection을 수행했습니다.

### OR_3

```text
105 trials
104 detected
1 missed

Detection Rate
= 104 / 105
= 99.05%
```

### OR_3_P2

```text
105 trials
96 detected
9 missed

Detection Rate
= 96 / 105
= 91.43%
```

P2에서 발생한 9회의 미탐은 특정 짧은 fault segment에 집중되었습니다.

```text
fault_5
→ 3 samples
→ 5회 모두 missed

fault_19
→ 10 samples
→ 4회 missed
```

즉 persistence 적용으로 발생한 탐지 손실은 모든 fault에 균등하게 나타난 것이 아니라 **짧은 fault event에 집중되는 특성**을 보였습니다.

반면 정상 cycle false alarm은:

```text
OR_3
1.17%

OR_3_P2
0.90%
```

으로 감소했습니다.

따라서 P2는 순간적인 이상 spike를 억제하는 대신 매우 짧은 event를 놓칠 수 있는 **탐지율-오경보 간 trade-off**를 갖습니다.

---

# 18. Why Three Detectors?

각 detector는 서로 다른 시간 해상도를 관찰합니다.

```text
1.0s Window
→ 상대적으로 긴 시간의 통계적 패턴

0.5s Window
→ 더 짧은 시간의 변화

Sample-level
→ 개별 sample 수준의 변화
```

따라서 단일 detector가 모든 이상 패턴을 동일하게 포착하기 어렵다는 문제를 보완합니다.

```text
Long-term Pattern
        │
        ├── 1.0s Window
        │
Short-term Pattern
        │
        ├── 0.5s Window
        │
Point-level Change
        │
        └── Sample-level
```

이 세 결과를 OR로 결합하면 일부 detector가 놓친 fault event를 다른 detector가 보완할 수 있습니다.

---

# 19. False Alarm Analysis

KAMPact는 False Alarm을 단순한 전체 FPR로만 보지 않습니다.

### Normal Cycle

정상 운전 cycle 중 alarm이 한 번이라도 발생하면 해당 cycle을 false-alarm cycle로 계산합니다.

```text
Normal Cycle
       │
       ├── No Alarm → Normal
       │
       └── Alarm    → False Alarm Cycle
```

### False Alarm Episode

연속적으로 발생하는 alarm은 하나의 episode로 계산합니다.

```text
0 0 1 1 1 0 0 1 0

→ 2 False Alarm Episodes
```

이를 통해 단순 alarm 횟수뿐 아니라 **오경보가 얼마나 반복적으로 발생하는지**를 분석합니다.

### Idle

Idle 구간은 별도로 계산합니다.

현재 ensemble 실험에서는:

```text
Idle FA Cycle Rate = 0%
```

로 기록되었습니다.

---

# 20. Repository Structure

```text
KAMPact/
│
├── data/
│   └── .gitkeep
│
├── result/
│   └── visualize_normal_outlier/
│
├── outputs/
│   ├── 5_2_mahalanobis_cv/
│   │   ├── cv_comparison.csv
│   │   ├── cv_repeat_results.csv
│   │   ├── cv_event_details.csv
│   │   └── cv_event_detection_frequency.csv
│   │
│   ├── 5_3_sample_level_cv/
│   │   ├── sample_cv_comparison.csv
│   │   ├── sample_cv_repeat_results.csv
│   │   ├── sample_cv_event_details.csv
│   │   └── sample_cv_event_detection_frequency.csv
│   │
│   └── 6_three_detector_ensemble/
│       ├── ensemble_summary.csv
│       ├── ensemble_repeat_results.csv
│       └── ensemble_event_details.csv
│
├── src/
│   ├── 1_visualize_normal_outlier.py
│   ├── 2_Classification_idle_sections.py
│   ├── 3_time_structure_analysis.py
│   ├── 4_make_window_dataset.py
│   ├── 5_run_mahalanobis.py
│   ├── 5_2_run_mahalanobis.py
│   ├── 5_3_run_sample_level_mahalanobis.py
│   └── 6_compare_three_detectors.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 21. Script Description

| Script                                | Purpose                                |
| ------------------------------------- | -------------------------------------- |
| `1_visualize_normal_outlier.py`       | Normal / Fault 시계열 시각화                 |
| `2_Classification_idle_sections.py`   | Normal 데이터의 Idle 구간 분류                 |
| `3_time_structure_analysis.py`        | Timestamp, sampling gap, segment 구조 분석 |
| `4_make_window_dataset.py`            | Window 생성 및 feature extraction         |
| `5_run_mahalanobis.py`                | 기본 Window Mahalanobis detector         |
| `5_2_run_mahalanobis.py`              | Group K-Fold 및 multi-scale Window 평가   |
| `5_3_run_sample_level_mahalanobis.py` | Sample-level causal Mahalanobis 평가     |
| `6_compare_three_detectors.py`        | 3개 detector ensemble 및 persistence 비교  |

---

# 22. Installation

### Requirements

```text
Python 3.11
```

필요한 패키지는 `requirements.txt`에 정의되어 있습니다.

```bash
pip install -r requirements.txt
```

주요 패키지:

```text
numpy
pandas
scikit-learn
matplotlib
tqdm
```

---

# 23. Reproduction

## Step 1. Visualization

```bash
python ./src/1_visualize_normal_outlier.py
```

---

## Step 2. Idle Classification

```bash
python ./src/2_Classification_idle_sections.py
```

---

## Step 3. Time Structure Analysis

```bash
python ./src/3_time_structure_analysis.py
```

---

## Step 4. Window Dataset

### 1.0s Window

```bash
python ./src/4_make_window_dataset.py \
    --window-sec 1.0 \
    --step-sec 0.1
```

### 0.5s Window

```bash
python ./src/4_make_window_dataset.py \
    --window-sec 0.5 \
    --step-sec 0.1
```

기본적으로 다음과 같은 결과가 생성됩니다.

```text
model_windows.csv
```

및 데이터 품질/segment 관련 결과 파일이 생성됩니다.

---

## Step 5-1. Basic Mahalanobis

```bash
python ./src/5_run_mahalanobis.py \
    --input result/modeling_dataset_1.0_0.1/model_windows.csv
```

---

## Step 5-2. Window-level Cross Validation

```bash
python ./src/5_2_run_mahalanobis.py \
    --inputs \
    result/modeling_dataset_1.0_0.1 \
    result/modeling_dataset_0.5_0.1
```

Multi-scale evaluation:

```bash
python ./src/5_2_run_mahalanobis.py \
    --multiscale \
    result/modeling_dataset_1.0_0.1 \
    result/modeling_dataset_0.5_0.1
```

---

## Step 5-3. Sample-level Detection

```bash
python ./src/5_3_run_sample_level_mahalanobis.py \
    --normal-path data/press_data_normal_with_idle.csv \
    --fault-path data/outlier_data.csv
```

---

## Step 6. Three-Detector Ensemble

최종 ensemble 비교:

```bash
python ./src/6_compare_three_detectors.py
```

현재 기본 설정은 다음과 같습니다.

```text
Window threshold = normal_quantile
Window quantile  = 0.9999
Window k         = 2

Sample feature   = raw_diff
Sample threshold = normal_quantile
Sample quantile  = 0.9999
Sample aggregation = mean
Sample agg-k     = 3

Covariance       = Ledoit-Wolf

Folds            = 5
Repeats          = 5
```

---

# 24. Output Files

## Window CV

```text
outputs/5_2_mahalanobis_cv/
```

주요 파일:

```text
cv_comparison.csv
cv_repeat_results.csv
cv_event_details.csv
cv_event_detection_frequency.csv
```

### `cv_comparison.csv`

전체 system의 평균 성능을 저장합니다.

### `cv_repeat_results.csv`

각 repeat별 상세 성능을 저장합니다.

### `cv_event_details.csv`

fault event별 detection / delay 정보를 저장합니다.

### `cv_event_detection_frequency.csv`

각 fault event가 얼마나 안정적으로 탐지되었는지 확인할 수 있습니다.

---

## Sample CV

```text
outputs/5_3_sample_level_cv/
```

주요 파일:

```text
sample_cv_comparison.csv
sample_cv_repeat_results.csv
sample_cv_event_details.csv
sample_cv_event_detection_frequency.csv
```

---

## Three-Detector Ensemble

```text
outputs/6_three_detector_ensemble/
```

주요 파일:

```text
ensemble_summary.csv
ensemble_repeat_results.csv
ensemble_event_details.csv
```

### `ensemble_summary.csv`

각 ensemble 전략의 평균 및 표준편차를 저장합니다.

### `ensemble_repeat_results.csv`

각 repeat의 개별 성능을 저장합니다.

### `ensemble_event_details.csv`

각 fault segment의 탐지 여부와 delay를 저장합니다.

---

# 25. Key Findings

현재 실험에서 확인된 핵심 결과는 다음과 같습니다.

### 1. Multi-scale detector

1.0s와 0.5s Window는 서로 다른 시간 범위의 이상 패턴을 관찰합니다.

### 2. Sample-level detector

Sample-level detector는 더 세밀한 시간 단위의 이상을 관찰할 수 있지만 window 통계 기반 모델과는 다른 특성을 보입니다.

### 3. OR Ensemble

세 detector를 OR 방식으로 결합하면 fault-event detection이 크게 증가했습니다.

```text
OR_3
Event Detection = 99.05%
Delay            = 0.470s
```

### 4. Persistence

OR_3에 2-sample persistence를 적용하면:

```text
Event Detection
99.05% → 91.43%

Delay
0.470s → 0.571s

Normal FA Cycle
1.17% → 0.90%
```

으로 변화했습니다.

즉 persistence는 순간적인 오경보를 억제하는 대신 짧은 fault event의 탐지율을 일부 희생합니다.

---

# 26. Limitations

현재 데이터와 평가 방식에는 다음과 같은 한계가 있습니다.

### Fault Onset Ground Truth

현재 Detection Delay는 물리적 장비에서 실제 고장이 시작된 시점을 직접 측정한 값이 아니라 데이터에서 정의한 fault onset을 기준으로 계산합니다.

### Short Fault Events

매우 짧은 fault는 Window 기반 detector나 persistence 조건에서 충분한 연속 정보를 확보하기 어렵습니다.

### Dataset Size

평가 대상 fault event 수가 제한적이므로 다양한 설비 상태와 운전 조건에서의 일반화 성능을 추가로 검증할 필요가 있습니다.

### False Alarm Generalization

현재 정상 및 Idle 데이터에서의 오경보를 분석했지만, 실제 현장에서는 더 다양한 부하·속도·운전 조건이 존재할 수 있습니다.

---

# 27. Future Work

향후에는 다음과 같은 확장이 가능합니다.

```text
Frequency-domain Features
        +
Additional Sensor Information
        +
Operating-condition-aware Detection
        +
More Diverse Fault Data
        +
Adaptive Threshold
```

특히 실제 운전 환경에서는 운전 조건에 따라 정상 분포 자체가 달라질 수 있으므로 운전 조건별 정상 모델 또는 adaptive threshold를 적용하는 방향을 고려할 수 있습니다.

---

# 28. Dataset / License

본 저장소는 공모전 및 프로젝트 목적의 분석 코드와 실험 결과를 제공합니다.

원본 데이터는 저장소에 포함하지 않습니다.

데이터의 사용 및 재배포는 해당 데이터셋 제공기관의 이용 조건과 공모전 규정을 따라야 합니다.

---

# 29. Summary

KAMPact는 다음과 같은 구조를 갖는 **multivariate time-series anomaly detection pipeline**입니다.

```text
Sensor Data
    ↓
Idle / Segment Analysis
    ↓
Window Feature Engineering
    ↓
Mahalanobis Detection
    ├── 1.0s Window
    ├── 0.5s Window
    └── Sample-level
          ↓
        OR_3
          ↓
   Persistence P2
          ↓
     Final Alarm
```

핵심은 단순한 고장 분류가 아니라,

```text
Fault Detection
      +
Early Detection
      +
False Alarm Analysis
```

를 동시에 고려하는 것입니다.

현재 최종 ensemble 설정은 **1.0s Window + 0.5s Window + Sample-level Mahalanobis → OR_3 → 2-sample Persistence(P2)** 입니다.
