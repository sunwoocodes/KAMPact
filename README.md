# KAMPact

**진동·전류 시계열 기반 프레스 유압펌프 이상 조기탐지 및 오경보 분석**

KAMPact는 프레스 유압펌프에서 수집된 **진동 및 전류 시계열 데이터**를 이용하여 정상 운전 패턴을 학습하고, 정상 패턴에서 벗어나는 이상을 탐지하는 예지보전 분석 프로젝트입니다.

단순한 고장 분류가 아니라 다음 세 가지를 함께 평가하는 것을 목표로 합니다.

```text
Fault Detection
      +
Detection Delay
      +
False Alarm Analysis
```

> **Detection Delay 주의:** 본 프로젝트의 delay는 실제 물리적 고장이 발생하기 전의 예지시간이 아니라, **데이터에서 정의된 fault onset 이후 최초 alarm까지의 시간**입니다.

---

## 1. Problem

프레스 공정의 유압펌프는 진동과 전류 신호를 통해 운전 상태를 관찰할 수 있습니다.

본 프로젝트에서는 다음 세 가지 센서 신호를 사용합니다.

```text
AI0_Vibration
AI1_Vibration
AI2_Current
```

주요 문제는 다음과 같습니다.

```text
정상 운전
   ↓
미세한 이상 변화
   ↓
이상 상태
```

이러한 변화를 가능한 한 빠르게 탐지하면서도 정상 운전 중 발생하는 false alarm을 줄이는 것이 핵심입니다.

---

## 2. Project Goals

KAMPact의 목표는 다음과 같습니다.

1. 정상 운전 데이터만으로 정상 분포를 모델링
2. 다양한 시간 범위에서 이상 신호 탐지
3. fault event 단위의 detection rate 측정
4. fault onset 이후 detection delay 측정
5. 정상 cycle 및 Idle cycle의 false alarm 분석
6. 서로 다른 detector를 결합하여 탐지 성능 보완
7. 짧은 fault event에서 persistence가 미치는 영향 분석

---

## 3. Dataset

원본 데이터는 저장소에 포함하지 않습니다.

사용 데이터:

```text
Normal : 20,000 rows
Fault  : 600 rows
```

Fault 데이터는 총 **21개 fault segment/event**로 구성됩니다.

주요 컬럼:

| Column            | Description  |
| ----------------- | ------------ |
| `TimeStamp`       | 측정 timestamp |
| `AI0_Vibration`   | 진동 센서 0      |
| `AI1_Vibration`   | 진동 센서 1      |
| `AI2_Current`     | 전류 센서        |
| `Equipment_state` | 설비 상태        |
| `Idle`            | Idle 상태 여부   |

### 데이터 상태

```text
Equipment_state = 0
→ 정상

Equipment_state >= 1
→ Fault
```

`Idle=1`인 구간은 정상 운전과 별도로 관리합니다.

Idle 데이터는 모델 학습이나 threshold 산정에 사용하지 않고 **평가 단계에서 false alarm만 측정**합니다.

---

## 4. Time-Series Structure

timestamp gap을 이용하여 데이터를 press cycle 단위의 segment로 분리합니다.

```text
Gap > 0.5 sec
    ↓
New Segment
```

이를 통해 서로 다른 운전 cycle이 하나의 연속 시계열로 연결되는 것을 방지합니다.

또한 Group-based split을 사용하여 동일 segment가 train/validation/test에 동시에 들어가는 **segment leakage를 방지**합니다.

---

## 5. Idle Handling

Normal 데이터에는 실제 장비가 동작하지 않는 Idle 구간이 포함되어 있습니다.

Idle 구간은 정상 운전과 다른 신호 특성을 가질 수 있으므로 모델 학습에서 제외합니다.

평가 구조:

```text
Normal operation
    ↓
Train / Validation / Test

Idle
    ↓
Test only
    ↓
False Alarm evaluation
```

현재 최종 ensemble 평가에서 반복마다:

```text
Normal cycles : 511
Idle cycles   : 88
```

을 사용합니다.

---

## 6. Window Feature Engineering

Window-based detector는 각 시간 구간에서 통계적 특징을 계산합니다.

각 센서마다 다음 5개 feature를 생성합니다.

```text
Mean
Std
RMS
Peak-to-Peak
Slope
```

센서 3개에 대해:

```text
3 × 5 = 15 features
```

추가로:

```text
AI0 ↔ AI1 correlation
```

을 사용합니다.

따라서 총:

```text
16 features
```

입니다.

### Feature 구성

```text
AI0_Vibration
 ├── mean
 ├── std
 ├── rms
 ├── ptp
 └── slope

AI1_Vibration
 ├── mean
 ├── std
 ├── rms
 ├── ptp
 └── slope

AI2_Current
 ├── mean
 ├── std
 ├── rms
 ├── ptp
 └── slope

AI0_AI1_corr
```

---

## 7. Window Scales

두 가지 시간 범위의 window detector를 사용합니다.

```text
1.0 sec Window
0.5 sec Window
```

공통 step:

```text
0.1 sec
```

즉 서로 다른 시간 범위에서 신호 패턴을 관찰합니다.

```text
1.0s
→ 비교적 긴 통계적 변화

0.5s
→ 짧은 시간의 변화
```

---

## 8. Mahalanobis Anomaly Detection

KAMPact는 정상 데이터의 다변량 분포를 학습하기 위해 Mahalanobis distance를 사용합니다.

Mahalanobis distance:

$$
D^2(x)=(x-\mu)^T\Sigma^{-1}(x-\mu)
$$

여기서:

* \(x\): 현재 feature vector
* \(\mu\): 정상 데이터 평균
* \(\Sigma\): 정상 데이터 covariance matrix

정상 분포에서 멀어질수록 anomaly score가 증가합니다.

### Covariance Estimator

다음 방법을 비교할 수 있습니다.

```text
Empirical Covariance
Ledoit-Wolf
OAS
```

최종 ensemble에서는:

```text
Ledoit-Wolf
```

를 사용합니다.

---

## 9. Signed Log Transformation

센서 feature의 scale 차이와 큰 값의 영향을 완화하기 위해 다음 변환을 적용합니다.

$$
x' = sign(x)\log(1+|x|)
$$

이를 통해 큰 값에 의한 Mahalanobis score의 과도한 변화를 완화합니다.

---

## 10. Threshold

Threshold는 두 가지 방법을 지원합니다.

```text
F1-based threshold
Normal-quantile threshold
```

### Normal Quantile

Normal-quantile 방식은 validation 데이터의 **정상 score만 사용**하여 threshold를 계산합니다.

현재 최종 ensemble 설정:

```text
Quantile = 0.9999
```

즉 validation 정상 분포의 매우 높은 분위수를 threshold로 사용합니다.

중요하게도:

```text
Train → 정상 데이터만
Threshold → validation 정상 score만
Test → 최종 평가
```

구조를 유지합니다.

Fault 데이터 및 test fold는 normal-quantile threshold 계산에 사용하지 않습니다.

---

## 11. Window Detector

Window detector는 다음 구조입니다.

```text
Raw Time Series
      ↓
Window
      ↓
16 Features
      ↓
Signed Log
      ↓
Mahalanobis Distance
      ↓
Threshold
      ↓
Alarm
```

사용하는 window:

```text
1.0s Window
0.5s Window
```

최종 ensemble에서는 두 결과를 각각 raw timeline으로 매핑하여 결합합니다.

---

## 12. Causal Window Mapping

Window feature는 window 내부의 데이터를 사용하여 계산되므로, alarm을 실제 detection timeline에서 미래 정보를 사용하지 않는 방식으로 처리해야 합니다.

KAMPact에서는 window alarm을 **window duration만큼 causal timeline 방향으로 이동**시켜 feature 계산 이후에 alarm이 활성화되도록 합니다.

개념적으로:

```text
Original Window
[start -------- end]

                  ↓ causal shift

Alarm Timeline
             [start -------- end]
```

따라서 window detector의 alarm이 feature 계산 이전 시점에 나타나지 않도록 구성합니다.

---

## 13. Sample-level Detector

`5_3_run_sample_level_mahalanobis.py`에서는 window 통계 대신 원본 sample 수준에서 Mahalanobis distance를 계산합니다.

현재 최종 ensemble에서는:

```text
Feature Set : raw_diff
Aggregation : mean
Aggregation k : 3
```

을 사용합니다.

### raw_diff

원본 센서 값과 연속 sample 간 차이를 함께 사용합니다.

```text
Raw
+
First Difference
```

즉 총 6개 sample-level feature를 사용합니다.

```text
AI0_Vibration
AI1_Vibration
AI2_Current

d_AI0_Vibration
d_AI1_Vibration
d_AI2_Current
```

---

## 14. Causal Mean Aggregation

sample score에 대해 최근 3개 sample의 평균을 계산합니다.

```text
t-2 ─┐
t-1 ─┼─→ Mean3 → Alarm
t   ─┘
```

미래 sample은 사용하지 않으므로 causal 구조를 유지합니다.

```text
Aggregation = mean3
```

은 sample-level detector의 순간적인 noise를 완화하면서도 지나치게 긴 aggregation으로 인한 delay 증가를 피하기 위한 설정입니다.

---

## 15. Three-Detector Ensemble

최종 ensemble은 서로 다른 시간 해상도의 세 detector를 결합합니다.

```text
1.0s Window Mahalanobis
          +
0.5s Window Mahalanobis
          +
Sample-level Mahalanobis
```

각 detector의 alarm을 원본 sample timeline으로 통합합니다.

### OR Combination

세 detector 중 하나라도 alarm을 발생시키면 후보 alarm을 생성합니다.

```text
1.0s Window ─┐
0.5s Window ─┼── OR ──→ OR_3
Sample       ┘
```

서로 다른 detector가 놓치는 fault를 상호 보완하는 것이 목적입니다.

---

## 16. Persistence Filter

OR_3는 민감도가 높은 대신 순간적인 score spike에 의해 false alarm이 발생할 수 있습니다.

이를 줄이기 위해 최종 ensemble에서 2-sample persistence를 적용합니다.

```text
OR_3
 ↓
2 consecutive alarm samples
 ↓
Final Alarm
```

### P2

```text
P2 = 2 consecutive alarm samples
```

즉 하나의 sample에서만 alarm이 발생한 경우 최종 alarm으로 확정하지 않습니다.

Persistence는:

```text
False Alarm 감소
        ↕
Short Fault Detection 감소
```

라는 trade-off를 가집니다.

---

## 17. Final Configuration Selection

최종 설정은 개발 과정에서 수행한 detector 및 hyperparameter 비교 결과를 바탕으로 고정했습니다.

최종 설정:

```text
Window detector
 ├── 1.0s
 ├── 0.5s
 ├── Ledoit-Wolf
 ├── normal quantile
 ├── q = 0.9999
 └── k = 2

Sample detector
 ├── raw_diff
 ├── Ledoit-Wolf
 ├── mean aggregation
 ├── k = 3
 └── q = 0.9999

Ensemble
 ├── OR_3
 └── Persistence P2
```

### 설정 선택 시 고려한 기준

단일 F1만을 기준으로 선택하지 않고:

```text
Event Detection
+
Detection Delay
+
Normal False Alarm
```

을 함께 비교했습니다.

또한 최종 P2는 **fault-event detection rate를 90% 이상 유지하면서 OR_3의 false alarm을 줄이는 절충안**으로 선택했습니다.

비교 결과:

```text
OR_3
Event Detection = 99.05%
Normal FA Cycle = 1.17%

OR_3 + P2
Event Detection = 91.43%
Normal FA Cycle = 0.90%
```

반면:

```text
2-of-3 Exact
Normal FA Cycle = 0.39%
Event Detection = 71.43%
```

으로 false alarm은 더 낮지만 동일한 90% 이상 event detection 기준을 만족하지 않습니다.

---

## 18. Evaluation Metrics

KAMPact에서는 sample-level F1뿐 아니라 event/cycle 단위 평가를 함께 사용합니다.

| Metric               | Description                          |
| -------------------- | ------------------------------------ |
| Event Detection Rate | fault event 중 1회 이상 탐지된 비율           |
| Detection Delay      | fault onset 이후 최초 alarm까지의 시간        |
| F1                   | sample-level alarm classification F1 |
| Precision            | alarm 중 실제 fault 비율                  |
| Recall               | 실제 fault 중 탐지 비율                     |
| Normal FA Cycle Rate | 정상 cycle 중 1회 이상 false alarm 발생 비율   |
| FA Episodes          | 정상 구간에서 발생한 연속 false alarm 묶음 수      |
| Idle FA Cycle Rate   | Idle cycle 중 1회 이상 false alarm 발생 비율 |

---

## 19. Detection Delay Definition

본 데이터에는 실제 물리적 고장이 발생한 정확한 timestamp가 별도로 제공되지 않습니다.

따라서 다음과 같이 정의합니다.

```text
Fault onset
    ↓
First alarm
    ↓
Detection Delay
```

### Fault onset

Fault segment의 시작 시점 또는 데이터에서 정의된 fault onset timestamp를 사용합니다.

현재 fault 데이터는 전 행이 `Equipment_state >= 1`이므로 fault segment의 첫 timestamp가 onset 기준이 됩니다.

### Window detector

Window alarm은 causal shifted timeline으로 매핑된 뒤 최초 alarm 시점을 사용합니다.

### Sample detector

fault segment의 첫 sample부터 최초 alarm sample까지의 시간차를 사용합니다.

### Delay 평균

중요하게도 **탐지에 성공한 event만 delay 평균에 포함**됩니다.

즉:

```text
Detected Event
→ delay 계산

Missed Event
→ delay = NaN
→ 평균에서 제외
```

따라서 detection rate가 서로 다른 detector 간 delay를 비교할 때는 **탐지된 event 집합이 서로 다를 수 있음**을 함께 고려해야 합니다.

---

## 20. Cross Validation

시계열 데이터의 leakage를 방지하기 위해 `group_id` 단위로 fold를 구성합니다.

하나의 segment는 하나의 fold에만 배정됩니다.

```text
Fold i
→ Test

Fold i+1
→ Validation

Remaining folds
→ Train
```

학습에는:

```text
Normal operation only
```

을 사용합니다.

### Individual Experiments

Window 및 sample-level 개별 실험:

```text
4-fold × 5 repeats
```

### Final Ensemble

최종 three-detector ensemble:

```text
5-fold × 5 repeats
```

repeat별 seed는:

```text
0
1
2
3
4
```

입니다.

---

## 21. No Independent Final Holdout

본 프로젝트에서는 별도의 미사용 독립 데이터가 제공되지 않아 **독립적인 final holdout set을 별도로 확보하지 않았습니다.**

따라서 최종 결과는:

```text
Independent Final Holdout Performance
```

가 아니라

```text
Group-based
5-fold × 5-repeat
Cross Validation Performance
```

입니다.

또한 최종 설정과 주요 hyperparameter를 개발 과정의 CV 결과를 참고하여 선택했기 때문에, **동일 CV 결과를 최종 성능으로 보고한 수치는 다소 낙관적일 가능성**이 있습니다.

향후 독립적인 설비·운전 조건의 데이터가 확보되면 별도의 external holdout을 이용하여 일반화 성능을 추가 검증할 필요가 있습니다.

---

## 22. Final Ensemble Result

다음 결과는 `outputs/6_three_detector_ensemble/ensemble_summary.csv`에 저장된 **5-fold × 5-repeat 결과**입니다.

| Detector / Ensemble       |   Event Detection |              Delay |           Sample F1 |  Normal FA Cycle | Idle FA Cycle |
| ------------------------- | ----------------: | -----------------: | ------------------: | ---------------: | ------------: |
| 1.0s Window               |     73.33 ± 2.61% |     1.156 ± 0.027s |     0.7422 ± 0.0146 |     0.47 ± 0.22% |         0.00% |
| 0.5s Window               |     76.19 ± 0.00% |     0.776 ± 0.015s |     0.7894 ± 0.0099 |     0.47 ± 0.18% |         0.00% |
| Sample-level              |     85.71 ± 0.00% |     0.571 ± 0.002s |     0.5817 ± 0.0040 |     0.98 ± 0.28% |         0.00% |
| OR_3                      | **99.05 ± 2.13%** | **0.470 ± 0.019s** | **0.8482 ± 0.0134** |     1.17 ± 0.14% |         0.00% |
| **OR_3 + P2**             |     91.43 ± 2.13% |     0.571 ± 0.014s |     0.8300 ± 0.0136 |     0.90 ± 0.11% |         0.00% |
| OR_3 + P3                 |     79.05 ± 2.61% |     0.729 ± 0.022s |     0.8109 ± 0.0133 | **0.67 ± 0.18%** |         0.00% |
| 2-of-3 Exact              |     71.43 ± 3.37% |     0.897 ± 0.013s |     0.7633 ± 0.0088 | **0.39 ± 0.14%** |         0.00% |
| 2-of-3 Tolerance          |     72.38 ± 2.13% |     0.854 ± 0.010s |     0.7764 ± 0.0097 |     0.43 ± 0.16% |         0.00% |
| OR_3 Sample Corroboration |     81.90 ± 2.13% |     0.776 ± 0.025s |     0.8068 ± 0.0142 |     0.59 ± 0.14% |         0.00% |

> **표 해석:** `±`는 5개 repeat 간 표준편차입니다.
> 최종 ensemble의 `Normal FA Cycle` 분모는 repeat별 **511개 normal cycles**, `Idle FA Cycle` 분모는 **88개 idle cycles**입니다.

---

## 23. Event-level Persistence Analysis

최종 ensemble에서는 **21개 fault event를 5개 반복 분할에서 평가**했습니다.

이는:

```text
21 events × 5 repeats
= 105 event-evaluation cases
```

를 의미하며, **105개의 독립적인 fault sample을 의미하지 않습니다.**

### OR_3

```text
21 events
×
5 repeats

= 105 evaluations

Detected = 104
Missed   = 1
```

따라서:

```text
104 / 105
= 99.05%
```

입니다.

### OR_3_P2

```text
105 evaluations

Detected = 96
Missed   = 9
```

따라서:

```text
96 / 105
= 91.43%
```

입니다.

---

## 24. P2 Miss Analysis

P2에서 발생한 9회의 미탐은 전체 fault event에 고르게 분포하지 않았습니다.

현재 `ensemble_event_details.csv` 기준으로:

| Fault Event | Samples | Missed / 5 Repeats |
| ----------- | ------: | -----------------: |
| `fault_5`   |       3 |                  5 |
| `fault_19`  |      10 |                  4 |
| Others      |       - |                  0 |

즉 P2의 전체 9회 미탐:

```text
fault_5  → 5회
fault_19 → 4회
----------------
Total    → 9회
```

를 두 개의 짧은 fault event가 전부 설명합니다.

특히:

```text
fault_5
→ 3 samples
→ 5회 모두 missed
```

로 나타났습니다.

이는 persistence 조건이 짧은 fault event에서 탐지 기회를 제한할 수 있음을 보여줍니다.

---

## 25. Persistence Trade-off

OR_3와 OR_3_P2 비교:

| Metric          |   OR_3 | OR_3 + P2 |
| --------------- | -----: | --------: |
| Event Detection | 99.05% |    91.43% |
| Detection Delay | 0.470s |    0.571s |
| Sample F1       | 0.8482 |    0.8300 |
| Normal FA Cycle |  1.17% |     0.90% |
| Idle FA Cycle   |  0.00% |     0.00% |

P2 적용 후:

```text
Event Detection
99.05% → 91.43%

Delay
0.470s → 0.571s

Normal FA Cycle
1.17% → 0.90%
```

로 변화합니다.

따라서 persistence는 false alarm을 줄이는 대신:

* 짧은 fault event의 탐지율 감소
* detection delay 증가

라는 trade-off를 발생시킵니다.

---

## 26. Why Three Detectors?

세 detector는 서로 다른 시간 해상도를 관찰합니다.

```text
1.0s Window
→ 상대적으로 긴 시간의 통계적 패턴

0.5s Window
→ 짧은 시간의 변화

Sample-level
→ sample 수준의 변화
```

따라서 하나의 detector가 놓치는 이상 패턴을 다른 detector가 보완할 수 있습니다.

개념적으로:

```text
Long-term Pattern
        │
        └── 1.0s Window

Short-term Pattern
        │
        └── 0.5s Window

Point-level Change
        │
        └── Sample-level
```

이를 OR 방식으로 결합하여 detector 간 상호 보완 효과를 확인했습니다.

---

## 27. False Alarm Analysis

### Normal Cycle

정상 cycle에서 alarm이 한 번이라도 발생하면 false-alarm cycle로 계산합니다.

```text
Normal Cycle
      │
      ├── No Alarm → Normal
      │
      └── Alarm    → False Alarm Cycle
```

최종 ensemble 평가에서:

```text
Normal Cycles = 511
```

입니다.

예를 들어 OR_3_P2의 평균:

```text
Normal FA Cycle = 0.90%
```

는 정상 cycle 가운데 false alarm이 발생한 cycle의 비율을 의미합니다.

---

### False Alarm Episode

연속적인 alarm은 하나의 episode로 계산합니다.

예:

```text
0 0 1 1 1 0 0 1 0
```

은:

```text
2 False Alarm Episodes
```

로 계산됩니다.

이를 통해 단순 alarm 횟수뿐 아니라 오경보가 반복적으로 발생하는 정도도 확인할 수 있습니다.

---

### Idle

Idle은 normal operation과 별도로 평가합니다.

최종 ensemble에서:

```text
Idle Cycles = 88
Idle FA Cycle Rate = 0.00%
```

입니다.

---

## 28. Repository Structure

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
├── LICENSE
└── README.md
```

> `LICENSE`는 저장소에 실제 파일을 추가한 뒤 유지합니다.

---

## 29. Script Description

| Script                                | Purpose                                    |
| ------------------------------------- | ------------------------------------------ |
| `1_visualize_normal_outlier.py`       | Normal / Fault 시계열 시각화                     |
| `2_Classification_idle_sections.py`   | Normal 데이터의 Idle 구간 분류                     |
| `3_time_structure_analysis.py`        | Timestamp, sampling gap, segment 구조 분석     |
| `4_make_window_dataset.py`            | Window 생성 및 feature extraction             |
| `5_run_mahalanobis.py`                | 기본 Window Mahalanobis detector             |
| `5_2_run_mahalanobis.py`              | Window-level Group K-Fold / multi-scale 평가 |
| `5_3_run_sample_level_mahalanobis.py` | Sample-level causal Mahalanobis 평가         |
| `6_compare_three_detectors.py`        | 3개 detector ensemble 및 persistence 비교      |

### Script Numbering

스크립트 번호는 실행 파이프라인을 위한 번호이며 README의 장 번호와 동일한 의미가 아닙니다.

```text
README
5장  → 데이터 / Idle 처리

Script
5_*.py → Mahalanobis 모델링 / 평가
```

따라서 README section number와 source script number를 동일한 단계 번호로 해석하지 않습니다.

---

## 30. Installation

### Python

테스트 환경:

```text
Python 3.11
```

### Install

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

`requirements.txt`는 주요 패키지의 호환 범위를 지정하고 있습니다.

현재 설정은 완전한 patch-level version lock이 아니라 **major/minor compatibility range 기반**입니다.

---

## 31. Reproducibility

최종 ensemble의 기본 설정:

```text
Folds   = 5
Repeats = 5
Seed    = 0
```

각 repeat는:

```text
seed = 0
seed = 1
seed = 2
seed = 3
seed = 4
```

를 사용합니다.

동일한 데이터와 동일한 Python/package 환경에서 실행하면 동일한 seed 기반 split을 재현할 수 있습니다.

---

## 32. Data Preparation

원본 CSV는 저장소에 포함하지 않습니다.

사용자가 데이터를 준비할 때 필요한 최소 schema:

```text
TimeStamp
AI0_Vibration
AI1_Vibration
AI2_Current
Equipment_state
Idle
```

정상 데이터:

```text
data/press_data_normal_with_idle.csv
```

고장 데이터:

```text
data/outlier_data.csv
```

원본 데이터가 없는 환경에서는 실제 모델 성능을 재현할 수 없으며, 저장소에는 분석 코드와 실험 결과만 제공합니다.

---

## 33. Reproduction

### Step 1. Visualization

```bash
python ./src/1_visualize_normal_outlier.py
```

### Step 2. Idle Classification

```bash
python ./src/2_Classification_idle_sections.py
```

### Step 3. Time Structure Analysis

```bash
python ./src/3_time_structure_analysis.py
```

### Step 4. Window Dataset

#### 1.0s

```bash
python ./src/4_make_window_dataset.py \
    --window-sec 1.0 \
    --step-sec 0.1
```

#### 0.5s

```bash
python ./src/4_make_window_dataset.py \
    --window-sec 0.5 \
    --step-sec 0.1
```

### Step 5-1. Basic Mahalanobis

```bash
python ./src/5_run_mahalanobis.py \
    --input result/modeling_dataset_1.0_0.1/model_windows.csv
```

### Step 5-2. Window-level CV

```bash
python ./src/5_2_run_mahalanobis.py \
    --inputs \
    result/modeling_dataset_1.0_0.1 \
    result/modeling_dataset_0.5_0.1
```

### Step 5-3. Sample-level

```bash
python ./src/5_3_run_sample_level_mahalanobis.py \
    --normal-path data/press_data_normal_with_idle.csv \
    --fault-path data/outlier_data.csv
```

### Step 6. Final Ensemble

```bash
python ./src/6_compare_three_detectors.py
```

기본 최종 설정:

```text
Window threshold  = normal_quantile
Window quantile   = 0.9999
Window k          = 2
Window covariance = Ledoit-Wolf

Sample feature    = raw_diff
Sample threshold  = normal_quantile
Sample quantile   = 0.9999
Sample aggregation= mean
Sample agg-k      = 3
Sample covariance = Ledoit-Wolf

Ensemble          = OR_3
Persistence       = P2

Folds             = 5
Repeats           = 5
Seed              = 0
```

---

## 34. Output Files

### Window CV

```text
outputs/5_2_mahalanobis_cv/
```

주요 결과:

```text
cv_comparison.csv
cv_repeat_results.csv
cv_event_details.csv
cv_event_detection_frequency.csv
```

### Sample CV

```text
outputs/5_3_sample_level_cv/
```

주요 결과:

```text
sample_cv_comparison.csv
sample_cv_repeat_results.csv
sample_cv_event_details.csv
sample_cv_event_detection_frequency.csv
```

### Final Ensemble

```text
outputs/6_three_detector_ensemble/
```

주요 결과:

```text
ensemble_summary.csv
ensemble_repeat_results.csv
ensemble_event_details.csv
```

#### `ensemble_summary.csv`

각 detector / ensemble의 평균 및 표준편차를 저장합니다.

#### `ensemble_repeat_results.csv`

각 repeat의 세부 성능을 저장합니다.

#### `ensemble_event_details.csv`

각 fault event의 detection 여부, delay, sample 수 등을 저장합니다.

---

## 35. Key Findings

### Multi-scale Detection

1.0s와 0.5s window는 서로 다른 시간 범위의 패턴을 관찰하며, sample-level detector는 더 짧은 시간 해상도에서 변화를 포착합니다.

### Detector Complementarity

단일 detector보다 여러 detector의 alarm을 결합함으로써 서로 다른 이상 패턴을 상호 보완할 수 있음을 확인했습니다.

### OR_3

```text
Event Detection = 99.05%
Delay            = 0.470s
```

로 높은 event coverage와 짧은 detection delay를 보였습니다.

### OR_3 + P2

```text
Event Detection = 91.43%
Delay            = 0.571s
Normal FA Cycle  = 0.90%
```

으로 OR_3보다 false alarm이 감소했습니다.

다만 persistence로 인해 짧은 fault event에서 일부 miss가 발생했으며, P2의 9회 미탐은 `fault_5`와 `fault_19`에 집중되었습니다.

---

## 36. Limitations

### 1. Independent Final Holdout 부재

독립적인 미사용 final holdout 데이터가 없어 최종 성능은 반복 교차검증 기반 결과입니다.

### 2. Hyperparameter Selection

주요 detector 구성 및 hyperparameter는 개발 과정의 CV 결과를 참고하여 선택되었습니다.

따라서 선택에 사용된 동일 CV 결과를 최종 성능으로 보고했기 때문에 실제 독립 데이터 성능보다 낙관적일 가능성이 있습니다.

### 3. Fault Onset Ground Truth

현재 데이터에서 실제 물리적 고장이 시작된 정확한 시각을 별도로 알 수 없으므로 detection delay는 dataset-defined fault onset을 기준으로 계산합니다.

### 4. Short Fault Events

매우 짧은 fault event는 window 또는 persistence 조건에서 충분한 연속 정보를 확보하기 어렵습니다.

특히:

```text
fault_5
→ 3 samples
```

와 같은 이벤트는 P2 조건에서 탐지에 불리합니다.

### 5. Dataset Size

fault event 수가 제한적이기 때문에 다양한 설비와 운전 조건에서의 일반화 성능을 추가적으로 검증할 필요가 있습니다.

### 6. Operating Condition

실제 현장에서는 부하, 속도 및 기타 운전 조건에 따라 정상 신호의 분포가 달라질 수 있습니다.

---

## 37. Future Work

향후 다음 방향으로 확장할 수 있습니다.

```text
Frequency-domain Features
        +
Operating-condition-aware Detection
        +
Adaptive Threshold
        +
Additional Sensor Information
        +
More Diverse Fault Data
        +
External Validation Dataset
```

특히 실제 현장 적용에서는 운전 조건별 정상 분포를 별도로 모델링하거나 adaptive threshold를 적용하는 방법을 고려할 수 있습니다.

---

## 38. Dataset / License

본 저장소는 분석 코드와 실험 결과를 제공합니다.

원본 데이터는 저장소에 포함하지 않습니다.

원본 데이터의 이용 및 재배포는 해당 데이터셋 제공기관의 이용 조건과 공모전 규정을 따라야 합니다.

본 저장소의 코드 라이선스는 저장소의 `LICENSE` 파일을 따릅니다.

---

## 39. Summary

KAMPact는 다변량 시계열 기반 anomaly detection pipeline으로 다음 구조를 사용합니다.

```text
Sensor Data
    ↓
Timestamp / Segment Analysis
    ↓
Idle Classification
    ↓
Window Feature Engineering
    ↓
Mahalanobis Detection
    ├── 1.0s Window
    ├── 0.5s Window
    └── Sample-level raw_diff
              ↓
            OR_3
              ↓
       Persistence P2
              ↓
         Final Alarm
```

핵심은 단순한 fault classification이 아니라:

```text
Fault Event Detection
        +
Detection Delay
        +
Normal False Alarm
        +
Idle False Alarm
```

을 함께 평가하는 것입니다.

최종 ensemble은 개발 과정의 비교 결과를 바탕으로:

```text
1.0s Window
+
0.5s Window
+
Sample-level raw_diff / mean3
        ↓
      OR_3
        ↓
Persistence P2
```

구성으로 고정했습니다.

최종 ensemble은 **5-fold × 5-repeat Group Cross Validation**으로 평가되었으며, 독립적인 final holdout 데이터는 포함하지 않습니다.
