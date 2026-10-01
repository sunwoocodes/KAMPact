# KAMPact

**진동·전류 시계열 기반 프레스 유압펌프 이상 조기탐지 및 오경보 분석**

KAMPact는 프레스 유압펌프에서 수집된 **진동 및 전류 시계열 데이터**를 이용하여 정상 운전 패턴을 학습하고, 정상 패턴에서 벗어나는 이상을 탐지하는 예지보전 분석 프로젝트입니다.

단순한 고장 분류가 아니라 다음 세 가지를 함께 평가합니다.

```text
Fault Event Detection
        +
Detection Delay
        +
False Alarm Analysis
```

> **Detection Delay 주의:** 본 프로젝트의 delay는 실제 물리적 고장이 발생하기 전의 예지시간이 아니라, **데이터에서 정의한 fault onset 이후 최초 alarm까지의 시간**입니다.

---

## 1. Problem

프레스 공정의 유압펌프는 진동 및 전류 신호를 통해 운전 상태를 관찰할 수 있습니다.

본 프로젝트에서는 다음 세 가지 센서 신호를 사용합니다.

```text
AI0_Vibration
AI1_Vibration
AI2_Current
```

주요 목표는 정상 운전에서 발생하는 미세한 신호 변화 또는 이상 패턴을 가능한 한 빠르게 탐지하면서, 정상 운전 중 발생하는 false alarm을 함께 분석하는 것입니다.

```text
정상 운전
    ↓
신호 변화
    ↓
이상 패턴
    ↓
Fault Detection
```

---

## 2. Project Goals

KAMPact의 주요 목표는 다음과 같습니다.

1. 정상 운전 데이터로 정상 분포 모델링
2. 서로 다른 시간 해상도에서 이상 탐지
3. fault event 단위 detection rate 측정
4. fault onset 이후 detection delay 측정
5. 정상 cycle 및 Idle cycle false alarm 분석
6. 서로 다른 detector의 상호 보완 효과 분석
7. persistence가 탐지율과 false alarm에 미치는 영향 분석

---

## 3. Dataset

원본 데이터는 저장소에 포함하지 않습니다.

사용 데이터:

```text
Normal : 20,000 rows
Fault  : 600 rows
```

Fault 데이터는 총 **21개 fault segment/event**로 구성됩니다.

### Input Schema

| Column            | Description  |
| ----------------- | ------------ |
| `TimeStamp`       | 측정 timestamp |
| `AI0_Vibration`   | 진동 센서 0      |
| `AI1_Vibration`   | 진동 센서 1      |
| `AI2_Current`     | 전류 센서        |
| `Equipment_state` | 설비 상태        |
| `Idle`            | Idle 상태 여부   |

### Data Label

```text
Equipment_state = 0
→ Normal

Equipment_state >= 1
→ Fault
```

`Idle=1`인 구간은 정상 운전과 별도로 관리합니다.

Idle 데이터는 모델 학습이나 threshold 산정에 사용하지 않고 **평가 단계에서 false alarm만 측정**합니다.

---

## 4. Time-Series Structure

timestamp gap을 이용하여 press cycle 단위의 segment를 생성합니다.

```text
Gap > 0.5 sec
    ↓
New Segment
```

이렇게 분리된 `group_id`를 기준으로 train / validation / test를 구성하여 동일 segment가 여러 데이터 분할에 들어가는 leakage를 방지합니다.

---

## 5. Idle Handling

Normal 데이터에는 장비가 실제로 동작하지 않는 Idle 구간이 포함되어 있습니다.

Idle은 정상 운전과 신호 분포가 다를 수 있으므로 모델 학습 및 threshold 계산에서 제외합니다.

```text
Normal Operation
       ↓
Train / Validation / Test

Idle
       ↓
Test Only
       ↓
False Alarm Evaluation
```

최종 ensemble 평가에서는 repeat마다:

```text
Normal Cycles = 511
Idle Cycles   = 88
```

을 사용합니다.

Idle cycle은 5개 fold에 분산되며, 각 repeat에서 각 cycle은 **자신이 배정된 하나의 test fold에서만** 평가됩니다. 따라서 한 repeat에서는 88개 Idle cycle이 평가되며, 동일 Idle cycle이 모든 fold에서 반복 평가되는 구조는 아닙니다.

---

## 6. Window Feature Engineering

Window-based detector는 각 시간 구간에서 통계적 특징을 추출합니다.

각 센서에서 다음 5개 feature를 계산합니다.

```text
Mean
Std
RMS
Peak-to-Peak
Slope
```

센서 3개:

```text
3 × 5 = 15 features
```

추가로:

```text
AI0 ↔ AI1 Correlation
```

을 사용합니다.

따라서 총 **16개 feature**를 사용합니다.

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

최종 ensemble에서는 두 가지 시간 범위의 window detector를 사용합니다.

```text
1.0 sec Window
0.5 sec Window
```

공통 step:

```text
0.1 sec
```

원본 시계열의 명목 sampling interval 역시 **0.1 sec**입니다.

다만 실제 timestamp gap은 완전히 일정하지 않을 수 있으므로 delay 계산에서는 실제 timestamp 차이를 사용합니다.

```text
1.0s Window
→ 비교적 긴 시간의 통계적 패턴

0.5s Window
→ 짧은 시간의 변화
```

---

## 8. Mahalanobis Anomaly Detection

KAMPact는 정상 데이터의 다변량 분포로부터 Mahalanobis distance를 계산합니다.

$$
D^2(x)=(x-\mu)^T\Sigma^{-1}(x-\mu)
$$

여기서:

* \(x\): 현재 feature vector
* \(\mu\): 정상 데이터 평균
* \(\Sigma\): 정상 데이터 covariance matrix

정상 분포에서 멀어질수록 anomaly score가 증가합니다.

### Covariance Estimator

다음 covariance estimator를 지원합니다.

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

feature scale 차이와 큰 값의 영향을 완화하기 위해 다음 변환을 적용합니다.

$$
x' = sign(x)\log(1+|x|)
$$

변환 후 Mahalanobis distance를 계산합니다.

---

## 10. Threshold

Threshold는 다음 두 방식을 지원합니다.

```text
F1-based threshold
Normal-quantile threshold
```

### Normal Quantile

Normal-quantile 방식에서는 validation 데이터 중 **정상 score만 사용**하여 threshold를 계산합니다.

최종 설정:

```text
Quantile = 0.9999
```

구조는 다음과 같습니다.

```text
Train
→ Normal samples only

Validation
→ Normal scores only for quantile threshold

Test
→ Final evaluation
```

Fault 데이터와 test fold는 normal-quantile threshold 산정에 사용하지 않습니다.

### Quantile 해석

`q = 0.9999`는 validation 정상 score의 상위 **0.01% 지점**입니다.

Validation 정상 score가 수천 개라면 이 값은 distribution의 최댓값에 매우 가까워질 수 있습니다. 따라서 상위 몇 개의 score에 threshold가 민감하게 영향을 받을 가능성이 있습니다.

현재 `ensemble_summary.csv`와 `ensemble_event_details.csv`에는 fold별 validation 정상 score의 정확한 개수가 저장되어 있지 않으므로, 존재하지 않는 값을 임의로 기재하지 않습니다. 따라서 본 README에서는 `q=0.9999`의 의미와 잠재적인 민감성만 명시합니다.

---

## 11. Window Detector

Window detector의 전체 과정:

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

최종 ensemble에서는:

```text
1.0s Window
0.5s Window
```

의 결과를 각각 raw timeline으로 매핑합니다.

---

## 12. Causal Window Mapping

Window feature 계산에 사용된 시간 구간보다 앞선 시점에 alarm이 발생하지 않도록 window alarm을 causal timeline으로 이동합니다.

개념적으로:

```text
Original Window
[start -------- end]

                  ↓ causal shift

Alarm Timeline
             [start -------- end]
```

즉 feature 계산에 사용된 window가 끝난 이후의 timeline에서 alarm이 활성화되도록 구성합니다.

이 방식으로 window detector가 미래 sample 정보를 사용하여 과거 시점에 alarm을 발생시키는 문제를 방지합니다.

---

## 13. Sample-level Detector

`5_3_run_sample_level_mahalanobis.py`에서는 window 통계 대신 원본 sample 수준에서 Mahalanobis distance를 계산합니다.

최종 ensemble 설정:

```text
Feature Set : raw_diff
Covariance  : Ledoit-Wolf
Aggregation : mean
Aggregation k = 3
```

### Feature Set

```text
Raw
+
First Difference
```

즉 다음 6개 feature를 사용합니다.

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

sample-level score에서 최근 3개 sample의 평균을 계산합니다.

```text
t-2 ─┐
t-1 ─┼──→ Mean3 ──→ Alarm
t   ─┘
```

미래 sample은 사용하지 않습니다.

```text
Aggregation = mean3
```

을 사용하여 순간적인 score spike를 완화하면서 지나치게 긴 aggregation에 따른 delay 증가를 줄입니다.

---

## 15. Three-Detector Ensemble

최종 ensemble은 세 detector를 결합합니다.

```text
1.0s Window Mahalanobis
          +
0.5s Window Mahalanobis
          +
Sample-level Mahalanobis
```

### OR_3

세 detector 중 하나라도 alarm이면 후보 alarm을 생성합니다.

```text
1.0s Window ─┐
0.5s Window ─┼── OR ──→ OR_3
Sample       ┘
```

OR 방식은 서로 다른 detector가 놓치는 fault를 상호 보완하기 위한 구성입니다.

### OR_3 Sample Corroboration

`OR_3 Sample Corroboration`은 1.0s 및 0.5s window alarm은 그대로 유지하면서, sample-level alarm이 발생한 현재 시점 및 이전 4개 raw sample 안에서 window alarm이 확인되는 경우 sample alarm을 추가로 허용하는 비교 구성입니다.

```text
Window Any ───────────────┐
                          ├──→ OR_3 Sample Corroboration
Sample Alarm ── AND ──────┘
              ↑
     현재 포함 최근 5 samples
     내 window alarm 존재
```

이는 최종 선택 구성에는 포함하지 않고 ensemble 비교 실험에만 사용합니다.

---

## 16. Persistence Filter

OR_3는 높은 민감도를 갖지만 짧은 score spike가 alarm으로 이어질 수 있습니다.

이를 줄이기 위해 persistence를 적용합니다.

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

입니다.

원본 명목 sampling interval이 0.1 sec이므로 두 번째 연속 alarm sample까지 확인하는 과정에서 약 0.1 sec의 추가 시간이 발생할 수 있습니다. 실제 timestamp gap이 다른 경우에는 실제 시간 차이가 적용됩니다.

Persistence는:

```text
False Alarm 감소
        ↕
Short Fault Detection 감소
```

라는 trade-off를 갖습니다.

---

## 17. Final Configuration Selection

최종 설정:

```text
Window Detector
 ├── 1.0s Window
 ├── 0.5s Window
 ├── Ledoit-Wolf
 ├── normal quantile
 ├── q = 0.9999
 └── k = 2

Sample Detector
 ├── raw_diff
 ├── Ledoit-Wolf
 ├── mean aggregation
 ├── k = 3
 └── q = 0.9999

Ensemble
 ├── OR_3
 └── Persistence P2
```

설정은 개발 과정에서 수행한 비교 결과를 바탕으로 결정했습니다.

비교 기준은:

```text
Event Detection
+
Detection Delay
+
Normal False Alarm
```

입니다.

### P2 선택 기준

`OR_3 + P2`는 **event detection rate를 90% 이상 유지하면서 OR_3의 normal false alarm을 줄이는 절충안**으로 선택했습니다.

다만 이 **90% 기준은 실험 전에 사전 등록한 기준이 아니라 결과를 비교한 뒤 적용한 사후적 기준**입니다.

따라서 이 기준과 이를 이용한 최종 구성 선택은 독립적인 confirmatory evaluation으로 해석해서는 안 됩니다.

비교 결과:

```text
OR_3
Event Detection = 99.05%
Normal FA Cycle = 1.17%

OR_3 + P2
Event Detection = 91.43%
Normal FA Cycle = 0.90%
```

`2-of-3 Exact`는:

```text
Event Detection = 71.43%
Normal FA Cycle = 0.39%
```

으로 false alarm은 더 낮지만 event detection rate가 90% 기준에 미치지 못합니다.

---

## 18. Evaluation Metrics

| Metric               | Description                          |
| -------------------- | ------------------------------------ |
| Event Detection Rate | fault event 중 1회 이상 탐지된 비율           |
| Detection Delay      | fault onset 이후 최초 alarm까지의 시간        |
| F1                   | sample-level alarm classification F1 |
| Precision            | alarm 중 실제 fault 비율                  |
| Recall               | 실제 fault 중 탐지된 비율                    |
| Normal FA Cycle Rate | 정상 cycle 중 1회 이상 false alarm 발생 비율   |
| FA Episodes          | 정상 구간에서 발생한 연속 false alarm 묶음 수      |
| Idle FA Cycle Rate   | Idle cycle 중 1회 이상 false alarm 발생 비율 |

---

## 19. Detection Delay Definition

현재 데이터에는 실제 물리적 고장이 발생한 정확한 timestamp가 별도로 제공되지 않습니다.

따라서 데이터에서 정의된 fault onset을 기준으로 delay를 계산합니다.

```text
Fault Onset
     ↓
First Alarm
     ↓
Detection Delay
```

Fault 데이터에서는 `Equipment_state >= 1`인 행으로 fault segment가 구성되어 있으므로 segment의 첫 timestamp를 onset 기준으로 사용합니다.

### Window Detector

Window alarm을 causal timeline으로 매핑한 후 최초 alarm 시점을 사용합니다.

### Sample Detector

Fault segment의 첫 sample부터 최초 alarm sample까지의 시간차를 사용합니다.

### Delay 평균

Detection Delay 평균은 **탐지된 event에서만 계산**합니다.

```text
Detected Event
→ delay 계산

Missed Event
→ delay = NaN
→ 평균에서 제외
```

따라서 탐지율이 서로 다른 detector 간 delay는 서로 다른 event 집합을 기반으로 할 수 있으므로 이를 함께 고려하여 해석해야 합니다.

---

## 20. Sampling Interval and Short Fault Events

원본 시계열의 명목 sampling interval은:

```text
0.1 sec
```

입니다.

따라서 sample 수를 시간 범위로 해석하면:

```text
3 samples
→ 첫 sample ~ 마지막 sample
→ 약 0.2 sec

10 samples
→ 첫 sample ~ 마지막 sample
→ 약 0.9 sec
```

입니다.

### `fault_5`

```text
3 samples
≈ 0.2 sec
```

이므로 0.5s 및 1.0s window보다 짧습니다.

이 event에서는 해당 window 길이를 확보하지 못하기 때문에 window dataset에 해당 window의 row가 생성되지 않을 수 있습니다. 따라서 window-only detector에서는 해당 event를 탐지할 수 없어 miss로 집계될 수 있습니다.

Sample-level detector는 window 길이와 관계없이 raw sample을 직접 사용하므로 별도로 평가할 수 있습니다.

### `fault_19`

```text
10 samples
≈ 0.9 sec
```

이므로 1.0s window보다 짧지만 0.5s window에서는 평가 가능한 길이입니다.

---

## 21. Cross Validation

시계열 leakage를 방지하기 위해 `group_id` 단위로 fold를 구성합니다.

```text
Fold i
→ Test

Fold i+1
→ Validation

Remaining folds
→ Train
```

### Train

학습에는 **non-Idle normal operation**만 사용합니다.

### Validation

Threshold 선택에 사용합니다.

Normal-quantile threshold의 경우 validation 정상 score만 사용합니다.

### Test

Test fold의 fault detection 및 normal / Idle false alarm을 평가합니다.

---

## 22. Individual Experiments vs Final Ensemble

### Individual Experiments

Window 및 Sample-level 개별 실험:

```text
4-fold × 5 repeats
```

### Final Ensemble

Three-Detector Ensemble:

```text
5-fold × 5 repeats
```

최종 ensemble의 기본 seed:

```text
0
```

repeat별 seed:

```text
0
1
2
3
4
```

입니다.

---

## 23. Final Holdout

본 프로젝트에서는 **미사용 독립 final holdout 데이터가 별도로 제공되지 않았기 때문에**, 기존 데이터를 다시 잘라 독립 holdout이라고 정의하지 않았습니다.

따라서 최종 결과는:

```text
Independent Final Holdout Performance
```

가 아니라:

```text
Group-based
5-fold × 5-repeat
Cross Validation Performance
```

입니다.

또한 최종 설정과 hyperparameter를 개발 과정의 CV 결과를 참고하여 선택했습니다.

따라서 동일 CV 결과를 최종 성능으로 보고한 수치는 **독립 데이터에 대한 성능보다 낙관적으로 추정되었을 가능성**이 있습니다.

---

## 24. Final Ensemble Result

다음 결과는 `outputs/6_three_detector_ensemble/ensemble_summary.csv`에 저장된 **5-fold × 5-repeat 결과**입니다.

| Detector / Ensemble       |   Event Detection |              Delay |           Sample F1 |  Normal FA Cycle | Idle FA Cycle |
| ------------------------- | ----------------: | -----------------: | ------------------: | ---------------: | ------------: |
| 1.0s Window               |     73.33 ± 2.61% |     1.156 ± 0.027s |     0.7422 ± 0.0146 |     0.47 ± 0.22% |         0.00% |
| 0.5s Window               |     76.19 ± 0.00% |     0.776 ± 0.015s |     0.7894 ± 0.0099 |     0.47 ± 0.18% |         0.00% |
| Sample-level              |     85.71 ± 0.00% |     0.571 ± 0.002s |     0.5817 ± 0.0040 |     0.98 ± 0.28% |         0.00% |
| OR_3                      | **99.05 ± 2.13%** | **0.470 ± 0.019s** | **0.8482 ± 0.0134** |     1.17 ± 0.14% |         0.00% |
| OR_3 + P2                 |     91.43 ± 2.13% |     0.571 ± 0.014s |     0.8300 ± 0.0136 |     0.90 ± 0.11% |         0.00% |
| OR_3 + P3                 |     79.05 ± 2.61% |     0.729 ± 0.022s |     0.8109 ± 0.0133 |     0.67 ± 0.18% |         0.00% |
| 2-of-3 Exact              |     71.43 ± 3.37% |     0.897 ± 0.013s |     0.7633 ± 0.0088 | **0.39 ± 0.14%** |         0.00% |
| 2-of-3 Tolerance          |     72.38 ± 2.13% |     0.854 ± 0.010s |     0.7764 ± 0.0097 |     0.43 ± 0.16% |         0.00% |
| OR_3 Sample Corroboration |     81.90 ± 2.13% |     0.776 ± 0.025s |     0.8068 ± 0.0142 |     0.59 ± 0.14% |         0.00% |

> **표 해석:** `±`는 5개 repeat 결과의 표준편차입니다. 서로 다른 seed로 생성된 group-fold 분할에 따른 변동을 반영하며, 제한된 fault event 표본 수 자체의 통계적 불확실성이나 confidence interval을 의미하지 않습니다.
>
> 최종 ensemble의 `Normal FA Cycle` 분모는 repeat별 **511개 normal cycles**, `Idle FA Cycle` 분모는 **88개 idle cycles**입니다.
>
> **Final selected configuration:** `OR_3 + P2`. 굵은 수치는 각 열의 최고/최저 결과를 표시하기 위한 것이며, 최종 선택 자체를 의미하지 않습니다.

---

## 25. Event-level Analysis

최종 ensemble에서는 **21개 fault event를 5개 반복 분할에서 평가**했습니다.

```text
21 events × 5 repeats
= 105 event-evaluation cases
```

이는 **105개의 독립적인 fault sample을 의미하지 않습니다.**

### OR_3

```text
105 event-evaluation cases

Detected = 104
Missed   = 1
```

따라서:

```text
104 / 105
= 99.05%
```

입니다.

OR_3의 유일한 미탐은:

```text
fault_19
repeat_seed = 2
```

에서 발생했습니다.

### OR_3 + P2

```text
105 event-evaluation cases

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

## 26. P2 Miss Analysis

P2에서 발생한 9회의 미탐은 특정 짧은 fault event에 집중되었습니다.

| Fault Event | Samples | Missed / 5 Repeats |
| ----------- | ------: | -----------------: |
| `fault_5`   |       3 |                  5 |
| `fault_19`  |      10 |                  4 |
| Others      |       - |                  0 |

즉:

```text
fault_5  → 5회
fault_19 → 4회
----------------
Total    → 9회
```

입니다.

특히 `fault_5`는 3 samples로 구성되어 모든 repeat에서 P2 조건을 충족하지 못했습니다.

이는 persistence가 짧은 fault event에서 탐지 기회를 감소시킬 수 있음을 보여줍니다.

---

## 27. Persistence Trade-off

OR_3와 OR_3 + P2 비교:

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

로 변화했습니다.

따라서 P2는 순간적인 false alarm을 줄이는 대신 detection rate와 delay 측면에서 trade-off를 발생시킵니다.

---

## 28. False Alarm Analysis

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

### False Alarm Episode

연속적으로 발생한 alarm은 하나의 episode로 계산합니다.

예:

```text
0 0 1 1 1 0 0 1 0
```

은:

```text
2 False Alarm Episodes
```

로 계산됩니다.

이를 통해 단순 alarm 횟수뿐 아니라 false alarm이 반복적으로 발생하는 정도를 분석할 수 있습니다.

### Idle

Idle cycle은 별도로 평가합니다.

```text
Idle Cycles = 88
Idle FA Cycle Rate = 0.00%
```

입니다.

---

## 29. Why Three Detectors?

세 detector는 서로 다른 시간 해상도를 관찰합니다.

```text
1.0s Window
→ 상대적으로 긴 시간의 통계적 패턴

0.5s Window
→ 짧은 시간의 변화

Sample-level
→ sample 수준의 변화
```

따라서 특정 시간 범위에서 발생하는 이상을 하나의 detector만으로 처리하는 대신 서로 다른 detector가 상호 보완하도록 구성합니다.

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

---

## 30. Repository Structure

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

---

## 31. Script Description

| Script                                | Purpose                                    |
| ------------------------------------- | ------------------------------------------ |
| `1_visualize_normal_outlier.py`       | Normal / Fault 시계열 시각화                     |
| `2_Classification_idle_sections.py`   | Normal 데이터의 Idle 구간 분류                     |
| `3_time_structure_analysis.py`        | Timestamp, sampling gap, segment 구조 분석     |
| `4_make_window_dataset.py`            | Window 생성 및 feature extraction             |
| `5_run_mahalanobis.py`                | 기본 Window Mahalanobis detector             |
| `5_2_run_mahalanobis.py`              | Window-level Group K-Fold / multi-scale 평가 |
| `5_3_run_sample_level_mahalanobis.py` | Sample-level causal Mahalanobis 평가         |
| `6_compare_three_detectors.py`        | Three-detector ensemble 및 persistence 비교   |

### Script Numbering

README의 장 번호와 source script의 번호는 동일한 의미가 아닙니다.

```text
README
5장
→ Idle 처리

Script
5_*.py
→ Mahalanobis 모델링 / 평가
```

스크립트 번호는 실행 파이프라인을 위한 번호입니다.

---

## 32. Installation

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

`requirements.txt`는 주요 패키지의 호환 범위를 지정합니다.

현재 requirements는 완전한 patch-level version lock이 아니라 **version range 기반**입니다.

---

## 33. Reproducibility

최종 ensemble의 기본 설정:

```text
Folds   = 5
Repeats = 5
Seed    = 0
```

repeat별 seed:

```text
0
1
2
3
4
```

동일한 데이터와 동일한 Python/package 환경에서 동일 seed를 사용하면 동일한 fold assignment를 재현할 수 있습니다.

---

## 34. Data Preparation

원본 CSV는 저장소에 포함하지 않습니다.

필요한 최소 schema:

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

원본 데이터가 없는 환경에서는 실제 모델 성능을 재현할 수 없습니다.

저장소에는 분석 코드와 실험 결과를 제공합니다.

---

## 35. Reproduction

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

#### 1.0s Window

```bash
python ./src/4_make_window_dataset.py \
    --window-sec 1.0 \
    --step-sec 0.1
```

#### 0.5s Window

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

### Step 5-3. Sample-level Detector

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

## 36. Output Files

### Window CV

```text
outputs/5_2_mahalanobis_cv/
```

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

```text
ensemble_summary.csv
ensemble_repeat_results.csv
ensemble_event_details.csv
```

### Output Description

`ensemble_summary.csv`

→ detector 및 ensemble별 평균과 표준편차

`ensemble_repeat_results.csv`

→ repeat별 성능

`ensemble_event_details.csv`

→ fault event별 detection 여부, delay, sample 수

---

## 37. Key Findings

### Multi-scale Detection

1.0s 및 0.5s window detector는 서로 다른 시간 범위의 통계적 패턴을 관찰하며, sample-level detector는 더 짧은 시간 해상도의 변화를 직접 관찰합니다.

### Detector Complementarity

단일 detector가 놓치는 fault를 다른 detector가 보완할 수 있도록 세 가지 시간 해상도를 결합했습니다.

### OR_3

```text
Event Detection = 99.05%
Delay            = 0.470s
```

을 기록했습니다.

OR_3의 105 event-evaluation 중:

```text
104 detected
1 missed
```

입니다.

### OR_3 + P2

```text
Event Detection = 91.43%
Delay            = 0.571s
Normal FA Cycle  = 0.90%
```

입니다.

OR_3에 비해 normal false-alarm cycle rate가 감소하지만 짧은 fault event의 일부가 탐지되지 않았습니다.

P2의 9회 미탐은:

```text
fault_5
fault_19
```

에 집중되었습니다.

---

## 38. Limitations

### 1. Independent Final Holdout 부재

독립적인 미사용 final holdout 데이터가 없어 최종 평가는 Group-based 5-fold × 5-repeat cross validation 기반입니다.

### 2. Post-hoc Configuration Selection

최종 detector 및 hyperparameter는 개발 과정의 CV 결과를 비교한 뒤 선택했습니다.

또한 `event detection >= 90%` 기준도 결과를 본 뒤 적용한 사후적 기준입니다.

따라서 최종 성능이 독립 데이터 성능보다 낙관적으로 추정되었을 가능성이 있습니다.

### 3. Quantile Threshold Sensitivity

`q=0.9999`는 validation 정상 score의 극단적인 상위 분위수입니다.

Validation score 수가 많을 경우 최대값에 매우 가까워질 수 있으며, 소수의 큰 score에 threshold가 민감하게 영향을 받을 수 있습니다.

현재 최종 output에는 fold별 validation 정상 score 개수가 저장되어 있지 않아 정확한 개수는 별도로 보고하지 않습니다.

### 4. Fault Onset Ground Truth

실제 물리적 고장 발생 시각이 별도로 제공되지 않기 때문에 detection delay는 데이터에서 정의된 fault onset을 기준으로 계산합니다.

따라서 본 결과를 실제 장비의 고장 이전 예지시간과 동일한 의미로 해석해서는 안 됩니다.

### 5. Short Fault Events

매우 짧은 fault event에서는 충분한 window 길이 또는 persistence 연속성을 확보하기 어렵습니다.

예:

```text
fault_5
→ 3 samples
→ 약 0.2 sec
```

이러한 event는 0.5s / 1.0s window 및 P2 조건에서 특히 불리합니다.

### 6. Limited Fault Events

fault event가 21개로 제한적이므로 다양한 설비 상태와 운전 조건에서의 일반화 성능을 추가로 검증할 필요가 있습니다.

### 7. Operating Condition

실제 현장에서는 부하, 속도 및 기타 운전 조건에 따라 정상 신호 분포가 달라질 수 있습니다.

---

## 39. Future Work

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
Independent External Validation
```

특히 실제 현장 적용에서는 운전 조건별 정상 분포를 별도로 모델링하거나 adaptive threshold를 적용하는 방법을 고려할 수 있습니다.

---

## 40. Dataset / License

본 저장소는 분석 코드와 실험 결과를 제공합니다.

원본 데이터는 저장소에 포함하지 않습니다.

원본 데이터의 이용 및 재배포는 해당 데이터셋 제공기관의 이용 조건과 공모전 규정을 따라야 합니다.

본 저장소의 **코드 라이선스는 `LICENSE` 파일**을 따릅니다.

코드의 라이선스와 원본 데이터의 이용 권한은 별개의 문제이며, 본 저장소의 코드 라이선스가 원본 데이터의 재배포 권한을 부여하지는 않습니다.

---

## 41. Summary

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

을 함께 분석하는 것입니다.

최종 선택 구성:

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

최종 ensemble은 **5-fold × 5-repeat Group Cross Validation**으로 평가했습니다.

독립적인 final holdout 데이터는 포함하지 않으며, 최종 결과는 반복 교차검증 기반 성능으로 해석해야 합니다.
