# KAMPact

### 진동·전류 시계열 기반 프레스 유압펌프 이상 조기탐지 및 오경보 분석

소성가공 프레스 공정에서 수집된 **진동 및 전류 시계열 데이터**를 기반으로 유압펌프의 이상 상태를 조기에 탐지하고, 정상 운전 및 유휴(Idle) 상태에서 발생하는 오경보를 함께 분석하는 예지보전 AI 프로젝트입니다.

본 프로젝트는 단순한 고장 분류(Classification)가 아니라 다음 세 가지를 동시에 고려하는 것을 목표로 합니다.

* 이상 이벤트를 얼마나 안정적으로 탐지하는가
* 이상 발생 이후 얼마나 빠르게 탐지하는가
* 정상 운전 및 Idle 상태에서 불필요한 경보가 얼마나 발생하는가

---

## 1. 문제 정의

프레스 유압펌프의 상태를 나타내는 센서 데이터에는 정상 운전 중에도 진동과 전류의 변동이 존재합니다. 또한 설비가 가동 중이지 않은 **Idle 상태**에서는 별도의 센서 패턴이 나타날 수 있습니다.

따라서 단순한 센서 임계값 기반 탐지만으로는 다음과 같은 문제가 발생할 수 있습니다.

```text
정상 운전
   ↓
센서 변동 발생
   ↓
이상으로 오인
   ↓
False Alarm
```

KAMPact는 정상 운전 상태의 다변량 분포를 학습한 후, 현재 상태가 정상 분포에서 얼마나 벗어났는지를 Mahalanobis Distance로 계산하여 이상 여부를 판단합니다.

---

## 2. 프로젝트 목표

### 핵심 목표

1. 진동·전류 시계열 기반 이상 상태 탐지
2. 이상 이벤트의 조기 탐지
3. 정상 운전 사이클의 오탐(False Alarm) 분석
4. Idle 상태에서의 오탐 분석
5. Window Scale에 따른 탐지 성능 및 탐지 지연 비교
6. Group K-Fold 기반 반복 검증을 통한 성능 안정성 확인

---

## 3. 데이터

본 프로젝트에서는 프레스 유압펌프의 다음 센서 데이터를 사용합니다.

| 컬럼                | 설명       |
| ----------------- | -------- |
| `TimeStamp`       | 센서 측정 시각 |
| `AI0_Vibration`   | 진동 센서 0  |
| `AI1_Vibration`   | 진동 센서 1  |
| `AI2_Current`     | 전류 센서    |
| `Equipment_state` | 설비 상태    |
| `Idle`            | 유휴 상태 여부 |

### 데이터 구성

```text
Normal Data
├─ 정상 운전 구간
└─ Idle 구간

Fault Data
└─ 이상 이벤트 구간
```

원본 데이터는 저장소에 포함하지 않으며, 다음 경로에 배치하여 사용합니다.

```text
data/
├─ press_data_normal.csv
├─ press_data_normal_with_idle.csv
└─ outlier_data.csv
```

---

## 4. 전체 분석 Pipeline

```text
Raw Sensor Data
       │
       ▼
┌────────────────────────────┐
│ 1. 시계열 시각화           │
│    Normal / Fault 비교     │
└────────────────────────────┘
       │
       ▼
┌────────────────────────────┐
│ 2. Idle 구간 분류          │
│    운전=0 / Idle=1         │
└────────────────────────────┘
       │
       ▼
┌────────────────────────────┐
│ 3. 시간 구조 분석          │
│    Sampling Gap / Segment  │
└────────────────────────────┘
       │
       ▼
┌────────────────────────────┐
│ 4. Window Dataset 생성     │
│    Segment → Window        │
│    Feature Extraction      │
└────────────────────────────┘
       │
       ▼
┌────────────────────────────┐
│ 5. Mahalanobis Anomaly     │
│    Detection               │
└────────────────────────────┘
       │
       ├───────────────┐
       ▼               ▼
   5_2 Event CV     5_3 Sample CV
       │               │
       ▼               ▼
 Multi-scale       Raw / Diff
 Event Detection   Sample Detection
 False Alarm       Causal Smoothing
 Idle Analysis
```

---

## 5. 시계열 전처리

### 5.1 Sampling Gap 분석

센서 timestamp의 연속성을 확인한 후 일정 수준 이상의 시간 간격이 발생하면 서로 다른 시계열 segment로 분리합니다.

현재 기본 기준:

```text
Gap > 0.5 sec
→ 새로운 Segment
```

이를 통해 서로 독립적인 프레스 사이클 또는 시계열 구간을 하나의 그룹으로 관리합니다.

---

### 5.2 Idle 구간 분리

정상 데이터 내부의 Idle 구간은 일반 운전 상태와 다른 센서 특성을 가질 수 있으므로 별도로 관리합니다.

```text
Idle = 0
→ 정상 운전

Idle = 1
→ 유휴 상태
```

Idle 데이터는 정상 운전 모델의 학습에 직접 사용하지 않고, **독립적인 오탐 평가용 데이터**로 활용합니다.

---

### 5.3 Segment 단위 데이터 분할

Window 단위 random split을 사용하면 같은 시계열에서 생성된 매우 유사한 window가 train/test에 동시에 포함될 수 있습니다.

이를 방지하기 위해 KAMPact는 다음 순서로 데이터를 분할합니다.

```text
Raw Data
   ↓
Segment 생성
   ↓
Segment 단위 Train / Validation / Test
   ↓
Window 생성
```

따라서 동일한 segment가 여러 split에 동시에 들어가는 **segment leakage를 방지**합니다.

---

## 6. Window Feature Engineering

각 시계열 window에서 센서별 통계 특징을 추출합니다.

### 센서별 특징

각 센서에 대해 다음 5개의 특징을 계산합니다.

```text
Mean
Standard Deviation
RMS
Peak-to-Peak (PTP)
Slope
```

3개 센서 × 5개 특징:

```text
3 × 5 = 15 features
```

추가로 두 진동 센서 간 상관관계를 계산합니다.

```text
AI0_AI1_corr
```

따라서 기본 입력 벡터는 총 **16개 특징**으로 구성됩니다.

```text
[AI0 statistics]
[AI1 statistics]
[AI2 statistics]
[AI0-AI1 correlation]
```

---

## 7. Anomaly Detection Model

### Mahalanobis Distance

KAMPact는 정상 운전 데이터의 다변량 분포를 학습하고 각 window가 정상 분포에서 얼마나 떨어져 있는지를 Mahalanobis Distance로 계산합니다.

개념적으로:

```text
정상 분포
    │
    ├── 가까움 → Normal
    │
    └── 멀어짐 → Anomaly
```

단순 Euclidean Distance와 달리 feature 간 공분산 구조를 반영할 수 있습니다.

---

### Covariance Estimation

다음 공분산 추정 방법을 지원합니다.

```text
Empirical Covariance
Ledoit-Wolf
OAS
```

기본 실험에서는 **Ledoit-Wolf**를 사용합니다.

---

### Signed Log Transformation

feature 값의 크기 차이 및 극단값 영향을 완화하기 위해 다음 형태의 변환을 적용합니다.

```text
sign(x) × log(1 + |x|)
```

이를 통해 양수/음수 feature를 모두 유지하면서 큰 값의 영향력을 완화합니다.

---

## 8. Threshold

이상 여부 판단을 위한 threshold는 두 가지 방법을 지원합니다.

### F1 기반 Threshold

Validation 데이터의 label을 이용하여 F1-score가 최대가 되는 threshold를 선택합니다.

```text
Validation
    ↓
여러 threshold 평가
    ↓
F1 최대 threshold 선택
```

### Normal Quantile Threshold

Validation 데이터 중 정상 데이터의 anomaly score만 사용하여 quantile 기반 threshold를 결정합니다.

예:

```text
Normal score
     ↓
99.5 percentile
     ↓
Threshold
```

두 방법은 서로 다른 목적을 가지므로 최종 평가에서 별도의 설정으로 비교합니다.

---

# 9. Event-level Evaluation

이 프로젝트의 목적은 단순한 sample classification이 아니므로 **Event 단위 평가**를 함께 수행합니다.

주요 평가지표:

| 지표                   | 설명                           |
| -------------------- | ---------------------------- |
| Event Detection Rate | 전체 이상 이벤트 중 탐지된 이벤트 비율       |
| Detection Delay      | 이상 onset 이후 첫 경보까지의 시간       |
| F1                   | window/sample 단위 분류 성능       |
| Precision            | 이상 경보 중 실제 이상 비율             |
| Recall               | 실제 이상 중 탐지된 비율               |
| Normal Window FPR    | 정상 window에서 발생한 오탐 비율        |
| FA Cycle Rate        | 정상 cycle 중 오탐이 한 번이라도 발생한 비율 |
| FA Episodes          | 연속적인 오탐을 하나의 episode로 묶은 횟수  |
| FA/hour              | 관측된 정상 데이터 시간당 오탐 episode    |
| Idle FA Cycle Rate   | Idle cycle에서 오탐이 발생한 비율      |

---

## 10. Detection Delay 정의

현재 데이터에서는 물리적인 고장 발생 시점을 직접 알 수 없기 때문에 다음과 같이 정의합니다.

### Window-based Model

```text
fault_onset_time
        ↓
첫 번째 alarm window의 window_end
```

### Sample-based Model

```text
fault segment 시작 시점
        ↓
첫 번째 alarm sample
```

따라서 본 프로젝트의 detection delay는 **물리적 고장 발생 이전의 예측 시간**이 아니라, 데이터에서 정의된 이상 onset 이후 최초 경보까지의 탐지 지연 시간입니다.

---

# 11. Group K-Fold Cross Validation

이상 이벤트와 정상 운전 사이클의 특성을 고려하여 `group_id` 단위의 K-Fold 교차검증을 사용합니다.

기본 구성:

```text
K = 4 folds
Repeats = 5
```

각 반복에서 fold를 새롭게 구성하여 결과를 `mean ± std` 형태로 집계합니다.

```text
Fold i
   → Test

Fold i+1
   → Validation

Remaining folds
   → Train
```

학습에는 **비-Idle 정상 데이터만** 사용하고, Idle 데이터는 모델 학습 및 threshold 설정에서 제외합니다.

---

# 12. Multi-scale Detection

단일 window 길이만 사용할 경우 짧은 이상 이벤트가 충분한 window를 확보하지 못할 수 있습니다.

이를 보완하기 위해 여러 window scale을 함께 사용할 수 있습니다.

예:

```text
1.0 sec Window
      +
0.5 sec Window
```

Multi-scale 시스템에서는 하나의 fault segment를 중복 평가하지 않도록 **가장 큰 window에서 평가 가능한 scale을 우선 사용**하고, 그렇지 않은 짧은 이벤트는 작은 window scale로 라우팅합니다.

이를 통해 긴 이벤트와 짧은 이벤트를 동시에 평가할 수 있습니다.

---

# 13. Sample-level Model

`5_3_run_sample_level_mahalanobis.py`에서는 window 통계 특징을 사용하는 대신 원본 0.1초 샘플에 직접 Mahalanobis Distance를 계산합니다.

지원 특징:

```text
raw
raw_diff
raw_diff_roll3
```

그리고 과거 k개 score만 사용하는 causal smoothing을 적용할 수 있습니다.

```text
현재 시점
   ↑
과거 k개 sample
   ↓
평균 / 최대
   ↓
Alarm
```

이 방식은 window-based 방식보다 더 빠른 반응이 가능하지만, window 통계가 제공하는 정보가 줄어들기 때문에 별도의 비교 실험으로 사용합니다.

---

# 14. 저장소 구조

```text
KAMPact/
│
├─ data/
│  └─ .gitkeep
│
├─ src/
│  ├─ 1_visualize_normal_outlier.py
│  ├─ 2_Classification_idle_sections.py
│  ├─ 3_time_structure_analysis.py
│  ├─ 4_make_window_dataset.py
│  ├─ 5_run_mahalanobis.py
│  ├─ 5_2_run_mahalanobis.py
│  └─ 5_3_run_sample_level_mahalanobis.py
│
├─ result/
│  └─ visialize_normal_outlier/
│
├─ outputs/
│  ├─ 5_2_mahalanobis_cv/
│  │  ├─ cv_comparison.csv
│  │  ├─ cv_repeat_results.csv
│  │  ├─ cv_event_details.csv
│  │  └─ cv_event_detection_frequency.csv
│  │
│  └─ 5_3_sample_level_cv/
│     ├─ sample_cv_comparison.csv
│     ├─ sample_cv_repeat_results.csv
│     ├─ sample_cv_event_details.csv
│     └─ sample_cv_event_detection_frequency.csv
│
├─ .gitignore
├─ requirements.txt
└─ README.md
```

---

# 15. 주요 실행 방법

## 15.1 Idle 데이터 생성

```bash
python ./src/2_Classification_idle_sections.py
```

---

## 15.2 시간 구조 분석

```bash
python ./src/3_time_structure_analysis.py
```

---

## 15.3 Window Dataset 생성

예를 들어:

```bash
python ./src/4_make_window_dataset.py \
    --window-sec 0.8 \
    --step-sec 0.1
```

생성되는 주요 파일:

```text
model_windows.csv
idle_windows.csv
segment_manifest.csv
event_manifest.csv
split_class_summary.csv
window_features_all.csv
```

---

## 15.4 기본 Mahalanobis 실행

```bash
python ./src/5_run_mahalanobis.py \
    --input result/modeling_dataset_0.8_0.1/model_windows.csv
```

---

## 15.5 Group K-Fold + Multi-scale 평가

```bash
python ./src/5_2_run_mahalanobis.py \
    --inputs result/modeling_dataset_1.0_0.1 \
             result/modeling_dataset_0.5_0.1
```

Multi-scale:

```bash
python ./src/5_2_run_mahalanobis.py \
    --multiscale \
    result/modeling_dataset_1.0_0.1 \
    result/modeling_dataset_0.5_0.1
```

---

## 15.6 Sample-level 비교 실험

```bash
python ./src/5_3_run_sample_level_mahalanobis.py \
    --normal-path data/press_data_normal_with_idle.csv \
    --fault-path data/outlier_data.csv
```

평활 길이 비교:

```bash
python ./src/5_3_run_sample_level_mahalanobis.py \
    --agg-k 1 3 5 10
```

---

# 16. 실험 결과

현재 저장소에 기록된 Group K-Fold 실험은 4-fold × 5-repeat 방식으로 수행되었습니다.

### Window-based Mahalanobis 실험

기록된 `1.0s + 0.5s` Multi-scale 실험에서는 다음 결과가 확인되었습니다.

| System      | Threshold | Evaluated Events | Event Detection | Mean Delay |    F1 | Precision | Recall | Normal FPR |
| ----------- | --------- | ---------------: | --------------: | ---------: | ----: | --------: | -----: | ---------: |
| 1.0s        | F1        |            16/21 |          98.75% |      1.07s | 0.939 |     0.954 |  0.926 |     0.153% |
| 0.5s        | F1        |            18/21 |         100.00% |      0.64s | 0.892 |     0.916 |  0.868 |     0.274% |
| 1.0s + 0.5s | F1        |            18/21 |          98.89% |      0.99s | 0.945 |     0.952 |  0.938 |     0.161% |

Multi-scale 실험에서는 Idle cycle 오탐률이 0%로 기록되었습니다.

> 주의: window 길이에 따라 평가 가능한 fault event의 수가 달라지므로 Event Detection Rate만 비교하지 않고 `평가 이벤트 수 / 전체 이벤트 수`를 함께 확인합니다.

---

# 17. Sample-level 비교 결과

Sample-level Mahalanobis는 원본 시계열에 직접 anomaly score를 계산하여 더 세밀한 시간 단위의 탐지를 시도합니다.

현재 실험에서는 `raw_diff` 기반 causal mean을 사용했을 때:

```text
k = 1
Event Detection Rate ≈ 85.7%
Mean Detection Delay ≈ 0.43 sec
Sample F1 ≈ 0.58
```

로 기록되었습니다.

이 결과는 sample-level 방식이 빠른 반응성을 제공할 수 있는 반면, window 통계 기반 모델과 비교했을 때 분류 성능 및 이벤트 탐지 안정성에 차이가 있음을 보여줍니다.

---

# 18. 프로젝트의 핵심 특징

KAMPact는 단순한 이상/정상 분류를 넘어 다음을 함께 평가합니다.

```text
Anomaly Detection
        +
Early Detection
        +
Event-level Evaluation
        +
Cycle-level False Alarm
        +
Idle False Alarm
        +
Multi-scale Detection
```

특히 동일한 프레스 cycle에서 생성된 window가 train/test에 동시에 포함되지 않도록 **segment/group 단위 검증 체계**를 적용하여 시계열 데이터에서 발생하기 쉬운 leakage를 방지하는 것을 핵심 설계 원칙으로 사용합니다.

---

## 19. 향후 개선 방향

* Detection Delay의 물리적 고장 발생 시점 기준 정교화
* Short Event에 대한 평가 기준 개선
* 다양한 시계열 feature 및 frequency-domain feature 추가
* Threshold와 Alarm Persistence의 결합
* 정상 운전과 Idle 상태의 특성 차이를 활용한 계층형 이상탐지
* 추가 센서 및 다양한 운전 조건에서의 일반화 성능 검증

---

## 20. Reproducibility

Python 환경에서 필요한 패키지는 `requirements.txt`를 통해 설치할 수 있습니다.

```bash
pip install -r requirements.txt
```

대용량 원본 CSV와 생성된 모델링 데이터는 `.gitignore`에 의해 Git 저장소에서 제외됩니다.

---

## License / Dataset

본 저장소는 공모전 프로젝트 및 연구 목적의 구현 코드를 제공합니다.

원본 데이터셋의 사용 및 재배포는 해당 데이터셋의 제공 조건과 공모전 규정을 따릅니다.
