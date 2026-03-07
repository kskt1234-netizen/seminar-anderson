# 데이터 클래스 불균형(Data Imbalance) 처리 가이드

머신러닝 프로젝트의 약 80% 이상은 비대칭 데이터 문제를 겪습니다. 타겟 클래스의 구성 비율이 매우 차이나는 **데이터 불균형(Class Imbalance)** 문제를 이해하고, 파이썬 기반 데이터셋을 활용해 파훼하는 실무적 방식을 살펴봅니다.

## 1. 클래스 불균형이란?
두 개 이상의 범주 중, 하나의 범주에만 데이터가 비정상적으로 쏠려있는 상태입니다.
- **실전 예시**: 암 진단 데이터 (정상 99%, 암 1%), 금융 사기 탐지, 공장 제조 공정에서의 불량품 탐지.
- **위험성**: 모델이 그냥 "전부 정상(다수 클래스)이다!"라고만 찍어도 모델 정확도(Accuracy)가 99%가 되어버리는 착시 현상이 일어납니다. 따라서 '진짜 찾아야 할 희귀한 타겟'의 식별력을 완전히 잃어버리는 치명적인 문제가 발생합니다.

## 2. 불균형 데이터 문제 해결하기

보통 **Resampling (재표본추출)** 기법을 사용하여, 데이터 개수를 물리적으로 맞춰주는 방식을 가장 먼저 적용합니다. 이 과정에서 파이썬의 `imbalanced-learn` 라이브러리가 표준처럼 쓰입니다.

### 2.1. 언더샘플링 (Undersampling)
다수(Majority) 클래스의 데이터 개수를 소수(Minority) 클래스 수준까지 대폭 버리는(삭제하는) 방식입니다.
- **Random Under Sampler**: 무작위로 다수 클래스의 관측치를 삭제합니다.
- **장점**: 연산 시간이 대폭 감축됩니다.
- **단점**: 유의미한 정보들까지 몽땅 버려질 수 있는 치명적 정보 손실(Information Loss)이 생깁니다.

### 2.2. 오버샘플링 (Oversampling)
소수 클래스의 데이터 개수를 다수 클래스 수준까지 뻥튀기하는(증식하는) 방식입니다. 학습 정보의 유실이 없어서 언더샘플링보다 보편적으로 쓰입니다.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: 가장 유명한 생성 방식. 단순히 복사하는게 아니라 소수 클래스 데이터 픽셀/지점들 사이를 보간(선으로 이어서 그 위의 임의 값을 찍음)하여 **가짜이면서도 실제와 비슷한 변종 데이터**를 새로 창조해냅니다.
- **장점**: 모델이 소수 클래스의 "패턴"을 확실히 배울 기회가 늘어납니다.
- **단점**: 데이터 수가 2배, 편향도가 심했다면 수백배 늘어나 학습 속도가 엄청나게 늘어질 수 있고 특정한 포인트에 과적합(Overfitting) 될 수 있습니다.

## 3. 실습 가이드

`imblearn` 라이브러리를 활용해 가상의 불균형 모델을 다듬는 코드 구조입니다.

### Step 1: 언더샘플링 적용 (RandomUnderSampler)
```python
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# 불균형한 X, y 데이터가 있다고 가정합니다. (예: 정상 90개, 불량 10개)
rus = RandomUnderSampler(random_state=42)

# 반드시 훈련 데이터(Train data)에만 핏하고 리샘플해야 합니다. (Test 타겟 변경 금지)
X_resampled, y_resampled = rus.fit_resample(X, y)

print("Before:", Counter(y)) # {0: 90, 1: 10}
print("After Under:", Counter(y_resampled)) # {0: 10, 1: 10}
```

### Step 2: 오버샘플링 적용 (SMOTE)
실제 현장에서 Baseline 잡을 때 무조건 한 번씩 돌려보는 SMOTE 적용법입니다.
```python
from imblearn.over_sampling import SMOTE

# SMOTE 초기화 
smote = SMOTE(random_state=42)

# 가상의 새로운 점들을 생성
X_smote, y_smote = smote.fit_resample(X, y)

print("Before:", Counter(y)) # {0: 90, 1: 10}
print("After SMOTE:", Counter(y_smote)) # {0: 90, 1: 90}
```

## 4. 실무 주의점 정리
1. **절대 테스트(Test) 데이터셋은 Resampling 하지 마세요**. 모델을 학습시키는(Train) 데이터에만 오버/언더샘플링을 적용해야 합니다. 현실 세계 비율을 왜곡해서 평가하면 안 되기 때문입니다.
2. 불균형 데이터 모델 평가에서는 **Accuracy 를 금기시** 하세요. `F1-score`, `Precision(정밀도)`, `Recall(재현율)` 그리고 `ROC-AUC` 메트릭을 반드시 살펴야 합니다.
