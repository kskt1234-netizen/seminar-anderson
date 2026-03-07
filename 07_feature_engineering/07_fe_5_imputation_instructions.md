# 피처 엔지니어링 5: 결측치 처리 (Missing Value Imputation)

현업 데이터의 80% 이상에 결측치가 존재합니다. 어떻게 처리하느냐가 모델 성능에 직접적인 영향을 줍니다.

---

## 1. 결측치 파악

```python
import pandas as pd

df.isna().sum()               # 열별 결측치 개수
df.isna().mean() * 100        # 열별 결측치 비율 (%)
df.isna().any(axis=1).sum()   # 결측치가 있는 행의 수

# 결측 패턴 한눈에 보기
df.info()
```

---

## 2. 단순 대치 (SimpleImputer)

가장 기본적인 방법입니다.

```python
from sklearn.impute import SimpleImputer

# 수치형: 평균으로 채우기
mean_imp   = SimpleImputer(strategy='mean')
# 수치형: 중앙값으로 채우기 (이상치 있을 때 권장)
median_imp = SimpleImputer(strategy='median')
# 범주형: 최빈값으로 채우기
mode_imp   = SimpleImputer(strategy='most_frequent')
# 상수로 채우기
const_imp  = SimpleImputer(strategy='constant', fill_value=0)

X_imputed = mean_imp.fit_transform(X)

# Pandas로 직접 처리 (빠른 방법)
df['age'].fillna(df['age'].mean(), inplace=True)
df['city'].fillna('Unknown', inplace=True)
```

---

## 3. KNN 대치 (KNNImputer)

유사한 샘플(K개 최근접 이웃)의 값을 사용하여 결측치를 채웁니다. 단순 평균보다 정교합니다.

```python
from sklearn.impute import KNNImputer

knn_imp = KNNImputer(n_neighbors=5, weights='uniform')
X_imputed = knn_imp.fit_transform(X)

# weights='distance': 가까운 이웃에 더 높은 가중치
knn_imp_w = KNNImputer(n_neighbors=5, weights='distance')
```

---

## 4. 반복 대치 (IterativeImputer)

다른 피처들로 회귀 모델을 학습하여 결측치를 예측합니다. 가장 정교한 방법입니다.

```python
from sklearn.experimental import enable_iterative_imputer  # 실험적 기능 활성화
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# 기본 (BayesianRidge 회귀 사용)
it_imp = IterativeImputer(max_iter=10, random_state=42)
X_imputed = it_imp.fit_transform(X)

# 커스텀 추정기 사용
rf_imp = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=42),
    max_iter=5,
    random_state=42
)
X_imputed = rf_imp.fit_transform(X)
```

---

## 5. 결측치 지시자 피처 (Missing Indicator)

"이 값이 결측이었다"는 사실 자체를 피처로 추가합니다. 결측 여부 자체가 정보를 가질 때 유용합니다.

```python
from sklearn.impute import MissingIndicator
from sklearn.pipeline import make_pipeline, FeatureUnion
import numpy as np

indicator = MissingIndicator(features='missing-only')
flags = indicator.fit_transform(X)
# 결측치가 있는 열에 대해 True/False 열이 추가됨

# 실용적 패턴: 대치 + 지시자를 동시에 추가
df['age_was_missing'] = df['age'].isna().astype(int)
df['age'] = df['age'].fillna(df['age'].median())
```

---

## 6. 결측치 처리 전략 선택

| 결측률 | 데이터 유형 | 권장 전략 |
|--------|-----------|---------|
| < 5% | 수치형 | SimpleImputer (mean/median) |
| 5~30% | 수치형 | KNNImputer |
| 5~30% | 복잡한 패턴 | IterativeImputer |
| > 50% | 모든 유형 | 해당 열 제거 고려 |
| 범주형 | 저결측 | most_frequent 또는 'Unknown' 카테고리 |
| 결측 자체가 의미 있음 | 모든 유형 | MissingIndicator 추가 |

> ⚠️ **항상 Train 데이터로만 fit하고, Test 데이터에는 transform만 적용하세요!**

---

## 7. 연습문제 (Pytest) 🚀

- **tests_fe/test_fe_5_imputation.py**: SimpleImputer, KNNImputer, IterativeImputer를 적용하고 결측치가 모두 채워졌는지, 통계적으로 올바른지 검증하세요.
