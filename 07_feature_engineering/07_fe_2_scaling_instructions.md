# 피처 엔지니어링 2: 스케일링 (Feature Scaling)

대부분의 ML 알고리즘(KNN, SVM, 선형 회귀, 신경망)은 피처의 **크기(Scale)**에 민감합니다. 예를 들어 나이(0~100)와 연봉(0~100,000,000)이 같이 있으면 연봉이 모델을 지배합니다. 스케일링으로 모든 피처가 동등한 영향력을 갖게 조정합니다.

---

## 1. 표준화 (StandardScaler) — Z-Score

평균을 0, 표준편차를 1로 변환합니다. 정규분포를 따르는 데이터에 적합합니다.

$$z = \frac{x - \mu}{\sigma}$$

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled.mean(axis=0))  # ≈ [0.0, 0.0, ...]
print(X_scaled.std(axis=0))   # ≈ [1.0, 1.0, ...]

# 저장된 통계 확인
print(scaler.mean_)    # 각 열의 평균
print(scaler.scale_)   # 각 열의 표준편차
```

---

## 2. 최솟값-최댓값 정규화 (MinMaxScaler)

[0, 1] 범위로 변환합니다. 이미지 픽셀, 신경망 입력에 자주 쓰입니다.

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))  # 범위 변경 가능
X_scaled = scaler.fit_transform(X)

# 모든 값이 [0.0, 1.0] 사이
assert X_scaled.min() >= 0.0
assert X_scaled.max() <= 1.0
```

> ⚠️ 이상치에 매우 민감합니다. 이상치가 있다면 `RobustScaler`를 사용하세요.

---

## 3. 로버스트 스케일러 (RobustScaler)

중앙값(Median)과 IQR을 사용하여 이상치의 영향을 줄입니다.

$$x' = \frac{x - Q_2}{Q_3 - Q_1}$$

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 스케일링에 사용된 통계
print(scaler.center_)  # 각 열의 중앙값 (Q2)
print(scaler.scale_)   # 각 열의 IQR (Q3 - Q1)
```

---

## 4. 정규화 (Normalizer) — L2 Norm

각 **행(샘플)**을 단위 벡터(길이=1)로 만듭니다. 텍스트 유사도, 코사인 유사도 계산에 사용합니다.

```python
from sklearn.preprocessing import Normalizer

norm = Normalizer(norm='l2')   # 'l1', 'l2', 'max' 선택 가능
X_normalized = norm.fit_transform(X)

# 각 행의 L2 노름 = 1
import numpy as np
norms = np.linalg.norm(X_normalized, axis=1)
# 모두 ≈ 1.0
```

> ⚠️ Normalizer는 **열(피처)이 아닌 행(샘플)**을 정규화합니다. 다른 스케일러와 반대입니다.

---

## 5. MaxAbsScaler

각 피처의 절댓값 최대치로 나눠 [-1, 1] 범위로 만듭니다. **희소 행렬(sparse matrix)**에 적합합니다.

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)
# 값 범위: [-1.0, 1.0]
```

---

## 6. 스케일러 선택 기준

| 상황 | 권장 스케일러 |
|------|------------|
| 정규분포에 가까운 데이터 | `StandardScaler` |
| 이미지, [0,1] 필요, 이상치 없음 | `MinMaxScaler` |
| 이상치 있는 데이터 | `RobustScaler` |
| 텍스트, 코사인 유사도 | `Normalizer` |
| 희소 행렬 (TF-IDF 등) | `MaxAbsScaler` |
| 트리 모델 (RF, XGBoost, LightGBM) | 스케일링 불필요 |

---

## 7. 핵심 주의사항 — Train/Test 분리

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # ✅ 학습 데이터로만 fit!
X_test_scaled  = scaler.transform(X_test)       # ✅ 테스트는 transform만!

# ❌ 절대 금지: scaler.fit_transform(X_test)
# 테스트 데이터의 통계로 fit하면 Data Leakage 발생!
```

---

## 8. 연습문제 (Pytest) 🚀

- **tests_fe/test_fe_2_scaling.py**: StandardScaler, MinMaxScaler, RobustScaler, Normalizer를 구현하고 각 스케일러의 수학적 성질을 검증하세요.
