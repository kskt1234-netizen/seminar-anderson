# 피처 엔지니어링 6: 구간화 (Binning / Discretization)

연속형 수치 변수를 범주형 구간으로 나눕니다. 비선형 패턴 포착, 이상치 영향 감소, 모델 해석력 향상에 도움이 됩니다.

---

## 1. 동일 너비 구간화 (Equal-Width) — `pd.cut`

전체 값의 범위를 동일한 너비로 나눕니다.

```python
import pandas as pd
import numpy as np

ages = pd.Series([5, 15, 22, 35, 42, 58, 73, 81])

# 4개 구간으로 자동 분할
bins_auto = pd.cut(ages, bins=4)

# 레이블 지정
bins_labeled = pd.cut(ages, bins=4, labels=['young', 'adult', 'middle', 'senior'])

# 경계 직접 지정
bins_custom = pd.cut(
    ages,
    bins=[0, 18, 35, 60, 100],
    labels=['teen', 'young_adult', 'adult', 'senior'],
    right=True    # 구간이 (left, right] 형태 (기본값)
)

# 결과 확인
print(pd.cut(ages, bins=4).value_counts())
```

---

## 2. 동일 빈도 구간화 (Equal-Frequency / Quantile) — `pd.qcut`

각 구간에 **동일한 수의 데이터**가 들어가도록 나눕니다. 왜곡된 분포에 더 효과적입니다.

```python
# 4분위수로 나누기
quartile_bins = pd.qcut(ages, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# 10분위수 (decile) — 레이블을 숫자로
decile_bins = pd.qcut(ages, q=10, labels=False)  # 0~9

# 중복 값이 많을 때 duplicates='drop' 옵션 사용
pd.qcut(ages, q=4, duplicates='drop')
```

---

## 3. Scikit-learn KBinsDiscretizer

파이프라인에 통합할 수 있는 sklearn 방식입니다.

```python
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

X = np.array([[20], [35], [42], [55], [70]])

# 균일 너비 + 원-핫 인코딩 출력
kbd = KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='uniform')
X_binned = kbd.fit_transform(X)

# encode 옵션:
# 'onehot-dense' — 원-핫 인코딩 (밀집 행렬)
# 'ordinal'      — 정수 레이블 (0, 1, 2, ...)

# strategy 옵션:
# 'uniform'  — 동일 너비 (pd.cut과 유사)
# 'quantile' — 동일 빈도 (pd.qcut과 유사)
# 'kmeans'   — K-Means 클러스터링 기반

# 구간 경계 확인
print(kbd.bin_edges_)
```

---

## 4. pd.cut vs pd.qcut 비교

```python
data = pd.Series([1, 1, 1, 2, 3, 10, 10, 100])

print("equal-width (pd.cut):")
print(pd.cut(data, bins=3).value_counts())
# 값이 치우쳐 있으면 특정 구간이 텅 빔

print("\nequal-frequency (pd.qcut):")
print(pd.qcut(data, q=3).value_counts())
# 각 구간에 균등하게 분배
```

---

## 5. 언제 구간화를 사용할까?

- 나이, 소득 등 사람이 이해하기 쉬운 범주를 만들 때
- 이상치의 영향을 완화하고 싶을 때
- 선형 모델에 비선형 관계를 추가하고 싶을 때
- 연속형 변수가 타겟과 계단식 관계를 가질 때

> ⚠️ 트리 기반 모델(RF, XGBoost)은 자체적으로 분할 기준을 찾으므로 구간화 효과가 제한적입니다.

---

## 6. 연습문제 (Pytest) 🚀

- **tests_fe/test_fe_6_binning.py**: `pd.cut`, `pd.qcut`, `KBinsDiscretizer`를 사용하여 구간화 함수를 구현하세요.
