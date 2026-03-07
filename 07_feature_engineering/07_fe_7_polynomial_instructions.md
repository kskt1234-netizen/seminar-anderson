# 피처 엔지니어링 7: 다항 피처 & 상호작용 (Polynomial & Interaction Features)

선형 모델은 피처 간 비선형 관계를 표현하지 못합니다. 다항 피처를 추가하면 선형 모델도 곡선적인 패턴을 학습할 수 있습니다.

---

## 1. PolynomialFeatures

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.array([[2, 3]])   # 피처: [a, b]

# degree=2: 1, a, b, a², a·b, b²
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)

print(poly.get_feature_names_out())
# ['1', 'x0', 'x1', 'x0^2', 'x0 x1', 'x1^2']
print(X_poly)
# [[1. 2. 3. 4. 6. 9.]]

# 상호작용 항만 생성 (자체 제곱항 제외)
inter = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_inter = inter.fit_transform(X)
print(inter.get_feature_names_out())
# ['x0', 'x1', 'x0 x1']

# 편향 항 제외 (선형 회귀의 intercept가 이미 처리)
poly_no_bias = PolynomialFeatures(degree=2, include_bias=False)
```

---

## 2. 도메인 지식 기반 상호작용 피처

모델이 자동으로 찾기 어려운, 의미 있는 조합을 직접 만듭니다.

```python
import pandas as pd

# 집 가격 예측
df['area_per_room']     = df['total_area'] / df['num_rooms']      # 방 하나당 면적
df['price_per_area']    = df['price'] / df['total_area']           # 단위 면적당 가격

# 이진 조합 (AND 조건)
df['garage_and_garden'] = (df['has_garage'] & df['has_garden']).astype(int)

# 비율 피처
df['debt_to_income']    = df['debt'] / (df['income'] + 1e-8)      # 부채비율
df['click_through_rate'] = df['clicks'] / (df['impressions'] + 1e-8)
```

---

## 3. 수학적 변환 (Mathematical Transformations)

```python
import numpy as np

# 로그 변환: 오른쪽으로 치우친(right-skewed) 분포를 정규화
df['log_income']  = np.log1p(df['income'])     # log(x + 1), x=0 안전 처리

# 제곱근 변환: 중간 강도의 왜곡 보정
df['sqrt_area']   = np.sqrt(df['area'])

# 제곱: 이차 관계 (U자형 패턴)
df['age_sq']      = df['age'] ** 2

# 역수: 거리 기반 피처 (가까울수록 영향 큼)
df['inv_dist']    = 1 / (df['distance'] + 1e-8)

# 절댓값: 부호보다 크기가 중요할 때
df['abs_change']  = df['price_change'].abs()
```

---

## 4. 변환 선택 기준

```python
import pandas as pd
import numpy as np

# 분포 왜곡 확인
skewness = df['income'].skew()

if abs(skewness) < 0.5:
    # 거의 대칭: 변환 불필요
    pass
elif abs(skewness) < 1.0:
    # 중간 왜곡: 제곱근 변환
    df['income_transformed'] = np.sqrt(df['income'])
else:
    # 심한 왜곡: 로그 변환
    df['income_transformed'] = np.log1p(df['income'])
```

---

## 5. 피처 수 폭발 주의

```
degree=2, 피처 100개 → C(100+2, 2) = 5,151개 피처
degree=3, 피처 100개 → C(100+3, 3) = 171,700개 피처
```

고차원에서는:
1. 중요 피처만 선택 후 PolynomialFeatures 적용
2. `interaction_only=True`로 교차항만 생성
3. 이후 피처 선택(Feature Selection)으로 불필요한 항 제거

---

## 6. 연습문제 (Pytest) 🚀

- **tests_fe/test_fe_7_polynomial.py**: PolynomialFeatures 적용, 수학적 변환 함수를 구현하고 생성된 피처의 정확성을 검증하세요.
