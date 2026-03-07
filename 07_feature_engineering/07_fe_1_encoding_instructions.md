# 피처 엔지니어링 1: 범주형 인코딩 (Categorical Encoding)

머신러닝 모델은 숫자만 이해합니다. 문자열로 된 범주형 데이터를 숫자로 변환하는 과정이 **인코딩**입니다.

---

## 1. 라벨 인코딩 (Label Encoding)

각 카테고리에 정수를 순서대로 할당합니다.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = ['cat', 'dog', 'cat', 'fish', 'dog']
encoded = le.fit_transform(y)
# array([0, 1, 0, 2, 1])   ← cat=0, dog=1, fish=2

# 역변환
original = le.inverse_transform([0, 1, 2])
# array(['cat', 'dog', 'fish'])

# 클래스 목록 확인
print(le.classes_)  # ['cat' 'dog' 'fish']
```

> ⚠️ **주의**: 다중 카테고리 **입력 피처**에 라벨 인코딩을 사용하면, 모델이 숫자의 크기를 순서(cat < dog < fish)로 해석합니다. 타겟 변수에만 사용하거나 트리 모델에만 사용하세요.

---

## 2. 순서형 인코딩 (Ordinal Encoding)

카테고리가 **명확한 순서**를 가질 때, 그 순서를 지정하여 인코딩합니다.

```python
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

df = pd.DataFrame({'grade': ['high', 'low', 'medium', 'high', 'low']})

oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])  # 순서 명시
encoded = oe.fit_transform(df[['grade']])
# [[2.], [0.], [1.], [2.], [0.]]
# low=0, medium=1, high=2
```

---

## 3. 원-핫 인코딩 (One-Hot Encoding)

각 카테고리를 독립된 0/1 이진 열로 분리합니다. **순서가 없는 카테고리**에 표준적으로 사용합니다.

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

df = pd.DataFrame({'city': ['Seoul', 'Busan', 'Seoul', 'Jeju']})

# sklearn 방식
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = ohe.fit_transform(df[['city']])
# [[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]  ← Busan, Seoul, Jeju 순서

# Pandas 방식 (빠른 탐색용)
dummies = pd.get_dummies(df['city'], prefix='city')
#    city_Busan  city_Jeju  city_Seoul
# 0       False       False        True
```

> ⚠️ **Dummy Variable Trap**: 카테고리가 n개면 n-1개 열로 충분합니다 (나머지 하나는 다른 열로 유추 가능). `drop='first'` 옵션으로 하나를 제거하세요.

---

## 4. 타겟 인코딩 (Target Encoding)

각 카테고리를 **타겟 변수의 평균값**으로 대체합니다. 카테고리 수가 매우 많을 때(고카디널리티) 효과적입니다.

```python
import pandas as pd

def target_encode(df: pd.DataFrame, column: str, target: str) -> pd.Series:
    # 학습 데이터 기준으로 각 카테고리의 타겟 평균 계산
    means = df.groupby(column)[target].mean()
    return df[column].map(means)

# 예시
df = pd.DataFrame({
    'city':    ['Seoul', 'Busan', 'Seoul', 'Jeju', 'Busan'],
    'revenue': [100, 80, 120, 60, 90]
})
df['city_encoded'] = target_encode(df, 'city', 'revenue')
# Seoul → 110.0, Busan → 85.0, Jeju → 60.0
```

> ⚠️ **Data Leakage 주의**: 반드시 학습 데이터로만 평균을 계산하고, 검증/테스트 데이터에 `map()`으로 적용하세요.

---

## 5. 빈도 인코딩 (Frequency Encoding)

카테고리의 **등장 빈도(Count 또는 비율)**로 대체합니다.

```python
freq_map = df['city'].value_counts().to_dict()
df['city_freq'] = df['city'].map(freq_map)

# 비율로 표현
ratio_map = df['city'].value_counts(normalize=True).to_dict()
df['city_ratio'] = df['city'].map(ratio_map)
```

---

## 6. 인코딩 방법 선택 기준

| 상황 | 권장 방법 |
|------|---------|
| 타겟 변수 (분류) | Label Encoding |
| 순서 있는 카테고리 (low/mid/high) | Ordinal Encoding |
| 순서 없는 카테고리, 카디널리티 낮음 | One-Hot Encoding |
| 카디널리티 높음 (수백 개 이상) | Target Encoding, Frequency Encoding |
| 트리 모델 (RF, XGBoost) | Label Encoding 가능 |
| 선형/신경망 모델 | One-Hot 권장 |

---

## 7. 연습문제 (Pytest) 🚀

- **tests_fe/test_fe_1_encoding.py**: `LabelEncoder`, `OrdinalEncoder`, `OneHotEncoder`, 타겟 인코딩 함수를 구현하세요.
