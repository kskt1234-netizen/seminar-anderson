# 피처 엔지니어링 3: 날짜/시간 피처 (Datetime Features)

시간 데이터에는 계절성, 주기성, 트렌드 등 다양한 패턴이 숨어 있습니다. 이를 ML 모델이 이해할 수 있는 수치형 피처로 변환합니다.

---

## 1. 기본 날짜 분해 (Basic Decomposition)

```python
import pandas as pd

df['datetime'] = pd.to_datetime(df['datetime'])

# 날짜 관련
df['year']        = df['datetime'].dt.year
df['month']       = df['datetime'].dt.month        # 1~12
df['day']         = df['datetime'].dt.day          # 1~31
df['quarter']     = df['datetime'].dt.quarter      # 1~4

# 시간 관련
df['hour']        = df['datetime'].dt.hour         # 0~23
df['minute']      = df['datetime'].dt.minute       # 0~59

# 요일 관련
df['dayofweek']   = df['datetime'].dt.dayofweek    # 0=월, 6=일
df['dayofyear']   = df['datetime'].dt.dayofyear    # 1~366
df['weekofyear']  = df['datetime'].dt.isocalendar().week

# 이진 피처
df['is_weekend']    = (df['datetime'].dt.dayofweek >= 5).astype(int)
df['is_month_start'] = df['datetime'].dt.is_month_start.astype(int)
df['is_month_end']   = df['datetime'].dt.is_month_end.astype(int)
```

---

## 2. 순환 인코딩 (Cyclical Encoding)

시간(0~23)을 그냥 숫자로 쓰면 23시와 0시가 완전히 다른 값이 됩니다.
사인/코사인 변환으로 **원형 연속성**을 보존합니다.

```python
import numpy as np

# 시간 (0~23) → 하루가 원형으로 연결됨
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# 월 (1~12) → 12월과 1월이 인접
df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

# 요일 (0~6) → 일요일과 월요일이 인접
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
```

**왜 sin/cos 두 개가 필요한가?**
sin만 쓰면 0시와 12시가 같은 값(0)이 됩니다. sin과 cos를 함께 쓰면 원 위의 모든 점이 고유한 (sin, cos) 쌍을 가집니다.

---

## 3. 시간 경과 피처 (Time Since Features)

```python
# 기준 날짜로부터 경과된 일수
reference_date = pd.Timestamp('2020-01-01')
df['days_since_ref'] = (df['datetime'] - reference_date).dt.days

# 이전 이벤트로부터 경과된 시간 (초 단위)
df_sorted = df.sort_values('datetime')
df_sorted['seconds_since_last'] = df_sorted['datetime'].diff().dt.total_seconds()
```

---

## 4. 시간대 피처 (Time Zone / Period)

```python
# 하루 중 시간대 분류
def get_time_period(hour: int) -> str:
    if 6 <= hour < 12:   return 'morning'
    if 12 <= hour < 18:  return 'afternoon'
    if 18 <= hour < 22:  return 'evening'
    return 'night'

df['time_period'] = df['hour'].apply(get_time_period)
# 이후 One-Hot Encoding 적용
```

---

## 5. 연습문제 (Pytest) 🚀

- **tests_fe/test_fe_3_datetime.py**: 날짜 분해, 순환 인코딩, 시간 경과 피처 함수를 구현하세요.
