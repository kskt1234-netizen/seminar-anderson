# 피처 엔지니어링 4: 시계열 피처 (Time Series Features)

시계열 데이터에서는 **과거 값의 패턴**을 피처로 만들어야 모델이 미래를 예측할 수 있습니다.

---

## 1. 래그 피처 (Lag Features)

과거 n 타임스텝의 값을 현재 예측에 활용합니다.

```python
import pandas as pd

# shift(n): n 행 아래로 밀기 → n 타임스텝 이전 값
df['lag_1'] = df['value'].shift(1)   # 1 타임스텝 전
df['lag_3'] = df['value'].shift(3)   # 3 타임스텝 전
df['lag_7'] = df['value'].shift(7)   # 7일 전 (주간 패턴 포착)

# ⚠️ 처음 n개 행은 NaN이 됩니다 → dropna() 또는 fillna() 필요
```

---

## 2. 이동 통계 (Rolling Statistics)

최근 N개 구간의 통계를 피처로 사용합니다.

```python
# 이동 평균 (Moving Average)
df['rolling_mean_3'] = df['value'].rolling(window=3).mean()
df['rolling_mean_7'] = df['value'].rolling(window=7).mean()

# 이동 표준편차 (변동성/리스크 지표)
df['rolling_std_7']  = df['value'].rolling(window=7).std()

# 이동 최댓값/최솟값
df['rolling_max_7']  = df['value'].rolling(window=7).max()
df['rolling_min_7']  = df['value'].rolling(window=7).min()

# 이동 범위 (최대 - 최소)
df['rolling_range_7'] = df['rolling_max_7'] - df['rolling_min_7']

# min_periods: 최소 데이터 수 (NaN 줄이기)
df['rolling_mean_7'] = df['value'].rolling(window=7, min_periods=1).mean()
```

---

## 3. 차분 (Differencing)

트렌드를 제거하고 변화량을 피처로 사용합니다.

```python
# 1차 차분: 현재 - 1 타임스텝 전 (=변화량)
df['diff_1'] = df['value'].diff(1)

# 7차 차분: 현재 - 7일 전 (주간 계절성 제거)
df['diff_7'] = df['value'].diff(7)

# 변화율 (%)
df['pct_change_1'] = df['value'].pct_change(1) * 100

# 2차 차분 (변화의 변화, 가속도 개념)
df['diff_2nd'] = df['diff_1'].diff(1)
```

---

## 4. 지수 가중 이동 평균 (EWM — Exponential Weighted Mean)

최근 데이터에 **기하급수적으로 더 높은 가중치**를 줍니다. 이동 평균보다 최신 변화에 민감하게 반응합니다.

```python
# span: 가중치 반감기 (클수록 과거 데이터를 더 오래 반영)
df['ewm_mean'] = df['value'].ewm(span=7, adjust=False).mean()
df['ewm_std']  = df['value'].ewm(span=7, adjust=False).std()

# alpha 직접 지정 (0~1, 클수록 최근 데이터 강조)
df['ewm_alpha'] = df['value'].ewm(alpha=0.3, adjust=False).mean()
```

---

## 5. 확장 통계 (Expanding Statistics)

현재 시점까지의 **누적(전체) 통계**를 피처로 사용합니다.

```python
df['expanding_mean'] = df['value'].expanding(min_periods=1).mean()  # 누적 평균
df['expanding_max']  = df['value'].expanding(min_periods=1).max()   # 누적 최대
df['expanding_std']  = df['value'].expanding(min_periods=1).std()   # 누적 표준편차
```

---

## 6. 그룹별 시계열 피처

실무에서는 상품별, 사용자별로 시계열을 분리해서 처리합니다.

```python
df = df.sort_values(['group_id', 'date'])

# 그룹별 lag
df['lag_1'] = df.groupby('group_id')['value'].shift(1)

# 그룹별 rolling (transform 사용)
df['roll_mean_7'] = (
    df.groupby('group_id')['value']
    .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
)
```

---

## 7. 연습문제 (Pytest) 🚀

- **tests_fe/test_fe_4_timeseries.py**: 래그, 이동 통계, 차분, EWM 피처 생성 함수를 구현하세요.
