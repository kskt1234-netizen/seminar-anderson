# 데이터 이상치(Outlier) 처리 가이드

데이터 분석과 머신러닝 최적화 과정에서 **이상치(Outlier)**는 통계적 모델의 성능을 크게 저하시킬 수 있는 주요 원인 중 하나입니다. 이 문서에서는 실무적으로 가장 자주 쓰이는 이상치 탐지 방법론과 그 처리 방법을 단계별로 상세히 살펴봅니다.

## 1. 이상치(Outlier)란?
전체 데이터의 보편적인 패턴에서 심하게 벗어난 극단적인 측정값을 의미합니다. 이는 단순 측정 오류일 수도 있고, 실제로 드물게 발생하는 유의미한 현상일 수도 있습니다.
- **예시**: 사람들의 몸무게 데이터를 조사하는데 500kg인 사람이 기록되어 있는 경우. (측정 오류 혹은 드문 케이스)

## 2. 이상치 탐지 방법론

### 2.1. Z-Score (표준점수) 기법
데이터가 **정규 분포**를 따른다는 가정 하에 사용할 수 있는 방법론입니다. 각각의 데이터 포인트가 평균(Mean)으로부터 표준편차(Standard Deviation)의 몇 배만큼 떨어져 있는지를 계산합니다.
- 보통 $|Z| > 3$ (표준편차 3배 이상) 인 범위를 통상적인 이상치로 규정합니다.

### 2.2. IQR (Interquartile Range) 사분위수 범위 기법
정규 분포를 띄지 않더라도 강력하게(Robust) 적용할 수 있는 가장 대표적인 통계적 탐지 방법론입니다.
- **Q1 (1사분위수)**: 데이터 하위 25% 지점의 값
- **Q3 (3사분위수)**: 데이터 하위 75% 지점의 값
- **IQR** = Q3 - Q1
- 정상 범위: `Q1 - 1.5 * IQR` 이상, `Q3 + 1.5 * IQR` 이하. 이 범위를 벗어나면 이상치로 취급합니다.

### 2.3. 시각화 기반 방법론 (Boxplot)
Seaborn 등의 라이브러리를 이용하여 Boxplot 상자 수염 그림을 그리면, IQR 계산 로직에 의해 튀어나온 점들이 선명하게 보입니다.

## 3. 실습을 통한 이상치 처리 방법

이제 파이썬(Python)과 판다스(Pandas)를 이용하여 어떻게 실무적으로 이상치를 쳐낼 수 있는지 실습을 진행해 보겠습니다. Scikit-Learn 데이터셋 보스턴 혹은 당뇨병 관련 데이터에서 자주 쓰입니다.

### Step 1: IQR 함수 생성하기
실무에서는 매번 코드를 치기보다 재사용 가능한 함수를 하나 만들어 파이프라인에 구축해둡니다.
```python
import pandas as pd
import numpy as np

def detect_outliers_iqr(df: pd.DataFrame, column: str):
    """지정된 컬럼의 IQR 범위, 상하한선, 그리고 이상치만 필터링한 데이터프레임을 반환합니다."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # 조건을 벗어나는 행들이 바로 이상치입니다.
    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outliers = df[outlier_mask]
    
    return outliers, lower_bound, upper_bound
```

### Step 2: 이상치 대응법 3가지

#### 1) 제거 (Drop/Remove)
- 데이터 건수가 충분하거나, 측정 에러가 명백할 때 아예 행 자체를 날려버립니다.
```python
# 정상 데이터만 취하는 조건
clean_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
```

#### 2) 대체 (Capping/Winsorizing) 
- 하한선보다 작은 값은 전부 하한선으로, 상한선보다 큰 값은 상한선 값으로 '자르는' 기법입니다. 데이터 손실을 막기 위해 널리 씁니다.
```python
# numpy 의 clip 함수를 활용하면 상/하한선 대체가 매우 쉽습니다.
df[column] = np.clip(df[column], lower_bound, upper_bound)
```

#### 3) 변수 변환 (Log Transformation)
- 값 자체가 너무 비대칭일 때, 로그를 씌워 정규분포에 가깝게 당겨오는 방법입니다.
```python
df[column] = np.log1p(df[column])
```

## 4. 요약 정리
- 머신러닝(Linear Regression 등)은 이상치에 매우 취약합니다.
- 탐지는 Boxplot 시각화 및 IQR 공식이 90% 이상 활용됩니다.
- 삭제, Clipping, 변환 중에서 데이터의 도메인 특성을 파악하여 신중히 결정해야 합니다.
