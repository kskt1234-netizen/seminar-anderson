# Pandas (판다스) 실전 가이드

안녕하세요! 이 가이드는 파이썬의 핵심 데이터 분석 라이브러리인 **Pandas(판다스)**를 실전(Pragmatic) 위주로 다룹니다.
데이터를 불러오고, 탐색하고, 정제(Cleaning)하고, 변형하는 모든 과정에서 Pandas는 엑셀(Excel)이나 SQL보다 훨씬 빠르고 유연한 도구가 되어 줍니다.

현업에서 마주치는 실제 데이터 다루기를 위해 구성되었으며, 본 문서를 익힌 후 준비된 `pytest` 문제들을 해결해 보시기 바랍니다.

---

## 1. Pandas의 핵심 자료구조

Pandas는 데이터를 다루기 위해 크게 두 가지 객체를 사용합니다.

1. **Series (시리즈)**: 1차원 배열 형태입니다. (하나의 열 데이터)
2. **DataFrame (데이터프레임)**: 2차원 표(테이블) 형태입니다. (여러 개의 Series가 모인 것)

```python
import pandas as pd

# Series 생성
s = pd.Series([10, 20, 30], index=['A', 'B', 'C'])

# DataFrame 생성
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['Seoul', 'Busan', 'New York']
}
df = pd.DataFrame(data)
```

---

## 2. 실전 데이터 불러오기 (Hugging Face `datasets` 활용)

이번 실습에서는 로컬의 `.csv` 파일뿐만 아니라, 요즘 AI/ML 실무에서 많이 쓰는 **Hugging Face `datasets`** 라이브러리를 통해 공공 데이터셋을 내려받고 Pandas 데이터프레임으로 변환하여 사용해 봅니다.

```python
# pip install datasets pandas
from datasets import load_dataset
import pandas as pd

# 허깅페이스에서 타이타닉 데이터셋 불러오기
dataset = load_dataset("titanic")

# 훈련(train) 셋을 가져와서 Pandas DataFrame으로 변환하기!
df = dataset['train'].to_pandas()
```

### 2-1. 데이터 살펴보기 (Inspection)
데이터가 로드되었으면 어떤 형태로 생겼는지 먼저 파악하는 것이 분석의 첫걸음입니다.

```python
df.head()       # 상위 5개 행 미리보기
df.tail(3)      # 하위 3개 행 미리보기
df.info()       # 열(Column) 구성, 결측치 양, 데이터 타입(dtype) 확인
df.describe()   # 숫자형 열들의 요약 통계(평균, 최소/최대, 4분위수) 제공
```

---

## 3. 데이터 선택 및 필터링 (Selection)

원하는 행(Row)과 열(Column)만 추출해내는 기술입니다.

### 3-1. 열 (Column) 추출
```python
names = df['Name']             # 1개의 열 추출 (Series 반환)
subset = df[['Name', 'Age']]   # 여러 개의 열 추출 (DataFrame 반환, 리스트 형태 주의 `[[]]`)
```

### 3-2. 행 (Row) 추출: `loc` 와 `iloc`
- **`loc` (Location)**: "이름(Label)" 기반 특정 속성/행을 추출합니다. (또는 조건문)
- **`iloc` (Integer Location)**: 컴퓨터의 "순서(Index 번호)" 기반으로 추출합니다.

```python
# 인덱스(순서)가 0부터 4까지인 행 다루기
print(df.iloc[0:5]) 

# 인덱스 "이름"이 10인 행의 모든 데이터 가져오기 (이 경우에는 숫자가 이름 역할을 함)
print(df.loc[10])

# loc을 이용한 조건문(Boolean Indexing) - 실무에서 가장 많이 씁니다!
# 예: 나이가 30 이상인 사람만 필터링
adults = df.loc[df['Age'] >= 30]

# 다중 조건은 '&' (AND) 와 '|' (OR) 사용 (괄호 필수!)
target = df.loc[(df['Age'] >= 30) & (df['City'] == 'Seoul')]
```

---

## 4. 데이터 정제 (Data Cleaning)

현업의 데이터는 깨끗하지 않습니다. 빈 값(NaN)이 있거나 중복이 많습니다.

### 4-1. 결측치 (Missing Values / NaN) 다루기
```python
df.isna().sum()         # 각 열별로 누락된 데이터가 몇 개인지 개수 세기

# 누락된 행(가로줄) 아예 삭제하기 (실무에서는 데이터 유실 때문에 주의!)
df_cleaned = df.dropna()

# 누락된 값을 평균이나 특정 문자로 채우기
df['Age'] = df['Age'].fillna(df['Age'].mean())  # 나이 결측치를 평균으로 채우기
df['City'] = df['City'].fillna('Unknown')       # 도시 결측치를 'Unknown' 텍스트로 채우기
```

### 4-2. 타입 변환과 문자열 처리
숫자가 문자로 되어있거나 날짜 형식을 맞춰야 할 때 씁니다.

```python
# 타입 변환 (예: 문자열 "100" -> 정수 100)
df['Price'] = df['Price'].astype(int)

# 문자열(String) 전처리: .str 메서드 활용
df['Name'] = df['Name'].str.lower()       # 전부 소문자로 변경
df['City'] = df['City'].str.replace(' ', '') # 공백 제거
```

---

## 5. 진정한 파워: 그룹화와 집계 (Groupby & Aggregation)

"성별에 따른 평균 나이"를 구하거나 "지역별 총매출액"을 구하는 가장 강력한 무기입니다. (SQL의 `GROUP BY`와 동일)

```python
# 성별(Sex)로 그룹을 나눈 뒤, 생존율(Survived)의 '평균'을 구하기
survived_by_sex = df.groupby('Sex')['Survived'].mean()
print(survived_by_sex)

# 여러 기준으로 동시에 묶고(예: 성별 및 선실등급별), 여러 통계값을 동시에 구하기 (agg 사용)
summary = df.groupby(['Sex', 'Pclass']).agg({
    'Fare': ['mean', 'max'], # 요금의 평균과 최댓값
    'Age': 'mean'            # 나이의 평균
})
```

---

## 6. 고급/실무 테크닉 (실질적인 회사 문제용)

실제 회사 데이터는 위 내용들만으로 끝나지 않는 다양한 패턴을 가집니다.

### 6-1. 데이터 병합 (Merge)
두 개의 엑셀/테이블을 **특정 키(Key)** 기준으로 이어붙일 때 (SQL `JOIN` 개념).

```python
df_merged = pd.merge(df_users, df_payments, left_on='user_id', right_on='customer_id', how='left')
```

### 6-2. 시간차 분석 및 행 이동 (`shift`)
"이전 달 대비 매출 증감률", "다음 페이지 클릭까지 걸린 시간" 같은 문제는 현재 행과 과거나 미래 행을 비교해야 합니다.
`shift(1)` 은 모든 데이터를 아래로(미래로) 한 칸씩 밉니다.

```python
# 유저별 결제일시 오름차순 정렬 후
df = df.sort_values(by=['user_id', 'payment_date'])

# '이전 결제일' 이라는 파생 열 생성
df['prev_payment_date'] = df.groupby('user_id')['payment_date'].shift(1)

# 재구매 기간 (현재 결제일 - 이전 결제일)
df['time_diff'] = df['payment_date'] - df['prev_payment_date']
```

### 6-3. 이동 평균 / 윈도우 함수 (`rolling`)
주식 차트나 로그 데이터에서 흔히 쓰이는 "최근 N주차 이동 평균선" 같은 개념입니다.

```python
# 날짜순 정렬 후, 최근 3일 치의 매출 '평균'을 이동하며 구하기
df['3_day_moving_avg'] = df['sales'].rolling(window=3).mean()
```

---

## 7. 연습문제 (Pytest) 안내 🚀

실제 손으로 쳐봐야 본인의 기술이 됩니다. 제공된 `pytest` 파일들을 열고, 빈 칸을 채워 모든 테스트를 `PASSED`로 만드세요!

- **test_pandas_1_basics.py**: `datasets` 로드 및 기초 추출
- **test_pandas_2_cleaning.py**: 결측치 처리 및 데이터 값 수정
- **test_pandas_3_aggregation.py**: `groupby` 와 다중 통계 분석, 병합
- **test_pandas_4_realworld.py**: ⭐️ 재구매 코호트 기간 분석, 이상치 탐지 등 **스타트업/테크 기업 실무형 트러블슈팅**

`pytest tests_pandas/` 를 실행하여 본인의 코드를 점검하세요!
