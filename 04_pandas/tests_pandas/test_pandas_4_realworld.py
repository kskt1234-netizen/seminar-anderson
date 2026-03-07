import pandas as pd
import numpy as np

def test_pandas_1_time_delta_churn():
    """
    회사 문제 1: 유저 서비스 재방문(재결제) 기간(Time Delta) 구하기
    
    데이터 팀에서 "유저들이 결제한 후, 다음 결제까지 평균 며칠이 걸리는지" 분석해달라고 합니다.
    각 'user_id' 별로, 결제(payment_date)가 여러 번 있을 때,
    '현재 결제일과 직전(이전) 결제일의 시간 차이(Days)'를 구하여 'days_since_last' 컬럼에 정수로 저장하세요.
    (첫 결제라 '이전 결제일'이 없는 경우는 NaN 또는 NaT 가 됩니다.)
    
    힌트: 데이터프레임을 먼저 정렬(sort_values)하고, groupby 한 뒤 .shift(1)를 활용하세요.
    결과값인 일(Days) 수는 .dt.days 를 활용해 정수화 할 수 있습니다.
    """
    df = pd.DataFrame({
        'user_id': [1, 2, 1, 2, 1],
        'payment_date': pd.to_datetime([
            '2024-01-01', 
            '2024-01-05', 
            '2024-01-15', # User 1의 두번째 결제 (+14일)
            '2024-01-10', # User 2의 두번째 결제 (+5일)
            '2024-02-04'  # User 1의 세번째 결제 (+20일)
        ])
    })
    
    # --- 코드를 작성하세요 ---
    
    # 임시 변수명을 사용하셔도 되고 바로 변형해도 됩니다.
    
    # -----------------------

    # 검증을 위해 user_id=1 의 데이터만 시간순으로 가져옵니다.
    user1 = df[df['user_id'] == 1].sort_values('payment_date')
    assert user1['days_since_last'].isna().iloc[0] # 첫 결제는 결제 차이가 없음 (NaN/NaT)
    assert user1['days_since_last'].iloc[1] == 14
    assert user1['days_since_last'].iloc[2] == 20
    
    user2 = df[df['user_id'] == 2].sort_values('payment_date')
    assert user2['days_since_last'].iloc[1] == 5


def test_pandas_2_anomaly_detection_rolling():
    """
    회사 문제 2: 이동 평균(Rolling Average)을 활용한 이상치(Anomaly) 탐지
    
    인프라 팀에서 서버의 분당 트래픽(Traffic) 로그를 전달했습니다.
    "최근 3분 간의 이동 평균(자신 포함) 대비, 자신의 트래픽이 2배를 초과하는 경우"를
    위험(Warning) 상태로 간주하려 합니다.
    
    1) 'traffic_3m_avg' 컬럼을 생성해 최근 3행의 평균 트래픽을 계산하세요. (rolling(window=3).mean() 활용)
    2) 'is_warning' 이라는 Boolean(True/False) 컬럼을 생성하여, 트래픽이 3분 평균보다 *엄격히 2배 초과*인지 표시하세요.
    (1, 2번째 데이터처럼 3분이 채워지지 않아 이전 값이 없는 경우 NaN이 나올 수 있는데, 이때 NaN 처리 혹은 무시하셔도 테스트는 통과합니다.)
    """
    df = pd.DataFrame({
        'time': pd.date_range('2024-01-01 00:00', periods=6, freq='min'),
        'traffic': [100, 110, 105, 500, 120, 130] 
        # 4번째 데이터(500)는 앞선 3개(110, 105, 500)의 평균인 약 238의 2배를 넘지 않지만 
        # 이상치로 판별해봅시다.
    })
    
    # --- 코드를 작성하세요 ---

    # -----------------------

    # 4번째 데이터 (인덱스 3) 검증 
    # rolling window 3 (indices 1,2,3 -> traffic: 110, 105, 500, mean = 715/3 = 238.33)
    # 500 > 238.33 * 2 (476.66) -> True!
    assert df.loc[3, 'is_warning'] == True
    
    # 나머지 정상 데이터는 False 또는 NaN 이어야 함
    assert df.loc[2, 'is_warning'] in (False, np.nan, pd.NA)
    assert df.loc[4, 'is_warning'] == False
    assert df.loc[5, 'is_warning'] == False


def test_pandas_3_complex_cohort_prep():
    """
    회사 문제 3: 코호트(Cohort) 분석 기틀 만들기
    
    유저가 처음 가입/구매한 '최초 결제월(Cohort Month)' 별로 유저들을 묶어서 볼 일이 많습니다.
    
    1) 유저별 최초 결제일을 구하여 'first_payment_date' 컬럼을 원본에 추가하세요. 
       (transform('min') 활용 추천)
    2) 'payment_date'와 'first_payment_date'에서 '년-월(YYYY-MM)' 형식의 문자열 혹은 Period 객체만을 추출하여
       각각 'order_month', 'cohort_month' 컬럼으로 저장하세요. (.dt.to_period('M') 추천)
    """
    df = pd.DataFrame({
        'user_id': ['A', 'A', 'B', 'B', 'C'],
        'payment_date': pd.to_datetime([
            '2024-01-15', '2024-03-10', # A는 1월 코호트
            '2024-02-05', '2024-02-18', # B는 2월 코호트
            '2024-03-01'                # C는 3월 코호트
        ])
    })
    
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    # 코호트 기준 월 정상 할당 검증
    # User A의 cohort_month는 항상 '2024-01' (또는 Period객체) 여야 합니다.
    a_cohorts = df[df['user_id'] == 'A']['cohort_month'].astype(str).tolist()
    assert all('2024-01' in c for c in a_cohorts)
    
    # User B의 cohort_month
    b_cohorts = df[df['user_id'] == 'B']['cohort_month'].astype(str).tolist()
    assert all('2024-02' in c for c in b_cohorts)
    
    # 주문 월 정상 추출 검증
    assert df.loc[1, 'order_month'].astype(str) == '2024-03' 
