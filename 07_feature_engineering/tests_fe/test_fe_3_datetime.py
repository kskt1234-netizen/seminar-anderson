import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def datetime_df():
    """다양한 날짜/시간 데이터를 포함하는 DataFrame을 생성합니다."""
    dates = pd.date_range(start='2024-01-01', periods=24 * 7, freq='h')  # 7일치 시간별 데이터
    df = pd.DataFrame({'datetime': dates})
    df['value'] = np.random.randn(len(df))
    return df


# =============================================================================
# 문제 1: 기본 날짜 분해
# =============================================================================

def extract_date_features(df: pd.DataFrame, col: str = 'datetime') -> pd.DataFrame:
    """
    주어진 datetime 컬럼에서 다음 피처를 추출하여 df에 추가하고 반환하세요.
    - year, month, day, hour, dayofweek (0=월, 6=일), quarter

    힌트: df[col].dt.year, df[col].dt.month, ...
    """
    df = df.copy()
    # --- 코드를 작성하세요 ---

    # -----------------------
    return df


def test_datetime_1_year(datetime_df):
    result = extract_date_features(datetime_df)
    assert result is not None
    assert 'year' in result.columns
    assert all(result['year'] == 2024)


def test_datetime_1_month(datetime_df):
    result = extract_date_features(datetime_df)
    assert 'month' in result.columns
    assert result['month'].between(1, 12).all()


def test_datetime_1_day(datetime_df):
    result = extract_date_features(datetime_df)
    assert 'day' in result.columns
    assert result['day'].between(1, 31).all()


def test_datetime_1_hour(datetime_df):
    result = extract_date_features(datetime_df)
    assert 'hour' in result.columns
    assert result['hour'].between(0, 23).all()
    # 0시부터 시작하고 24시간 주기로 반복
    assert result['hour'].iloc[0] == 0
    assert result['hour'].iloc[23] == 23
    assert result['hour'].iloc[24] == 0


def test_datetime_1_dayofweek(datetime_df):
    result = extract_date_features(datetime_df)
    assert 'dayofweek' in result.columns
    assert result['dayofweek'].between(0, 6).all()


def test_datetime_1_quarter(datetime_df):
    result = extract_date_features(datetime_df)
    assert 'quarter' in result.columns
    # 1월은 1분기
    assert all(result['quarter'] == 1)


# =============================================================================
# 문제 2: 이진 날짜 피처
# =============================================================================

def extract_binary_features(df: pd.DataFrame, col: str = 'datetime') -> pd.DataFrame:
    """
    다음 이진 피처를 df에 추가하고 반환하세요.
    - is_weekend: 토요일(5) 또는 일요일(6)이면 1, 아니면 0 (int 타입)
    - is_month_start: 월의 첫째 날이면 True
    - is_month_end: 월의 마지막 날이면 True
    """
    df = df.copy()
    # --- 코드를 작성하세요 ---

    # -----------------------
    return df


def test_datetime_2_is_weekend(datetime_df):
    result = extract_binary_features(datetime_df)
    assert 'is_weekend' in result.columns
    # 2024-01-01은 월요일이므로 주말이 아닙니다
    assert result['is_weekend'].iloc[0] == 0
    # 2024-01-06(토)부터 주말
    saturday_rows = result[result['datetime'].dt.dayofweek == 5]
    assert all(saturday_rows['is_weekend'] == 1)


def test_datetime_2_is_month_start(datetime_df):
    result = extract_binary_features(datetime_df)
    assert 'is_month_start' in result.columns
    # 2024-01-01이 포함되어 있으므로 월 시작이 있어야 합니다
    assert result['is_month_start'].any()


# =============================================================================
# 문제 3: 순환 인코딩 (Cyclical Encoding)
# =============================================================================

def cyclical_encode(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    """
    df[col]을 순환 인코딩(사인/코사인)하여 두 개의 새 컬럼을 추가하고 반환하세요.
    새 컬럼 이름: f'{col}_sin', f'{col}_cos'
    공식:
        sin = sin(2π × value / period)
        cos = cos(2π × value / period)
    """
    df = df.copy()
    # --- 코드를 작성하세요 ---

    # -----------------------
    return df


def test_datetime_3_cyclical_columns(datetime_df):
    result = extract_date_features(datetime_df)
    result = cyclical_encode(result, 'hour', 24)
    assert 'hour_sin' in result.columns
    assert 'hour_cos' in result.columns


def test_datetime_3_cyclical_range(datetime_df):
    result = extract_date_features(datetime_df)
    result = cyclical_encode(result, 'hour', 24)
    # 사인/코사인 값은 [-1, 1] 범위 내
    assert result['hour_sin'].between(-1, 1).all()
    assert result['hour_cos'].between(-1, 1).all()


def test_datetime_3_cyclical_periodicity(datetime_df):
    result = extract_date_features(datetime_df)
    result = cyclical_encode(result, 'hour', 24)
    # 0시와 24시(=0시)의 값이 동일해야 합니다 (주기성)
    hour_0_sin = result[result['hour'] == 0]['hour_sin'].iloc[0]
    # 0시의 sin(2π × 0 / 24) = sin(0) = 0
    assert abs(hour_0_sin - 0.0) < 1e-6


def test_datetime_3_cyclical_12hour(datetime_df):
    result = extract_date_features(datetime_df)
    result = cyclical_encode(result, 'hour', 24)
    # 12시의 sin(2π × 12 / 24) = sin(π) ≈ 0
    hour_12 = result[result['hour'] == 12].iloc[0]
    assert abs(hour_12['hour_sin']) < 1e-6
    # 12시의 cos(2π × 12 / 24) = cos(π) = -1
    assert abs(hour_12['hour_cos'] - (-1.0)) < 1e-6


# =============================================================================
# 문제 4: 시간 경과 피처 (Time Since)
# =============================================================================

def add_time_since(df: pd.DataFrame, col: str, reference: pd.Timestamp) -> pd.DataFrame:
    """
    reference 날짜로부터 col까지 경과된 일수(days_since)를 정수로 추가하고 반환하세요.
    컬럼 이름: 'days_since'
    """
    df = df.copy()
    # --- 코드를 작성하세요 ---

    # -----------------------
    return df


def test_datetime_4_days_since(datetime_df):
    ref = pd.Timestamp('2024-01-01')
    result = add_time_since(datetime_df, 'datetime', ref)
    assert 'days_since' in result.columns
    # 2024-01-01의 경과 일수는 0
    assert result['days_since'].iloc[0] == 0
    # 2024-01-02의 경과 일수는 1
    assert result['days_since'].iloc[24] == 1


def test_datetime_4_days_since_non_negative(datetime_df):
    ref = pd.Timestamp('2023-01-01')
    result = add_time_since(datetime_df, 'datetime', ref)
    assert all(result['days_since'] >= 0)
