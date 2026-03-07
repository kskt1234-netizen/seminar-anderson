import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def ts_df():
    """단순 시계열 데이터를 생성합니다."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    values = np.cumsum(np.random.randn(30)) + 50  # random walk
    return pd.DataFrame({'date': dates, 'value': values})


@pytest.fixture
def group_ts_df():
    """그룹별 시계열 데이터를 생성합니다."""
    np.random.seed(0)
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    records = []
    for group in ['A', 'B']:
        for d in dates:
            records.append({'date': d, 'group': group, 'value': np.random.randint(10, 100)})
    df = pd.DataFrame(records).sort_values(['group', 'date']).reset_index(drop=True)
    return df


# =============================================================================
# 문제 1: 래그 피처 (Lag Features)
# =============================================================================

def add_lag_features(df: pd.DataFrame, col: str, lags: list) -> pd.DataFrame:
    """
    df[col]에 대해 lags 리스트에 명시된 래그 피처를 추가하고 반환하세요.
    컬럼 이름: f'lag_{n}' (예: lag_1, lag_3, lag_7)
    힌트: df[col].shift(n)
    """
    df = df.copy()
    # --- 코드를 작성하세요 ---

    # -----------------------
    return df


def test_ts_1_lag_columns(ts_df):
    result = add_lag_features(ts_df, 'value', [1, 3, 7])
    assert 'lag_1' in result.columns
    assert 'lag_3' in result.columns
    assert 'lag_7' in result.columns


def test_ts_1_lag_1_values(ts_df):
    result = add_lag_features(ts_df, 'value', [1])
    # lag_1의 1번째 행은 NaN
    assert pd.isna(result['lag_1'].iloc[0])
    # lag_1의 2번째 행은 원본의 1번째 행 값
    assert abs(result['lag_1'].iloc[1] - ts_df['value'].iloc[0]) < 1e-10


def test_ts_1_lag_values_match(ts_df):
    result = add_lag_features(ts_df, 'value', [3])
    # 인덱스 3부터는 lag_3 값이 원본의 0번째 값과 일치해야 합니다
    assert abs(result['lag_3'].iloc[3] - ts_df['value'].iloc[0]) < 1e-10


# =============================================================================
# 문제 2: 이동 통계 (Rolling Statistics)
# =============================================================================

def add_rolling_features(df: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    """
    df[col]에 대해 window 크기의 이동 통계를 추가하고 반환하세요.
    추가할 컬럼:
    - f'rolling_mean_{window}': 이동 평균
    - f'rolling_std_{window}':  이동 표준편차
    - f'rolling_max_{window}':  이동 최댓값
    - f'rolling_min_{window}':  이동 최솟값
    min_periods=1 을 사용하여 초반 NaN을 최소화하세요.
    """
    df = df.copy()
    # --- 코드를 작성하세요 ---

    # -----------------------
    return df


def test_ts_2_rolling_columns(ts_df):
    result = add_rolling_features(ts_df, 'value', 7)
    assert 'rolling_mean_7' in result.columns
    assert 'rolling_std_7'  in result.columns
    assert 'rolling_max_7'  in result.columns
    assert 'rolling_min_7'  in result.columns


def test_ts_2_rolling_mean_first_row(ts_df):
    result = add_rolling_features(ts_df, 'value', 7)
    # min_periods=1이면 첫 행의 rolling_mean은 첫 값과 같아야 합니다
    assert abs(result['rolling_mean_7'].iloc[0] - ts_df['value'].iloc[0]) < 1e-10


def test_ts_2_rolling_max_gte_min(ts_df):
    result = add_rolling_features(ts_df, 'value', 7)
    # rolling_max는 항상 rolling_min 이상이어야 합니다
    assert all(result['rolling_max_7'] >= result['rolling_min_7'])


def test_ts_2_rolling_mean_correct(ts_df):
    result = add_rolling_features(ts_df, 'value', 3)
    # 인덱스 2 (3번째 행)의 rolling_mean_3은 처음 3개 값의 평균
    expected = ts_df['value'].iloc[:3].mean()
    assert abs(result['rolling_mean_3'].iloc[2] - expected) < 1e-10


# =============================================================================
# 문제 3: 차분 (Differencing)
# =============================================================================

def add_diff_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    df[col]에 대해 다음 차분 피처를 추가하고 반환하세요.
    - 'diff_1': 1차 차분 (현재 - 1 타임스텝 전)
    - 'diff_7': 7차 차분 (현재 - 7 타임스텝 전)
    - 'pct_change_1': 1차 변화율 (%) → pct_change(1) * 100
    힌트: df[col].diff(n), df[col].pct_change(n)
    """
    df = df.copy()
    # --- 코드를 작성하세요 ---

    # -----------------------
    return df


def test_ts_3_diff_columns(ts_df):
    result = add_diff_features(ts_df, 'value')
    assert 'diff_1' in result.columns
    assert 'diff_7' in result.columns
    assert 'pct_change_1' in result.columns


def test_ts_3_diff_1_nan(ts_df):
    result = add_diff_features(ts_df, 'value')
    # 첫 행은 diff_1이 NaN
    assert pd.isna(result['diff_1'].iloc[0])


def test_ts_3_diff_1_correct(ts_df):
    result = add_diff_features(ts_df, 'value')
    # 두 번째 행의 diff_1 = 두 번째 값 - 첫 번째 값
    expected = ts_df['value'].iloc[1] - ts_df['value'].iloc[0]
    assert abs(result['diff_1'].iloc[1] - expected) < 1e-10


def test_ts_3_diff_7_nan_count(ts_df):
    result = add_diff_features(ts_df, 'value')
    # 처음 7개 행은 diff_7이 NaN
    assert result['diff_7'].iloc[:7].isna().all()
    # 8번째 행부터는 유효
    assert not pd.isna(result['diff_7'].iloc[7])


# =============================================================================
# 문제 4: 지수 가중 이동 평균 (EWM)
# =============================================================================

def add_ewm_features(df: pd.DataFrame, col: str, span: int) -> pd.DataFrame:
    """
    df[col]에 대해 EWM 피처를 추가하고 반환하세요.
    - f'ewm_mean_{span}': EWM 평균 (span=span, adjust=False)
    힌트: df[col].ewm(span=span, adjust=False).mean()
    """
    df = df.copy()
    # --- 코드를 작성하세요 ---

    # -----------------------
    return df


def test_ts_4_ewm_columns(ts_df):
    result = add_ewm_features(ts_df, 'value', 7)
    assert 'ewm_mean_7' in result.columns


def test_ts_4_ewm_no_nan(ts_df):
    result = add_ewm_features(ts_df, 'value', 7)
    # EWM은 adjust=False이면 NaN이 없어야 합니다
    assert not result['ewm_mean_7'].isna().any()


def test_ts_4_ewm_first_equals_first(ts_df):
    result = add_ewm_features(ts_df, 'value', 7)
    # 첫 번째 EWM 값은 첫 번째 원본 값과 동일
    assert abs(result['ewm_mean_7'].iloc[0] - ts_df['value'].iloc[0]) < 1e-10


# =============================================================================
# 문제 5: 그룹별 래그 피처
# =============================================================================

def add_group_lag(df: pd.DataFrame, group_col: str, value_col: str, lag: int) -> pd.DataFrame:
    """
    그룹별로 value_col의 lag 피처를 추가하고 반환하세요.
    컬럼 이름: f'lag_{lag}'
    각 그룹 내에서 독립적으로 shift를 적용해야 합니다.
    힌트: df.groupby(group_col)[value_col].shift(lag)
    """
    df = df.copy()
    # --- 코드를 작성하세요 ---

    # -----------------------
    return df


def test_ts_5_group_lag_columns(group_ts_df):
    result = add_group_lag(group_ts_df, 'group', 'value', 1)
    assert 'lag_1' in result.columns


def test_ts_5_group_lag_first_per_group(group_ts_df):
    result = add_group_lag(group_ts_df, 'group', 'value', 1)
    # 각 그룹의 첫 번째 행은 lag_1이 NaN이어야 합니다
    for g in ['A', 'B']:
        first_row = result[result['group'] == g].iloc[0]
        assert pd.isna(first_row['lag_1']), f"그룹 {g}의 첫 행 lag_1이 NaN이 아닙니다."


def test_ts_5_group_lag_no_cross_contamination(group_ts_df):
    result = add_group_lag(group_ts_df, 'group', 'value', 1)
    # 그룹 A의 마지막 행의 lag_1이 그룹 B의 첫 행에 영향을 주어서는 안 됩니다
    group_b_first = result[result['group'] == 'B'].iloc[0]
    assert pd.isna(group_b_first['lag_1'])
