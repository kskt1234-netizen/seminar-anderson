import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes


@pytest.fixture
def age_series():
    return pd.Series([5, 12, 18, 22, 30, 35, 42, 55, 60, 73, 81, 90])


@pytest.fixture
def diabetes_df():
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame.copy()
    return df


# =============================================================================
# 문제 1: pd.cut — 동일 너비 구간화
# =============================================================================

def cut_into_bins(s: pd.Series, n_bins: int, labels: list) -> pd.Series:
    """
    pd.cut()을 사용하여 Series를 n_bins개의 동일 너비 구간으로 나누세요.
    labels 리스트로 각 구간에 이름을 붙이세요.
    결측치가 생길 수 있으므로 include_lowest=True를 사용하세요.
    반환값: 카테고리형 Series
    """
    result = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return result


def test_binning_1_cut_length(age_series):
    labels = ['child', 'youth', 'adult', 'senior']
    result = cut_into_bins(age_series, n_bins=4, labels=labels)
    assert result is not None
    assert len(result) == len(age_series)


def test_binning_1_cut_labels(age_series):
    labels = ['child', 'youth', 'adult', 'senior']
    result = cut_into_bins(age_series, n_bins=4, labels=labels)
    # 사용된 카테고리가 labels와 일치해야 합니다
    used_cats = set(result.dropna().unique())
    assert used_cats.issubset(set(labels))


def test_binning_1_cut_no_nan(age_series):
    labels = ['child', 'youth', 'adult', 'senior']
    result = cut_into_bins(age_series, n_bins=4, labels=labels)
    assert not result.isna().any(), "결측치가 있습니다. include_lowest=True를 사용하세요."


# =============================================================================
# 문제 2: pd.cut — 경계 직접 지정
# =============================================================================

def cut_with_custom_bins(s: pd.Series) -> pd.Series:
    """
    pd.cut()으로 나이 데이터를 다음 경계로 구간화하세요.
    경계: [0, 18, 35, 60, 100]
    레이블: ['teen', 'young_adult', 'adult', 'senior']
    반환값: 카테고리형 Series
    """
    result = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return result


def test_binning_2_custom_boundaries(age_series):
    result = cut_with_custom_bins(age_series)
    assert result is not None
    # 5세 → teen, 22세 → young_adult, 42세 → adult, 73세 → senior
    assert result.iloc[0] == 'teen'       # 5
    assert result.iloc[3] == 'young_adult' # 22
    assert result.iloc[6] == 'adult'       # 42
    assert result.iloc[9] == 'senior'      # 73


# =============================================================================
# 문제 3: pd.qcut — 동일 빈도 구간화
# =============================================================================

def qcut_into_quantiles(s: pd.Series, q: int) -> pd.Series:
    """
    pd.qcut()을 사용하여 Series를 q분위수로 구간화하세요.
    레이블은 사용하지 않고 기본 구간 표시를 사용하세요. (labels=False 로 정수 반환)
    duplicates='drop'을 사용하세요.
    반환값: 정수 레이블 Series (0, 1, ..., q-1)
    """
    result = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return result


def test_binning_3_qcut_length(age_series):
    result = qcut_into_quantiles(age_series, q=4)
    assert result is not None
    assert len(result) == len(age_series)


def test_binning_3_qcut_n_unique(age_series):
    result = qcut_into_quantiles(age_series, q=4)
    # 4분위수이므로 최대 4개의 고유 값
    assert result.nunique() <= 4


def test_binning_3_qcut_equal_frequency(diabetes_df):
    """당뇨병 데이터의 bmi 컬럼을 4분위로 나누면 각 구간이 비슷한 빈도를 가집니다."""
    result = qcut_into_quantiles(diabetes_df['bmi'], q=4)
    counts = result.value_counts()
    # 각 구간의 빈도 차이가 전체의 10% 이하여야 합니다
    total = len(result)
    assert counts.max() - counts.min() <= total * 0.10


# =============================================================================
# 문제 4: KBinsDiscretizer
# =============================================================================

def discretize_with_kbins(X: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """
    KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)를 사용하여
    X를 구간화하세요.
    반환값: 정수 레이블로 구간화된 numpy array (shape 유지)
    """
    result = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return result


def test_binning_4_kbins_shape(diabetes_df):
    X = diabetes_df[['bmi', 'age']].values
    result = discretize_with_kbins(X, n_bins=5, strategy='uniform')
    assert result is not None
    assert result.shape == X.shape


def test_binning_4_kbins_n_unique(diabetes_df):
    X = diabetes_df[['bmi']].values
    result = discretize_with_kbins(X, n_bins=5, strategy='quantile')
    # 5개 구간이므로 고유 값은 최대 5개 (0~4)
    unique_vals = np.unique(result)
    assert len(unique_vals) <= 5


def test_binning_4_kbins_range(diabetes_df):
    X = diabetes_df[['bmi', 'age']].values
    result = discretize_with_kbins(X, n_bins=4, strategy='uniform')
    # 레이블이 0 ~ n_bins-1 범위 안에 있어야 합니다
    assert result.min() >= 0
    assert result.max() <= 3  # n_bins - 1


# =============================================================================
# 문제 5: 구간화 후 One-Hot Encoding
# =============================================================================

def bin_and_onehot(s: pd.Series, n_bins: int) -> pd.DataFrame:
    """
    1. pd.cut()으로 n_bins개 구간으로 나눈 후
    2. pd.get_dummies()로 원-핫 인코딩하세요.
    반환값: 원-핫 인코딩된 DataFrame
    """
    result = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return result


def test_binning_5_onehot_shape(age_series):
    result = bin_and_onehot(age_series, n_bins=4)
    assert result is not None
    assert result.shape[0] == len(age_series)
    assert result.shape[1] == 4  # n_bins개 열


def test_binning_5_onehot_binary(age_series):
    result = bin_and_onehot(age_series, n_bins=4)
    # 각 행의 합은 정확히 1 (하나의 구간에만 속함)
    row_sums = result.sum(axis=1)
    assert all(row_sums == 1)
