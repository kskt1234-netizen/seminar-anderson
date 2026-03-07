import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes


@pytest.fixture
def iris_with_missing():
    """일부 값에 결측치를 인위적으로 추가한 Iris 데이터셋을 반환합니다."""
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    np.random.seed(42)
    # 약 15%의 값을 결측치로 만듭니다
    mask = np.random.rand(*X.shape) < 0.15
    X[mask] = np.nan
    return X.values  # numpy array


@pytest.fixture
def diabetes_with_missing():
    """일부 값에 결측치를 인위적으로 추가한 Diabetes 데이터셋을 반환합니다."""
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data.copy()
    np.random.seed(7)
    mask = np.random.rand(*X.shape) < 0.10
    X[mask] = np.nan
    return X.values


# =============================================================================
# 문제 1: SimpleImputer — 평균 대치
# =============================================================================

def impute_with_mean(X: np.ndarray) -> np.ndarray:
    """
    SimpleImputer(strategy='mean')을 사용하여 결측치를 열별 평균으로 채우세요.
    반환값: 결측치가 없는 numpy array
    """
    result = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return result


def test_imputation_1_no_missing_after_mean(iris_with_missing):
    result = impute_with_mean(iris_with_missing)
    assert result is not None
    assert not np.isnan(result).any(), "결측치가 남아 있습니다."


def test_imputation_1_mean_shape(iris_with_missing):
    result = impute_with_mean(iris_with_missing)
    assert result.shape == iris_with_missing.shape


def test_imputation_1_mean_value(iris_with_missing):
    result = impute_with_mean(iris_with_missing)
    # 대치된 값은 원본(결측치 제외)의 열 평균과 같아야 합니다
    for col in range(iris_with_missing.shape[1]):
        col_data = iris_with_missing[:, col]
        col_mean = np.nanmean(col_data)
        missing_mask = np.isnan(col_data)
        imputed_values = result[missing_mask, col]
        np.testing.assert_allclose(imputed_values, col_mean, atol=1e-6)


# =============================================================================
# 문제 2: SimpleImputer — 중앙값 대치
# =============================================================================

def impute_with_median(X: np.ndarray) -> np.ndarray:
    """
    SimpleImputer(strategy='median')을 사용하여 결측치를 열별 중앙값으로 채우세요.
    반환값: 결측치가 없는 numpy array
    """
    result = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return result


def test_imputation_2_no_missing_after_median(iris_with_missing):
    result = impute_with_median(iris_with_missing)
    assert result is not None
    assert not np.isnan(result).any()


def test_imputation_2_median_value(iris_with_missing):
    result = impute_with_median(iris_with_missing)
    for col in range(iris_with_missing.shape[1]):
        col_data = iris_with_missing[:, col]
        col_median = np.nanmedian(col_data)
        missing_mask = np.isnan(col_data)
        imputed_values = result[missing_mask, col]
        np.testing.assert_allclose(imputed_values, col_median, atol=1e-6)


# =============================================================================
# 문제 3: KNNImputer
# =============================================================================

def impute_with_knn(X: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """
    KNNImputer(n_neighbors=n_neighbors)를 사용하여 결측치를 채우세요.
    반환값: 결측치가 없는 numpy array
    """
    result = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return result


def test_imputation_3_no_missing_knn(iris_with_missing):
    result = impute_with_knn(iris_with_missing, n_neighbors=5)
    assert result is not None
    assert not np.isnan(result).any(), "KNN 대치 후에도 결측치가 남아 있습니다."


def test_imputation_3_knn_shape(iris_with_missing):
    result = impute_with_knn(iris_with_missing)
    assert result.shape == iris_with_missing.shape


def test_imputation_3_knn_non_negative(iris_with_missing):
    """Iris 데이터는 모두 양수이므로 대치된 값도 양수여야 합니다."""
    result = impute_with_knn(iris_with_missing)
    assert result.min() >= 0


# =============================================================================
# 문제 4: IterativeImputer
# =============================================================================

def impute_with_iterative(X: np.ndarray) -> np.ndarray:
    """
    IterativeImputer를 사용하여 결측치를 채우세요.
    max_iter=10, random_state=42를 사용하세요.
    힌트: from sklearn.experimental import enable_iterative_imputer
          from sklearn.impute import IterativeImputer
    반환값: 결측치가 없는 numpy array
    """
    result = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return result


def test_imputation_4_no_missing_iterative(iris_with_missing):
    result = impute_with_iterative(iris_with_missing)
    assert result is not None
    assert not np.isnan(result).any(), "반복 대치 후에도 결측치가 남아 있습니다."


def test_imputation_4_iterative_shape(iris_with_missing):
    result = impute_with_iterative(iris_with_missing)
    assert result.shape == iris_with_missing.shape


# =============================================================================
# 문제 5: Missing Indicator (결측치 지시자 피처)
# =============================================================================

def add_missing_indicator(X: np.ndarray) -> np.ndarray:
    """
    MissingIndicator를 사용하여 결측치가 있는 열에 대한 지시자(True/False) 행렬을 반환하세요.
    features='missing-only' 옵션을 사용하세요.
    반환값: 결측치가 있는 열에 대한 True/False numpy array
    """
    result = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return result


def test_imputation_5_indicator_shape(iris_with_missing):
    result = add_missing_indicator(iris_with_missing)
    assert result is not None
    # 행 수는 원본과 같아야 합니다
    assert result.shape[0] == iris_with_missing.shape[0]
    # 결측치가 있는 열만 포함하므로 열 수는 원본 이하
    assert result.shape[1] <= iris_with_missing.shape[1]


def test_imputation_5_indicator_dtype(iris_with_missing):
    result = add_missing_indicator(iris_with_missing)
    # 불리언 타입이어야 합니다
    assert result.dtype == bool


def test_imputation_5_indicator_matches_missing(iris_with_missing):
    result = add_missing_indicator(iris_with_missing)
    from sklearn.impute import MissingIndicator
    indicator = MissingIndicator(features='missing-only')
    expected = indicator.fit_transform(iris_with_missing)
    # 원본 결측 위치와 지시자가 일치해야 합니다
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# 문제 6: 결측치 비율 계산
# =============================================================================

def missing_ratio(X: np.ndarray) -> np.ndarray:
    """
    각 열(피처)의 결측치 비율(0.0~1.0)을 numpy array로 반환하세요.
    힌트: np.isnan(X).mean(axis=0)
    """
    result = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return result


def test_imputation_6_missing_ratio_range(iris_with_missing):
    ratios = missing_ratio(iris_with_missing)
    assert ratios is not None
    assert ratios.shape == (iris_with_missing.shape[1],)
    assert all(0.0 <= r <= 1.0 for r in ratios)


def test_imputation_6_missing_ratio_correct(iris_with_missing):
    ratios = missing_ratio(iris_with_missing)
    expected = np.isnan(iris_with_missing).mean(axis=0)
    np.testing.assert_allclose(ratios, expected, atol=1e-10)
