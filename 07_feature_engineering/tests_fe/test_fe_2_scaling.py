import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.datasets import load_iris, load_diabetes


@pytest.fixture
def iris_X():
    iris = load_iris(as_frame=True)
    return iris.data.values  # numpy array


@pytest.fixture
def diabetes_X():
    diabetes = load_diabetes(as_frame=True)
    return diabetes.data.values


# =============================================================================
# 문제 1: StandardScaler
# =============================================================================

def apply_standard_scaler(X: np.ndarray) -> tuple:
    """
    StandardScaler를 적용하세요.
    반환값: (X_scaled, scaler)
    - X_scaled: 변환된 numpy array
    - scaler: 학습된 StandardScaler 객체
    """
    X_scaled = None
    scaler = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return X_scaled, scaler


def test_scaling_1_standard_mean(iris_X):
    X_scaled, scaler = apply_standard_scaler(iris_X)
    assert X_scaled is not None
    # 각 열의 평균이 0에 가까워야 합니다
    col_means = X_scaled.mean(axis=0)
    np.testing.assert_allclose(col_means, np.zeros(iris_X.shape[1]), atol=1e-6)


def test_scaling_1_standard_std(iris_X):
    X_scaled, scaler = apply_standard_scaler(iris_X)
    # 각 열의 표준편차가 1에 가까워야 합니다
    col_stds = X_scaled.std(axis=0)
    np.testing.assert_allclose(col_stds, np.ones(iris_X.shape[1]), atol=1e-6)


def test_scaling_1_standard_shape(iris_X):
    X_scaled, _ = apply_standard_scaler(iris_X)
    assert X_scaled.shape == iris_X.shape


def test_scaling_1_standard_scaler_attributes(iris_X):
    _, scaler = apply_standard_scaler(iris_X)
    assert hasattr(scaler, 'mean_')
    assert hasattr(scaler, 'scale_')
    np.testing.assert_allclose(scaler.mean_, iris_X.mean(axis=0), atol=1e-6)


# =============================================================================
# 문제 2: MinMaxScaler
# =============================================================================

def apply_minmax_scaler(X: np.ndarray) -> tuple:
    """
    MinMaxScaler를 적용하세요. feature_range는 기본값 (0, 1)을 사용하세요.
    반환값: (X_scaled, scaler)
    """
    X_scaled = None
    scaler = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return X_scaled, scaler


def test_scaling_2_minmax_range(iris_X):
    X_scaled, _ = apply_minmax_scaler(iris_X)
    assert X_scaled is not None
    assert X_scaled.min() >= -1e-9     # 최솟값 ≥ 0
    assert X_scaled.max() <= 1 + 1e-9  # 최댓값 ≤ 1


def test_scaling_2_minmax_column_min_max(iris_X):
    X_scaled, _ = apply_minmax_scaler(iris_X)
    # 각 열의 최솟값은 정확히 0, 최댓값은 정확히 1
    np.testing.assert_allclose(X_scaled.min(axis=0), np.zeros(iris_X.shape[1]), atol=1e-6)
    np.testing.assert_allclose(X_scaled.max(axis=0), np.ones(iris_X.shape[1]),  atol=1e-6)


def test_scaling_2_minmax_shape(iris_X):
    X_scaled, _ = apply_minmax_scaler(iris_X)
    assert X_scaled.shape == iris_X.shape


# =============================================================================
# 문제 3: RobustScaler
# =============================================================================

def apply_robust_scaler(X: np.ndarray) -> tuple:
    """
    RobustScaler를 적용하세요.
    반환값: (X_scaled, scaler)
    """
    X_scaled = None
    scaler = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return X_scaled, scaler


def test_scaling_3_robust_median(iris_X):
    X_scaled, scaler = apply_robust_scaler(iris_X)
    assert X_scaled is not None
    # RobustScaler는 중앙값으로 center합니다
    # 따라서 변환 후 각 열의 중앙값은 0에 가까워야 합니다
    col_medians = np.median(X_scaled, axis=0)
    np.testing.assert_allclose(col_medians, np.zeros(iris_X.shape[1]), atol=1e-6)


def test_scaling_3_robust_scaler_attributes(iris_X):
    _, scaler = apply_robust_scaler(iris_X)
    assert hasattr(scaler, 'center_')
    assert hasattr(scaler, 'scale_')
    # center_는 원본 데이터의 중앙값이어야 합니다
    np.testing.assert_allclose(scaler.center_, np.median(iris_X, axis=0), atol=1e-6)


def test_scaling_3_robust_shape(iris_X):
    X_scaled, _ = apply_robust_scaler(iris_X)
    assert X_scaled.shape == iris_X.shape


# =============================================================================
# 문제 4: Normalizer (행 단위 정규화)
# =============================================================================

def apply_normalizer(X: np.ndarray, norm: str = 'l2') -> np.ndarray:
    """
    Normalizer를 적용하세요.
    반환값: X_normalized (각 행의 L2 노름이 1인 numpy array)
    """
    X_normalized = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return X_normalized


def test_scaling_4_normalizer_l2_norm(iris_X):
    X_norm = apply_normalizer(iris_X, norm='l2')
    assert X_norm is not None
    # 각 행의 L2 노름이 1이어야 합니다
    row_norms = np.linalg.norm(X_norm, axis=1)
    np.testing.assert_allclose(row_norms, np.ones(iris_X.shape[0]), atol=1e-6)


def test_scaling_4_normalizer_shape(iris_X):
    X_norm = apply_normalizer(iris_X)
    assert X_norm.shape == iris_X.shape


def test_scaling_4_normalizer_l1(iris_X):
    X_norm = apply_normalizer(iris_X, norm='l1')
    # 각 행의 L1 노름(절댓값 합계)이 1이어야 합니다
    row_norms = np.abs(X_norm).sum(axis=1)
    np.testing.assert_allclose(row_norms, np.ones(iris_X.shape[0]), atol=1e-6)


# =============================================================================
# 문제 5: Train/Test 분리 후 스케일링 (Data Leakage 방지)
# =============================================================================

def scale_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    올바른 방법으로 Train/Test를 스케일링하세요.
    - X_train으로만 fit()
    - X_test에는 transform()만 적용
    반환값: (X_train_scaled, X_test_scaled, scaler)
    """
    X_train_scaled = None
    X_test_scaled  = None
    scaler = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return X_train_scaled, X_test_scaled, scaler


def test_scaling_5_no_leakage(iris_X):
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(iris_X, test_size=0.3, random_state=42)

    X_train_scaled, X_test_scaled, scaler = scale_train_test(X_train, X_test)

    assert X_train_scaled is not None
    assert X_test_scaled is not None

    # Train 스케일링 결과 검증: 평균 ≈ 0
    np.testing.assert_allclose(X_train_scaled.mean(axis=0), np.zeros(iris_X.shape[1]), atol=1e-6)

    # scaler는 train 데이터의 통계를 사용해야 합니다
    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0), atol=1e-6)

    # Test 데이터의 mean은 0이 아닐 수 있습니다 (train 통계로 변환하므로)
    # 하지만 shape는 유지
    assert X_test_scaled.shape == X_test.shape
