import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_iris, load_diabetes


@pytest.fixture
def simple_X():
    """간단한 2차원 입력 데이터"""
    return np.array([[2.0, 3.0], [4.0, 5.0], [1.0, 0.0]])


@pytest.fixture
def diabetes_df():
    diabetes = load_diabetes(as_frame=True)
    return diabetes.frame.copy()


# =============================================================================
# 문제 1: PolynomialFeatures (degree=2, bias 포함)
# =============================================================================

def apply_polynomial_features(X: np.ndarray, degree: int = 2) -> tuple:
    """
    PolynomialFeatures(degree=degree, include_bias=True)를 적용하세요.
    반환값: (X_poly, poly)
    - X_poly: 변환된 numpy array
    - poly: 학습된 PolynomialFeatures 객체
    """
    X_poly = None
    poly = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return X_poly, poly


def test_poly_1_n_features(simple_X):
    X_poly, poly = apply_polynomial_features(simple_X, degree=2)
    assert X_poly is not None
    # 피처 2개, degree=2, bias 포함: 1, x0, x1, x0^2, x0*x1, x1^2 → 6개
    assert X_poly.shape[1] == 6


def test_poly_1_bias_column(simple_X):
    X_poly, poly = apply_polynomial_features(simple_X, degree=2)
    # bias 열은 모두 1이어야 합니다
    assert all(X_poly[:, 0] == 1.0)


def test_poly_1_original_features(simple_X):
    X_poly, poly = apply_polynomial_features(simple_X, degree=2)
    # 원본 피처가 보존되어야 합니다 (열 1, 2)
    np.testing.assert_array_equal(X_poly[:, 1], simple_X[:, 0])
    np.testing.assert_array_equal(X_poly[:, 2], simple_X[:, 1])


def test_poly_1_squared_features(simple_X):
    X_poly, poly = apply_polynomial_features(simple_X, degree=2)
    feature_names = poly.get_feature_names_out()
    # x0^2 열 값 확인
    x0_sq_idx = list(feature_names).index('x0^2')
    np.testing.assert_allclose(X_poly[:, x0_sq_idx], simple_X[:, 0] ** 2)


def test_poly_1_interaction_term(simple_X):
    X_poly, poly = apply_polynomial_features(simple_X, degree=2)
    feature_names = poly.get_feature_names_out()
    # x0*x1 교차항 확인
    inter_idx = list(feature_names).index('x0 x1')
    np.testing.assert_allclose(X_poly[:, inter_idx], simple_X[:, 0] * simple_X[:, 1])


# =============================================================================
# 문제 2: 상호작용 피처만 생성 (interaction_only=True)
# =============================================================================

def apply_interaction_only(X: np.ndarray) -> tuple:
    """
    PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)를 적용하세요.
    자체 제곱항(x0^2, x1^2)은 제외하고 교차항(x0*x1)만 생성합니다.
    반환값: (X_inter, poly)
    """
    X_inter = None
    poly = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return X_inter, poly


def test_poly_2_interaction_n_features(simple_X):
    X_inter, poly = apply_interaction_only(simple_X)
    assert X_inter is not None
    # 피처 2개, interaction_only=True, bias=False: x0, x1, x0*x1 → 3개
    assert X_inter.shape[1] == 3


def test_poly_2_no_squared_terms(simple_X):
    X_inter, poly = apply_interaction_only(simple_X)
    feature_names = list(poly.get_feature_names_out())
    # 제곱항이 없어야 합니다
    assert 'x0^2' not in feature_names
    assert 'x1^2' not in feature_names
    assert 'x0 x1' in feature_names


# =============================================================================
# 문제 3: 수학적 변환 — 로그, 제곱근
# =============================================================================

def apply_log_transform(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    df의 columns 리스트에 있는 열에 log(x+1) 변환을 적용하세요.
    원본 열은 유지하고, 새 열 이름은 f'log_{col}'로 하세요.
    힌트: np.log1p()
    """
    df = df.copy()
    # --- 코드를 작성하세요 ---

    # -----------------------
    return df


def test_poly_3_log_columns(diabetes_df):
    result = apply_log_transform(diabetes_df, ['bmi', 'age'])
    assert result is not None
    assert 'log_bmi' in result.columns
    assert 'log_age' in result.columns


def test_poly_3_log_values(diabetes_df):
    result = apply_log_transform(diabetes_df, ['bmi'])
    expected = np.log1p(diabetes_df['bmi'].values)
    np.testing.assert_allclose(result['log_bmi'].values, expected, atol=1e-10)


def test_poly_3_original_preserved(diabetes_df):
    result = apply_log_transform(diabetes_df, ['bmi'])
    # 원본 열이 그대로 남아 있어야 합니다
    np.testing.assert_array_equal(result['bmi'].values, diabetes_df['bmi'].values)


# =============================================================================
# 문제 4: 수학적 변환 — 제곱, 역수
# =============================================================================

def apply_power_transforms(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    df[col]에 대해 다음 변환을 적용하세요.
    - f'{col}_sq':  제곱 (x^2)
    - f'{col}_sqrt': 제곱근 (sqrt(x), 음수 없다고 가정)
    - f'{col}_inv': 역수 (1 / (x + 1e-8))
    반환값: 새 열이 추가된 DataFrame (원본 열 유지)
    """
    df = df.copy()
    # --- 코드를 작성하세요 ---

    # -----------------------
    return df


def test_poly_4_power_columns(diabetes_df):
    result = apply_power_transforms(diabetes_df, 'bmi')
    assert 'bmi_sq'   in result.columns
    assert 'bmi_sqrt' in result.columns
    assert 'bmi_inv'  in result.columns


def test_poly_4_squared_values(diabetes_df):
    result = apply_power_transforms(diabetes_df, 'bmi')
    expected = diabetes_df['bmi'].values ** 2
    np.testing.assert_allclose(result['bmi_sq'].values, expected, atol=1e-10)


def test_poly_4_sqrt_values(diabetes_df):
    result = apply_power_transforms(diabetes_df, 'bmi')
    expected = np.sqrt(diabetes_df['bmi'].values)
    np.testing.assert_allclose(result['bmi_sqrt'].values, expected, atol=1e-10)


def test_poly_4_inv_values(diabetes_df):
    result = apply_power_transforms(diabetes_df, 'bmi')
    expected = 1 / (diabetes_df['bmi'].values + 1e-8)
    np.testing.assert_allclose(result['bmi_inv'].values, expected, atol=1e-6)


# =============================================================================
# 문제 5: 직접 상호작용 피처 생성
# =============================================================================

def create_ratio_feature(df: pd.DataFrame, numerator: str, denominator: str, new_col: str) -> pd.DataFrame:
    """
    df[numerator] / (df[denominator] + 1e-8) 비율 피처를 new_col로 추가하세요.
    반환값: 새 열이 추가된 DataFrame
    """
    df = df.copy()
    # --- 코드를 작성하세요 ---

    # -----------------------
    return df


def test_poly_5_ratio_column(diabetes_df):
    result = create_ratio_feature(diabetes_df, 'bmi', 'age', 'bmi_per_age')
    assert 'bmi_per_age' in result.columns


def test_poly_5_ratio_values(diabetes_df):
    result = create_ratio_feature(diabetes_df, 'bmi', 'age', 'bmi_per_age')
    expected = diabetes_df['bmi'].values / (diabetes_df['age'].values + 1e-8)
    np.testing.assert_allclose(result['bmi_per_age'].values, expected, atol=1e-6)
