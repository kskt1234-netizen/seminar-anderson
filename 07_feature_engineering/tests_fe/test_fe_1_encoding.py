import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.datasets import load_iris


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'city':    ['Seoul', 'Busan', 'Seoul', 'Jeju', 'Busan', 'Seoul'],
        'grade':   ['high',  'low',   'medium','high', 'low',   'medium'],
        'revenue': [120,      80,      100,     60,     85,      110],
    })


# =============================================================================
# 문제 1: Label Encoding
# =============================================================================

def label_encode_series(s: pd.Series) -> tuple:
    """
    주어진 Series를 LabelEncoder로 인코딩하세요.

    반환값: (encoded_array, encoder)
    - encoded_array: fit_transform 결과 (numpy array)
    - encoder: 학습된 LabelEncoder 객체
    """
    encoded_array = None
    encoder = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return encoded_array, encoder


def test_encoding_1_label_encode_values(sample_df):
    encoded, le = label_encode_series(sample_df['city'])
    assert encoded is not None
    assert len(encoded) == len(sample_df)
    # 같은 도시는 같은 인코딩 값을 가져야 합니다
    assert encoded[0] == encoded[2] == encoded[5]   # Seoul
    assert encoded[1] == encoded[4]                 # Busan


def test_encoding_1_label_encode_classes(sample_df):
    encoded, le = label_encode_series(sample_df['city'])
    assert set(le.classes_) == {'Seoul', 'Busan', 'Jeju'}


def test_encoding_1_label_encode_inverse(sample_df):
    encoded, le = label_encode_series(sample_df['city'])
    restored = le.inverse_transform(encoded)
    assert list(restored) == list(sample_df['city'])


# =============================================================================
# 문제 2: Ordinal Encoding
# =============================================================================

def ordinal_encode_grade(df: pd.DataFrame) -> np.ndarray:
    """
    df의 'grade' 컬럼을 순서형 인코딩하세요.
    순서: low=0, medium=1, high=2
    OrdinalEncoder를 사용하고 변환 결과를 1차원 numpy array로 반환하세요.
    """
    result = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return result


def test_encoding_2_ordinal_order(sample_df):
    encoded = ordinal_encode_grade(sample_df)
    assert encoded is not None
    # grade: ['high','low','medium','high','low','medium']
    # 기댓값:   [2,    0,    1,      2,    0,    1]
    expected = np.array([2., 0., 1., 2., 0., 1.])
    np.testing.assert_array_equal(encoded, expected)


def test_encoding_2_ordinal_shape(sample_df):
    encoded = ordinal_encode_grade(sample_df)
    assert encoded.shape == (6,)


# =============================================================================
# 문제 3: One-Hot Encoding
# =============================================================================

def onehot_encode_city(df: pd.DataFrame) -> pd.DataFrame:
    """
    df의 'city' 컬럼을 원-핫 인코딩하세요.
    결과는 city_ 접두사가 붙은 열들을 포함하는 DataFrame이어야 합니다.
    pd.get_dummies()를 사용하세요.
    """
    result = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return result


def test_encoding_3_onehot_shape(sample_df):
    result = onehot_encode_city(sample_df)
    assert result is not None
    # Busan, Jeju, Seoul 3개 도시 → 3개 열
    assert result.shape[1] == 3
    assert result.shape[0] == len(sample_df)


def test_encoding_3_onehot_columns(sample_df):
    result = onehot_encode_city(sample_df)
    # 열 이름에 city_가 포함되어야 합니다
    assert all('city_' in col for col in result.columns)


def test_encoding_3_onehot_binary(sample_df):
    result = onehot_encode_city(sample_df)
    # 각 행은 정확히 하나의 1과 나머지 0으로 구성
    row_sums = result.sum(axis=1)
    assert all(row_sums == 1)


# =============================================================================
# 문제 4: Target Encoding
# =============================================================================

def target_encode(df: pd.DataFrame, column: str, target: str) -> pd.Series:
    """
    df의 column을 target 변수의 평균으로 인코딩하세요.
    각 카테고리 값을 해당 카테고리의 target 평균으로 대체한 Series를 반환하세요.
    """
    result = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return result


def test_encoding_4_target_encode_values(sample_df):
    encoded = target_encode(sample_df, 'city', 'revenue')
    assert encoded is not None

    # Seoul: (120+100+110)/3 = 110.0
    seoul_encoded = encoded[sample_df['city'] == 'Seoul']
    assert all(abs(seoul_encoded - 110.0) < 1e-6)

    # Busan: (80+85)/2 = 82.5
    busan_encoded = encoded[sample_df['city'] == 'Busan']
    assert all(abs(busan_encoded - 82.5) < 1e-6)


def test_encoding_4_target_encode_length(sample_df):
    encoded = target_encode(sample_df, 'city', 'revenue')
    assert len(encoded) == len(sample_df)


# =============================================================================
# 문제 5: Frequency Encoding
# =============================================================================

def frequency_encode(df: pd.DataFrame, column: str) -> pd.Series:
    """
    df의 column에서 각 카테고리의 등장 빈도(count)로 대체한 Series를 반환하세요.
    """
    result = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return result


def test_encoding_5_frequency_values(sample_df):
    encoded = frequency_encode(sample_df, 'city')
    assert encoded is not None

    # Seoul은 3번 등장
    seoul_freq = encoded[sample_df['city'] == 'Seoul']
    assert all(seoul_freq == 3)

    # Busan은 2번 등장
    busan_freq = encoded[sample_df['city'] == 'Busan']
    assert all(busan_freq == 2)

    # Jeju는 1번 등장
    jeju_freq = encoded[sample_df['city'] == 'Jeju']
    assert all(jeju_freq == 1)


def test_encoding_5_frequency_length(sample_df):
    encoded = frequency_encode(sample_df, 'city')
    assert len(encoded) == len(sample_df)
