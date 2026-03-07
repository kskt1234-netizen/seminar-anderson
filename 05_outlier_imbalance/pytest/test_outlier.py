import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

# ---------- TODO 영역 ----------
# 학생들은 이 밑의 두 함수를 완성해야 합니다.
# 1. IQR 기반 이상치 필터링
# 2. IQR 기반 이상치 대체 (Clipping)

def detect_iqr_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    DataFrame과 column 이름이 주어졌을 때, 해당 컬럼의 IQR 방식을 이용한 '이상치 행(row)'만 
    추출하여 반환하는 함수를 작성하세요.
    
    [알고리즘]
    - Q1: 하위 25% (0.25)
    - Q3: 하위 75% (0.75)
    - IQR: Q3 - Q1
    - 이상치 기준: Q1 - (1.5 * IQR) 미만 이거나, Q3 + (1.5 * IQR) 초과
    """
    # 이 부분을 직접 구현하세요.
    pass

def cap_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    DataFrame과 column 이름이 주어졌을 때, IQR 기준을 벗어나는 이상치를 
    해당 상한선/하한선(Lower Bound, Upper Bound) 값으로 대체(Clipping)하여 
    데이터프레임의 기존 열(컬럼) 값을 덮어씌운 후 반환하세요.
    - np.clip() 내장 함수를 사용할 것을 매우 권장합니다.
    """
    # 기존 데이터프레임을 변형시키는 것을 막기 위해 명시적으로 copy를 사용합니다.
    # df_copy = df.copy() (필요 시 활용)
    
    # 이 부분을 직접 구현하세요.
    pass


# ---------- Pytest 검증 영역 ----------
# 학생들이 코딩 후 pytest 를 돌렸을 때 정답 여부를 채점하는 구간

@pytest.fixture
def diabetes_df():
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    # 실습용 타겟 변수. diabetes 데이터에서는 보통 bmi 등에서 이상치가 나옵니다.
    return df

def test_detect_iqr_outliers(diabetes_df):
    outliers = detect_iqr_outliers(diabetes_df, 'bmi')
    
    assert outliers is not None, "None이 반환되었습니다. 코드를 작성하세요!"
    assert isinstance(outliers, pd.DataFrame), "결과물은 DataFrame 형태여야 합니다."
    
    # 정답 검증용 로직
    q1 = diabetes_df['bmi'].quantile(0.25)
    q3 = diabetes_df['bmi'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    expected = diabetes_df[(diabetes_df['bmi'] < lower) | (diabetes_df['bmi'] > upper)]
    
    assert len(outliers) == len(expected), "도출된 이상치의 개수가 정답과 다릅니다."
    assert list(outliers.index) == list(expected.index), "도출된 이상치의 인덱스가 정답 위치와 다릅니다. 조건을 다시 확인하세요."

def test_cap_outliers(diabetes_df):
    result_df = cap_outliers(diabetes_df.copy(), 'bmi')
    
    assert result_df is not None, "None이 반환되었습니다. 코드를 작성하세요!"
    
    q1 = diabetes_df['bmi'].quantile(0.25)
    q3 = diabetes_df['bmi'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    # 대체된 결과의 max, min 확인
    assert result_df['bmi'].max() <= upper + 1e-6, "상한선이 제대로 깎이지(Clipping) 않았습니다."
    assert result_df['bmi'].min() >= lower - 1e-6, "하한선이 제대로 적용되지 않았습니다."
    
    # 정상 범위는 일치하는지 체크
    normal_mask = (diabetes_df['bmi'] >= lower) & (diabetes_df['bmi'] <= upper)
    np.testing.assert_array_almost_equal(
        result_df.loc[normal_mask, 'bmi'],
        diabetes_df.loc[normal_mask, 'bmi'],
        err_msg="이상치 범위를 벗어나지 않은 정상 값들까지 변질되었습니다! clip의 로직을 조심하세요."
    )
