import pandas as pd
import numpy as np

def test_pandas_1_handling_nans():
    """
    문제 1: 결측치(NaN) 처리하기
    주어진 df의 'Score' 열에 존재하는 NaN 값들을 해당 열의 '평균(mean)'으로 채우고,
    그 결과를 기존 데이터프레임 원본에 반영하여 `df_filled` 변수에 할당하세요. (fillna 활용)
    """
    df = pd.DataFrame({
        'Student': ['A', 'B', 'C', 'D'],
        'Score': [80.0, np.nan, 90.0, np.nan] # 평균은 85.0
    })
    
    df_filled = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    assert df_filled['Score'].isna().sum() == 0, "여전히 NaN 값이 존재합니다."
    assert list(df_filled['Score']) == [80.0, 85.0, 90.0, 85.0]


def test_pandas_2_data_type_conversion():
    """
    문제 2: 데이터 타입 변환하기
    'Price' 열은 현재 숫자가 아닌 문자열 형태로 저장되어 있습니다 (예: '1000').
    이를 정수형(int)으로 변환하여 다시 'Price' 열에 저장한 데이터프레임 `df_converted`를 만드세요.
    """
    df = pd.DataFrame({
        'Item': ['Apple', 'Banana', 'Cherry'],
        'Price': ['1500', '2000', '3000']
    })
    
    df_converted = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    assert df_converted['Price'].dtype in (np.int32, np.int64)
    assert df_converted['Price'].sum() == 6500


def test_pandas_3_string_cleaning():
    """
    문제 3: 문자열 데이터 정제하기
    'City' 열의 데이터들이 대소문자가 섞여있고 앞뒤로 공백이 존재합니다.
    모든 글자를 소문자(lower)로 바꾸고 띄어쓰기 공백을 제거(strip() 이나 replace())한 뒤 
    원래 열을 덮어씌운 `df_cleaned`를 구하세요. (.str 접근자 활용)
    """
    df = pd.DataFrame({
        'Person': ['John', 'Jane', 'Sam'],
        'City': ['  Seoul  ', 'busan', 'NEW YORK ']
    })
    
    df_cleaned = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    expected_cities = ['seoul', 'busan', 'newyork']
    assert list(df_cleaned['City']) == expected_cities


def test_pandas_4_drop_duplicates():
    """
    문제 4: 중복 데이터 제거하기
    동일한 데이터가 두 번 입력된 상황입니다. 'Email' 열을 기준으로 중복된 행을 찾아, 
    나중에 들어온(두번째) 행을 유지(keep='last')하고 이전 데이터를 지운 `df_unique`를 작성하세요.
    """
    df = pd.DataFrame({
        'ID': [1, 2, 3, 2],
        'Name': ['Alice', 'Bob', 'Charlie', 'Bob_updated'],
        'Email': ['a@a.com', 'b@b.com', 'c@c.com', 'b@b.com']
    })
    
    df_unique = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    assert len(df_unique) == 3
    # 인덱스 초기화를 진행하지 않았다면 3번 인덱스가 남아야함
    assert list(df_unique['Name'].values) == ['Alice', 'Charlie', 'Bob_updated']
