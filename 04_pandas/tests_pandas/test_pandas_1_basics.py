import pandas as pd
from datasets import load_dataset

def test_pandas_1_load_and_inspect():
    """
    문제 1: datasets 라이브러리를 사용하여 'titanic' 데이터셋을 불러오세요.
    훈련(train) 셋을 가져와 Pandas DataFrame으로 변환(to_pandas)하여 `df` 변수에 할당하세요.
    """
    df = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    assert isinstance(df, pd.DataFrame), "데이터 타입이 DataFrame이 아닙니다."
    assert "Survived" in df.columns, "Titanic 데이터셋이 정상적으로 로드되지 않았습니다."
    assert len(df) == 891, "행의 개수가 891개가 아닙니다."


def test_pandas_2_extract_columns():
    """
    문제 2: 위 데이터를 가상으로 만들었습니다.
    'Name' 과 'Age' 두 개의 컬럼(열)만 추출한 데이터프레임을 `subset_df` 에 할당하세요.
    """
    data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['Seoul', 'Busan', 'New York']
    }
    df = pd.DataFrame(data)
    
    subset_df = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    assert list(subset_df.columns) == ['Name', 'Age']
    assert len(subset_df) == 3


def test_pandas_3_boolean_indexing():
    """
    문제 3: 주어진 데이터프레임에서 나이가 30 이상이고(AND),
    도시는 'Seoul'이 아닌 사람들의 데이터만 추출해 `filtered_df`에 할당하세요. (loc 활용)
    """
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'Dave'],
        'Age': [25, 30, 35, 40],
        'City': ['Seoul', 'Busan', 'Seoul', 'Jeju']
    }
    df = pd.DataFrame(data)
    
    filtered_df = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    assert len(filtered_df) == 2
    assert list(filtered_df['Name'].values) == ['Bob', 'Dave']


def test_pandas_4_iloc_usage():
    """
    문제 4: iloc 위치 기반 인덱싱
    데이터프레임 `df`의 위에서부터 두번째 행(인덱스 1)부터 끝까지,
    그리고 컬럼(열)은 첫번째(0), 세번째(2) 열만을 추출한 데이터프레임을 `iloc_df`에 할당하세요.
    """
    data = {
        'A': [1, 2, 3, 4],
        'B': [10, 20, 30, 40],
        'C': [100, 200, 300, 400]
    }
    df = pd.DataFrame(data)
    
    iloc_df = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    assert iloc_df.shape == (3, 2)
    assert list(iloc_df.columns) == ['A', 'C']
    assert iloc_df.iloc[0, 1] == 200 # 원래 데이터의 B행 C열 값
