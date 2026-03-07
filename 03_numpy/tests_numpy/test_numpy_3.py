import numpy as np

def test_numpy_1_boolean_masking():
    """
    문제 1: 불리언 마스킹(Boolean Masking) 이해하기
    주어진 배열에서 50 초과 (50 불포함)인 값들만 추출하여 `answer_arr`에 할당하세요.
    """
    arr = np.array([10, 55, 30, 80, 45, 90, 20])
    answer_arr = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    np.testing.assert_array_equal(answer_arr, np.array([55, 80, 90]))


def test_numpy_2_multiple_conditions():
    """
    문제 2: 다중 조건 불리언 마스킹
    주어진 배열에서 20 이상 "그리고" 70 미만인 값들만 추출하여 `answer_arr`에 할당하세요.
    (Numpy에서는 파이썬의 `and`, `or` 대신 `&`, `|` 기호를 사용하고 조건들을 괄호()로 묶어야 합니다)
    """
    arr = np.array([5, 20, 35, 65, 70, 85, 100])
    answer_arr = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    np.testing.assert_array_equal(answer_arr, np.array([20, 35, 65]))


def test_numpy_3_vectorization():
    """
    문제 3: 벡터화 연산 (Vectorization)
    주어진 배열의 모든 요소에 15를 더하고, 그 결과에 2를 곱한 새로운 배열을 `answer_arr`에 할당하세요.
    (for문을 사용하지 말고 Numpy 배열 사칙연산을 수행하세요)
    """
    arr = np.array([1, 2, 3, 4, 5])
    answer_arr = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    np.testing.assert_array_equal(answer_arr, np.array([32, 34, 36, 38, 40]))


def test_numpy_4_broadcasting():
    """
    문제 4: 브로드캐스팅 (Broadcasting)
    형태(shape)가 다른 배열 간의 연산입니다. 
    (3x3) 배열의 '각 열'마다 특정 값들을 순서대로 빼고 싶습니다.
    (3x3) 행렬(matrix)에서 1차원 배열(vector)을 바로 빼면 (broadcasting에 의해)
    각 행마다 vector 값이 빠지게 됩니다. 코드를 작성해 `answer_arr`를 구하세요.
    """
    matrix = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ])
    vector = np.array([1, 2, 3])
    
    answer_arr = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    expected = np.array([
        [9, 18, 27],
        [39, 48, 57],
        [69, 78, 87]
    ])
    np.testing.assert_array_equal(answer_arr, expected)


def test_numpy_5_aggregation():
    """
    문제 5: 집합(Aggregation) 함수
    제공된 2차원 배열에 대해 아래 3가지 통계값을 구하세요:
    - `total_sum` : 배열 전체 요소의 합
    - `col_mean`  : 각 '열(column)'의 평균값 (힌트: axis 사용)
    - `row_max`   : 각 '행(row)'의 최댓값 (힌트: axis 사용)
    """
    arr_2d = np.array([
        [1, 5, 2],
        [4, 3, 6],
        [7, 8, 9]
    ])
    
    total_sum = None
    col_mean = None
    row_max = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    assert total_sum == 45
    np.testing.assert_array_equal(col_mean, np.array([4., 5.33333333, 5.66666667]))
    np.testing.assert_array_equal(row_max, np.array([5, 6, 9]))
