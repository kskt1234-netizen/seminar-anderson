import numpy as np

def test_numpy_1_array_creation():
    """
    문제 1: 다음과 같은 Numpy 배열을 만드세요.
    1부터 5까지의 정수가 들어있는 1차원 배열을 `answer_arr` 변수에 할당하세요.
    """
    answer_arr = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------
    
    assert isinstance(answer_arr, np.ndarray), "리스트가 아닌 numpy array로 생성해야 합니다."
    assert answer_arr.shape == (5,), "크기가 올바르지 않습니다."
    np.testing.assert_array_equal(answer_arr, np.array([1, 2, 3, 4, 5]))


def test_numpy_2_ones_matrix():
    """
    문제 2: 모든 원소가 1.0으로 채워진 3행 4열 (3 x 4) 2차원 배열을 생성하세요.
    """
    answer_arr = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    assert answer_arr.shape == (3, 4), "shape이 올바르지 않습니다."
    assert answer_arr.dtype in (np.float32, np.float64), "실수형 데이터 타입이어야 합니다."
    np.testing.assert_array_equal(answer_arr, np.ones((3, 4)))


def test_numpy_3_arange_and_type():
    """
    문제 3: 10부터 50까지 (50 미만 아님, 50 포함) 2의 간격으로 숫자를 생성하고(arange 사용),
    해당 배열의 요소 개수를 반환하는 함수를 만드세요.
    """
    
    # --- 코드를 작성하세요 ---
    arr = None
    count_of_elements = None
    # -----------------------

    assert arr[0] == 10
    assert arr[-1] == 50
    assert count_of_elements == 21


def test_numpy_4_linspace():
    """
    문제 4: 0부터 1 사이를 정확히 5등분한 지점들의 값을 가지는 배열 생성하세요 (linspace 사용).
    결과는 [0.0, 0.25, 0.5, 0.75, 1.0] 형태가 되어야 합니다.
    """
    answer_arr = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    assert answer_arr.shape == (5,)
    np.testing.assert_allclose(answer_arr, [0.0, 0.25, 0.5, 0.75, 1.0])
