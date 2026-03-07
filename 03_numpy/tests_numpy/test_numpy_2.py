import numpy as np

def test_numpy_1_indexing_1d():
    """
    문제 1: 주어진 배열에서 숫자 45를 가져와 `answer` 변수에 할당하세요.
    (음수 인덱싱을 사용해도 좋습니다)
    """
    arr = np.array([10, 20, 30, 40, 45, 50, 60])
    answer = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    assert answer == 45


def test_numpy_2_slicing_1d():
    """
    문제 2: 주어진 배열에서 30부터 60까지 해당하는 부분을 슬라이싱하여 `answer_arr`에 할당하세요.
    """
    arr = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    answer_arr = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    np.testing.assert_array_equal(answer_arr, np.array([30, 40, 50, 60]))


def test_numpy_3_indexing_2d():
    """
    문제 3: 3x3 이차원 배열에서 정중앙에 있는 값(5)을 가져와 `answer`에 할당하세요.
    """
    arr_2d = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    answer = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    assert answer == 5


def test_numpy_4_slicing_2d():
    """
    문제 4: 위 3x3 배열을 대상으로 슬라이싱을 진행하세요.
    첫번째 행과 두번째 행으로 이루어지고, 두번째 열부터 끝 열까지 구성된 2x2 하위 배열을 반환해야 합니다.
    기대되는 결과 형태:
    [[2, 3],
     [5, 6]]
    """
    arr_2d = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    answer_arr = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    expected = np.array([[2, 3], [5, 6]])
    np.testing.assert_array_equal(answer_arr, expected)


def test_numpy_5_reshape():
    """
    문제 5: 1부터 12까지 순서대로 나열된 1차원 배열이 있습니다. 
    이 배열을 3행 4열 (3 x 4) 형태의 2차원 배열로 변경(reshape)하여 `answer_arr`에 할당하세요.
    """
    arr = np.arange(1, 13)
    answer_arr = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    assert answer_arr.shape == (3, 4)
    np.testing.assert_array_equal(answer_arr, np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]]))
