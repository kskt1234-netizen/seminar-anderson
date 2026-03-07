import numpy as np

# 학생들의 과목별 시험 성적 (모의 데이터셋)
# 행(Row): 각 학생 (학생 A, B, C, D, E)
# 열(Column): 과목 점수 (수학, 영어, 과학)
student_scores = np.array([
    [85, 92, 78],  # 학생 A
    [65, 70, 58],  # 학생 B
    [95, 88, 92],  # 학생 C
    [70, 60, 65],  # 학생 D
    [90, 85, 88]   # 학생 E
])

def test_data_1_student_averages():
    """
    문제 1: 각 학생의 전체 과목 평균(average) 점수를 구하세요.
    결과는 학생 수 (5명) 만큼의 길이를 가지는 1차원 배열이어야 합니다.
    """
    student_avgs = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    assert student_avgs.shape == (5,)
    np.testing.assert_allclose(student_avgs, [85.0, 64.333333, 91.666667, 65.0, 87.666667])


def test_data_2_subject_max():
    """
    문제 2: 각 과목별로 최고점을 구하세요.
    결과는 과목 수 (3과목) 만큼의 길이를 가지는 1차원 배열이어야 합니다.
    """
    subject_max_scores = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    assert subject_max_scores.shape == (3,)
    np.testing.assert_array_equal(subject_max_scores, [95, 92, 92])


def test_data_3_finding_low_scores():
    """
    문제 3: 영어 과목(두번째 열, 인덱스 1)의 점수가 80점 미만인 학생들의 
    '수학 점수(첫번째 열)'만 추출해 `low_english_math_scores` 배열에 할당하세요.
    
    (힌트: 먼저 영어 과목 점수가 80점 미만인 불리언 마스크를 만들고, 
    해당 마스크를 수학 점수 열에 씌워 추출합니다.)
    """
    low_english_math_scores = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    # 학생 B, D가 영어 80점 미만이므로 그들의 수학 점수인 [65, 70]이 답이 됩니다.
    np.testing.assert_array_equal(low_english_math_scores, [65, 70])


def test_data_4_min_max_normalization():
    """
    문제 4: (데이터 전처리 맛보기)
    수학 점수 편차가 큽니다. 전체 데이터를 '최소-최대 정규화(Min-Max Normalization)' 방식으로 변환하세요.
    변환 공식: (x - 전체최소값) / (전체최대값 - 전체최소값)
    
    수학 점수 열만 따로 추출하지 말고, 전체 데이터셋(student_scores)에 대해 
    가장 작은 값이 0, 가장 큰 값이 1이 되도록 전체 변환된 `normalized_scores`를 만드세요.
    """
    normalized_scores = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    # 전체 최소값은 58, 전체 최대값은 95
    assert np.isclose(normalized_scores.min(), 0.0)
    assert np.isclose(normalized_scores.max(), 1.0)
    assert normalized_scores.shape == (5, 3)
    
    expected_first_row = (np.array([85, 92, 78]) - 58) / (95 - 58)
    np.testing.assert_allclose(normalized_scores[0], expected_first_row)
