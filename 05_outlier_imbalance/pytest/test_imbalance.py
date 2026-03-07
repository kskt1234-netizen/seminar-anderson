import pytest
import numpy as np
from sklearn.datasets import make_classification
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    pytest.skip("imbalanced-learn 모듈이 설치되어 있지 않습니다. pip install imbalanced-learn 명령어로 터미널에서 설치하세요.", allow_module_level=True)

# ---------- TODO 영역 ----------
# 학생들은 이 밑의 두 함수를 완성해야 합니다.
# 1. RandomUnderSampler 를 활용한 언더샘플링 적용
# 2. SMOTE 를 활용한 오버샘플링 적용

def apply_random_undersampling(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> tuple:
    """
    주어진 불균형 데이터셋 X, y 특징 집합에 대해 imblearn의 'RandomUnderSampler'를 선언 및 적용하여,
    다운샘플링 처리된 (X_resampled, y_resampled) 튜플을 반환하세요.
    - 반드시 random_state 값을 Sampler 초기화 파라미터로 넘겨주어 재현성을 확보해야 합니다.
    - fit_resample 메서드를 활용하세요.
    """
    # 이 부분을 직접 구현하세요.
    pass

def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> tuple:
    """
    주어진 불균형 데이터셋 X, y 에 대해 imblearn의 'SMOTE' 기법을 적용하여
    오버샘플링(생성) 처리된 (X_resampled, y_resampled) 튜플을 반환하세요.
    - 반드시 random_state 값을 초기화 시 넘겨야 합니다.
    """
    # 이 부분을 직접 구현하세요.
    pass


# ---------- Pytest 검증 영역 ----------
# 학생들이 코딩 후 pytest 를 돌렸을 때 정답 여부를 채점하는 구간

@pytest.fixture
def imbalanced_data():
    # 극단적으로 불균형한 세트 생성 (9:1 비율)
    X, y = make_classification(n_samples=2000, n_features=5, n_redundant=0, 
                               weights=[0.9, 0.1], random_state=123)
    return X, y

def test_apply_random_undersampling(imbalanced_data):
    X, y = imbalanced_data
    minority_class_count = np.sum(y == 1)
    
    assert np.sum(y == 0) > np.sum(y == 1) * 5, "시작 데이터는 불균형해야 합니다."
    
    res_tuple = apply_random_undersampling(X, y)
    assert res_tuple is not None, "None이 반환되었습니다. 언더샘플링 로직을 추가하세요!"
    
    X_res, y_res = res_tuple
    
    # 1:1 로 맞춰졌는지 (y==0 객체 수와 y==1 객체 수가 동일해야함)
    assert np.sum(y_res == 0) == np.sum(y_res == 1), "클래스 결과 비율이 1:1로 추출되지 않았습니다."
    # 언더 스케일 되었으니 소수 클래스 수량만큼으로 전체가 깎여야함
    assert np.sum(y_res == 1) == minority_class_count, "언더샘플링은 소수 클래스의 수는 보존해야 합니다."
    assert len(X_res) == len(y_res), "X와 y 길이는 일치해야 합니다."

def test_apply_smote(imbalanced_data):
    X, y = imbalanced_data
    majority_class_count = np.sum(y == 0)
    minority_class_count = np.sum(y == 1)
    
    res_tuple = apply_smote(X, y)
    assert res_tuple is not None, "None이 반환되었습니다. SMOTE 코드를 구성하세요!"
    
    X_res, y_res = res_tuple
    
    # 다수 클래스와 동일하게 증축되었는가
    assert np.sum(y_res == 0) == np.sum(y_res == 1), "SMOTE 이후 클래스 비율이 1:1이 아닙니다."
    # 다수 클래스는 기존 양 보존
    assert np.sum(y_res == 0) == majority_class_count, "다수 클래스의 데이터 수는 손실되면 안 됩니다."
    # 소수 클래스는 증식
    assert np.sum(y_res == 1) > minority_class_count, "소수 클래스의 데이터가 생성되지 않았습니다."
    assert len(X_res) == len(y_res), "X 데이터와 y 데이터의 길이가 일치해야 합니다."
