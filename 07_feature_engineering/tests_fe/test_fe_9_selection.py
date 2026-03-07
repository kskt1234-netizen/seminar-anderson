import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def cancer_data():
    bc = load_breast_cancer(as_frame=True)
    X = bc.data
    y = bc.target
    return X, y


@pytest.fixture
def iris_data():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    return X, y


# =============================================================================
# 문제 1: VarianceThreshold — 낮은 분산 피처 제거
# =============================================================================

def remove_low_variance(X: pd.DataFrame, threshold: float = 0.01) -> tuple:
    """
    VarianceThreshold(threshold=threshold)를 사용하여 낮은 분산 피처를 제거하세요.
    반환값: (X_selected, selector)
    - X_selected: 선택된 피처만 남긴 numpy array
    - selector: 학습된 VarianceThreshold 객체
    """
    X_selected = None
    selector = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return X_selected, selector


def test_selection_1_variance_threshold(cancer_data):
    X, y = cancer_data
    X_scaled = StandardScaler().fit_transform(X)
    X_selected, selector = remove_low_variance(pd.DataFrame(X_scaled), threshold=0.01)
    assert X_selected is not None
    # 선택된 피처 수는 원본 이하
    assert X_selected.shape[1] <= X.shape[1]
    assert X_selected.shape[0] == X.shape[0]


# =============================================================================
# 문제 2: SelectKBest — F-통계 기반 피처 선택
# =============================================================================

def select_k_best_f(X, y, k: int = 5) -> tuple:
    """
    SelectKBest(score_func=f_classif, k=k)를 사용하여 상위 k개 피처를 선택하세요.
    반환값: (X_selected, selector)
    - X_selected: 선택된 피처 numpy array
    - selector: 학습된 SelectKBest 객체
    """
    X_selected = None
    selector = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return X_selected, selector


def test_selection_2_kbest_shape(cancer_data):
    X, y = cancer_data
    k = 10
    X_selected, selector = select_k_best_f(X.values, y, k=k)
    assert X_selected is not None
    assert X_selected.shape == (X.shape[0], k)


def test_selection_2_kbest_n_selected(cancer_data):
    X, y = cancer_data
    k = 5
    _, selector = select_k_best_f(X.values, y, k=k)
    selected_mask = selector.get_support()
    assert selected_mask.sum() == k


def test_selection_2_kbest_scores(cancer_data):
    X, y = cancer_data
    _, selector = select_k_best_f(X.values, y, k=5)
    # 점수가 계산되어야 합니다
    assert hasattr(selector, 'scores_')
    assert len(selector.scores_) == X.shape[1]
    assert not np.isnan(selector.scores_).any()


# =============================================================================
# 문제 3: SelectKBest — Mutual Information
# =============================================================================

def select_k_best_mi(X, y, k: int = 5) -> tuple:
    """
    SelectKBest(score_func=mutual_info_classif, k=k)를 사용하여 상위 k개 피처를 선택하세요.
    random_state=42를 사용하세요.
    반환값: (X_selected, selector)
    """
    X_selected = None
    selector = None
    # --- 코드를 작성하세요 ---
    # 힌트: from sklearn.feature_selection import SelectKBest, mutual_info_classif
    # -----------------------
    return X_selected, selector


def test_selection_3_mi_shape(iris_data):
    X, y = iris_data
    k = 2
    X_selected, _ = select_k_best_mi(X.values, y, k=k)
    assert X_selected is not None
    assert X_selected.shape == (X.shape[0], k)


# =============================================================================
# 문제 4: RFE (Recursive Feature Elimination)
# =============================================================================

def apply_rfe(X, y, n_features: int = 5) -> tuple:
    """
    LogisticRegression을 estimator로 사용하여 RFE(n_features_to_select=n_features)를 적용하세요.
    LogisticRegression은 max_iter=1000, random_state=42를 사용하세요.
    반환값: (selected_mask, ranking)
    - selected_mask: 선택된 피처의 boolean array (True = 선택됨)
    - ranking: 각 피처의 순위 array (1 = 가장 중요)
    """
    selected_mask = None
    ranking = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return selected_mask, ranking


def test_selection_4_rfe_mask(iris_data):
    X, y = iris_data
    mask, ranking = apply_rfe(X.values, y, n_features=2)
    assert mask is not None
    assert mask.sum() == 2  # 2개 선택
    assert len(mask) == X.shape[1]


def test_selection_4_rfe_ranking(iris_data):
    X, y = iris_data
    mask, ranking = apply_rfe(X.values, y, n_features=2)
    assert ranking is not None
    # 선택된 피처의 순위는 1이어야 합니다
    assert all(ranking[mask] == 1)
    # 순위는 1부터 n_features까지의 정수
    assert ranking.min() == 1
    assert ranking.max() == X.shape[1] - 2 + 1  # n_features_to_select 제외


# =============================================================================
# 문제 5: 트리 기반 피처 중요도
# =============================================================================

def get_tree_importance(X: pd.DataFrame, y) -> pd.Series:
    """
    RandomForestClassifier(n_estimators=100, random_state=42)로 학습하고
    피처 중요도를 pd.Series로 반환하세요.
    인덱스: 피처 이름, 값: 중요도
    내림차순으로 정렬하세요.
    """
    importance_series = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return importance_series


def test_selection_5_importance_length(cancer_data):
    X, y = cancer_data
    importance = get_tree_importance(X, y)
    assert importance is not None
    assert len(importance) == X.shape[1]


def test_selection_5_importance_sum(cancer_data):
    X, y = cancer_data
    importance = get_tree_importance(X, y)
    # 모든 피처 중요도의 합은 1.0 이어야 합니다
    assert abs(importance.sum() - 1.0) < 1e-6


def test_selection_5_importance_sorted(cancer_data):
    X, y = cancer_data
    importance = get_tree_importance(X, y)
    # 내림차순 정렬 확인
    vals = importance.values
    assert all(vals[i] >= vals[i+1] for i in range(len(vals)-1))


# =============================================================================
# 문제 6: PCA — 차원 축소
# =============================================================================

def apply_pca(X: np.ndarray, n_components) -> tuple:
    """
    1. StandardScaler로 먼저 스케일링하세요.
    2. PCA(n_components=n_components)를 적용하세요.
    반환값: (X_pca, pca)
    - X_pca: 변환된 numpy array
    - pca: 학습된 PCA 객체
    """
    X_pca = None
    pca = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return X_pca, pca


def test_selection_6_pca_shape_fixed(cancer_data):
    X, _ = cancer_data
    n_components = 5
    X_pca, pca = apply_pca(X.values, n_components=n_components)
    assert X_pca is not None
    assert X_pca.shape == (X.shape[0], n_components)


def test_selection_6_pca_explained_variance(cancer_data):
    X, _ = cancer_data
    X_pca, pca = apply_pca(X.values, n_components=10)
    assert hasattr(pca, 'explained_variance_ratio_')
    # 각 분산 비율은 0~1 사이
    assert all(0 <= v <= 1 for v in pca.explained_variance_ratio_)
    # 누적 분산 비율은 증가해야 합니다
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    assert all(cumsum[i] <= cumsum[i+1] for i in range(len(cumsum)-1))


def test_selection_6_pca_95_variance(cancer_data):
    """분산 95% 유지하도록 n_components를 자동 결정"""
    X, _ = cancer_data
    X_pca, pca = apply_pca(X.values, n_components=0.95)
    # 누적 분산 비율이 95% 이상이어야 합니다
    assert pca.explained_variance_ratio_.sum() >= 0.95
    # 차원이 줄어야 합니다
    assert X_pca.shape[1] < X.shape[1]


def test_selection_6_pca_uncorrelated(cancer_data):
    """PCA의 주성분들은 서로 상관이 없어야 합니다 (직교)."""
    X, _ = cancer_data
    X_pca, _ = apply_pca(X.values, n_components=5)
    corr_matrix = np.corrcoef(X_pca.T)
    # 대각선 제외 상관계수는 0에 가까워야 합니다
    off_diag = corr_matrix - np.eye(5)
    assert np.abs(off_diag).max() < 1e-6
