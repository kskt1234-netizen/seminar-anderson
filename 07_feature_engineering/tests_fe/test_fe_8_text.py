import pytest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


@pytest.fixture
def sample_corpus():
    return [
        "I love machine learning and deep learning",
        "machine learning is great for data science",
        "deep learning is a subset of machine learning",
        "data science requires statistics and programming",
        "I love programming and data science",
    ]


@pytest.fixture
def sample_df(sample_corpus):
    return pd.DataFrame({'text': sample_corpus})


# =============================================================================
# 문제 1: 기본 텍스트 통계 피처
# =============================================================================

def extract_text_stats(df: pd.DataFrame, col: str = 'text') -> pd.DataFrame:
    """
    df[col]에서 다음 텍스트 통계 피처를 추출하여 df에 추가하고 반환하세요.
    - 'text_length':   전체 문자 수 (공백 포함)
    - 'word_count':    단어 수 (공백으로 구분)
    - 'unique_words':  고유 단어 수 (소문자 변환 후)
    - 'avg_word_len':  평균 단어 길이 (문자 수 / 단어 수)
    - 'digit_count':   숫자 문자 수
    """
    df = df.copy()
    # --- 코드를 작성하세요 ---

    # -----------------------
    return df


def test_text_1_text_length(sample_df):
    result = extract_text_stats(sample_df)
    assert result is not None
    assert 'text_length' in result.columns
    # 첫 번째 텍스트의 길이 확인
    expected_len = len(sample_df['text'].iloc[0])
    assert result['text_length'].iloc[0] == expected_len


def test_text_1_word_count(sample_df):
    result = extract_text_stats(sample_df)
    assert 'word_count' in result.columns
    # "I love machine learning and deep learning" → 7 단어
    assert result['word_count'].iloc[0] == 7


def test_text_1_unique_words(sample_df):
    result = extract_text_stats(sample_df)
    assert 'unique_words' in result.columns
    # 첫 번째 문장의 고유 단어 수 (소문자)
    words = set(sample_df['text'].iloc[0].lower().split())
    assert result['unique_words'].iloc[0] == len(words)


def test_text_1_avg_word_len(sample_df):
    result = extract_text_stats(sample_df)
    assert 'avg_word_len' in result.columns
    assert result['avg_word_len'].gt(0).all()


def test_text_1_digit_count(sample_df):
    df_with_digits = pd.DataFrame({'text': ["Hello 2024 world 42", "no digits here"]})
    result = extract_text_stats(df_with_digits)
    assert result['digit_count'].iloc[0] == 6  # 2, 0, 2, 4, 4, 2
    assert result['digit_count'].iloc[1] == 0


# =============================================================================
# 문제 2: CountVectorizer (Bag of Words)
# =============================================================================

def apply_count_vectorizer(corpus: list, max_features: int = 20) -> tuple:
    """
    CountVectorizer(max_features=max_features)를 적용하세요.
    반환값: (X_bow, vectorizer)
    - X_bow: 변환된 밀집 numpy array (toarray() 적용)
    - vectorizer: 학습된 CountVectorizer 객체
    """
    X_bow = None
    vectorizer = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return X_bow, vectorizer


def test_text_2_bow_shape(sample_corpus):
    X_bow, cv = apply_count_vectorizer(sample_corpus, max_features=20)
    assert X_bow is not None
    # 행 수 = 문서 수
    assert X_bow.shape[0] == len(sample_corpus)
    # 열 수 ≤ max_features
    assert X_bow.shape[1] <= 20


def test_text_2_bow_non_negative(sample_corpus):
    X_bow, _ = apply_count_vectorizer(sample_corpus)
    assert X_bow.min() >= 0, "BoW 값은 음수일 수 없습니다."


def test_text_2_bow_machine_learning(sample_corpus):
    """'machine'과 'learning'은 모든 문서에 등장하므로 빈도가 0보다 커야 합니다."""
    X_bow, cv = apply_count_vectorizer(sample_corpus, max_features=50)
    feature_names = list(cv.get_feature_names_out())
    if 'machine' in feature_names:
        machine_idx = feature_names.index('machine')
        # 'machine'이 등장하는 문서의 카운트 > 0
        assert X_bow[:, machine_idx].sum() > 0


def test_text_2_bow_feature_names(sample_corpus):
    _, cv = apply_count_vectorizer(sample_corpus)
    features = cv.get_feature_names_out()
    assert len(features) > 0
    assert all(isinstance(f, str) for f in features)


# =============================================================================
# 문제 3: TfidfVectorizer
# =============================================================================

def apply_tfidf(corpus: list, max_features: int = 20) -> tuple:
    """
    TfidfVectorizer(max_features=max_features)를 적용하세요.
    반환값: (X_tfidf, vectorizer)
    - X_tfidf: 변환된 밀집 numpy array (toarray() 적용)
    - vectorizer: 학습된 TfidfVectorizer 객체
    """
    X_tfidf = None
    vectorizer = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return X_tfidf, vectorizer


def test_text_3_tfidf_shape(sample_corpus):
    X_tfidf, _ = apply_tfidf(sample_corpus, max_features=20)
    assert X_tfidf is not None
    assert X_tfidf.shape[0] == len(sample_corpus)
    assert X_tfidf.shape[1] <= 20


def test_text_3_tfidf_range(sample_corpus):
    X_tfidf, _ = apply_tfidf(sample_corpus)
    # TF-IDF 값은 0 이상이어야 합니다
    assert X_tfidf.min() >= 0


def test_text_3_tfidf_normalized(sample_corpus):
    """기본 TfidfVectorizer는 각 행을 L2 정규화합니다."""
    X_tfidf, _ = apply_tfidf(sample_corpus, max_features=50)
    row_norms = np.linalg.norm(X_tfidf, axis=1)
    # 문서가 비어있지 않으면 L2 노름은 1에 가까워야 합니다
    np.testing.assert_allclose(row_norms, np.ones(len(sample_corpus)), atol=1e-6)


def test_text_3_tfidf_rare_words_high_score(sample_corpus):
    """드문 단어는 흔한 단어보다 TF-IDF 점수가 높아야 합니다."""
    X_tfidf, cv = apply_tfidf(sample_corpus, max_features=50)
    feature_names = list(cv.get_feature_names_out())
    # 'machine'은 모든 문서에 등장하므로 IDF가 낮음
    # 특정 문서에만 등장하는 단어는 IDF가 높음
    if 'machine' in feature_names:
        machine_idx = feature_names.index('machine')
        machine_idf = cv.idf_[machine_idx]
        # 최대 IDF는 machine의 IDF보다 커야 합니다
        assert cv.idf_.max() >= machine_idf


# =============================================================================
# 문제 4: N-gram TF-IDF
# =============================================================================

def apply_ngram_tfidf(corpus: list, ngram_range: tuple = (1, 2), max_features: int = 30) -> tuple:
    """
    TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)를 적용하세요.
    반환값: (X_tfidf, vectorizer)
    """
    X_tfidf = None
    vectorizer = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    return X_tfidf, vectorizer


def test_text_4_bigram_features(sample_corpus):
    X_tfidf, cv = apply_ngram_tfidf(sample_corpus, ngram_range=(1, 2))
    assert X_tfidf is not None
    feature_names = list(cv.get_feature_names_out())
    # bigram이 포함되어야 합니다 (공백이 있는 피처 이름)
    has_bigram = any(' ' in f for f in feature_names)
    assert has_bigram, "bigram 피처가 생성되지 않았습니다."


def test_text_4_ngram_shape(sample_corpus):
    X_tfidf, cv = apply_ngram_tfidf(sample_corpus, ngram_range=(1, 2), max_features=30)
    assert X_tfidf.shape[0] == len(sample_corpus)
    assert X_tfidf.shape[1] <= 30
