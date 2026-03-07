# 피처 엔지니어링 8: 텍스트 피처 (Text Features)

자연어 텍스트를 ML 모델이 처리할 수 있는 숫자 벡터로 변환합니다.

---

## 1. 기본 텍스트 통계 피처

텍스트의 구조적 특성을 수치로 추출합니다.

```python
import pandas as pd

df['text_length']      = df['text'].str.len()
df['word_count']       = df['text'].str.split().str.len()
df['char_count']       = df['text'].str.replace(' ', '').str.len()
df['avg_word_len']     = df['char_count'] / (df['word_count'] + 1e-8)
df['sentence_count']   = df['text'].str.count(r'[.!?]+')
df['digit_count']      = df['text'].str.count(r'\d')
df['upper_count']      = df['text'].str.count(r'[A-Z]')
df['punctuation_count'] = df['text'].str.count(r'[^\w\s]')
df['unique_word_count'] = df['text'].apply(lambda x: len(set(x.lower().split())))
```

---

## 2. BoW (Bag of Words) — CountVectorizer

단어의 등장 횟수를 카운트한 행렬을 생성합니다.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love machine learning",
    "machine learning is great",
    "I love deep learning too"
]

cv = CountVectorizer(
    max_features=100,       # 빈도 상위 100 단어만 사용
    stop_words='english',   # 불용어 제거 (the, is, ...)
    min_df=2,               # 최소 2개 문서에 등장한 단어만
    ngram_range=(1, 1)      # 단어 단독 (unigram)
)
X_bow = cv.fit_transform(corpus)   # 희소 행렬 (sparse matrix)
print(cv.get_feature_names_out())
```

---

## 3. TF-IDF (Term Frequency - Inverse Document Frequency)

단어 빈도와 문서 희귀도를 결합한 가중치입니다. 흔한 단어(the, is)는 낮은 점수, 희귀하고 중요한 단어는 높은 점수를 받습니다.

$$TF\text{-}IDF(t, d) = TF(t, d) \times \log\frac{N}{DF(t)}$$

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),      # 단어 단독 + 2연속 단어 쌍
    stop_words='english',
    sublinear_tf=True,       # TF에 log 적용 (빈도 폭발 방지)
    min_df=2
)
X_tfidf = tfidf.fit_transform(corpus)
print(X_tfidf.shape)   # (문서 수, 피처 수)
```

---

## 4. N-gram

연속된 N개 단어의 조합을 피처로 사용합니다.

```python
# "I love machine" 에서:
# Unigram (1-gram): "I", "love", "machine"
# Bigram  (2-gram): "I love", "love machine"
# Trigram (3-gram): "I love machine"

tfidf_ngram = TfidfVectorizer(ngram_range=(1, 3))
X_ngram = tfidf_ngram.fit_transform(corpus)
```

---

## 5. HashingVectorizer — 어휘 사전 없이 해시

어휘 사전을 메모리에 저장하지 않고 해시 함수로 직접 벡터화합니다. 스트리밍 데이터에 유용합니다.

```python
from sklearn.feature_extraction.text import HashingVectorizer

hv = HashingVectorizer(
    n_features=2**16,   # 해시 공간 크기 (고정)
    alternate_sign=False,
    norm='l2'
)
X_hashed = hv.transform(corpus)
# fit 없이 바로 transform 가능 → 온라인 학습 가능
```

---

## 6. 텍스트 전처리 파이프라인 예시

```python
import re

def preprocess_text(text: str) -> str:
    text = text.lower()                          # 소문자
    text = re.sub(r'<.*?>', '', text)            # HTML 태그 제거
    text = re.sub(r'http\S+', '', text)          # URL 제거
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # 특수문자 제거
    text = re.sub(r'\s+', ' ', text).strip()     # 연속 공백 제거
    return text

df['cleaned'] = df['text'].apply(preprocess_text)
```

---

## 7. 연습문제 (Pytest) 🚀

- **tests_fe/test_fe_8_text.py**: 기본 텍스트 통계 피처, CountVectorizer, TfidfVectorizer를 구현하고 행렬의 형태와 값을 검증하세요.
