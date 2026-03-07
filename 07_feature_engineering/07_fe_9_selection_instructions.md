# 피처 엔지니어링 9: 피처 선택 & 차원 축소 (Feature Selection & Dimensionality Reduction)

불필요한 피처는 노이즈를 증가시키고, 학습을 느리게 하며, 과적합을 유발합니다. 좋은 피처를 골라내거나 차원을 줄이는 것도 핵심 기술입니다.

---

## 1. 분산 기반 필터 (Variance Threshold)

분산이 거의 없는 피처(상수에 가까운 피처)를 제거합니다.

```python
from sklearn.feature_selection import VarianceThreshold

# 분산이 0.1 이하인 피처 제거
selector = VarianceThreshold(threshold=0.1)
X_selected = selector.fit_transform(X)

# 제거된 피처 확인
kept_mask = selector.get_support()
```

---

## 2. 상관계수 기반 필터 (Correlation Filter)

```python
import pandas as pd
import numpy as np

# 타겟과의 상관관계 (높을수록 유용한 피처)
corr_target = df.corr()['target'].abs().sort_values(ascending=False)
print(corr_target.head(10))

# 피처 간 다중공선성 제거 (상관계수 0.9 이상인 피처 쌍에서 하나 제거)
corr_matrix = df.drop('target', axis=1).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
df_filtered = df.drop(columns=to_drop)
```

---

## 3. 단변량 통계 선택 (SelectKBest)

각 피처와 타겟 간의 통계적 관계를 계산하여 상위 K개를 선택합니다.

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression

# 분류 — F-검정 (선형 관계, 빠름)
selector_f = SelectKBest(score_func=f_classif, k=5)
X_sel = selector_f.fit_transform(X, y)

# 분류 — 상호 정보량 (비선형 관계도 포착, 느림)
selector_mi = SelectKBest(score_func=mutual_info_classif, k=5)
X_sel_mi = selector_mi.fit_transform(X, y)

# 회귀 — F-검정
selector_reg = SelectKBest(score_func=f_regression, k=5)

# 선택된 피처 확인
selected_mask = selector_f.get_support()
scores = selector_f.scores_
```

---

## 4. 재귀적 피처 제거 (RFE — Recursive Feature Elimination)

모델을 반복 학습하며 덜 중요한 피처를 하나씩 제거합니다.

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression(max_iter=1000)

# 고정 개수로 선택
rfe = RFE(estimator=estimator, n_features_to_select=5)
rfe.fit(X, y)

selected_mask = rfe.support_   # True: 선택됨
ranking       = rfe.ranking_   # 1: 선택된 피처

# 교차검증으로 최적 피처 수 자동 결정
rfecv = RFECV(estimator=estimator, cv=5, scoring='accuracy')
rfecv.fit(X, y)
print(f"최적 피처 수: {rfecv.n_features_}")
```

---

## 5. 트리 기반 피처 중요도 (Tree Feature Importance)

랜덤포레스트, XGBoost 등 트리 모델은 자동으로 피처 중요도를 계산합니다.

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=feature_names)
top10 = importances.nlargest(10)
print(top10)
```

---

## 6. Permutation Importance

모델 학습 후 피처 값을 무작위로 섞어 성능 저하를 측정합니다. 어떤 모델에도 적용 가능합니다.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,
    random_state=42
)

perm_imp = pd.Series(result.importances_mean, index=feature_names)
print(perm_imp.sort_values(ascending=False).head(10))
```

---

## 7. PCA (Principal Component Analysis) — 차원 축소

피처 간 분산을 최대로 보존하면서 차원을 줄입니다. 피처 선택이 아닌 **피처 변환**입니다.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# PCA 전 반드시 스케일링!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 고정 차원으로 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)         # 각 PC의 분산 설명 비율
print(pca.explained_variance_ratio_.cumsum()) # 누적 분산 설명 비율

# 분산 95% 유지하도록 자동으로 차원 결정
pca_auto = PCA(n_components=0.95)
X_pca_auto = pca_auto.fit_transform(X_scaled)
print(f"선택된 PC 수: {pca_auto.n_components_}")
```

---

## 8. 피처 선택 방법 비교

| 방법 | 속도 | 모델 필요 | 비선형 포착 | 특징 |
|------|------|---------|-----------|------|
| VarianceThreshold | 매우 빠름 | 없음 | 불가 | 상수 피처 제거 |
| Correlation | 빠름 | 없음 | 불가 | 다중공선성 제거 |
| SelectKBest (F) | 빠름 | 없음 | 불가 | 선형 관계 가정 |
| SelectKBest (MI) | 보통 | 없음 | 가능 | 비선형도 탐지 |
| RFE | 느림 | 필요 | 모델 의존 | 반복적, 정교함 |
| Tree Importance | 빠름 | 필요 | 가능 | 트리 기반 |
| Permutation | 보통 | 필요 | 가능 | 모델 무관 |
| PCA | 빠름 | 없음 | 불가 | 새 축 생성 |

---

## 9. 연습문제 (Pytest) 🚀

- **tests_fe/test_fe_9_selection.py**: SelectKBest, RFE, RandomForest 중요도, PCA를 적용하고 선택된 피처 수와 차원 축소 결과를 검증하세요.
