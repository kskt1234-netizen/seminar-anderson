# DuckDB 기초 가이드

## DuckDB란?

**DuckDB**는 서버 없이 프로세스 내에서 실행되는 **인메모리 OLAP(분석용) SQL 데이터베이스**입니다.

| 특징 | 설명 |
|------|------|
| 서버 불필요 | SQLite처럼 파일 하나 또는 메모리만으로 동작 |
| 컬럼 기반 저장 | 집계 연산이 Pandas보다 훨씬 빠름 |
| Pandas 완벽 연동 | DataFrame을 바로 SQL로 쿼리 |
| 파일 직접 쿼리 | CSV, Parquet, JSON을 로드 없이 쿼리 |
| 표준 SQL 지원 | PostgreSQL 문법과 거의 동일 |

```python
pip install duckdb
```

---

## 1. 연결 및 기본 쿼리

```python
import duckdb

# 인메모리 연결 (가장 기본)
conn = duckdb.connect()

# 파일로 저장하는 연결
conn = duckdb.connect("mydb.duckdb")

# 쿼리 실행 방법들
conn.execute("SELECT 42 AS answer").fetchone()    # (42,)          ← 튜플 1개
conn.execute("SELECT 42 AS answer").fetchall()    # [(42,)]        ← 튜플 리스트
conn.execute("SELECT 42 AS answer").fetchdf()     # Pandas DataFrame ← 가장 많이 씀
```

---

## 2. Pandas DataFrame ↔ DuckDB

DuckDB의 가장 강력한 기능입니다. Pandas DataFrame을 **그대로 SQL 테이블처럼** 쿼리합니다.

```python
import duckdb
import pandas as pd

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "score": [85, 92, 78],
    "dept": ["eng", "mkt", "eng"]
})

# 방법 1: register()로 이름 붙이기
conn = duckdb.connect()
conn.register("employees", df)
result = conn.execute("SELECT * FROM employees WHERE score > 80").fetchdf()

# 방법 2: Python 변수명을 SQL에서 직접 참조 (빠른 탐색용)
result = duckdb.query("SELECT * FROM df WHERE score > 80").df()

# 방법 3: SQL → Pandas (가장 간결)
result = duckdb.sql("SELECT dept, AVG(score) FROM df GROUP BY dept").df()
```

---

## 3. 기본 SQL 문법

### SELECT, WHERE, ORDER BY, LIMIT
```sql
-- 전체 조회
SELECT * FROM employees LIMIT 5;

-- 특정 컬럼만
SELECT name, score FROM employees;

-- 조건 필터링
SELECT * FROM employees
WHERE score >= 85 AND dept = 'eng';

-- 정렬
SELECT * FROM employees ORDER BY score DESC;
```

### 집계 함수 & GROUP BY
```sql
SELECT
    dept,
    COUNT(*)        AS headcount,
    AVG(score)      AS avg_score,
    MAX(score)      AS max_score,
    MIN(score)      AS min_score,
    SUM(score)      AS total_score
FROM employees
GROUP BY dept
HAVING AVG(score) > 80   -- 그룹 조건 (WHERE의 집계 버전)
ORDER BY avg_score DESC;
```

### JOIN
```sql
-- 두 테이블 연결
SELECT e.name, e.score, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept = d.dept_id;
```

---

## 4. CSV / Parquet 파일 직접 쿼리

DuckDB의 핵심 장점 중 하나입니다. 파일을 메모리에 전부 올리지 않고도 SQL로 바로 쿼리합니다.

```python
# CSV 직접 쿼리 (pandas의 read_csv 없이!)
result = conn.execute("""
    SELECT dept, AVG(score)
    FROM read_csv_auto('data.csv')
    GROUP BY dept
""").fetchdf()

# Parquet 직접 쿼리
result = conn.execute("""
    SELECT * FROM read_parquet('data.parquet') LIMIT 100
""").fetchdf()

# DuckDB → Parquet 저장
conn.execute("COPY (SELECT * FROM employees) TO 'output.parquet' (FORMAT PARQUET)")
```

---

## 5. Pandas vs DuckDB 비교

```python
import duckdb
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']

# ─────────────── Pandas 방식 ───────────────
result_pd = (
    df.groupby('target')
    .agg(avg_sepal=('sepal_length', 'mean'),
         max_petal=('petal_length', 'max'),
         count=('target', 'count'))
    .reset_index()
)

# ─────────────── DuckDB 방식 ───────────────
result_ddb = duckdb.sql("""
    SELECT
        target,
        AVG(sepal_length) AS avg_sepal,
        MAX(petal_length) AS max_petal,
        COUNT(*)          AS count
    FROM df
    GROUP BY target
    ORDER BY target
""").df()
```

DuckDB는 데이터가 수백만 행 이상일 때 Pandas보다 월등히 빠릅니다.

---

## 6. 연습문제 (Pytest) 🚀

- **test_duckdb_1_basics.py**: Pandas DataFrame을 DuckDB로 등록하고 다양한 SQL 쿼리를 작성하세요.
