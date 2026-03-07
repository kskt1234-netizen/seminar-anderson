# DuckDB 중급 가이드

기초 SQL을 넘어, 실무 데이터 분석에 자주 쓰이는 고급 기법들을 다룹니다.

---

## 1. CTE (Common Table Expression) — `WITH` 절

복잡한 쿼리를 단계별로 분리하여 가독성을 높입니다. 임시 뷰처럼 동작합니다.

```sql
WITH
  -- Step 1: species별 평균 계산
  species_avg AS (
    SELECT
        target,
        AVG(sepal_length) AS avg_sepal,
        AVG(petal_length) AS avg_petal
    FROM iris
    GROUP BY target
  ),
  -- Step 2: 평균 이상인 species만 필터링
  large_species AS (
    SELECT * FROM species_avg WHERE avg_petal > 3.0
  )
-- Step 3: 최종 결과
SELECT * FROM large_species ORDER BY avg_petal DESC;
```

CTE는 중첩 서브쿼리보다 훨씬 읽기 쉽고, 같은 CTE를 여러 번 참조할 수 있습니다.

---

## 2. 윈도우 함수 (Window Functions)

`GROUP BY`와 달리 **행을 유지하면서** 그룹 내 집계를 계산합니다.

```sql
SELECT
    target,
    sepal_length,

    -- ① 순위 함수
    ROW_NUMBER() OVER (PARTITION BY target ORDER BY sepal_length DESC) AS row_num,
    RANK()       OVER (PARTITION BY target ORDER BY sepal_length DESC) AS rank,
    DENSE_RANK() OVER (PARTITION BY target ORDER BY sepal_length DESC) AS dense_rank,

    -- ② 이전/다음 행 값
    LAG(sepal_length,  1, 0) OVER (PARTITION BY target ORDER BY sepal_length) AS prev_val,
    LEAD(sepal_length, 1, 0) OVER (PARTITION BY target ORDER BY sepal_length) AS next_val,

    -- ③ 누적 집계
    SUM(sepal_length) OVER (PARTITION BY target ORDER BY sepal_length) AS cumulative_sum,

    -- ④ 파티션 전체 통계
    AVG(sepal_length) OVER (PARTITION BY target) AS species_avg,
    MIN(sepal_length) OVER (PARTITION BY target) AS species_min,
    MAX(sepal_length) OVER (PARTITION BY target) AS species_max

FROM iris;
```

### ROW_NUMBER vs RANK vs DENSE_RANK

| 점수 | ROW_NUMBER | RANK | DENSE_RANK |
|------|-----------|------|-----------|
| 100  | 1         | 1    | 1         |
| 100  | 2         | 1    | 1         |
| 90   | 3         | 3    | 2         |
| 80   | 4         | 4    | 3         |

- **ROW_NUMBER**: 항상 고유한 번호, 동점이어도 순서대로
- **RANK**: 동점 있으면 같은 순위, 다음 순위는 건너뜀 (1,1,3)
- **DENSE_RANK**: 동점 있으면 같은 순위, 다음 순위는 연속 (1,1,2)

---

## 3. QUALIFY — 윈도우 함수 결과로 행 필터링

윈도우 함수 결과를 바로 WHERE처럼 사용합니다. 서브쿼리 없이 간결하게 작성합니다.

```sql
-- 각 species(target)에서 sepal_length 상위 3개 행만 추출
SELECT *
FROM iris
QUALIFY ROW_NUMBER() OVER (PARTITION BY target ORDER BY sepal_length DESC) <= 3;

-- 각 그룹에서 중앙값 이상인 행만
SELECT *
FROM iris
QUALIFY sepal_length >= AVG(sepal_length) OVER (PARTITION BY target);
```

---

## 4. 이동 집계 (Sliding Window)

`ROWS BETWEEN`으로 윈도우 프레임 범위를 직접 지정합니다.

```sql
SELECT
    sepal_length,
    -- 현재 행 포함 이전 2개 행의 평균 (rolling mean 3)
    AVG(sepal_length) OVER (
        ORDER BY sepal_length
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_avg_3,

    -- 전체 누적 합계
    SUM(sepal_length) OVER (
        ORDER BY sepal_length
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_sum
FROM iris;
```

---

## 5. PIVOT — 행을 열로 변환

```sql
-- DuckDB PIVOT 문법
PIVOT iris
ON target          -- 이 열의 고유 값들이 새 열 이름이 됨
USING AVG(sepal_length) AS avg_sepal, COUNT(*) AS cnt
GROUP BY (SELECT NULL);  -- 전체를 하나의 그룹으로
```

---

## 6. DuckDB에서 Pandas 대체하기

실무에서 Pandas로 하던 복잡한 연산을 DuckDB SQL로 더 빠르게 처리합니다.

```python
import duckdb
import pandas as pd

# Pandas groupby + merge를 SQL로 대체
result = duckdb.sql("""
    WITH species_stats AS (
        SELECT
            target,
            AVG(sepal_length) AS avg_sepal,
            STDDEV(sepal_length) AS std_sepal
        FROM df
        GROUP BY target
    )
    SELECT
        d.*,
        s.avg_sepal,
        -- 표준화 점수 (Z-score) 계산
        (d.sepal_length - s.avg_sepal) / s.std_sepal AS z_score
    FROM df d
    JOIN species_stats s ON d.target = s.target
""").df()
```

---

## 7. Parquet 읽기/쓰기

DuckDB는 Parquet을 가장 효율적으로 처리합니다.

```python
conn = duckdb.connect()

# DataFrame을 Parquet으로 저장
conn.execute("COPY (SELECT * FROM df) TO 'iris.parquet' (FORMAT PARQUET)")

# Parquet 파일을 SQL로 바로 쿼리
result = conn.execute("""
    SELECT target, COUNT(*), AVG(sepal_length)
    FROM read_parquet('iris.parquet')
    GROUP BY target
""").fetchdf()

# 여러 Parquet 파일을 한번에 쿼리
result = conn.execute("SELECT * FROM read_parquet('data/*.parquet')").fetchdf()
```

---

## 8. 연습문제 (Pytest) 🚀

- **test_duckdb_2_intermediate.py**: CTE, 윈도우 함수, QUALIFY, PIVOT 등을 사용하는 쿼리를 작성하세요.
