import pytest
import duckdb
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


@pytest.fixture
def conn():
    """인메모리 DuckDB 연결 및 Iris 데이터셋을 테이블로 등록합니다."""
    con = duckdb.connect()
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['target'].map(species_map)
    con.register('iris', df)
    yield con
    con.close()


@pytest.fixture
def sales_conn():
    """매출 데이터를 시뮬레이션하는 DuckDB 연결을 생성합니다."""
    con = duckdb.connect()
    np.random.seed(42)
    n = 100
    sales_df = pd.DataFrame({
        'date':       pd.date_range('2024-01-01', periods=n, freq='D'),
        'product_id': np.random.choice(['A', 'B', 'C'], size=n),
        'revenue':    np.random.randint(100, 1000, size=n).astype(float),
        'quantity':   np.random.randint(1, 50, size=n),
    })
    con.register('sales', sales_df)
    yield con
    con.close()


# =============================================================================
# 문제 1: ROW_NUMBER() 윈도우 함수
# =============================================================================

def test_intermediate_1_row_number(conn):
    """
    문제 1: 각 species 내에서 sepal_length 기준 내림차순으로 ROW_NUMBER()를 부여하세요.
    컬럼: species, sepal_length, row_num
    결과를 `result`에 할당하세요.
    힌트: ROW_NUMBER() OVER (PARTITION BY species ORDER BY sepal_length DESC)
    """
    result = None
    # --- SQL을 작성하세요 ---

    # -----------------------

    assert result is not None
    assert 'row_num' in result.columns
    # 각 species별로 row_num이 1부터 시작해야 합니다
    min_row_num = result.groupby('species')['row_num'].min()
    assert all(min_row_num == 1)
    # 각 species는 50개 행이므로 max row_num은 50
    max_row_num = result.groupby('species')['row_num'].max()
    assert all(max_row_num == 50)


# =============================================================================
# 문제 2: QUALIFY로 각 그룹 상위 N개 추출
# =============================================================================

def test_intermediate_2_qualify_top_n(conn):
    """
    문제 2: 각 species에서 sepal_length가 가장 큰 상위 3개 행만 추출하세요.
    QUALIFY 절을 사용하세요.
    결과를 `result`에 할당하세요.
    힌트: QUALIFY ROW_NUMBER() OVER (...) <= 3
    """
    result = None
    # --- SQL을 작성하세요 ---

    # -----------------------

    assert result is not None
    # 3개 species × 3개 = 9개 행
    assert len(result) == 9
    # 각 species별로 정확히 3개
    counts_per_species = result.groupby('species').size()
    assert all(counts_per_species == 3)


# =============================================================================
# 문제 3: RANK() vs DENSE_RANK() 비교
# =============================================================================

def test_intermediate_3_rank_vs_dense_rank(conn):
    """
    문제 3: setosa species의 sepal_width에 대해
    RANK()와 DENSE_RANK()를 동시에 계산하세요.
    컬럼: sepal_width, rank_val, dense_rank_val
    결과를 sepal_width 오름차순으로 정렬하여 `result`에 할당하세요.
    """
    result = None
    # --- SQL을 작성하세요 ---

    # -----------------------

    assert result is not None
    assert 'rank_val' in result.columns
    assert 'dense_rank_val' in result.columns
    # dense_rank는 항상 rank 이하여야 합니다 (동점 시 빈 순위가 없음)
    assert all(result['dense_rank_val'] <= result['rank_val'])
    # 둘 다 1에서 시작해야 합니다
    assert result['rank_val'].min() == 1
    assert result['dense_rank_val'].min() == 1


# =============================================================================
# 문제 4: LAG / LEAD 윈도우 함수
# =============================================================================

def test_intermediate_4_lag_lead(conn):
    """
    문제 4: versicolor species를 sepal_length 오름차순으로 정렬하고,
    각 행에 대해:
    - prev_sepal: 이전 행의 sepal_length (없으면 NULL)
    - next_sepal: 다음 행의 sepal_length (없으면 NULL)
    - diff_from_prev: 현재 sepal_length - 이전 sepal_length
    를 계산하세요. 결과를 `result`에 할당하세요.
    """
    result = None
    # --- SQL을 작성하세요 ---

    # -----------------------

    assert result is not None
    assert 'prev_sepal' in result.columns
    assert 'next_sepal' in result.columns
    # 첫 행은 이전 값이 없으므로 NULL
    assert pd.isna(result.iloc[0]['prev_sepal'])
    # 마지막 행은 다음 값이 없으므로 NULL
    assert pd.isna(result.iloc[-1]['next_sepal'])


# =============================================================================
# 문제 5: 누적 합계 (Cumulative SUM)
# =============================================================================

def test_intermediate_5_cumulative_sum(conn):
    """
    문제 5: 전체 데이터를 sepal_length 오름차순으로 정렬하고
    sepal_length의 누적 합계(cumulative_sum)를 계산하세요.
    결과를 `result`에 할당하세요.
    힌트: SUM(...) OVER (ORDER BY sepal_length ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """
    result = None
    # --- SQL을 작성하세요 ---

    # -----------------------

    assert result is not None
    assert 'cumulative_sum' in result.columns
    # 마지막 행의 누적 합계는 전체 합계와 같아야 합니다
    total = conn.execute("SELECT SUM(sepal_length) FROM iris").fetchone()[0]
    assert abs(result['cumulative_sum'].iloc[-1] - total) < 1e-4


# =============================================================================
# 문제 6: CTE (Common Table Expression) 활용
# =============================================================================

def test_intermediate_6_cte(conn):
    """
    문제 6: CTE를 사용하여 다음 두 단계를 구현하세요.
    Step 1 (CTE): species별 평균 sepal_length를 계산 (avg_sepal_length 컬럼)
    Step 2 (메인 쿼리): 원본 iris 테이블과 CTE를 JOIN하여
                       각 행에 해당 species의 평균을 붙이고,
                       개별값 - 평균 = deviation 컬럼을 추가하세요.
    결과를 `result`에 할당하세요.
    """
    result = None
    # --- SQL을 작성하세요 ---

    # -----------------------

    assert result is not None
    assert 'deviation' in result.columns
    assert len(result) == 150
    # 각 species의 deviation 평균은 0에 가까워야 합니다
    avg_deviation_by_species = result.groupby('species')['deviation'].mean()
    for val in avg_deviation_by_species:
        assert abs(val) < 1e-6


# =============================================================================
# 문제 7: 이동 평균 (Rolling Average with Window Frame)
# =============================================================================

def test_intermediate_7_rolling_window(conn):
    """
    문제 7: iris 데이터를 sepal_length 오름차순으로 정렬하고,
    현재 행과 이전 2개 행을 포함한 3행 이동 평균(rolling_avg_3)을 계산하세요.
    결과를 `result`에 할당하세요.
    힌트: AVG(...) OVER (ORDER BY sepal_length ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
    """
    result = None
    # --- SQL을 작성하세요 ---

    # -----------------------

    assert result is not None
    assert 'rolling_avg_3' in result.columns
    # 첫 행의 이동 평균은 자기 자신과 같아야 합니다 (이전 행이 없으므로)
    assert abs(result.iloc[0]['rolling_avg_3'] - result.iloc[0]['sepal_length']) < 1e-6


# =============================================================================
# 문제 8: Parquet 저장 및 읽기
# =============================================================================

def test_intermediate_8_parquet_roundtrip(conn, tmp_path):
    """
    문제 8: iris 테이블을 Parquet 파일로 저장하고,
    그 파일을 다시 DuckDB로 읽어 행 수와 컬럼 수를 확인하세요.

    1. tmp_path / 'iris.parquet' 경로에 iris 테이블을 Parquet으로 저장하세요.
       힌트: COPY (SELECT * FROM iris) TO '경로' (FORMAT PARQUET)
    2. 저장한 파일을 read_parquet()으로 읽어 `result`에 할당하세요.
    """
    parquet_path = str(tmp_path / 'iris.parquet')

    result = None
    # --- 코드를 작성하세요 ---

    # -----------------------

    assert result is not None
    assert len(result) == 150
    assert 'sepal_length' in result.columns
