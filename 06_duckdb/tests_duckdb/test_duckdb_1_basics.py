import pytest
import duckdb
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer


@pytest.fixture
def conn():
    """인메모리 DuckDB 연결 및 Iris 데이터셋을 테이블로 등록합니다."""
    con = duckdb.connect()

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
    # target 숫자를 종 이름으로 변환
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['target'].map(species_map)

    con.register('iris', df)
    yield con
    con.close()


@pytest.fixture
def cancer_conn():
    """인메모리 DuckDB 연결 및 Breast Cancer 데이터셋을 테이블로 등록합니다."""
    con = duckdb.connect()
    bc = load_breast_cancer(as_frame=True)
    df = bc.frame.copy()
    con.register('cancer', df)
    yield con
    con.close()


# =============================================================================
# 문제 1: 전체 행 조회 및 LIMIT
# =============================================================================

def test_basics_1_select_all_limit(conn):
    """
    문제 1: iris 테이블에서 모든 컬럼을 조회하되, 상위 10개 행만 가져오세요.
    결과를 `result`에 할당하세요.
    """
    result = None
    # --- SQL을 작성하세요 ---
    sql = ""  # 여기에 SQL 문자열을 작성하세요
    # result = conn.execute(sql).fetchdf()
    # -----------------------

    assert result is not None, "result가 None입니다. SQL을 실행하세요."
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10
    assert 'sepal_length' in result.columns


# =============================================================================
# 문제 2: 특정 컬럼 선택
# =============================================================================

def test_basics_2_select_columns(conn):
    """
    문제 2: iris 테이블에서 sepal_length, petal_length, species 세 컬럼만 조회하세요.
    결과를 `result`에 할당하세요.
    """
    result = None
    # --- SQL을 작성하세요 ---

    # -----------------------

    assert result is not None
    assert list(result.columns) == ['sepal_length', 'petal_length', 'species']
    assert len(result) == 150


# =============================================================================
# 문제 3: WHERE 조건 필터링
# =============================================================================

def test_basics_3_where_filter(conn):
    """
    문제 3: sepal_length가 6.0보다 크고 species가 'virginica'인 행만 조회하세요.
    결과를 `result`에 할당하세요.
    """
    result = None
    # --- SQL을 작성하세요 ---

    # -----------------------

    assert result is not None
    assert all(result['sepal_length'] > 6.0)
    assert all(result['species'] == 'virginica')


# =============================================================================
# 문제 4: ORDER BY 정렬
# =============================================================================

def test_basics_4_order_by(conn):
    """
    문제 4: petal_length 기준으로 내림차순 정렬하고 상위 5개 행을 조회하세요.
    결과를 `result`에 할당하세요.
    """
    result = None
    # --- SQL을 작성하세요 ---

    # -----------------------

    assert result is not None
    assert len(result) == 5
    # 정렬이 올바른지 확인
    petal_lengths = result['petal_length'].tolist()
    assert petal_lengths == sorted(petal_lengths, reverse=True)


# =============================================================================
# 문제 5: GROUP BY + 집계 함수
# =============================================================================

def test_basics_5_group_by_aggregation(conn):
    """
    문제 5: species별로 그룹화하여 다음 통계를 계산하세요.
    - count: 행 수
    - avg_sepal_length: sepal_length의 평균 (소수점 4자리 반올림)
    - max_petal_length: petal_length의 최댓값
    결과를 species 알파벳 오름차순으로 정렬하고 `result`에 할당하세요.
    """
    result = None
    # --- SQL을 작성하세요 ---

    # -----------------------

    assert result is not None
    assert len(result) == 3   # 3개 종
    assert list(result.columns) == ['species', 'count', 'avg_sepal_length', 'max_petal_length']
    assert result['count'].sum() == 150
    # setosa의 max petal_length는 1.9
    setosa_row = result[result['species'] == 'setosa']
    assert float(setosa_row['max_petal_length'].values[0]) == pytest.approx(1.9)


# =============================================================================
# 문제 6: HAVING 절로 그룹 필터링
# =============================================================================

def test_basics_6_having(conn):
    """
    문제 6: species별 평균 sepal_width를 계산하고,
    평균이 3.0 이상인 species만 조회하세요.
    결과를 `result`에 할당하세요.
    """
    result = None
    # --- SQL을 작성하세요 ---

    # -----------------------

    assert result is not None
    assert all(result['avg_sepal_width'] >= 3.0)


# =============================================================================
# 문제 7: pandas DataFrame → DuckDB 직접 쿼리
# =============================================================================

def test_basics_7_query_pandas_df_directly():
    """
    문제 7: 아래 pandas DataFrame `employee_df`를 DuckDB에 등록하지 않고
    duckdb.sql() 또는 duckdb.query()로 직접 쿼리하세요.
    부서별 평균 급여를 계산하여 `result`에 할당하세요.
    힌트: duckdb.sql("SELECT ... FROM employee_df ...").df()
    """
    employee_df = pd.DataFrame({
        'name':   ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve'],
        'dept':   ['eng',   'mkt', 'eng',     'hr',   'mkt'],
        'salary': [90000,   70000, 95000,     65000,  72000]
    })

    result = None
    # --- SQL을 작성하세요 ---

    # -----------------------

    assert result is not None
    assert 'dept' in result.columns
    assert 'avg_salary' in result.columns
    eng_avg = result[result['dept'] == 'eng']['avg_salary'].values[0]
    assert eng_avg == pytest.approx(92500.0)


# =============================================================================
# 문제 8: JOIN
# =============================================================================

def test_basics_8_join():
    """
    문제 8: orders와 customers 두 DataFrame을 DuckDB에 등록하고
    customer_id를 기준으로 LEFT JOIN하여 이름과 주문 금액을 조회하세요.
    결과를 amount 내림차순으로 정렬하여 `result`에 할당하세요.
    """
    customers = pd.DataFrame({
        'id':   [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie']
    })
    orders = pd.DataFrame({
        'order_id':    [101, 102, 103, 104],
        'customer_id': [1,   2,   1,   3],
        'amount':      [500, 300, 700, 200]
    })

    result = None
    con = duckdb.connect()
    # --- 코드를 작성하세요 ---
    # con.register('customers', customers)
    # con.register('orders', orders)
    # sql = "..."
    # result = con.execute(sql).fetchdf()
    # -----------------------
    con.close()

    assert result is not None
    assert 'name' in result.columns
    assert 'amount' in result.columns
    assert len(result) == 4
    # 내림차순 정렬 확인
    amounts = result['amount'].tolist()
    assert amounts == sorted(amounts, reverse=True)
