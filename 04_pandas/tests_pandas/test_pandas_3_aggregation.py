import pandas as pd

def test_pandas_1_simple_groupby():
    """
    문제 1: 기초 그룹화 (Groupby)
    백화점 매출 데이터입니다. 
    'Category'별로 묶어(groupby) 총 매출액('Sales'의 sum)을 구한 시리즈(Series)를 `category_sales`에 할당하세요.
    """
    df = pd.DataFrame({
        'Category': ['Fashion', 'Electronics', 'Fashion', 'Food', 'Electronics'],
        'Sales': [100, 500, 150, 50, 600]
    })
    
    category_sales = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    assert category_sales['Fashion'] == 250
    assert category_sales['Electronics'] == 1100
    assert category_sales['Food'] == 50


def test_pandas_2_multiple_aggregation():
    """
    문제 2: 다중 집계 연산 (Multiple Aggregation)
    이번에는 .agg() 를 사용하여 각 카테고리별로 '최대 매출값(max)'과 '거래 건수(count)'를 동시에 구하세요.
    결과 데이터프레임을 `agg_df` 에 할당하고 컬럼은 .reset_index() 하여 멀티인덱스를 방지하세요.
    """
    df = pd.DataFrame({
        'Category': ['Fashion', 'Electronics', 'Fashion', 'Food', 'Electronics'],
        'Sales': [100, 500, 150, 50, 600]
    })
    
    agg_df = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    # agg_df는 최소한 Category 열 1개, Sales 열에 대해 max, count 2개 등 총 3개의 열이 있어야 합니다
    # 컬럼 이름을 플랫화하지 않았다면 튜플 구조일 수 있으나 검증에서는 값만 확인합니다
    fashion_row = agg_df[agg_df['Category'] == 'Fashion']
    assert len(fashion_row) == 1
    Fashion_max = fashion_row.iloc[0, 1] if isinstance(agg_df.columns[1], str) else fashion_row.iloc[0]['Sales']['max']
    assert Fashion_max == 150


def test_pandas_3_merge_dataframes():
    """
    문제 3: 데이터프레임 병합 (Merge)
    유저 정보 테이블(`df_users`)과 유저 구매 내역 테이블(`df_orders`)이 분리되어 있습니다.
    두 데이터프레임을 '유저 아이디'를 기준으로 LEFT JOIN 하여 `df_merged` 에 할당하세요.
    단, df_users의 아이디 열은 'id' 이고, df_orders의 아이디 열은 'user_id' 입니다. (left_on, right_on 활용)
    """
    df_users = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['John', 'Paul', 'George']
    })
    
    df_orders = pd.DataFrame({
        'order_id': [101, 102],
        'user_id': [1, 3], # Paul(2번유저)은 구매 내역 없음
        'amount': [5000, 7000]
    })
    
    df_merged = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    assert len(df_merged) == 3, "Left Join이므로 3개 행이 유지되어야 합니다."
    # 2번 유저 Paul의 amount 누락값(NaN) 체크
    paul_row = df_merged[df_merged['name'] == 'Paul']
    assert pd.isna(paul_row['amount'].iloc[0])


def test_pandas_4_pivot_table():
    """
    문제 4: 피벗 테이블 (Pivot Table)
    월별 카테고리별 평균 환불 금액을 엑셀의 피벗테이블처럼 정리합니다.
    행(index)을 'Month'로, 열(columns)을 'Category'로 설정하고, 
    값(values)은 'Refund'의 평균(mean)으로 채운 피벗 테이블을 `pivot_df`에 할당하세요.
    """
    df = pd.DataFrame({
        'Month': ['Jan', 'Jan', 'Feb', 'Feb', 'Jan', 'Feb'],
        'Category': ['A', 'B', 'A', 'B', 'A', 'B'],
        'Refund': [100, 200, 150, 300, 120, 250]
    })
    
    pivot_df = None
    # --- 코드를 작성하세요 ---
    
    # -----------------------

    # Jan, Category A의 평균 환불은 (100+120)/2 = 110
    assert pivot_df.loc['Jan', 'A'] == 110
    # Feb, Category B의 평균 환불은 (300+250)/2 = 275
    assert pivot_df.loc['Feb', 'B'] == 275
