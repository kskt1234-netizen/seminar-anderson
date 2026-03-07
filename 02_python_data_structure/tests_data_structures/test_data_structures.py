import pytest
import heapq
from collections import defaultdict, Counter, deque, namedtuple


# =============================================================================
# 섹션 1: 리스트 (List)
# =============================================================================

def test_list_1_creation_and_indexing():
    """
    문제 1: 1부터 10까지의 정수 리스트를 만들고 (list comprehension 사용),
    첫 번째 원소, 마지막 원소, 인덱스 2~4 슬라이스를 각 변수에 저장하세요.
    """
    nums = None            # 1~10 리스트
    first = None           # 첫 번째 원소
    last = None            # 마지막 원소
    middle_slice = None    # 인덱스 2~4 (포함)
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert nums == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert first == 1
    assert last == 10
    assert middle_slice == [3, 4, 5]


def test_list_2_methods():
    """
    문제 2: 아래 리스트에 다음 작업을 순서대로 수행하세요.
    1. 끝에 99 추가
    2. 인덱스 0 위치에 0 삽입
    3. 값 3을 제거
    4. 마지막 원소를 꺼내어 popped에 저장
    """
    nums = [1, 2, 3, 4, 5]
    popped = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert 0 in nums
    assert 99 not in nums       # pop으로 제거됨
    assert popped == 99
    assert 3 not in nums
    assert nums[0] == 0


def test_list_3_slicing_reverse():
    """
    문제 3: 슬라이싱만 사용하여 리스트를 뒤집으세요. (reverse() 메서드 사용 금지)
    """
    nums = [1, 2, 3, 4, 5]
    reversed_nums = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert reversed_nums == [5, 4, 3, 2, 1]
    assert nums == [1, 2, 3, 4, 5]   # 원본은 변경되지 않아야 합니다


def test_list_4_comprehension_filter():
    """
    문제 4: 0~29의 정수 중 3의 배수이면서 짝수인 수를 리스트 컴프리헨션으로 구하세요.
    """
    result = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert result == [0, 6, 12, 18, 24]


def test_list_5_nested_flatten():
    """
    문제 5: 중첩 리스트를 한 줄 리스트 컴프리헨션으로 평탄화(flatten)하세요.
    """
    nested = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    flat = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert flat == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_list_6_sort_with_key():
    """
    문제 6: 단어 리스트를 길이 오름차순으로 정렬하되,
    같은 길이면 알파벳순으로 정렬하세요. (sorted + key 또는 tuple 키 사용)
    """
    words = ["banana", "apple", "fig", "cherry", "date", "kiwi"]
    sorted_words = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert sorted_words == ["fig", "date", "kiwi", "apple", "banana", "cherry"]


def test_list_7_count_and_index():
    """
    문제 7: 리스트에서 값 2의 등장 횟수와 첫 번째 인덱스를 구하세요.
    """
    nums = [1, 2, 3, 2, 4, 2, 5]
    count_of_2 = None
    first_index_of_2 = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert count_of_2 == 3
    assert first_index_of_2 == 1


# =============================================================================
# 섹션 2: 튜플 (Tuple)
# =============================================================================

def test_tuple_1_packing_unpacking():
    """
    문제 1: 튜플 (10, 20, 30, 40, 50)을 만들고,
    첫 번째 원소를 first에, 나머지를 rest 리스트에 언패킹하세요. (*rest 사용)
    """
    t = None               # (10, 20, 30, 40, 50) 튜플
    first = None
    rest = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert t == (10, 20, 30, 40, 50)
    assert first == 10
    assert rest == [20, 30, 40, 50]


def test_tuple_2_swap():
    """
    문제 2: 튜플 언패킹을 사용하여 a와 b의 값을 임시 변수 없이 교환하세요.
    """
    a, b = 1, 2
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert a == 2
    assert b == 1


def test_tuple_3_namedtuple():
    """
    문제 3: namedtuple로 'Point3D'를 정의하고 (x, y, z 필드),
    Point3D(1, 2, 3)을 만들어 x, y, z 값과 딕셔너리 변환을 검증하세요.
    """
    Point3D = None   # namedtuple 정의
    p = None         # Point3D(1, 2, 3)
    p_dict = None    # _asdict() 결과
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert p.x == 1
    assert p.y == 2
    assert p.z == 3
    assert p[0] == 1   # 인덱스로도 접근 가능
    assert p_dict == {'x': 1, 'y': 2, 'z': 3}


def test_tuple_4_immutability():
    """
    문제 4: 튜플은 변경 불가능합니다. 튜플 원소를 변경하려 하면 TypeError가 발생해야 합니다.
    """
    t = (1, 2, 3)
    with pytest.raises(TypeError):
        t[0] = 99  # type: ignore


def test_tuple_5_as_dict_key():
    """
    문제 5: 튜플은 딕셔너리 키로 사용 가능합니다.
    좌표 → 이름 딕셔너리를 만드세요.
    """
    locations = None   # {(0, 0): 'origin', (1, 0): 'right', (0, 1): 'up'}
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert locations[(0, 0)] == 'origin'
    assert locations[(1, 0)] == 'right'
    assert locations[(0, 1)] == 'up'


# =============================================================================
# 섹션 3: 셋 (Set)
# =============================================================================

def test_set_1_deduplication():
    """
    문제 1: 중복이 있는 리스트에서 set을 사용하여 고유 원소를 구하고,
    다시 정렬된 리스트로 변환하세요.
    """
    nums = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    unique_sorted = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert unique_sorted == [1, 2, 3, 4, 5, 6, 9]


def test_set_2_operations():
    """
    문제 2: 두 셋에 대해 합집합, 교집합, 차집합, 대칭차집합을 구하세요.
    """
    a = {1, 2, 3, 4, 5}
    b = {4, 5, 6, 7, 8}
    union = None           # 합집합
    intersection = None    # 교집합
    diff_a_b = None        # a - b (a에만 있는 것)
    sym_diff = None        # 대칭차집합
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert union == {1, 2, 3, 4, 5, 6, 7, 8}
    assert intersection == {4, 5}
    assert diff_a_b == {1, 2, 3}
    assert sym_diff == {1, 2, 3, 6, 7, 8}


def test_set_3_membership():
    """
    문제 3: set을 사용하여 두 리스트의 공통 원소를 빠르게 찾으세요.
    """
    list_a = list(range(0, 1000, 3))  # 3의 배수
    list_b = list(range(0, 1000, 7))  # 7의 배수
    common = None   # 공통 원소를 정렬된 리스트로 반환
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert all(x % 3 == 0 and x % 7 == 0 for x in common)
    assert common == sorted(set(common))   # 정렬 확인


def test_set_4_comprehension():
    """
    문제 4: 문자열 리스트에서 단어 길이의 고유 집합을 셋 컴프리헨션으로 구하세요.
    """
    words = ["hello", "world", "hi", "python", "go", "rust", "java"]
    unique_lengths = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert unique_lengths == {2, 4, 5, 6}


def test_set_5_frozenset_as_key():
    """
    문제 5: frozenset을 딕셔너리 키로 사용하세요.
    {frozenset({1, 2}): 'pair', frozenset({1, 2, 3}): 'triple'} 딕셔너리를 만드세요.
    """
    d = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert d[frozenset({2, 1})] == 'pair'       # 순서 무관
    assert d[frozenset({3, 1, 2})] == 'triple'


# =============================================================================
# 섹션 4: 딕셔너리 (Dictionary)
# =============================================================================

def test_dict_1_basic_operations():
    """
    문제 1: 딕셔너리에서 안전하게 값 가져오기와 갱신을 수행하세요.
    - 'score' 키가 없을 때 get()으로 기본값 0 반환
    - update()로 여러 키를 한번에 갱신
    - 'age' 키를 삭제하되 없어도 에러가 없도록 pop() 사용
    """
    d = {'name': 'Alice', 'age': 30}
    score = None           # get으로 'score' 조회, 기본값 0
    d_updated = None       # update로 {'age': 31, 'city': 'Seoul'} 적용한 딕셔너리
    removed_age = None     # pop으로 'age' 제거, 없으면 None 반환
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert score == 0
    assert d_updated['age'] == 31
    assert d_updated['city'] == 'Seoul'
    assert removed_age == 31
    assert 'age' not in d


def test_dict_2_comprehension():
    """
    문제 2: 딕셔너리 컴프리헨션으로 다음을 구하세요.
    - 1~5의 제곱 딕셔너리: {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
    - 기존 딕셔너리에서 값이 짝수인 항목만 필터링
    """
    squares_dict = None
    original = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 6}
    even_only = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert squares_dict == {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
    assert even_only == {'b': 2, 'd': 4, 'e': 6}


def test_dict_3_invert():
    """
    문제 3: 딕셔너리의 키와 값을 뒤집으세요. (값이 고유하다고 가정)
    """
    original = {'a': 1, 'b': 2, 'c': 3}
    inverted = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert inverted == {1: 'a', 2: 'b', 3: 'c'}


def test_dict_4_merge():
    """
    문제 4: 두 딕셔너리를 병합하세요. 충돌 시 d2의 값이 우선합니다.
    Python 3.9+ 방식({**d1, **d2} 또는 d1 | d2)을 사용하세요.
    """
    d1 = {'a': 1, 'b': 2, 'c': 3}
    d2 = {'b': 99, 'd': 4}
    merged = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert merged == {'a': 1, 'b': 99, 'c': 3, 'd': 4}


def test_dict_5_group_by():
    """
    문제 5: 단어 리스트를 첫 글자를 키로 그룹화하는 딕셔너리를 만드세요.
    결과: {'a': ['apple', 'avocado'], 'b': ['banana'], 'c': ['cherry', 'coconut']}
    """
    words = ['apple', 'banana', 'cherry', 'avocado', 'coconut']
    grouped = None
    # --- 코드를 작성하세요 ---
    # 힌트: defaultdict(list) 활용 권장
    # -----------------------
    assert sorted(grouped['a']) == ['apple', 'avocado']
    assert grouped['b'] == ['banana']
    assert sorted(grouped['c']) == ['cherry', 'coconut']


# =============================================================================
# 섹션 5: Counter
# =============================================================================

def test_counter_1_word_frequency():
    """
    문제 1: 문장의 단어 빈도를 Counter로 구하고, 가장 많이 등장한 단어 2개를 반환하세요.
    """
    sentence = "the quick brown fox jumps over the lazy dog the fox"
    word_counts = None    # Counter 객체
    top_2 = None          # most_common(2) 결과
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert word_counts['the'] == 3
    assert word_counts['fox'] == 2
    assert word_counts['zebra'] == 0   # 없는 단어는 0
    assert top_2[0][0] == 'the'
    assert top_2[0][1] == 3


def test_counter_2_arithmetic():
    """
    문제 2: Counter 산술 연산을 수행하세요.
    """
    c1 = Counter(a=4, b=2, c=0, d=-2)
    c2 = Counter(a=1, b=2, c=3)
    combined = None    # c1 + c2 (양수인 것만)
    diff = None        # c1 - c2 (양수인 것만)
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert combined['a'] == 5
    assert combined['b'] == 4
    assert combined['c'] == 3
    assert 'd' not in combined   # 음수는 제거됨
    assert diff['a'] == 3
    assert 'b' not in diff       # 0은 제거됨
    assert 'c' not in diff       # 음수는 제거됨


def test_counter_3_char_count():
    """
    문제 3: 문자열의 각 문자 빈도를 Counter로 구하고,
    가장 많이 등장한 문자와 그 빈도를 반환하세요.
    """
    s = "mississippi"
    counter = None
    most_common_char = None   # 가장 많이 등장한 문자
    most_common_count = None  # 그 빈도
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert counter['s'] == 4
    assert counter['p'] == 2
    assert most_common_char == 's'
    assert most_common_count == 4


# =============================================================================
# 섹션 6: defaultdict
# =============================================================================

def test_defaultdict_1_list_grouping():
    """
    문제 1: defaultdict(list)를 사용하여 학생들을 학년별로 그룹화하세요.
    """
    students = [
        ('Alice', 3), ('Bob', 2), ('Charlie', 3),
        ('Dave', 1), ('Eve', 2), ('Frank', 1),
    ]
    by_grade = None   # defaultdict(list): {학년: [이름 리스트]}
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert sorted(by_grade[3]) == ['Alice', 'Charlie']
    assert sorted(by_grade[2]) == ['Bob', 'Eve']
    assert sorted(by_grade[1]) == ['Dave', 'Frank']


def test_defaultdict_2_counting():
    """
    문제 2: defaultdict(int)를 사용하여 각 문자의 빈도를 계산하세요.
    (Counter를 사용하지 말고 직접 구현하세요)
    """
    text = "hello world"
    freq = None   # defaultdict(int)
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert freq['l'] == 3
    assert freq['o'] == 2
    assert freq['h'] == 1
    assert freq['z'] == 0   # defaultdict는 없는 키도 기본값 반환


# =============================================================================
# 섹션 7: deque (스택 & 큐)
# =============================================================================

def test_deque_1_queue_fifo():
    """
    문제 1: deque를 FIFO 큐로 사용하세요.
    1, 2, 3을 순서대로 enqueue하고, dequeue한 순서가 [1, 2, 3]인지 확인하세요.
    """
    q = deque()
    dequeued = []
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert dequeued == [1, 2, 3]
    assert len(q) == 0


def test_deque_2_stack_lifo():
    """
    문제 2: deque를 LIFO 스택으로 사용하세요.
    1, 2, 3을 순서대로 push하고, pop한 순서가 [3, 2, 1]인지 확인하세요.
    """
    stack = deque()
    popped = []
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert popped == [3, 2, 1]


def test_deque_3_maxlen():
    """
    문제 3: maxlen=3인 deque를 만들어 최근 3개 값만 유지하는 슬라이딩 윈도우를 구현하세요.
    1~6을 순서대로 append했을 때, 최종 deque는 [4, 5, 6]이어야 합니다.
    """
    window = None   # maxlen=3인 deque
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert list(window) == [4, 5, 6]


def test_deque_4_rotate():
    """
    문제 4: deque.rotate(n)은 오른쪽으로 n칸 회전합니다.
    [1, 2, 3, 4, 5]를 2칸 오른쪽 회전하면 [4, 5, 1, 2, 3]이 됩니다.
    """
    d = deque([1, 2, 3, 4, 5])
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert list(d) == [4, 5, 1, 2, 3]


# =============================================================================
# 섹션 8: 힙 (Heap)
# =============================================================================

def test_heap_1_min_heap():
    """
    문제 1: 리스트 [5, 1, 8, 3, 9, 2]를 힙으로 변환하고,
    원소를 하나씩 꺼내어 정렬된 결과를 만드세요.
    """
    nums = [5, 1, 8, 3, 9, 2]
    sorted_result = []
    # --- 코드를 작성하세요 ---
    # 힌트: heapq.heapify() → 반복 heapq.heappop()
    # -----------------------
    assert sorted_result == [1, 2, 3, 5, 8, 9]


def test_heap_2_k_smallest():
    """
    문제 2: 리스트에서 가장 작은 3개 원소를 heapq.nsmallest()로 구하세요.
    """
    nums = [7, 3, 1, 9, 5, 8, 2, 4, 6]
    smallest_3 = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert smallest_3 == [1, 2, 3]


def test_heap_3_k_largest():
    """
    문제 3: 리스트에서 가장 큰 3개 원소를 heapq.nlargest()로 구하세요.
    """
    nums = [7, 3, 1, 9, 5, 8, 2, 4, 6]
    largest_3 = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert largest_3 == [9, 8, 7]


def test_heap_4_max_heap():
    """
    문제 4: 파이썬 heapq는 최소 힙입니다. 최대 힙을 흉내내세요.
    [3, 1, 4, 1, 5, 9, 2, 6]에서 가장 큰 값부터 꺼내는 순서를 반환하세요.
    힌트: 값에 음수를 붙여 push, pop 후 다시 음수로 전환.
    """
    nums = [3, 1, 4, 1, 5, 9, 2, 6]
    max_order = []   # 가장 큰 값부터 꺼낸 순서
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert max_order == [9, 6, 5, 4, 3, 2, 1, 1]


def test_heap_5_priority_queue():
    """
    문제 5: (우선순위, 데이터) 튜플을 사용하여 우선순위 큐를 구현하세요.
    우선순위가 낮은 숫자일수록 먼저 처리됩니다.
    push: (3, 'low'), (1, 'high'), (2, 'medium')
    pop 순서: 'high' → 'medium' → 'low'
    """
    pq = []
    result = []
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert result == ['high', 'medium', 'low']


# =============================================================================
# 섹션 9: 문자열 (String)
# =============================================================================

def test_string_1_methods():
    """
    문제 1: 문자열 처리 메서드를 사용하세요.
    - strip으로 양쪽 공백 제거
    - upper로 대문자 변환
    - replace로 'World'를 'Python'으로 교체
    - split으로 쉼표 기준 분리
    """
    s = "  Hello, World!  "
    stripped = None
    uppercased = None
    replaced = None
    words = None   # "apple,banana,cherry" 를 split
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert stripped == "Hello, World!"
    assert uppercased == "HELLO, WORLD!"
    assert replaced == "  Hello, Python!  "
    assert words == ["apple", "banana", "cherry"]


def test_string_2_fstring_formatting():
    """
    문제 2: f-string을 사용하여 다음 형식으로 포매팅하세요.
    - name='Alice', score=95.678일 때: "Alice: 95.68점" (소수점 2자리)
    - price=1500000일 때: "₩1,500,000" (천 단위 콤마)
    - text='hi'를 너비 10에 오른쪽 정렬: "        hi"
    """
    name, score = 'Alice', 95.678
    price = 1500000
    text = 'hi'

    result1 = None   # "Alice: 95.68점"
    result2 = None   # "₩1,500,000"
    result3 = None   # "        hi"
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert result1 == "Alice: 95.68점"
    assert result2 == "₩1,500,000"
    assert result3 == "        hi"
    assert len(result3) == 10


def test_string_3_palindrome():
    """
    문제 3: 문자열이 회문(palindrome)인지 확인하는 함수를 구현하세요.
    대소문자 무시, 공백 무시.
    """
    def is_palindrome(s: str) -> bool:
        # --- 코드를 작성하세요 ---
        pass
        # -----------------------

    assert is_palindrome("racecar") is True
    assert is_palindrome("A man a plan a canal Panama") is True
    assert is_palindrome("hello") is False
    assert is_palindrome("Was it a car or a cat I saw") is True


def test_string_4_anagram():
    """
    문제 4: 두 문자열이 애너그램(같은 문자로 구성)인지 확인하세요.
    Counter를 활용하세요.
    """
    def is_anagram(s1: str, s2: str) -> bool:
        # --- 코드를 작성하세요 ---
        pass
        # -----------------------

    assert is_anagram("listen", "silent") is True
    assert is_anagram("hello", "world") is False
    assert is_anagram("triangle", "integral") is True


def test_string_5_join_and_find():
    """
    문제 5: 다음을 구현하세요.
    - 리스트를 '-'로 join
    - 문자열에서 'py'가 등장하는 모든 인덱스를 리스트로 반환 (find + while 루프 사용)
    """
    words = ["hello", "world", "python"]
    joined = None
    # --- 코드를 작성하세요 ---

    # -----------------------
    assert joined == "hello-world-python"

    s = "python is pythonic and py is short for python"
    indices = []
    # --- 코드를 작성하세요 ---
    # 힌트: s.find('py', start) 를 반복 사용
    # -----------------------
    assert indices == [0, 10, 24, 43]


# =============================================================================
# 섹션 10: 제너레이터 (Generator)
# =============================================================================

def test_generator_1_expression():
    """
    문제 1: 제너레이터 표현식으로 1~100의 홀수 제곱합을 구하세요.
    리스트 컴프리헨션이 아닌 제너레이터 표현식 (괄호 사용)으로 구현하세요.
    """
    total = None
    # --- 코드를 작성하세요 ---
    # 힌트: sum(x**2 for x ...)
    # -----------------------
    assert total == 166650


def test_generator_2_function():
    """
    문제 2: yield를 사용하여 무한 피보나치 수열 제너레이터를 만드세요.
    처음 10개 값을 리스트로 반환하세요.
    """
    def fibonacci():
        # --- 코드를 작성하세요 ---
        pass
        # -----------------------

    fib = fibonacci()
    result = [next(fib) for _ in range(10)]
    assert result == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


def test_generator_3_chunked():
    """
    문제 3: 리스트를 n개씩 나누는 제너레이터 함수를 구현하세요.
    chunked([1,2,3,4,5,6,7], 3) → [1,2,3], [4,5,6], [7]
    """
    def chunked(lst: list, n: int):
        # --- 코드를 작성하세요 ---
        pass
        # -----------------------

    result = list(chunked([1, 2, 3, 4, 5, 6, 7], 3))
    assert result == [[1, 2, 3], [4, 5, 6], [7]]


# =============================================================================
# 섹션 11: 종합 문제 (Combined)
# =============================================================================

def test_combined_1_top_k_words():
    """
    종합 문제 1: 텍스트에서 상위 3개 빈도 단어를 (단어, 빈도) 튜플 리스트로 반환하세요.
    Counter와 most_common을 활용하세요.
    """
    text = "to be or not to be that is the question to be is to live"

    def top_k_words(text: str, k: int) -> list:
        # --- 코드를 작성하세요 ---
        pass
        # -----------------------

    result = top_k_words(text, 3)
    assert result[0] == ('to', 4)
    assert result[1] == ('be', 3)
    assert result[2][1] == 2   # 빈도가 2인 단어 중 하나


def test_combined_2_lru_cache_with_deque():
    """
    종합 문제 2: deque와 dict를 사용하여 용량이 3인 LRU 캐시를 구현하세요.
    - get(key): 값 반환, 없으면 -1
    - put(key, value): 저장. 용량 초과 시 가장 오래된 항목 제거
    """
    class LRUCache:
        def __init__(self, capacity: int):
            # --- 코드를 작성하세요 ---
            pass

        def get(self, key: int) -> int:
            # --- 코드를 작성하세요 ---
            pass

        def put(self, key: int, value: int) -> None:
            # --- 코드를 작성하세요 ---
            pass

    cache = LRUCache(3)
    cache.put(1, 10)
    cache.put(2, 20)
    cache.put(3, 30)
    assert cache.get(1) == 10
    cache.put(4, 40)          # 용량 초과 → 가장 오래된 것(2) 제거
    assert cache.get(2) == -1  # 이미 제거됨
    assert cache.get(3) == 30
    assert cache.get(4) == 40


def test_combined_3_word_ladder():
    """
    종합 문제 3: deque를 이용한 BFS로 최단 변환 단어 수를 구하세요.
    한 번에 한 글자씩 바꾸어 begin_word → end_word로 도달하는 최소 단계를 반환하세요.
    (단어 리스트 안의 단어만 경유 가능)
    """
    def word_ladder_length(begin_word: str, end_word: str, word_list: list) -> int:
        """도달 불가능하면 0 반환"""
        # --- 코드를 작성하세요 ---
        # 힌트: BFS, deque, set 활용
        pass
        # -----------------------

    word_list = ["hot", "dot", "dog", "lot", "log", "cog"]
    assert word_ladder_length("hit", "cog", word_list) == 5
    # hit → hot → dot → dog → cog (5단계)

    word_list2 = ["hot", "dot", "dog"]
    assert word_ladder_length("hit", "cog", word_list2) == 0
    # 도달 불가
