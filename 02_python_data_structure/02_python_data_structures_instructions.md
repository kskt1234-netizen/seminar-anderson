# Python 자료구조 (Data Structures) 완벽 가이드

파이썬의 내장 자료구조와 `collections` 모듈을 완전히 이해하면 알고리즘 효율성과 코드 가독성이 크게 향상됩니다.

---

## 1. 리스트 (List)

**순서가 있고, 변경 가능한(mutable)** 가장 범용적인 자료구조입니다.

```python
# 생성
nums = [3, 1, 4, 1, 5, 9, 2, 6]
mixed = [1, "hello", 3.14, True, None]

# 인덱싱 & 슬라이싱
nums[0]     # 3       ← 정방향
nums[-1]    # 6       ← 역방향
nums[1:4]   # [1, 4, 1]
nums[::2]   # [3, 4, 5, 2]  ← 2칸 간격
nums[::-1]  # [6, 2, 9, 5, 1, 4, 1, 3]  ← 뒤집기
```

### 주요 메서드
```python
nums.append(7)          # 끝에 추가 → [3, 1, 4, ..., 6, 7]
nums.insert(0, 99)      # 특정 위치에 삽입
nums.extend([10, 11])   # 다른 이터러블 이어붙이기 (vs append는 단일 원소)
nums.remove(1)          # 값으로 첫 번째 항목 제거 (없으면 ValueError)
nums.pop()              # 마지막 원소 제거 & 반환
nums.pop(0)             # 인덱스로 제거 & 반환
nums.index(4)           # 값의 인덱스 반환
nums.count(1)           # 값의 등장 횟수
nums.sort()             # 제자리 정렬 (원본 변경)
nums.sort(reverse=True) # 내림차순
nums.reverse()          # 제자리 뒤집기 (원본 변경)
nums.clear()            # 전체 삭제
copy = nums.copy()      # 얕은 복사
```

### 리스트 컴프리헨션
```python
squares  = [x**2 for x in range(10)]
evens    = [x for x in range(20) if x % 2 == 0]
matrix   = [[i * j for j in range(1, 4)] for i in range(1, 4)]
flat     = [x for row in matrix for x in row]   # 중첩 리스트 펼치기
```

---

## 2. 튜플 (Tuple)

**순서가 있고, 변경 불가능한(immutable)** 자료구조입니다. 딕셔너리 키, 함수 반환값 등에 사용됩니다.

```python
# 생성
point = (3, 4)
single = (42,)    # ← 원소 1개 튜플은 콤마 필수!
empty = ()

# 패킹 & 언패킹
x, y = point                  # 언패킹
a, *rest = (1, 2, 3, 4, 5)    # *로 나머지 수집: a=1, rest=[2,3,4,5]
first, *mid, last = range(5)  # first=0, mid=[1,2,3], last=4

# 메서드 (2개만 있음)
point.count(3)   # 값 등장 횟수
point.index(4)   # 값의 인덱스

# 튜플 ↔ 리스트 변환
list(point)      # [3, 4]
tuple([1, 2, 3]) # (1, 2, 3)
```

### namedtuple
```python
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(3, 4)
print(p.x, p.y)    # 3 4
print(p[0], p[1])  # 3 4 ← 인덱스로도 접근 가능
print(p._asdict())  # {'x': 3, 'y': 4}
```

---

## 3. 셋 (Set)

**순서 없고, 중복 없는** 자료구조입니다. 집합 연산과 중복 제거에 탁월합니다.

```python
# 생성
s = {1, 2, 3, 3, 2}  # {1, 2, 3}  ← 중복 자동 제거
empty_set = set()     # {} 는 빈 딕셔너리이므로 set() 사용!

# 추가 / 제거
s.add(4)
s.remove(1)      # 없으면 KeyError
s.discard(99)    # 없어도 에러 없음
s.pop()          # 임의 원소 제거

# 집합 연산
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

a | b             # 합집합 {1,2,3,4,5,6}    (a.union(b))
a & b             # 교집합 {3,4}             (a.intersection(b))
a - b             # 차집합 {1,2}             (a.difference(b))
a ^ b             # 대칭차집합 {1,2,5,6}     (a.symmetric_difference(b))
{1,2}.issubset(a)    # 부분집합 여부
a.issuperset({1,2})  # 상위집합 여부
a.isdisjoint({5,6})  # 교집합이 없으면 True

# 셋 컴프리헨션
unique_lens = {len(word) for word in ["hello", "hi", "hey", "world"]}
```

### frozenset (불변 셋)
```python
fs = frozenset([1, 2, 3])
# 딕셔너리 키나 다른 셋의 원소로 사용 가능
```

---

## 4. 딕셔너리 (Dictionary)

**키-값 쌍**, Python 3.7+에서 삽입 순서가 보장됩니다.

```python
# 생성
d = {'name': 'Alice', 'age': 30, 'city': 'Seoul'}
d = dict(name='Alice', age=30)
d = dict(zip(['a', 'b', 'c'], [1, 2, 3]))

# 접근
d['name']          # 'Alice'  ← 없으면 KeyError
d.get('score', 0)  # 기본값 반환 (없을 때 0)

# 추가 / 수정 / 삭제
d['email'] = 'alice@test.com'
d.update({'age': 31, 'country': 'KR'})   # 여러 키 한번에 갱신
del d['city']
popped = d.pop('email', None)   # 제거 & 반환, 기본값 지정 가능

# 순회
for key in d:           pass   # 키
for val in d.values():  pass   # 값
for k, v in d.items():  pass   # 키-값 쌍

# 유용한 메서드
d.keys()              # dict_keys 뷰
d.values()            # dict_values 뷰
d.items()             # dict_items 뷰
'name' in d           # 키 존재 여부 (True)
d.setdefault('score', 0)  # 키가 없을 때만 기본값 삽입
```

### 딕셔너리 컴프리헨션
```python
squares = {x: x**2 for x in range(6)}
filtered = {k: v for k, v in d.items() if v is not None}
inverted = {v: k for k, v in d.items()}   # 키-값 뒤집기
```

### `defaultdict`
```python
from collections import defaultdict

# 존재하지 않는 키에 접근해도 에러 없이 기본값 생성
dd = defaultdict(list)
dd['fruits'].append('apple')
dd['fruits'].append('banana')
# {'fruits': ['apple', 'banana']}

word_count = defaultdict(int)
for word in "the quick brown fox the".split():
    word_count[word] += 1
```

### `Counter`
```python
from collections import Counter

c = Counter("aababcabcabc")
# Counter({'a': 5, 'b': 4, 'c': 3})

c.most_common(2)   # [('a', 5), ('b', 4)]
c['a']             # 5
c['z']             # 0 (없는 키도 0 반환)

# 산술 연산
c1 = Counter(a=3, b=2)
c2 = Counter(a=1, b=4)
c1 + c2   # Counter({'b': 6, 'a': 4})
c1 - c2   # Counter({'a': 2})  ← 음수는 제거
```

---

## 5. 문자열 (String)

**불변(immutable)** 시퀀스입니다.

```python
s = "  Hello, World!  "

# 검색 & 판단
s.find("World")     # 9  (없으면 -1)
s.index("World")    # 9  (없으면 ValueError)
"Hello" in s        # True
s.startswith("  H") # True
s.endswith("!  ")   # True
s.count("l")        # 3

# 변환
s.strip()           # "Hello, World!"  (양쪽 공백 제거)
s.lstrip()          # "Hello, World!  "
s.upper()           # "  HELLO, WORLD!  "
s.lower()           # "  hello, world!  "
s.replace("World", "Python")  # "  Hello, Python!  "
s.split(", ")       # ['  Hello', 'World!  ']
", ".join(["a", "b", "c"])  # "a, b, c"

# 포매팅
name, age = "Alice", 30
f"이름: {name}, 나이: {age}"               # f-string (권장)
"이름: {}, 나이: {}".format(name, age)     # .format()
f"{3.14159:.2f}"    # "3.14"  ← 소수점 2자리
f"{1000000:,}"      # "1,000,000"  ← 천 단위 콤마
f"{'left':<10}"     # "left      "  ← 왼쪽 정렬, 너비 10

# 문자 검사
"abc123".isdigit()   # False
"123".isdigit()      # True
"abc".isalpha()      # True
"abc123".isalnum()   # True
```

---

## 6. 스택 (Stack) — LIFO

리스트를 스택으로 사용합니다. `append()`로 push, `pop()`으로 pop.

```python
stack = []
stack.append(1)   # push
stack.append(2)
stack.append(3)
top = stack.pop()  # pop → 3 (마지막 삽입이 먼저 나옴)

print(stack[-1])   # 최상단 확인 (pop하지 않고)
```

---

## 7. 큐 (Queue) — FIFO

`collections.deque`를 사용합니다. 리스트의 `pop(0)`은 O(n)이지만 `deque.popleft()`는 O(1)입니다.

```python
from collections import deque

q = deque()
q.append(1)      # 오른쪽에 추가 (enqueue)
q.append(2)
q.append(3)
front = q.popleft()  # 왼쪽에서 제거 (dequeue) → 1

# 양방향 연산
q.appendleft(0)   # 왼쪽에 추가
q.pop()           # 오른쪽에서 제거

# 크기 제한 큐 (최근 N개 유지)
recent = deque(maxlen=3)
for i in range(5):
    recent.append(i)
# deque([2, 3, 4], maxlen=3)  ← 자동으로 오래된 것 제거
```

---

## 8. 힙 (Heap) — 우선순위 큐

`heapq` 모듈은 **최소 힙(Min-Heap)**을 구현합니다. 항상 가장 작은 값이 `heap[0]`에 위치합니다.

```python
import heapq

heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 1)
heapq.heappush(heap, 3)

heapq.heappop(heap)   # 1  ← 가장 작은 값 꺼내기
heap[0]               # 현재 최솟값 확인 (꺼내지 않음)

# 리스트를 힙으로 변환
nums = [3, 1, 4, 1, 5, 9]
heapq.heapify(nums)    # O(n), in-place

# 최대/최소 N개
heapq.nlargest(3, nums)   # [9, 5, 4]
heapq.nsmallest(3, nums)  # [1, 1, 3]

# 최대 힙: 값에 음수를 붙여서 사용
heapq.heappush(heap, -10)   # -10이 최솟값 → pop하면 -(-10)=10 최대값
```

---

## 9. 컴프리헨션 & 제너레이터 (Comprehensions & Generators)

```python
# 리스트 컴프리헨션 (메모리에 전부 생성)
squares = [x**2 for x in range(100)]

# 딕셔너리 컴프리헨션
sq_dict = {x: x**2 for x in range(5)}

# 셋 컴프리헨션
sq_set = {x**2 for x in range(-5, 6)}  # 중복 제거됨

# 제너레이터 표현식 (lazy evaluation, 메모리 효율적)
gen = (x**2 for x in range(100))
next(gen)      # 0
next(gen)      # 1
sum(gen)       # 나머지 합산 (한 번만 순회 가능!)

# 제너레이터 함수
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib = fibonacci()
[next(fib) for _ in range(8)]  # [0, 1, 1, 2, 3, 5, 8, 13]
```

---

## 10. 정렬 (Sorting)

```python
nums = [3, 1, 4, 1, 5, 9, 2, 6]

# sorted(): 새 리스트 반환 (원본 유지)
asc  = sorted(nums)
desc = sorted(nums, reverse=True)

# .sort(): 제자리 정렬 (원본 변경, 반환값 None)
nums.sort()

# key 함수로 정렬 기준 지정
words = ["banana", "apple", "cherry", "fig"]
sorted(words, key=len)              # 길이 순: ['fig', 'apple', ...]
sorted(words, key=str.lower)        # 대소문자 무시
sorted(words, key=lambda w: w[-1]) # 마지막 글자 기준

# 복잡한 객체 정렬
people = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
sorted(people, key=lambda p: p["age"])  # 나이 순

from operator import itemgetter, attrgetter
sorted(people, key=itemgetter("age"))   # 딕셔너리에는 itemgetter
sorted(people, key=itemgetter("age", "name"))  # 다중 키 정렬
```

---

## 11. 연습문제 (Pytest) 🚀

- **tests_data_structures/test_data_structures.py**: 각 자료구조의 동작을 구현하고 테스트를 통과하세요.
