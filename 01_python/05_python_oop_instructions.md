# Python OOP (객체지향 프로그래밍) 실습 가이드

파이썬의 OOP는 데이터(속성)와 동작(메서드)을 하나의 **클래스**로 묶어 코드의 재사용성과 유지보수성을 극적으로 높여줍니다.

---

## 1. 클래스(Class)와 객체(Object)

클래스는 **설계도**, 객체(인스턴스)는 그 설계도로 찍어낸 **실체**입니다.

```python
class Dog:
    # __init__: 객체 생성 시 자동 호출되는 "생성자"
    def __init__(self, name: str, age: int):
        self.name = name  # 인스턴스 속성 (Attribute)
        self.age = age

    def bark(self) -> str:  # 인스턴스 메서드
        return f"{self.name} says: Woof!"

my_dog = Dog("Rex", 3)
print(my_dog.bark())  # "Rex says: Woof!"
print(my_dog.name)    # "Rex"
```

---

## 2. 메서드의 세 가지 종류

```python
class Counter:
    total_count = 0  # 클래스 변수 (모든 인스턴스가 공유)

    def __init__(self):
        self.count = 0  # 인스턴스 변수 (각 인스턴스 고유)

    def increment(self):           # 인스턴스 메서드: 첫 인자가 self
        self.count += 1
        Counter.total_count += 1

    @classmethod
    def get_total(cls) -> int:     # 클래스 메서드: 첫 인자가 cls
        return cls.total_count

    @staticmethod
    def is_valid(n: int) -> bool:  # 정적 메서드: self/cls 없음, 유틸리티 함수
        return n >= 0

c1, c2 = Counter(), Counter()
c1.increment()
c1.increment()
c2.increment()
print(Counter.get_total())   # 3
print(Counter.is_valid(-1))  # False
```

---

## 3. 상속 (Inheritance) & `super()`

부모 클래스의 속성과 메서드를 자식 클래스가 물려받습니다.

```python
class Animal:
    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        raise NotImplementedError("서브클래스에서 반드시 구현하세요!")

class Dog(Animal):
    def speak(self) -> str:        # 메서드 오버라이딩 (Override)
        return "Woof!"

class Cat(Animal):
    def __init__(self, name: str, indoor: bool = True):
        super().__init__(name)     # 부모 __init__ 호출
        self.indoor = indoor

    def speak(self) -> str:
        return "Meow!"

# 다형성: 타입이 달라도 같은 인터페이스로 호출 가능
animals = [Dog("Rex"), Cat("Luna")]
for a in animals:
    print(a.speak())   # "Woof!", "Meow!"
```

---

## 4. 추상 클래스 (Abstract Class)

인스턴스화를 막고, 서브클래스가 반드시 구현해야 하는 메서드를 강제합니다.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

    @abstractmethod
    def perimeter(self) -> float:
        pass

    def describe(self) -> str:    # 공통 메서드는 일반 메서드로 정의 가능
        return f"넓이: {self.area():.2f}, 둘레: {self.perimeter():.2f}"

# Shape()  →  TypeError 발생! (직접 인스턴스화 불가)
```

---

## 5. 캡슐화 (Encapsulation) & `@property`

데이터를 숨기고 유효성 검사를 통해 접근하도록 강제합니다.

```python
class Temperature:
    ABSOLUTE_ZERO = -273.15

    def __init__(self, celsius: float):
        self.celsius = celsius     # setter를 통해 유효성 검사

    @property
    def celsius(self) -> float:   # Getter
        return self._celsius

    @celsius.setter
    def celsius(self, value: float):  # Setter
        if value < self.ABSOLUTE_ZERO:
            raise ValueError(f"절대 영도({self.ABSOLUTE_ZERO}°C) 아래로 내려갈 수 없습니다.")
        self._celsius = value

    @property
    def fahrenheit(self) -> float:    # 읽기 전용 파생 속성
        return self._celsius * 9 / 5 + 32
```

---

## 6. 매직 메서드 (Magic / Dunder Methods)

`__이름__` 형태의 메서드로, 파이썬 내장 연산자를 클래스에 적용할 수 있게 합니다.

| 메서드 | 호출 시점 | 예시 |
|--------|----------|------|
| `__str__` | `str(obj)`, `print(obj)` | 사람이 읽기 좋은 표현 |
| `__repr__` | 콘솔에서 객체 출력 | 디버깅용 표현 |
| `__len__` | `len(obj)` | 길이 반환 |
| `__eq__` | `obj1 == obj2` | 동등 비교 |
| `__lt__` | `obj1 < obj2` | 크기 비교 |
| `__add__` | `obj1 + obj2` | 덧셈 연산 |
| `__mul__` | `obj1 * scalar` | 곱셈 연산 |
| `__abs__` | `abs(obj)` | 절댓값/크기 |

```python
class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Vector(x={self.x}, y={self.y})"

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float) -> 'Vector':
        return Vector(self.x * scalar, self.y * scalar)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __abs__(self) -> float:    # 벡터의 크기 (유클리드 거리)
        return (self.x ** 2 + self.y ** 2) ** 0.5

v1 = Vector(3, 4)
print(abs(v1))         # 5.0
print(v1 + Vector(1, 1))  # Vector(x=4, y=5)
print(v1 * 2)          # Vector(x=6, y=8)
```

---

## 7. 연습문제 (Pytest) 🚀

`tests_python/` 디렉토리 안의 파일들을 열고 빈 칸을 완성하세요.

- **test_oop.py**: `BankAccount`, `Animal`/`Dog`/`Cat`, `Shape`/`Rectangle`/`Circle`, `Temperature`, `Vector` 구현
