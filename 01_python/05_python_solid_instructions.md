# SOLID 원칙 실습 가이드

소프트웨어 설계의 5가지 핵심 원칙입니다. 이 원칙들을 따르면 코드가 변경에 유연하고, 테스트하기 쉬우며, 팀원이 이해하기 쉬워집니다.

---

## S — Single Responsibility Principle (단일 책임 원칙)

> "하나의 클래스는 단 하나의 이유로만 변경되어야 한다."

한 클래스가 너무 많은 책임을 지면, 한 기능을 바꿀 때 의도치 않게 다른 기능이 깨집니다.

**❌ 나쁜 예시:**
```python
class UserManager:          # 이 클래스가 너무 많은 일을 합니다!
    def create_user(self, name, email):
        # DB에 저장 (저장 책임)
        pass
    def send_welcome_email(self, email):
        # 이메일 발송 (알림 책임)
        pass
    def log_action(self, action):
        # 로그 기록 (로깅 책임)
        pass
```

**✅ 좋은 예시:**
```python
class UserRepository:   # 저장만 담당
    def save(self, user): ...

class EmailService:     # 이메일만 담당
    def send_welcome(self, email): ...

class Logger:           # 로깅만 담당
    def log(self, message): ...
```

---

## O — Open/Closed Principle (개방-폐쇄 원칙)

> "소프트웨어 요소는 확장에는 열려있고, 수정에는 닫혀있어야 한다."

새로운 기능을 추가할 때 **기존 코드를 수정하지 않고** 새 클래스만 추가합니다.

**❌ 나쁜 예시:**
```python
def calculate_discount(order_type, amount):
    if order_type == "regular":
        return amount * 0.05
    elif order_type == "premium":
        return amount * 0.10
    # 새 타입 추가 시 이 함수를 직접 수정해야 함 → 기존 로직 파괴 위험!
```

**✅ 좋은 예시:**
```python
from abc import ABC, abstractmethod

class DiscountStrategy(ABC):
    @abstractmethod
    def calculate(self, amount: float) -> float:
        pass

class RegularDiscount(DiscountStrategy):
    def calculate(self, amount): return amount * 0.05

class PremiumDiscount(DiscountStrategy):
    def calculate(self, amount): return amount * 0.10

# 새 타입 추가: 기존 코드 수정 없이 새 클래스만 추가!
class StudentDiscount(DiscountStrategy):
    def calculate(self, amount): return amount * 0.15
```

---

## L — Liskov Substitution Principle (리스코프 치환 원칙)

> "부모 클래스 자리에 자식 클래스를 넣어도 프로그램이 정상 동작해야 한다."

**❌ 나쁜 예시:**
```python
class Bird:
    def fly(self) -> str:
        return "날고 있습니다."

class Penguin(Bird):
    def fly(self) -> str:
        raise Exception("펭귄은 날 수 없습니다!")  # LSP 위반!
        # Bird 타입으로 사용할 때 예외 발생 → 프로그램 파괴
```

**✅ 좋은 예시:**
```python
from abc import ABC, abstractmethod

class Bird(ABC):
    @abstractmethod
    def move(self) -> str: pass     # "이동"이라는 공통 행동으로 추상화

class FlyingBird(Bird):
    def move(self): return "날고 있습니다."

class SwimmingBird(Bird):
    def move(self): return "헤엄치고 있습니다."

class Eagle(FlyingBird): pass       # 독수리: 날아서 이동
class Penguin(SwimmingBird): pass   # 펭귄: 헤엄쳐서 이동

# 어떤 Bird 서브클래스를 넣어도 정상 동작!
def make_move(bird: Bird) -> str:
    return bird.move()
```

---

## I — Interface Segregation Principle (인터페이스 분리 원칙)

> "클라이언트는 자신이 사용하지 않는 메서드에 의존하도록 강요받아서는 안 된다."

하나의 거대한 인터페이스보다 **여러 개의 작은 인터페이스**가 낫습니다.

**❌ 나쁜 예시:**
```python
class IWorker(ABC):
    def work(self): ...
    def eat(self): ...   # 로봇은 먹지 않음! 불필요한 의존성
    def sleep(self): ... # 로봇은 잠자지 않음! 불필요한 의존성
```

**✅ 좋은 예시:**
```python
class IWorkable(ABC):
    @abstractmethod
    def work(self) -> str: pass

class IEatable(ABC):
    @abstractmethod
    def eat(self, food: str) -> str: pass

class ISleepable(ABC):
    @abstractmethod
    def sleep(self, hours: int) -> str: pass

class HumanWorker(IWorkable, IEatable, ISleepable):
    # 인간은 세 가지 모두 구현
    ...

class RobotWorker(IWorkable):
    # 로봇은 일만 구현 → 불필요한 메서드 없음!
    ...
```

---

## D — Dependency Inversion Principle (의존성 역전 원칙)

> "고수준 모듈은 저수준 모듈에 의존해선 안 된다. 둘 다 추상화에 의존해야 한다."

구체 구현 클래스를 직접 생성하지 말고 **추상화(인터페이스)에 의존**하세요.
구체 구현체는 외부에서 **주입(Dependency Injection)**받습니다.

**❌ 나쁜 예시:**
```python
class NotificationService:
    def __init__(self):
        self.sender = EmailSender()    # 구체 구현에 직접 의존!
                                       # SMS로 바꾸려면 이 클래스를 수정해야 함
```

**✅ 좋은 예시:**
```python
from abc import ABC, abstractmethod

class INotificationSender(ABC):       # 추상화(인터페이스)
    @abstractmethod
    def send(self, recipient: str, message: str) -> bool: pass

class EmailSender(INotificationSender):    # 저수준 구현체 A
    def send(self, recipient, message): ...

class SMSSender(INotificationSender):      # 저수준 구현체 B
    def send(self, recipient, message): ...

class NotificationService:                 # 고수준 모듈
    def __init__(self, sender: INotificationSender):  # 추상화에 의존!
        self.sender = sender

    def notify(self, recipient, message):
        return self.sender.send(recipient, message)

# 런타임에 구현체를 자유롭게 교체 가능
svc_email = NotificationService(EmailSender())
svc_sms   = NotificationService(SMSSender())
```

---

## 연습문제 (Pytest) 🚀

- **test_solid.py**: 각 SOLID 원칙별로 주어진 골격(skeleton) 코드를 완성하여 테스트를 통과하세요.
