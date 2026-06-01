## 미들웨어(Middleware)

미들웨어는 에이전트 실행의 모든 단계를 제어하고 커스터마이징하는 방법을 제공합니다.

핵심 에이전트 루프는 모델을 호출하고, 모델이 실행할 도구를 선택하도록 한 다음, 더 이상 도구를 호출하지 않으면 종료하는 것을 포함합니다.

![](../assets/langgraph-middleware.avif)

미들웨어는 각 단계 전후에 후크를 노출합니다.

- 에이전트 시작 전/후
- 모델 호출 전/후
- 도구 실행 전/후


## Human in the Loop Middleware

### 개요

Human in the Loop Middleware는 AI 시스템의 의사결정 과정에 사람의 개입을 가능하게 하는 중간 계층입니다. 자동화된 프로세스 중 특정 시점에서 사람의 검토, 승인 또는 수정을 요구할 수 있습니다.

### 주요 특징

* **검증 단계 추가**: AI의 출력을 사람이 검토하고 승인하는 단계 삽입
* **오류 방지**: 중요한 결정에 대한 사람의 최종 확인으로 오류 최소화
* **유연한 개입**: 필요에 따라 자동/수동 모드 전환 가능
* **피드백 루프**: 사람의 수정 사항을 학습 데이터로 활용

### Parameters

**`timeout`**
* **타입**: `int` 또는 `float`
* **기본값**: `None`
* **설명**: 사람의 응답을 기다리는 최대 시간(초)
* **사용법**: timeout 초과 시 기본 동작 실행 또는 예외 발생
```python
middleware = HumanInTheLoopMiddleware(timeout=300)  # 5분
```

**`approval_required`**
* **타입**: `bool`
* **기본값**: `True`
* **설명**: 사람의 명시적 승인이 필요한지 여부
* **사용법**: `False`로 설정 시 검토만 하고 자동 진행
```python
middleware = HumanInTheLoopMiddleware(approval_required=True)
```

**`callback_function`**
* **타입**: `callable`
* **기본값**: `None`
* **설명**: 사람의 개입이 필요할 때 호출되는 함수
* **사용법**: 알림, 로깅, UI 표시 등의 커스텀 동작 정의
```python
def notify_user(data):
    print(f"Review needed: {data}")

middleware = HumanInTheLoopMiddleware(callback_function=notify_user)
```

**`intervention_condition`**
* **타입**: `callable` 또는 `str`
* **기본값**: `"always"`
* **설명**: 사람 개입이 필요한 조건 정의
* **사용법**: 함수 또는 조건 문자열로 지정
```python
# 함수로 조건 정의
def check_confidence(result):
    return result.confidence < 0.8

middleware = HumanInTheLoopMiddleware(intervention_condition=check_confidence)

# 문자열로 조건 정의
middleware = HumanInTheLoopMiddleware(intervention_condition="low_confidence")
```

**`retry_limit`**
* **타입**: `int`
* **기본값**: `3`
* **설명**: 사람의 응답을 요청하는 최대 재시도 횟수
* **사용법**: 응답이 없을 때 재시도 횟수 제한
```python
middleware = HumanInTheLoopMiddleware(retry_limit=5)
```

**`fallback_action`**
* **타입**: `str` 또는 `callable`
* **기본값**: `"reject"`
* **설명**: timeout 또는 응답 실패 시 수행할 동작
* **옵션**: `"approve"`, `"reject"`, `"skip"`, 또는 커스텀 함수
* **사용법**:
```python
middleware = HumanInTheLoopMiddleware(fallback_action="approve")

# 커스텀 fallback
def custom_fallback(context):
    return context.get("default_value")

middleware = HumanInTheLoopMiddleware(fallback_action=custom_fallback)
```

**`notification_channels`**
* **타입**: `list`
* **기본값**: `["console"]`
* **설명**: 알림을 전송할 채널 목록
* **옵션**: `"console"`, `"email"`, `"slack"`, `"webhook"` 등
* **사용법**:
```python
middleware = HumanInTheLoopMiddleware(
    notification_channels=["email", "slack"]
)
```

**`store_feedback`**
* **타입**: `bool`
* **기본값**: `True`
* **설명**: 사람의 피드백을 저장할지 여부
* **사용법**: 학습 데이터로 활용하기 위해 피드백 저장
```python
middleware = HumanInTheLoopMiddleware(store_feedback=True)
```

**`priority_level`**
* **타입**: `str` 또는 `int`
* **기본값**: `"normal"`
* **설명**: 개입 요청의 우선순위
* **옵션**: `"low"`, `"normal"`, `"high"`, `"critical"` 또는 1-5
* **사용법**:
```python
middleware = HumanInTheLoopMiddleware(priority_level="high")
```