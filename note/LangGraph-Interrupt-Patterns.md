# LangGraph 인터럽트 패턴 정리

이 문서는 `langchain-kr` 레포의 `17-LangGraph` 튜토리얼에 나온 **`interrupt_before` / `interrupt_after`**, 그리고 **`update_state`** 사용 방식을 패턴별로 정리한 것입니다.

**주요 출처:** `17-LangGraph/01-Core-Features/07-LangGraph-Manual-State-Update.ipynb` (아래 `update_state`·인터럽트 비교 섹션)

---

## 패턴 1: `compile` 시점에 고정 — 특정 노드 **직전**에 멈춤

`human_feedback` 노드가 돌기 **전**에 그래프가 멈춥니다. 체크포인터와 같이 쓰는 경우가 많습니다.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

graph = builder.compile(
    interrupt_before=["human_feedback"],
    checkpointer=memory,
)
```

**출처:** `17-LangGraph/03-Use-Cases/10-LangGraph-Research-Assistant.ipynb`

같은 방식으로 `human` 노드 전에 멈추는 예:

```python
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["human"],
)
```

**출처:** `17-LangGraph/01-Core-Features/08-LangGraph-State-Customization.ipynb`

---

## 패턴 2: `stream` / `invoke` 호출마다 **한 번만** 오버라이드 — 노드 **직전**에 멈춤

그래프는 `compile`만 하고, **실행할 때** `interrupt_before`를 넘깁니다. (예: `tools` 실행 전에 멈춰서 `tool_calls` 검사)

```python
for event in graph.stream(
    input=input,
    config=config,
    stream_mode="values",  # 또는 "updates"
    interrupt_before=["tools"],
):
    ...
```

**출처:**

- `17-LangGraph/01-Core-Features/06-LangGraph-Human-In-the-Loop.ipynb`
- `17-LangGraph/01-Core-Features/07-LangGraph-Manual-State-Update.ipynb`
- `17-LangGraph/01-Core-Features/05-LangGraph-Streaming-Outputs.ipynb` (첫 번째 예)

**참고:** `06-LangGraph-Human-In-the-Loop.ipynb`의 한 셀에서는 `graph_builder.compile(checkpointer=memory)`만 하고, **인터럽트는 전부 `stream(..., interrupt_before=[...])`에만** 둔 형태입니다.

---

## 패턴 3: `stream`에서 노드 **직후**에 멈춤 — `interrupt_after`

`tools` 노드가 **끝난 뒤** 스트림이 끊깁니다. (다음 노드로 가기 전에 관찰·개입)

```python
for event in graph.stream(
    input=input,
    config=config,
    stream_mode="updates",
    interrupt_after=["tools"],
):
    ...
```

**출처:** `17-LangGraph/01-Core-Features/05-LangGraph-Streaming-Outputs.ipynb`

---

## StateGraph의 `update_state` 메서드

- mock 데이터를 넣고 실행 흐름을 시뮬레이션하는 개념과 맞닿아 있습니다.

`update_state` 메서드는 주어진 값으로 그래프의 상태를 업데이트합니다. 이 메서드는 마치 **`as_node`에서 값이 온 것처럼** 동작합니다.

### 매개변수

- `config` (`RunnableConfig`): 실행 구성
- `values` (`Optional[Union[dict[str, Any], Any]]`): 업데이트할 값들
- `as_node` (`Optional[str]`): 값의 출처로 간주할 노드 이름. 기본값은 `None`

### 반환값

- `RunnableConfig`

### 주요 기능

1. 체크포인터를 통해 이전 상태를 로드하고 새로운 상태를 저장합니다.
2. 서브그래프에 대한 상태 업데이트를 처리합니다.
3. `as_node`가 지정되지 않은 경우, 마지막으로 상태를 업데이트한 노드를 찾습니다.
4. 지정된 노드의 writer들을 실행하여 상태를 업데이트합니다.
5. 업데이트된 상태를 체크포인트에 저장합니다.

### 주요 로직 (요약)

1. 체크포인터를 확인하고, 없으면 `ValueError`를 발생시킵니다.
2. 서브그래프에 대한 업데이트인 경우, 해당 서브그래프의 `update_state` 메서드를 호출합니다.
3. 이전 체크포인트를 로드하고, 필요한 경우 `as_node`를 결정합니다.
4. 지정된 노드의 writer들을 사용하여 상태를 업데이트합니다.
5. 업데이트된 상태를 새로운 체크포인트로 저장합니다.

### 참고

- 이 메서드는 그래프의 상태를 **수동으로** 업데이트할 때 사용됩니다.
- 체크포인터를 사용하여 상태의 버전 관리와 지속성을 보장합니다.
- `as_node`를 지정하지 않으면 자동으로 결정되지만, 모호한 경우 오류가 발생할 수 있습니다.
- 상태 업데이트 중 SharedValues에 쓰기 작업은 허용되지 않습니다.

### 노트북 예시: `tools` 노드가 낸 것처럼 메시지 주입 후 재개

`State`의 `messages`가 `add_messages`로 `Annotated` 처리된 경우, `update_state`로 넘긴 메시지는 **기존 목록을 덮어쓰지 않고 추가**됩니다. `as_node="tools"`로 지정하면 해당 노드가 방금 실행된 것처럼 처리되어 이후 `stream(None, config)`로 이어갈 수 있습니다.

```python
graph.update_state(
    config,
    {"messages": new_messages},
    as_node="tools",
)
```

### 노트북 예시: `chatbot`에서 온 것처럼 최종 메시지 덧붙이기

```python
from langchain_core.messages import AIMessage

graph.update_state(
    config,
    {
        "messages": [
            AIMessage(content="마지막으로 최종 메시지를 추가하여 마무리 합니다.")
        ]
    },
    as_node="chatbot",
)
```

(`as_node`를 `tools`로 두면 사이클이 다시 도는 등, **어느 노드에서 온 업데이트인지**에 따라 다음 라우팅이 달라질 수 있음 — 노트북 주석 참고.)

### 타임 트래블 / 리플레이 맥락에서의 `update_state`

과거 체크포인트 기준 `config`로 상태를 바꾼 뒤, 반환된 `RunnableConfig`를 넘겨 리플레이할 수 있습니다.

```python
updated_state = graph.update_state(
    to_replay_state.config,
    {"messages": [new_message]},
)
for event in graph.stream(None, updated_state, stream_mode="values"):
    ...
```

**출처:** `07-LangGraph-Manual-State-Update.ipynb`

---

## `interrupt_before` vs `interrupt_after` 핵심 정리

LangGraph에서 `update_state`를 활용하면, 중단 지점에 따라 **사전 차단** 또는 **사후 교정** 전략을 선택할 수 있습니다.

### 1) `interrupt_before` (노드 실행 전 중단)

타겟 노드가 실행되기 **직전** 그래프를 멈춥니다.

- **동작 흐름**: 이전 노드 종료 → 그래프 일시 정지 → 상태 확인/수정 → 타겟 노드 실행
- **주요 목적**: 입력값(Input) 통제, 권한 체크, 악성 프롬프트 사전 필터링

이 시점의 `update_state`는 크게 두 가지 방식으로 사용합니다.

#### 전략 A. 입력값만 수정 후 그대로 실행 (파라미터 변조)

- **방법**: `update_state`로 상태만 수정하고 `as_node`는 지정하지 않은 채 Resume
- **효과**: 타겟 노드는 실행되지만, 방금 주입한 **수정된 입력값(안전한 상태)** 기준으로 동작

#### 전략 B. 타겟 노드를 완전히 건너뛰기 (Mocking & Skip)

- **방법**: `update_state`에 Mock 결과를 넣고 `as_node="타겟_노드명"`을 지정한 뒤 Resume
- **효과**: 엔진이 "타겟 노드가 이미 실행되어 결과를 냈다"고 간주하여, 실제 타겟 노드 로직은 실행하지 않고 다음 노드로 이동
- **보안 장점**: 위험한 API 호출/DB 변경 노드를 실행 전에 차단 가능 (완전한 사전 방어)

### 2) `interrupt_after` (노드 실행 후 중단)

타겟 노드가 실행을 마치고 상태를 갱신한 **직후**, 다음 노드로 라우팅되기 전에 멈춥니다.

- **동작 흐름**: 타겟 노드 실행 및 결과 반환 → 그래프 일시 정지 → 상태 확인/수정 → 다음 노드 이동
- **주요 목적**: 결과값(Output) 검증, 환각(Hallucination) 교정, 포맷 오류 수정

이 시점의 `update_state`는 본질적으로 **결과 덮어쓰기(Overwrite)** 입니다.

- **방법**: 교정된 데이터를 `update_state`로 주입하고 `as_node="타겟_노드명"` 명시
  - 직전 실행 노드가 타겟 노드라 생략 가능할 때도 있지만, 아키텍처 안정성을 위해 명시 권장
- **효과**: 타겟 노드의 부정확한 출력이 있더라도, 주입한 값으로 결과를 교체하여 다음 노드가 수정된 상태를 사용

- **주의점**: 타겟 노드 자체는 **이미 실행됨**
  - 따라서 부작용(Side-effect)이 있는 액션 노드(API 호출, DB 변경 등)에는 사후 대응이 늦을 수 있음
  - 주로 부작용이 없는 LLM 추론 결과 검수/교정에 적합

**출처:** `07-LangGraph-Manual-State-Update.ipynb`

---

## 패턴 요약 표

| 패턴 | API                                     | 의미 (한 줄)                                                |
| ---- | --------------------------------------- | ----------------------------------------------------------- |
| A    | `compile(..., interrupt_before=[노드])` | 그 스레드 실행에서 항상 그 노드 **앞**에서 멈춤             |
| B    | `stream(..., interrupt_before=[노드])`  | 이번 호출만 그 노드 **앞**에서 멈춤 (`compile`과 병행 가능) |
| C    | `stream(..., interrupt_after=[노드])`   | 이번 호출만 그 노드 **뒤**에서 멈춤                         |

---

## 워크스페이스에서 인터럽트가 쓰인 노트북 (요약)

| 파일                                                      | 사용 방식                                                                  |
| --------------------------------------------------------- | -------------------------------------------------------------------------- |
| `01-Core-Features/06-LangGraph-Human-In-the-Loop.ipynb`   | 주로 `stream(..., interrupt_before=["tools"])`                             |
| `01-Core-Features/05-LangGraph-Streaming-Outputs.ipynb`   | `interrupt_before=["tools"]` / `interrupt_after=["tools"]` (스트리밍 실습) |
| `01-Core-Features/07-LangGraph-Manual-State-Update.ipynb` | `stream(..., interrupt_before=["tools"])` 등                               |
| `01-Core-Features/08-LangGraph-State-Customization.ipynb` | `compile(..., interrupt_before=["human"])`                                 |
| `03-Use-Cases/10-LangGraph-Research-Assistant.ipynb`      | `compile(..., interrupt_before=["human_feedback"], checkpointer=memory)`   |

다른 노트북 출력에 `interrupt_before`가 보이는 경우, **에러 스택에 포함된 LangGraph 라이브러리 시그니처**일 수 있으므로 실제 튜토리얼 코드인지 구분해야 합니다.

---

## (참고) 이 튜토리얼에는 없음: `interrupt()` 함수

노드 **안**에서 `from langgraph.types import interrupt` 후 `answer = interrupt(...)` 하는 방식은 **`17-LangGraph` 노트북들에는 없고**, 클론된 `langgraph-main`의 테스트·공식 구현 쪽에 많이 나옵니다.

---

## 관련 인덱스

`STUDY_INDEX.md`의 LangGraph Human-in-the-loop 항목과 함께 보면 됩니다.




두 방식 모두 LangGraph에서 **Human-in-the-Loop (HITL, 인간 개입)** 파이프라인을 구현할 때 실행을 일시 중지(Pause)하고 재개(Resume)하기 위해 사용되지만, **제어의 위치, 유연성, 그리고 상태 관리 방식**에서 큰 차이가 있습니다.

폐쇄망 환경에서 안전하고 통제된 AI 에이전트를 구축할 때 이 두 가지 방식의 특성을 정확히 이해하는 것은 보안 승인 프로세스 설계에 매우 중요합니다.

---

## 1. `interrupt_after=["tools"]` (정적, 노드 단위 제어)

그래프 실행 시점(`stream` 또는 `invoke`)에 **그래프 외부에서 브레이크포인트(Breakpoint)를 설정**하는 방식입니다.

* **동작 방식:** 그래프가 실행되다가 `tools` 노드의 작업이 끝나면 무조건 실행을 멈추고 현재 상태(State)를 반환합니다.
* **재개 방법:** 외부에서 상태를 검사하거나 수정한 뒤, 동일한 `thread_id`를 가진 `config`를 사용하여 `graph.stream(None, config=config)`을 호출해 다음 노드부터 실행을 이어갑니다.

### 장단점 (Pros & Cons)

* **장점 (Pros):**
* **명확한 감사 트레일(Audit Trail):** 특정 노드 전/후로 무조건 멈추기 때문에 전체적인 상태의 스냅샷을 검증하기 좋습니다.
* **코드 변경 최소화:** 그래프 내부 노드의 코드를 수정할 필요 없이 호출부에서만 제어할 수 있습니다.


* **단점 (Cons):**
* **유연성 부족:** 조건에 상관없이 항상 해당 노드에서 멈춥니다. (예: 위험한 API 호출일 때만 멈추고 싶어도 분기 처리가 까다로움)
* **상태 덮어쓰기 복잡성:** 인간이 개입하여 흐름을 바꾸려면 State 전체를 조작(State Update)해야 하므로 데이터가 오염될 위험이 존재합니다.



## 2. `interrupt()` 와 `Command` (동적, 노드 내부 제어)

LangGraph 최신 버전에서 도입된 방식으로, **노드 내부의 로직에 따라 동적으로 실행을 중단하고 재개**하는 방식입니다.

* **동작 방식:** 노드 함수 내부에서 `interrupt("승인 필요")` 함수를 호출하면 그 즉시 노드 실행이 중단됩니다. 특정 조건(예: 민감한 데이터 접근, 위험한 쿼리 감지 등)이 충족될 때만 선택적으로 멈출 수 있습니다.
* **재개 방법:** 사용자가 입력을 제공하면, `graph.stream(Command(resume="사용자 응답 데이터"), config=config)`의 형태로 그래프에 값을 직접 전달하며 재개합니다.

### 장단점 (Pros & Cons)

* **장점 (Pros):**
* **세밀한 조건부 제어:** LLM이 생성한 파라미터를 검사하여, 보안상 위험한 특정 도구(Tool)를 사용할 때만 승인을 요청하는 등 동적인 파이프라인 구성이 가능합니다.
* **안전한 데이터 전달:** 전체 State를 조작하지 않고, `Command(resume=...)`을 통해 중단된 노드에 필요한 결과값만 깔끔하게 주입할 수 있어 상태 오염을 방지합니다.


* **단점 (Cons):**
* **그래프 복잡도 증가:** 노드 내부에 중단 및 재개 후의 로직(Resume 값을 처리하는 코드)이 포함되어야 하므로 함수 자체가 다소 길어질 수 있습니다.



---

## 🔒 보안 및 아키텍처 관점에서의 접근 가이드

보안을 최우선으로 하는 시스템을 설계하신다면 다음과 같이 두 방식을 혼합하거나 목적에 맞게 분리하는 것을 권장합니다.

1. **`interrupt_before`/`interrupt_after` 활용:**
* **시스템 전체의 안전장치(Fail-safe) 역할.** 폐쇄망 환경에서 외부 시스템이나 DB에 접근하는 노드 실행 직전(`interrupt_before=["db_access"]`)에 무조건 관리자 세션의 승인을 받도록 하는 '강제 체크포인트'로 활용하기 좋습니다.


2. **`interrupt()` / `Command` 활용:**
* **사용자 피드백 루프 및 동적 검증 역할.** RAG 시스템에서 LLM이 문서를 검색하기 전, 검색 쿼리가 적절한지 사용자에게 확인받고 쿼리를 수정(Refine)하여 다시 주입해야 할 때 매우 효과적입니다.



현재 구상 중인 시스템에서 이 Human-in-the-Loop 기능을 주로 어떤 작업(예: DB 쿼리 실행 전 승인, RAG 검색어 수정, 민감 정보 필터링 등)에 적용하려고 계획 중이신가요?
