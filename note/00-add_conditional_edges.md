# `add_conditional_edges` 사용 정리 (`01-Core-Features`)

`add_conditional_edges(source, path, path_map=None, then=None)`는  
`source` 노드 실행 후 `path` 함수 반환값으로 다음 노드를 결정합니다.

---

## 1) `tools_condition` 그대로 사용 (가장 기본)

### 공통 형태

```python
graph_builder.add_conditional_edges("chatbot", tools_condition)
```

- `tools_condition`이 내부적으로 `"tools"` 또는 `END`를 반환
- 별도 `path_map`을 생략해도, 반환값과 동일한 이름의 노드/`END`로 라우팅

### 사용 파일

- `03-LangGraph-Agent.ipynb` (유사 형태 + 커스텀 라우팅도 함께 사용)
- `04-LangGraph-Agent-With-Memory.ipynb`
- `05-LangGraph-Streaming-Outputs.ipynb`
- `06-LangGraph-Human-In-the-Loop.ipynb`
- `07-LangGraph-Manual-State-Update.ipynb`
- `10-LangGraph-ToolNode.ipynb`
- `15-LangGraph-Streaming-Steps.ipynb` (메인 그래프 2회 + 서브그래프 1회)

---

## 2) 커스텀 라우팅 + 명시적 `path_map` (dict)

### 예시 A: 도구 호출 여부 분기

```python
def route_tools(state):
    ...
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

graph_builder.add_conditional_edges(
    source="chatbot",
    path=route_tools,
    path_map={"tools": "tools", END: END},
)
```

- 반환값 `"tools"` -> `"tools"` 노드
- 반환값 `END` -> `END`

파일: `03-LangGraph-Agent.ipynb`

### 예시 B: 노드명 차이가 있는 매핑

```python
def should_continue(state: MessagesState):
    ...
    if not last_message.tool_calls:
        return END
    return "tool"

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tool": "tool", END: END},
)
```

파일: `09-LangGraph-DeleteMessages.ipynb` (초반 예제)

### 예시 C: 사람이 개입할지 분기

```python
def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    return tools_condition(state)

graph_builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", END: END},
)
```

파일: `08-LangGraph-State-Customization.ipynb`

---

## 3) 커스텀 라우팅 + `path_map` 생략 (반환값=노드명)

### 예시 A: 대화 길이에 따른 요약 분기

```python
def should_continue(state: State) -> Literal["summarize_conversation", END]:
    if len(state["messages"]) > 6:
        return "summarize_conversation"
    return END

workflow.add_conditional_edges("conversation", should_continue)
```

- 반환 `"summarize_conversation"` -> 동일 이름 노드
- 반환 `END` -> 종료

파일: `12-LangGraph-Add-Conversation-Summary.ipynb`

### 예시 B: 액션/메시지삭제 분기

```python
def should_continue(state: MessagesState) -> Literal["action", "delete_messages"]:
    ...
    if not last_message.tool_calls:
        return "delete_messages"
    return "action"

workflow.add_conditional_edges("agent", should_continue)
```

- 반환 `"action"` -> `"action"` 노드
- 반환 `"delete_messages"` -> `"delete_messages"` 노드

파일: `09-LangGraph-DeleteMessages.ipynb` (후반 예제)

---

## 4) 브랜칭/Fan-out: `Sequence[str]` 반환

### 예시: 한 번에 여러 노드로 분기

```python
def route_bc_or_cd(state: State) -> Sequence[str]:
    if state["which"] == "cd":
        return ["c", "d"]
    return ["b", "c"]
```

- 리스트를 반환하면 fan-out (병렬 브랜치)로 처리

파일: `11-LangGraph-Branching.ipynb`

---

## 5) `path_map`에 list 전달 (축약 문법)

```python
intermediates = ["b", "c", "d"]
builder.add_conditional_edges("a", route_bc_or_cd, intermediates)
```

위 코드는 개념적으로 아래와 동일합니다.

```python
{"b": "b", "c": "c", "d": "d"}
```

- 즉, 리스트 원소를 자기 자신으로 매핑하는 shorthand
- `route_bc_or_cd`가 반환한 문자열/문자열 리스트가 이 집합 안에서 해석됨

파일: `11-LangGraph-Branching.ipynb`

---

## 6) `then` 파라미터 (참고 코드)

`11-LangGraph-Branching.ipynb`에 주석 예시로 존재:

```python
# builder.add_conditional_edges(
#     "a",
#     route_bc_or_cd,
#     intermediates,
#     then="e",
# )
```

- fan-out 이후 공통 후속 노드를 간결하게 지정할 때 사용
- 현재 노트북에서는 주석(참고용) 상태

---

## 빠른 체크리스트

- 반환값이 노드명과 같으면 `path_map` 생략 가능
- 반환값 키와 실제 노드명을 다르게 쓰면 `path_map` 명시 필요
- 여러 노드 동시 분기 시 `Sequence[str]` 반환
- `path_map`에 list를 주면 identity mapping shorthand
