### 차이점
접근 할 수 있는 항목은 일부 동일하나 사용처가 다름

네, LangChain과 LangGraph 생태계(특히 최근 버전)에서 두 개념은 이름이 비슷하지만 **사용되는 위치와 목적**이 명확하게 다릅니다.

간단히 비유하자면 `Runtime`은 애플리케이션 전체를 관리하는 '총괄 매니저'이고, `ToolRuntime`은 개별 도구(Tool)가 총괄 매니저의 자원에 접근할 수 있게 해주는 '전용 출입증'입니다.

---

### 1. `Runtime` (런타임 컨텍스트)

LangGraph 파이프라인(그래프) **전체의 실행 환경**을 의미합니다.

* **사용 위치:** 주로 노드(Node) 함수 내부나 동적으로 LLM 모델을 선택할 때 프레임워크가 주입해 줍니다.
* **주요 역할:** * 한 번의 실행(Run)이 유지되는 동안 필요한 **불변 데이터**를 보관합니다.
* 사용자 ID, 데이터베이스 커넥션 풀 등 LLM의 프롬프트에는 들어가지 않지만 백엔드 시스템 로직상 필요한 전역 설정들이 담깁니다.



### 2. `ToolRuntime` (도구 실행 컨텍스트)

LLM이 호출하는 **개별 도구(Tool) 내부**에서 그래프의 핵심 자원에 접근할 수 있도록 여러 기능을 하나로 묶어놓은 객체입니다.

* **사용 위치:** `@tool` 데코레이터로 만든 함수 내부에서 매개변수(`runtime: ToolRuntime`)로 선언하여 사용합니다. (프레임워크가 자동으로 주입해 줍니다.)
* **주요 역할:** 도구는 원래 메인 파이프라인과 격리된 함수라 외부 상황을 모릅니다. 하지만 `ToolRuntime`을 넘겨받으면 도구 내부에서도 다음과 같은 막강한 작업이 가능해집니다.
1. **`runtime.state`:** 현재 그래프의 상태(State)나 과거 대화 내역을 읽어올 수 있습니다.
2. **`runtime.context`:** `Runtime`이 가지고 있던 사용자 ID나 DB 세션 같은 백엔드 정보에 접근할 수 있습니다.
3. **`runtime.stream_writer`:** 이전에 질문하셨던 `stream_mode="custom"`과 연결되는 핵심 기능입니다. 도구 내부에서 긴 작업을 수행할 때, 이 라이터를 통해 **프론트엔드로 실시간 중간 과정(Event)을 쏠 수 있습니다.**



---

### 💡 한눈에 보는 비교

| 구분 | `Runtime` | `ToolRuntime` |
| --- | --- | --- |
| **적용 범위** | 그래프 전체 (Global / Node 레벨) | 특정 도구 내부 (Local / Tool 레벨) |
| **접근 방식** | 프레임워크가 노드에 전달 | 도구 함수의 파라미터로 선언하여 주입받음 |
| **주요 목적** | 실행 환경 세팅, DB 연결 관리, 전역 상태 제어 | 도구 안에서 메인 상태(State) 참조, 실시간 스트리밍(Custom Stream) 발송 |

### 코드 예시: ToolRuntime의 활용

```python
from langchain.tools import tool, ToolRuntime

@tool
def my_database_search_tool(query: str, runtime: ToolRuntime = None) -> str:
    """DB를 검색하는 도구입니다."""
    
    # 1. Context 활용: LLM에게는 숨겨진 민감 정보(사용자 권한 등)를 안전하게 가져옴
    user_id = runtime.context.get("user_id")
    
    # 2. Custom Streaming: 도구 실행 중간에 프론트엔드로 실시간 상태 전송
    if runtime.stream_writer:
        runtime.stream_writer({"status": "DB 검색 시작...", "user": user_id})
        
    # 3. State 참조: 현재까지 그래프에 쌓인 대화나 변수 확인
    current_memory = runtime.state.get("messages", [])
    
    # ... (실제 DB 검색 로직) ...
    
    return "검색 완료된 데이터"

```

즉, `ToolRuntime`은 단순한 함수 쪼가리였던 Tool을 **메인 시스템의 상태를 읽고 클라이언트와 직접 소통할 수 있는 강력한 모듈**로 업그레이드해 주는 객체라고 보시면 됩니다.

---
### Runtime

LangChain의 `create_agent`는 내부적으로 LangGraph의 런타임을 사용합니다. Runtime은 에이전트 실행 중 도구와 미들웨어에서 접근할 수 있는 컨텍스트 정보를 제공합니다.

**Runtime 구성 요소:**

| 구성 요소 | 설명 |
|:---|:---|
| **Context** | 사용자 ID, 데이터베이스 연결 등 정적 정보 |
| **Store** | 장기 메모리를 위한 `BaseStore` 인스턴스 |
| **Stream Writer** | `"custom"` 스트림 모드로 정보 스트리밍 |

런타임 정보는 도구와 미들웨어 내에서 `runtime` 매개변수를 통해 액세스할 수 있습니다.

> 참고 문서: [LangChain Runtime](https://docs.langchain.com/oss/python/langchain/runtime)


----

### ToolRuntime

도구는 에이전트 상태, 런타임 컨텍스트 및 장기 메모리에 액세스할 수 있을 때 가장 강력합니다. 이를 통해 도구는 컨텍스트 인식 결정을 내리고, 응답을 개인화하며, 대화 전반에 걸쳐 정보를 유지할 수 있습니다.

`ToolRuntime` 매개변수를 통해 다음 런타임 정보에 액세스할 수 있습니다:

| 속성 | 설명 |
|:---|:---|
| **state** | 실행을 통해 흐르는 변경 가능한 데이터 (메시지, 카운터, 커스텀 필드) |
| **context** | 사용자 ID, 세션 세부 정보 등 불변 구성 정보 |
| **store** | 대화 전반에 걸친 영구 장기 메모리 |
| **stream_writer** | 도구 실행 중 커스텀 업데이트 스트리밍 |
| **config** | 실행을 위한 RunnableConfig |
| **tool_call_id** | 현재 도구 호출의 고유 ID |




