---

## 1. 어댑터 방식 (Wrapper Node Pattern)

메인 그래프와 서브그래프가 서로의 상태(State)를 전혀 몰라도 되는 **완벽한 결합도 분리** 방식입니다.

* **Spring 비유:** 계층 간 통신 시 `DTO`를 따로 두고 `Mapper`를 통해 값을 변환하여 전달하는 방식.
* **장점:** 완벽한 상태 격리 보장 (보안 유리), 서브 모듈의 재사용성 극대화.
* **단점:** 상태를 매핑하는 보일러플레이트 코드가 필요함.

```python
from typing import TypedDict
from langgraph.graph import StateGraph

# 1. 상태 정의 (A와 B의 교집합이 전혀 없음)
class MainState(TypedDict):
    user_request: str
    final_response: str

class SubState(TypedDict):
    search_keyword: str
    rag_context: str

# 2. 서브그래프 B (독립적)
# ... (sub_builder 구성 및 compile() 완료되었다고 가정) ...
# subgraph = sub_builder.compile()

# 3. ★ 어댑터 노드 (A -> B -> A 매핑) ★
def rag_adapter_node(state: MainState):
    # [A -> B] Request Mapping
    sub_input = {"search_keyword": state["user_request"] + " (보안 필터링됨)"}
    
    # 서브그래프 실행 (독립된 네임스페이스에서 동작)
    sub_output = subgraph.invoke(sub_input)
    
    # [B -> A] Response Mapping
    return {"final_response": f"검색된 내용: {sub_output['rag_context']}"}

# 4. 메인 그래프 조립
main_builder = StateGraph(MainState)
main_builder.add_node("rag_adapter", rag_adapter_node) # 어댑터를 노드로 등록

```

---

## 2. 직접 주입 방식 (Native Composition)

프레임워크가 상태 변환을 알아서 처리하도록, 메인 상태가 서브 상태의 필드를 포함(Superset)하는 방식입니다.

* **Spring 비유:** `Entity` 객체 하나를 컨트롤러부터 서비스, 레포지토리까지 통째로 들고 다니며 사용하는 방식.
* **장점:** 매핑 코드가 사라져 코드가 매우 간결해짐.
* **단점:** 메인 상태가 서브 모듈의 임시 데이터까지 떠안아 비대해지고 결합도가 높아짐.

```python
from typing import TypedDict
from langgraph.graph import StateGraph

# 1. 서브 상태 정의
class SubState(TypedDict):
    rag_query: str   # 메인과 공유해야 함
    rag_context: str # 메인과 공유해야 함

# 2. 메인 상태 정의 (★ SubState의 필드를 반드시 포함해야 함 ★)
class MainState(TypedDict):
    user_request: str
    rag_query: str     # 서브그래프 입력용
    rag_context: str   # 서브그래프 출력용
    final_response: str

# 3. 메인 그래프용 노드 (매핑 없이 값만 세팅)
def prepare_rag(state: MainState):
    # 어댑터 없이, 메인 상태에 서브그래프가 쓸 값을 미리 넣어둠
    return {"rag_query": state["user_request"]} 

# 4. 메인 그래프 조립
main_builder = StateGraph(MainState)
main_builder.add_node("prepare", prepare_rag)

# ★ 서브그래프를 일반 노드처럼 직접 꽂아 넣음 ★
# LangGraph가 MainState에서 'rag_query', 'rag_context'만 알아서 추출/병합함
main_builder.add_node("subgraph_node", subgraph) 

# 엣지 연결
main_builder.add_edge("prepare", "subgraph_node")

```

---

### 💡 요약 및 아키텍처 제언

폐쇄망 환경에서 동작하는 RAG 시스템을 구축하시며 **보안과 각 방법의 장단점**을 중요하게 생각하신다면, '어댑터 방식'을 우선적으로 고려해 보시기를 권장합니다.

초기 매핑 코드 작성의 번거로움은 있지만, 임시로 조회된 민감한 벡터 데이터나 프롬프트 검수 이력 등이 메인 파이프라인의 상태를 오염시키지 않도록 원천 차단할 수 있어 시스템의 안정성과 보안성이 크게 향상됩니다.

현재 구상 중이신 RAG 파이프라인의 서브 모듈들은 다른 그래프에서도 재사용될 여지가 많은 편인가요?