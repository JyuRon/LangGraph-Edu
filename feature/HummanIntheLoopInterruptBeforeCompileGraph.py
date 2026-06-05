"""
참고 문서:
note/LangGraph-Interrupt-Patterns.md
/langgraph-v1-tutorial/PART02-에이전트/Ch05-Human-in-the-Loop/06-LangGraph-Human-In-the-Loop.ipynb

핵심:
``compile(..., interrupt_before=[...])`` 로 특정 노드 실행 **전**에 그래프를 멈춘다.
체크포인터와 ``thread_id`` 가 필요하므로 ``InMemorySaver`` 로 컴파일한다.
"""

from __future__ import annotations

from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool, tool
from util.news import GoogleNews
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel


########## 1. 상태 정의 ##########
# 상태 정의
class InterruptBeforeState(TypedDict):
    """interrupt_before HITL 챗봇 상태.

    messages: 대화 메시지 리스트
    - add_messages 리듀서로 새 메시지가 누적됨 (덮어쓰기 아님)
    """

    messages: Annotated[list[AnyMessage], add_messages]


########## 2. 도구 정의 및 바인딩 ##########
# 키워드로 뉴스 검색하는 도구 생성
@tool
def search_keyword(query: str) -> list[dict[str, str]]:
    """Look up news by keyword"""
    news_tool = GoogleNews()
    return news_tool.search_by_keyword(query, k=5)


class HumanIntheLoopInterruptBeforeCompileGraph(BaseGraph):
    """``compile(interrupt_before=[...])`` 로 tools 노드 실행 전 사람 개입 지점을 고정하는 챗봇.

    체크포ин터가 필요하므로 ``InMemorySaver`` 를 사용한다.
    외부에서는 ``g = HumanIntheLoopInterruptBeforeCompileGraph()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        interrupt_before: list[str] | None = None,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._tools: list[BaseTool] = [search_keyword]
        self._llm: BaseChatModel = init_chat_model(model)
        self._llm_with_tools = self._llm.bind_tools(self._tools)
        # compile 시점 interrupt_before 대상 (기본: tools 노드 실행 전)
        self._interrupt_before = (
            interrupt_before if interrupt_before is not None else ["tools"]
        )
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    ########## 3. 노드 추가 ##########
    # 챗봇 함수 정의
    def _chatbot(self, state: InterruptBeforeState) -> dict[str, list[AnyMessage]]:
        # 메시지 호출 및 반환
        message = self._llm_with_tools.invoke(state["messages"])
        return {"messages": [message]}

    def _compile_graph(self) -> CompiledStateGraph:
        # 상태 그래프 생성
        builder = StateGraph(InterruptBeforeState)

        # 챗봇 노드 추가
        builder.add_node("chatbot", self._chatbot)

        # 도구 노드 생성 및 추가
        builder.add_node("tools", ToolNode(tools=self._tools))

        # 조건부 엣지
        builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )

        ########## 4. 엣지 추가 ##########

        # tools > chatbot
        builder.add_edge("tools", "chatbot")

        # START > chatbot
        builder.add_edge(START, "chatbot")

        ########## 5. MemorySaver 추가 ##########

        # 메모리 저장소 초기화 (interrupt_before 에 필수)
        memory = InMemorySaver()
        return builder.compile(
            checkpointer=memory,
            interrupt_before=self._interrupt_before,
        )


__all__ = [
    "HumanIntheLoopInterruptBeforeCompileGraph",
    "InterruptBeforeState",
    "search_keyword",
]
