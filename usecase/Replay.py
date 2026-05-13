"""
참고 문서:
/langgraph-v1-tutorial/PART01-LangGraph-기초/Ch01-그래프-생성하기/01-QuickStart-LangGraph-Tutorial.ipynb
(체크포인트·상태 이력·time travel 실습용 그래프)


Replay는 해당 파일이 아닌 ipynb를 확인할것!
"""

from __future__ import annotations

from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel


class ReplayState(TypedDict):
    """체크포인트·상태 이력 테스트용 그래프 상태.

    messages: 대화 메시지 리스트
    - add_messages 리듀서로 새 메시지가 누적됨 (덮어쓰기 아님)
    """

    messages: Annotated[list[AnyMessage], add_messages]


class Replay(BaseGraph):
    """상태 이력(get_state_history)·update_state·재실행용 체크포인트 그래프.

    ``InMemorySaver`` 로 컴파일한다. 외부에서는 ``g = Replay()`` 뒤
    ``g.invoke(...)`` / ``g.stream(...)`` / ``g.graph.get_state_history(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        tavily_max_results: int = 2,
        load_env: bool = True,
        langsmith_project: str | None = "LangChain-V1-Tutorial",
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        # 도구와 LLM 설정
        self._tools: list[BaseTool] = [TavilySearch(max_results=tavily_max_results)]
        self._llm: BaseChatModel = init_chat_model(model)
        self._llm_with_tools_tt = self._llm.bind_tools(self._tools)
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    def _chatbot_tt(self, state: ReplayState) -> dict[str, list[AnyMessage]]:
        """상태 관리 테스트용 챗봇"""
        return {"messages": [self._llm_with_tools_tt.invoke(state["messages"])]}

    def _compile_graph(self) -> CompiledStateGraph:
        # 상태 관리 테스트를 위한 체크포인트 기반 그래프
        graph_builder = StateGraph(ReplayState)

        # 그래프 구성
        graph_builder.add_node("chatbot", self._chatbot_tt)
        tool_node_tt = ToolNode(tools=self._tools)
        graph_builder.add_node("tools", tool_node_tt)

        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge(START, "chatbot")

        # 메모리와 함께 컴파일
        memory_tt = InMemorySaver()
        return graph_builder.compile(checkpointer=memory_tt)


__all__ = ["Replay", "ReplayState"]
