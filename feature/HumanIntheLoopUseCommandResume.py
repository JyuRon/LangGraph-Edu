"""
참고 문서:
/langgraph-v1-tutorial/PART01-LangGraph-기초/Ch01-그래프-생성하기/01-QuickStart-LangGraph-Tutorial.ipynb
"""

from __future__ import annotations

from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt
from typing_extensions import TypedDict

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel


@tool
def human_assistance(query: str) -> str:
    """Request assistance from an expert(human)."""
    # interrupt를 호출하여 실행 일시 중지
    # 사람의 응답을 기다림
    human_response = interrupt({"query": query})

    print(human_response)

    # 사람의 응답 반환
    return human_response["data"]


class HumanHitlState(TypedDict):
    """Human-in-the-loop(interrupt 도구) 챗봇 상태.

    messages: 대화 메시지 리스트
    - add_messages 리듀서로 새 메시지가 누적됨 (덮어쓰기 아님)
    """

    messages: Annotated[list[AnyMessage], add_messages]


class HumanIntheLoopUseCommand(BaseGraph):
    """interrupt + ``Command(resume=...)`` 로 사람 개입을 받는 챗봇 + ToolNode 루프.

    interrupt 사용 시 체크포인터가 필요하므로 ``InMemorySaver`` 로 컴파일한다.
    외부에서는 ``g = HumanIntheLoopUseCommand()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._tools: list[BaseTool] = [human_assistance]
        self._llm: BaseChatModel = init_chat_model(model)
        self._llm_with_human_tools = self._llm.bind_tools(self._tools)
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    def _chatbot_with_human(self, state: HumanHitlState) -> dict[str, list[AnyMessage]]:
        """Human interruption 을 요청할 수 있는 챗봇 노드."""
        message = self._llm_with_human_tools.invoke(state["messages"])

        # interrupt 중 병렬 도구 호출 방지
        # (재개 시 도구 호출이 반복되는 것을 방지)
        if hasattr(message, "tool_calls"):
            assert (
                len(message.tool_calls) <= 1
            ), "병렬 도구 호출은 interrupt와 함께 사용할 수 없습니다"

        return {"messages": [message]}

    def _compile_graph(self) -> CompiledStateGraph:
        # 메모리와 함께 컴파일 (interrupt 에 필수)
        memory_hitl = InMemorySaver()
        builder = StateGraph(HumanHitlState)
        builder.add_node("chatbot_with_human", self._chatbot_with_human)
        builder.add_node("tools", ToolNode(tools=self._tools))

        builder.add_conditional_edges("chatbot_with_human", tools_condition)
        builder.add_edge("tools", "chatbot_with_human")
        builder.add_edge(START, "chatbot_with_human")

        return builder.compile(checkpointer=memory_hitl)


__all__ = ["HumanHitlState", "HumanIntheLoopUseCommand", "human_assistance"]
