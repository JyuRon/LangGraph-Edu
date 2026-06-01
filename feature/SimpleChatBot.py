"""
참고 문서:
/langgraph-v1-tutorial/PART01-LangGraph-기초/Ch01-그래프-생성하기/01-QuickStart-LangGraph-Tutorial.ipynb
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel


class ChatBotState(TypedDict):
    """챗봇 그래프 상태.

    messages: 대화 메시지 리스트
    - add_messages 리듀서로 새 메시지가 누적됨 (덮어쓰기 아님)
    """

    messages: Annotated[list[AnyMessage], add_messages]


class SimpleChatBot(BaseGraph):
    """단일 LLM 노드 그래프(START → chatbot → END).

    외부에서는 ``a = SimpleChatBot()`` 뒤 ``a.invoke(...)`` / ``a.stream(...)``,
    LangGraph **노드 구조**는 ``a.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        load_env: bool = True,
        langsmith_project: str | None = "LangChain-V1-Tutorial",
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._llm: BaseChatModel = init_chat_model(model)
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    def _chatbot(self, state: ChatBotState) -> dict[str, list[AnyMessage]]:
        response = self._llm.invoke(state["messages"])
        return {"messages": [response]}

    def _compile_graph(self) -> CompiledStateGraph:
        builder = StateGraph(ChatBotState)
        builder.add_node("chatbot", self._chatbot)
        builder.add_edge(START, "chatbot")
        builder.add_edge("chatbot", END)
        return builder.compile()


__all__ = ["ChatBotState", "SimpleChatBot"]
