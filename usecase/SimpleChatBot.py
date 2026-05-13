"""
참고 문서:
/langgraph-v1-tutorial/PART01-LangGraph-기초/Ch01-그래프-생성하기/01-QuickStart-LangGraph-Tutorial.ipynb
"""

from __future__ import annotations

from typing import Annotated, Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from util import logging
from util.chat_model_enums import LangChainChatModel


class ChatBotState(TypedDict):
    """챗봇 그래프 상태.

    messages: 대화 메시지 리스트
    - add_messages 리듀서로 새 메시지가 누적됨 (덮어쓰기 아님)
    """

    messages: Annotated[list[AnyMessage], add_messages]


class SimpleChatBot:
    """단일 LLM 노드 그래프(START → chatbot → END).

    외부에서는 인스턴스를 만든 뒤 ``.graph`` / ``.invoke`` / ``.stream`` 으로 사용한다.

    Example:
        >>> bot = SimpleChatBot(model=LangChainChatModel.OPENAI_GPT_4O_MINI)
        >>> bot.invoke({"messages": [{"role": "user", "content": "안녕"}]})
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        load_env: bool = True,
        langsmith_project: str | None = "LangChain-V1-Tutorial",
    ) -> None:
        if load_env:
            load_dotenv(override=True)
        if langsmith_project:
            logging.langsmith(langsmith_project)

        self._llm: BaseChatModel = init_chat_model(model)
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def graph(self) -> CompiledStateGraph:
        return self._graph

    def _chatbot(self, state: ChatBotState) -> dict[str, list[AnyMessage]]:
        response = self._llm.invoke(state["messages"])
        return {"messages": [response]}

    def _compile_graph(self) -> CompiledStateGraph:
        builder = StateGraph(ChatBotState)
        builder.add_node("chatbot", self._chatbot)
        builder.add_edge(START, "chatbot")
        builder.add_edge("chatbot", END)
        return builder.compile()

    def invoke(
        self,
        state: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """컴파일된 그래프 ``invoke``를 그대로 호출한다. ``state``는 보통 ``ChatBotState`` 형태."""
        return self._graph.invoke(state, config=config, **kwargs)

    def stream(
        self,
        state: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ):
        """컴파일된 그래프 ``stream``을 그대로 호출한다."""
        return self._graph.stream(state, config=config, **kwargs)


__all__ = ["ChatBotState", "SimpleChatBot"]


if __name__ == "__main__":
    from util.graphs import visualize_graph

    bot = SimpleChatBot()
    print("StateGraph 생성 및 컴파일 완료!")
    print("실행 흐름: START → chatbot → END")
    visualize_graph(bot.graph)
