"""
참고 문서:
/langgraph-v1-tutorial/PART01-LangGraph-기초/Ch01-그래프-생성하기/01-QuickStart-LangGraph-Tutorial.ipynb
(Command 로 ToolMessage 상태를 갱신하는 human_review 예제)
"""

from __future__ import annotations

from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel


# 확장된 State 정의
class CustomState(TypedDict):
    """커스텀 필드가 추가된 상태"""

    messages: Annotated[list[AnyMessage], add_messages]
    human_feedback: str  # 사람의 피드백

"""
    ToolMessage를 직접 갱신
"""
@tool
def human_review(
    human_feedback: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Request human review for information."""
    # 인간에게 검토 요청
    human_response = interrupt(
        {"question": "이 정보가 맞나요?", "human_feedback": human_feedback}
    )

    feedback = human_response.get("human_feedback", "")

    if feedback.strip() == "":
        # 사용자가 AI 의 답변에 동의하는 경우
        tool_content = human_response
    else:
        # 사용자가 AI 의 답변에 동의하지 않는 경우
        tool_content = f"# 사용자에 의해 수정된 피드백: {feedback}"

    return Command(
        update={
            "messages": [ToolMessage(tool_content, tool_call_id=tool_call_id)]
        }
    )


class UpdateStateUseCommand(BaseGraph):
    """interrupt 후 ``Command`` 로 메시지 상태를 갱신하는 챗봇 + ToolNode 루프.

    체크포인터가 필요하므로 ``InMemorySaver`` 로 컴파일한다.
    외부에서는 ``g = UpdateStateUseCommand()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
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

        self._tools: list[BaseTool] = [human_review]
        self._llm: BaseChatModel = init_chat_model(model)
        self._llm_with_custom_tools = self._llm.bind_tools(self._tools)
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    def _chatbot_custom(self, state: CustomState) -> dict[str, list[AnyMessage]]:
        """커스텀 상태를 사용하는 챗봇"""
        message = self._llm_with_custom_tools.invoke(state["messages"])

        if hasattr(message, "tool_calls"):
            assert len(message.tool_calls) <= 1,  "병렬 도구 호출은 interrupt와 함께 사용할 수 없습니다"

        return {"messages": [message]}

    def _compile_graph(self) -> CompiledStateGraph:
        # 새로운 그래프 구성
        builder = StateGraph(CustomState)  # CustomState 사용

        # 노드와 엣지 추가
        builder.add_node("chatbot", self._chatbot_custom)
        builder.add_node("tools", ToolNode(tools=self._tools))

        builder.add_conditional_edges("chatbot", tools_condition)
        builder.add_edge("tools", "chatbot")
        builder.add_edge(START, "chatbot")

        # 컴파일 (interrupt 에 필수)
        memory_custom = InMemorySaver()
        return builder.compile(checkpointer=memory_custom)


__all__ = ["CustomState", "UpdateStateUseCommand", "human_review"]
