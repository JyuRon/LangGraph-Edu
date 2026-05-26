"""
참고 문서:
PART02-에이전트/Ch02-에이전트/02-LangGraph-Tools.ipynb

핵심:
``ToolRuntime`` 매개변수로 도구 실행 시 ``state``·``context`` 등 런타임 정보에 접근한다.
``runtime: ToolRuntime`` 은 LLM 도구 스키마에 노출되지 않고 자동 주입된다.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, cast

from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, RemoveMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import NotRequired

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel

_SYSTEM_PROMPT = "You are a helpful assistant."



class CustomContext(BaseModel):
    """``create_agent`` 의 ``context_schema`` — 런타임 ``context`` 로 전달."""

    user_preferences: dict[str, Any] | None = None


class CustomState(AgentState):
    """커스텀 상태 스키마 (``state_schema`` 로 쓸 때 참고).

    ``messages`` 는 ``AgentState`` 에 ``add_messages`` 리듀서로 정의되어 있다.
    """

    user_name: NotRequired[Annotated[list[AnyMessage], "The user's name"]]
    



# ToolRuntime 상태 업데이트 예시
# User Name 업데이트 도구
@tool
def update_user_name(new_name: str, runtime: ToolRuntime) -> Command:
    """Update the user's name."""
    return Command(
        update={
            "user_name": new_name,  # user_name 상태에 업데이트
            "messages": [
                ToolMessage(
                    content=f"Successfully updated user name to {new_name}",
                    tool_call_id=runtime.tool_call_id,  # runtime 에서 얻어온 tool_call_id 정보를 활용하여 업데이트
                )
            ],
        }
    )


# ToolRuntime 상태 업데이트 예시
@tool
def clear_messages(runtime: ToolRuntime) -> Command:
    """Clear all messages from the conversation history except the one whose tool_call_id matches the id we don't want to delete."""
    from langchain.messages import AIMessage

    messages = runtime.state.get("messages", [])

    to_remove_messages = []
    tool_call_id = runtime.tool_call_id

    for m in messages:
        # 단일 메시지에 여러개의 tool call이 있을 수 있음
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            # Tool Call ID 가 일치하지 않으면 삭제. Tool Call ID 가 일치하면 유지.
            if not any(call.get("id") == tool_call_id for call in m.tool_calls):
                to_remove_messages.append(m)
        else:
            to_remove_messages.append(m)

    removals = [RemoveMessage(id=m.id) for m in to_remove_messages]
    return Command(
        update={
            "messages": removals
            + [
                ToolMessage(
                    content=f"Successfully cleared all previous messages. Total of {len(removals)} deleted messages.",
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )





class ToolRuntimeAgent(BaseGraph):
    """``ToolRuntime`` 으로 ``state``·``context`` 에 접근하는 ``create_agent`` 데모.

    외부에서는 ``g = ToolRuntimeAgent()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
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

        self._llm: BaseChatModel = init_chat_model(model)
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    def _compile_graph(self) -> CompiledStateGraph:
        agent = create_agent(
            self._llm,
            tools=[update_user_name,clear_messages],
            system_prompt=_SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
            context_schema=CustomContext,
            state_schema=CustomState,
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "CustomContext",
    "CustomState",
    "ToolRuntimeAgent",
]
