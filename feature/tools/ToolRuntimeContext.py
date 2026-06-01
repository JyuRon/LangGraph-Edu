"""
참고 문서:
PART02-에이전트/Ch02-에이전트/02-LangGraph-Tools.ipynb

핵심:
``ToolRuntime`` 매개변수로 도구 실행 시 ``state``·``context`` 등 런타임 정보에 접근한다.
``runtime: ToolRuntime`` 은 LLM 도구 스키마에 노출되지 않고 자동 주입된다.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import RemoveMessage
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
    user_id: str | None = None


class CustomState(AgentState):
    """커스텀 상태 스키마 (``state_schema`` 로 쓸 때 참고).

    ``messages`` 는 ``AgentState`` 에 ``add_messages`` 리듀서로 정의되어 있다.
    """

    user_preferences: NotRequired[dict[str, Any]]



# ToolRuntime 사용 예시
# 현재 대화 상태 접근
@tool
def summarize_conversation(runtime: ToolRuntime) -> str:
    """Summarize the conversation so far."""
    # state 에서 메시지 접근
    messages = runtime.state.get("messages", [])
    human_msgs = sum(1 for m in messages if isinstance(m, HumanMessage))
    ai_msgs = sum(1 for m in messages if isinstance(m, AIMessage))
    tool_msgs = sum(1 for m in messages if isinstance(m, ToolMessage))
    return (
        f"Conversation has {human_msgs} user messages, "
        f"{ai_msgs} AI responses, and {tool_msgs} tool results"
    )


# ToolRuntime 사용 예시
@tool
def get_user_preference(
    preference_name: Literal["food", "coding", "sports"],
    runtime: ToolRuntime,  # ToolRuntime 매개변수는 모델에 보이지 않습니다 (자동 주입)
) -> str:
    """Get a user preference value."""

    # context는 state에 저장하지 않고 별도의 context 객체로 inject됨
    preferences: dict[str, Any] = {}
    ctx = cast(CustomContext | None, getattr(runtime, "context", None))
    if ctx is not None:
        # context dict 내 user_preferences
        preferences = ctx.user_preferences or {}
    return preferences.get(preference_name, "Have no information")



# 사용자 데이터베이스 시뮬레이션
USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com",
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com",
    },
}


@tool
def get_account_info(runtime: ToolRuntime[CustomContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"
    return "User not found"







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
            tools=[summarize_conversation, get_user_preference, get_account_info],
            system_prompt=_SYSTEM_PROMPT,
            # checkpointer=InMemorySaver(),
            context_schema=CustomContext,
            # state_schema=CustomState,
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "CustomContext",
    "CustomState",
    "ToolRuntimeAgent",
    "get_user_preference",
    "summarize_conversation",
]
