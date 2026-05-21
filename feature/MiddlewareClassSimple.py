"""
참고 문서:
PART02-에이전트/Ch02-에이전트/01-LangGraph-Agents.ipynb

핵심:
``AgentMiddleware`` 를 상속한 클래스로 ``state_schema`` 와 ``before_model`` 훅을 정의하고,
``AgentState`` 를 확장한 ``user_preferences`` 를 모델 호출 전에 로깅한다.
"""

from __future__ import annotations

from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from typing_extensions import NotRequired

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel
from util.messages import depth_colors

_MAGENTA, _RESET = depth_colors[1], depth_colors["reset"]


# 커스텀 상태 스키마 정의
class CustomState(AgentState):
    """``AgentState`` 확장 — ``user_preferences`` 로 호출 시 사용자 설정을 전달."""

    user_preferences: NotRequired[dict[str, Any]]


class CustomMiddleware(AgentMiddleware[CustomState]):
    """``state_schema`` 와 ``before_model`` 훅을 가진 클래스 기반 미들웨어."""

    state_schema = CustomState
    tools = []

    def before_model(
        self, state: CustomState, runtime: Runtime
    ) -> dict[str, Any] | None:
        # 모델 호출 전 커스텀 로직
        prefs = state.get("user_preferences") or {}
        print(
            f"{_MAGENTA}\n\n모델 호출 전 user_preferences: {prefs}{_RESET}"
        )
        return None


class MiddlewareClassSimpleAgent(BaseGraph):
    """``AgentMiddleware`` 서브클래스로 ``CustomState``·``before_model`` 을 쓰는 ``create_agent`` 데모.

    외부에서는 ``g = MiddlewareClassSimpleAgent()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
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
            model=self._llm,
            tools=[],
            middleware=[CustomMiddleware()],
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "CustomMiddleware",
    "CustomState",
    "MiddlewareClassSimpleAgent",
]
