"""
참고 문서:
PART02-에이전트/Ch02-에이전트/01-LangGraph-Agents.ipynb
test/01-LangGraph-Middleware.ipynb

핵심:
``ModelCallLimitMiddleware`` 로 스레드·실행 단위 모델 호출 횟수를 제한하고,
한도 초과 시 ``exit_behavior`` 에 따라 종료(``end``) 또는 예외(``error``)한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.base import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel

ExitBehavior = Literal["end", "error"]

_DEFAULT_THREAD_LIMIT = 3  # 스레드당 최대 3회 호출 (실행 전반)
_DEFAULT_RUN_LIMIT = 2  # 실행당 최대 2회 호출 (단일 호출)
_DEFAULT_EXIT_BEHAVIOR: ExitBehavior = "end"  # 또는 "error"로 예외 발생
_LIMIT_EXCEEDED_PREFIX = "Model call limits exceeded:"


# 간단한 도구 정의
@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"It's sunny in {city}!"


def make_model_call_limit_middleware(
    *,
    thread_limit: int | None = _DEFAULT_THREAD_LIMIT,
    run_limit: int | None = _DEFAULT_RUN_LIMIT,
    exit_behavior: ExitBehavior = _DEFAULT_EXIT_BEHAVIOR,
) -> ModelCallLimitMiddleware:
    """``ModelCallLimitMiddleware`` 인스턴스를 생성한다."""
    return ModelCallLimitMiddleware(
        thread_limit=thread_limit,
        run_limit=run_limit,
        exit_behavior=exit_behavior,
    )


def messages_contain_limit_exceeded(
    messages: Sequence[BaseMessage | dict[str, Any]],
) -> bool:
    """``exit_behavior=\"end\"`` 시 주입된 한도 초과 ``AIMessage`` 존재 여부."""
    for msg in messages:
        content = msg.content if isinstance(msg, BaseMessage) else msg.get("content", "")
        if isinstance(content, str) and content.startswith(_LIMIT_EXCEEDED_PREFIX):
            return True
    return False


class MiddlewareModelCallLimitAgent(BaseGraph):
    """``ModelCallLimitMiddleware`` 가 붙은 ``create_agent`` 데모.

    외부에서는 ``g = MiddlewareModelCallLimitAgent()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.

    ``thread_limit`` / ``run_limit`` / ``exit_behavior`` 를 생성자에서 넘긴다.
    ``thread_limit`` 는 스레드 간 호출 수를 누적하므로 ``checkpointer=True``(기본)와
    ``thread_id`` config 를 함께 쓴다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        thread_limit: int | None = _DEFAULT_THREAD_LIMIT,
        run_limit: int | None = _DEFAULT_RUN_LIMIT,
        exit_behavior: ExitBehavior = _DEFAULT_EXIT_BEHAVIOR,
        tools: list[Any] | None = None,
        checkpointer: bool = True,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._llm: BaseChatModel = init_chat_model(model)
        self._thread_limit = thread_limit
        self._run_limit = run_limit
        self._exit_behavior: ExitBehavior = exit_behavior
        self._tools = [get_weather] if tools is None else tools
        self._use_checkpointer = checkpointer
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def thread_limit(self) -> int | None:
        return self._thread_limit

    @property
    def run_limit(self) -> int | None:
        return self._run_limit

    @property
    def exit_behavior(self) -> ExitBehavior:
        return self._exit_behavior

    def _compile_graph(self) -> CompiledStateGraph:
        middleware = make_model_call_limit_middleware(
            thread_limit=self._thread_limit,
            run_limit=self._run_limit,
            exit_behavior=self._exit_behavior,
        )
        agent = create_agent(
            model=self._llm,
            tools=self._tools,
            middleware=[middleware],
            checkpointer=MemorySaver() if self._use_checkpointer else None,
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "ExitBehavior",
    "MiddlewareModelCallLimitAgent",
    "ModelCallLimitExceededError",
    "_DEFAULT_EXIT_BEHAVIOR",
    "_DEFAULT_RUN_LIMIT",
    "_DEFAULT_THREAD_LIMIT",
    "_LIMIT_EXCEEDED_PREFIX",
    "get_weather",
    "make_model_call_limit_middleware",
    "messages_contain_limit_exceeded",
]
