"""
참고 문서:
PART02-에이전트/Ch02-에이전트/01-LangGraph-Agents.ipynb
test/01-LangGraph-Middleware.ipynb

핵심:
``ToolCallLimitMiddleware`` 로 스레드·실행 단위 **도구 호출** 횟수를 제한한다.
``tool_name`` 으로 특정 도구만 제한하거나 전역(``__all__``)으로 적용할 수 있으며,
``exit_behavior`` 에 따라 차단 메시지(``continue``), 종료(``end``), 예외(``error``)를 선택한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.base import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel

ExitBehavior = Literal["continue", "error", "end"]

# 모든 도구 호출 제한 (원본 노트북 기본값)
_DEFAULT_GLOBAL_THREAD_LIMIT = 20
_DEFAULT_GLOBAL_RUN_LIMIT = 10
# 특정 도구 제한
_DEFAULT_TOOL_THREAD_LIMIT = 5
_DEFAULT_TOOL_RUN_LIMIT = 3
_DEFAULT_EXIT_BEHAVIOR: ExitBehavior = "continue"

_TOOL_LIMIT_BLOCKED_PREFIX = "Tool call limit exceeded"
_TOOL_LIMIT_END_MARKER = "call limit reached"


# 간단한 도구 정의
@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"It's sunny in {city}!"


@tool
def get_time(city: str) -> str:
    """Get the current local time for a given city."""
    return f"It's noon in {city}!"


def make_tool_call_limit_middleware(
    *,
    tool_name: str | None = None,
    thread_limit: int | None = None,
    run_limit: int | None = None,
    exit_behavior: ExitBehavior = _DEFAULT_EXIT_BEHAVIOR,
) -> ToolCallLimitMiddleware:
    """``ToolCallLimitMiddleware`` 인스턴스를 생성한다."""
    return ToolCallLimitMiddleware(
        tool_name=tool_name,
        thread_limit=thread_limit,
        run_limit=run_limit,
        exit_behavior=exit_behavior,
    )


def make_dual_tool_limit_middlewares(
    *,
    global_thread_limit: int = _DEFAULT_GLOBAL_THREAD_LIMIT,
    global_run_limit: int = _DEFAULT_GLOBAL_RUN_LIMIT,
    tool_name: str = "get_weather",
    tool_thread_limit: int = _DEFAULT_TOOL_THREAD_LIMIT,
    tool_run_limit: int = _DEFAULT_TOOL_RUN_LIMIT,
    exit_behavior: ExitBehavior = _DEFAULT_EXIT_BEHAVIOR,
) -> list[ToolCallLimitMiddleware]:
    """전역 + 특정 도구 이중 제한 미들웨어 (원본 노트북 패턴)."""
    # 모든 도구 호출 제한
    global_limiter = make_tool_call_limit_middleware(
        thread_limit=global_thread_limit,
        run_limit=global_run_limit,
        exit_behavior=exit_behavior,
    )
    # 특정 도구 제한
    tool_limiter = make_tool_call_limit_middleware(
        tool_name=tool_name,
        thread_limit=tool_thread_limit,
        run_limit=tool_run_limit,
        exit_behavior=exit_behavior,
    )
    return [global_limiter, tool_limiter]


def messages_contain_blocked_tool_call(
    messages: Sequence[BaseMessage | dict[str, Any]],
) -> bool:
    """``exit_behavior=\"continue\"`` / ``\"end\"`` 시 차단된 ``ToolMessage`` 존재 여부."""
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content
            status = msg.status
        elif isinstance(msg, dict) and msg.get("type") == "tool":
            content = msg.get("content", "")
            status = msg.get("status")
        else:
            continue
        if status == "error" and isinstance(content, str) and _TOOL_LIMIT_BLOCKED_PREFIX in content:
            return True
    return False


def messages_contain_tool_limit_end(
    messages: Sequence[BaseMessage | dict[str, Any]],
) -> bool:
    """``exit_behavior=\"end\"`` 시 주입된 한도 초과 ``AIMessage`` 존재 여부."""
    for msg in messages:
        content = msg.content if isinstance(msg, BaseMessage) else msg.get("content", "")
        if isinstance(content, str) and _TOOL_LIMIT_END_MARKER in content:
            return True
    return False


class MiddlewareToolCallLimitAgent(BaseGraph):
    """``ToolCallLimitMiddleware`` 가 붙은 ``create_agent`` 데모.

    외부에서는 ``g = MiddlewareToolCallLimitAgent()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.

    기본은 원본 노트북과 같이 **전역 + ``get_weather`` 전용** 이중 미들웨어입니다.
    ``middlewares`` 를 넘기면 목록을 그대로 사용하고, ``dual_limiters=False`` 이면
    단일 미들웨어(``tool_name`` / ``thread_limit`` / ``run_limit`` / ``exit_behavior``)만 씁니다.
    ``thread_limit`` 누적 테스트에는 ``checkpointer=True``(기본)와 ``thread_id`` 가 필요합니다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        tool_name: str | None = None,
        thread_limit: int | None = _DEFAULT_GLOBAL_THREAD_LIMIT,
        run_limit: int | None = _DEFAULT_GLOBAL_RUN_LIMIT,
        exit_behavior: ExitBehavior = _DEFAULT_EXIT_BEHAVIOR,
        dual_limiters: bool = True,
        global_thread_limit: int = _DEFAULT_GLOBAL_THREAD_LIMIT,
        global_run_limit: int = _DEFAULT_GLOBAL_RUN_LIMIT,
        tool_thread_limit: int = _DEFAULT_TOOL_THREAD_LIMIT,
        tool_run_limit: int = _DEFAULT_TOOL_RUN_LIMIT,
        limited_tool_name: str = "get_weather",
        middlewares: list[ToolCallLimitMiddleware] | None = None,
        tools: list[Any] | None = None,
        checkpointer: bool = True,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._llm: BaseChatModel = init_chat_model(model)
        self._tool_name = tool_name
        self._thread_limit = thread_limit
        self._run_limit = run_limit
        self._exit_behavior: ExitBehavior = exit_behavior
        self._dual_limiters = dual_limiters
        self._global_thread_limit = global_thread_limit
        self._global_run_limit = global_run_limit
        self._tool_thread_limit = tool_thread_limit
        self._tool_run_limit = tool_run_limit
        self._limited_tool_name = limited_tool_name
        self._middlewares_override = middlewares
        self._tools = [get_weather, get_time] if tools is None else tools
        self._use_checkpointer = checkpointer
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def tool_name(self) -> str | None:
        return self._tool_name

    @property
    def thread_limit(self) -> int | None:
        return self._thread_limit

    @property
    def run_limit(self) -> int | None:
        return self._run_limit

    @property
    def exit_behavior(self) -> ExitBehavior:
        return self._exit_behavior

    def _build_middlewares(self) -> list[ToolCallLimitMiddleware]:
        if self._middlewares_override is not None:
            return self._middlewares_override
        if self._dual_limiters:
            return make_dual_tool_limit_middlewares(
                global_thread_limit=self._global_thread_limit,
                global_run_limit=self._global_run_limit,
                tool_name=self._limited_tool_name,
                tool_thread_limit=self._tool_thread_limit,
                tool_run_limit=self._tool_run_limit,
                exit_behavior=self._exit_behavior,
            )
        return [
            make_tool_call_limit_middleware(
                tool_name=self._tool_name,
                thread_limit=self._thread_limit,
                run_limit=self._run_limit,
                exit_behavior=self._exit_behavior,
            )
        ]

    def _compile_graph(self) -> CompiledStateGraph:
        agent = create_agent(
            model=self._llm,
            tools=self._tools,
            middleware=self._build_middlewares(),
            checkpointer=MemorySaver() if self._use_checkpointer else None,
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "ExitBehavior",
    "MiddlewareToolCallLimitAgent",
    "ToolCallLimitExceededError",
    "_DEFAULT_EXIT_BEHAVIOR",
    "_DEFAULT_GLOBAL_RUN_LIMIT",
    "_DEFAULT_GLOBAL_THREAD_LIMIT",
    "_DEFAULT_TOOL_RUN_LIMIT",
    "_DEFAULT_TOOL_THREAD_LIMIT",
    "_TOOL_LIMIT_BLOCKED_PREFIX",
    "_TOOL_LIMIT_END_MARKER",
    "get_time",
    "get_weather",
    "make_dual_tool_limit_middlewares",
    "make_tool_call_limit_middleware",
    "messages_contain_blocked_tool_call",
    "messages_contain_tool_limit_end",
]
