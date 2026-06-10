"""
참고 문서:
PART02-에이전트/Ch02-에이전트/01-LangGraph-Agents.ipynb
test/01-LangGraph-Middleware.ipynb

핵심:
``ToolRetryMiddleware`` 로 실패한 **도구 호출**을 지수 백오프·지터와 함께
자동 재시도한다 (``wrap_tool_call``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain.agents.middleware._retry import OnFailure, RetryOn
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolRuntime

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel

# 원본 노트북 기본값
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_FACTOR = 2.0
_DEFAULT_INITIAL_DELAY = 1.0
_DEFAULT_MAX_DELAY = 60.0
_DEFAULT_JITTER = True
_DEFAULT_ON_FAILURE: OnFailure = "continue"

_TOOL_RETRY_FAILED_PREFIX = "failed after"


# 간단한 도구 정의
@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"It's sunny in {city}!"


def make_flaky_api_tool(
    *,
    fail_count: int = 2,
    exception_type: type[Exception] = ConnectionError,
    message: str = "temporary network error",
) -> tuple[BaseTool, Callable[[], int]]:
    """테스트용: 처음 ``fail_count`` 회는 예외, 이후 성공하는 도구와 호출 횟수 조회 함수."""
    state = {"calls": 0}

    @tool
    def flaky_api(city: str) -> str:
        """Simulate a flaky external API that may fail temporarily."""
        state["calls"] += 1
        if state["calls"] <= fail_count:
            raise exception_type(message)
        return f"API OK in {city}!"

    return flaky_api, lambda: state["calls"]


def make_tool_retry_middleware(
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    tools: list[BaseTool | str] | None = None,
    retry_on: RetryOn = (Exception,),
    on_failure: OnFailure = _DEFAULT_ON_FAILURE,
    backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
    initial_delay: float = _DEFAULT_INITIAL_DELAY,
    max_delay: float = _DEFAULT_MAX_DELAY,
    jitter: bool = _DEFAULT_JITTER,
) -> ToolRetryMiddleware:
    """``ToolRetryMiddleware`` 인스턴스를 생성한다."""
    return ToolRetryMiddleware(
        max_retries=max_retries,
        tools=tools,
        retry_on=retry_on,
        on_failure=on_failure,
        backoff_factor=backoff_factor,
        initial_delay=initial_delay,
        max_delay=max_delay,
        jitter=jitter,
    )


def _default_tool_handler(request: ToolCallRequest) -> ToolMessage:
    """``ToolCallRequest`` 에서 도구를 1회 실행해 ``ToolMessage`` 를 반환한다."""
    if request.tool is None:
        msg = "ToolCallRequest.tool is required for default handler"
        raise ValueError(msg)
    result = request.tool.invoke(request.tool_call["args"])
    content = result if isinstance(result, str) else str(result)
    return ToolMessage(
        content=content,
        tool_call_id=request.tool_call["id"],
        name=request.tool.name,
    )


def invoke_tool_with_retry(
    middleware: ToolRetryMiddleware,
    tool: BaseTool,
    args: dict[str, Any],
    *,
    tool_call_id: str = "test-call-1",
    handler: Callable[[ToolCallRequest], ToolMessage] | None = None,
) -> ToolMessage:
    """``wrap_tool_call`` 로 도구 1건을 실행한다 (LLM 없이 미들웨어 검증용)."""
    request = ToolCallRequest(
        tool_call={
            "name": tool.name,
            "args": args,
            "id": tool_call_id,
            "type": "tool_call",
        },
        tool=tool,
        state={"messages": []},
        runtime=ToolRuntime(
            state={"messages": []},
            context=None,
            config={},
            stream_writer=lambda _: None,
            tool_call_id=tool_call_id,
            store=None,
        ),
    )
    effective_handler = handler or _default_tool_handler
    result = middleware.wrap_tool_call(request, effective_handler)
    if not isinstance(result, ToolMessage):
        msg = f"Expected ToolMessage, got {type(result).__name__}"
        raise TypeError(msg)
    return result


def is_tool_retry_failure_message(message: ToolMessage) -> bool:
    """재시도 소진 후 ``on_failure='continue'`` 로 반환된 오류 ``ToolMessage`` 인지 확인."""
    return (
        message.status == "error"
        and isinstance(message.content, str)
        and _TOOL_RETRY_FAILED_PREFIX in message.content
    )


class MiddlewareToolRetryAgent(BaseGraph):
    """``ToolRetryMiddleware`` 가 붙은 ``create_agent`` 데모.

    외부에서는 ``g = MiddlewareToolRetryAgent()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.

    기본은 원본 노트북과 같이 최대 3회 재시도·지수 백오프·지터이다.
    ``middleware`` 로 인스턴스를 통째로 바꿀 수 있다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        middleware: ToolRetryMiddleware | None = None,
        tools: list[Any] | None = None,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._llm: BaseChatModel = init_chat_model(model)
        self._middleware_override = middleware
        self._tools = [get_weather] if tools is None else tools
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def middleware(self) -> ToolRetryMiddleware:
        return self._build_middleware()

    def _build_middleware(self) -> ToolRetryMiddleware:
        if self._middleware_override is not None:
            return self._middleware_override
        return make_tool_retry_middleware()

    def _compile_graph(self) -> CompiledStateGraph:
        agent = create_agent(
            model=self._llm,
            tools=self._tools,
            middleware=[self._build_middleware()],
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "MiddlewareToolRetryAgent",
    "_DEFAULT_BACKOFF_FACTOR",
    "_DEFAULT_INITIAL_DELAY",
    "_DEFAULT_JITTER",
    "_DEFAULT_MAX_DELAY",
    "_DEFAULT_MAX_RETRIES",
    "_DEFAULT_ON_FAILURE",
    "_TOOL_RETRY_FAILED_PREFIX",
    "get_weather",
    "invoke_tool_with_retry",
    "is_tool_retry_failure_message",
    "make_flaky_api_tool",
    "make_tool_retry_middleware",
]
