"""
참고 문서:
PART02-에이전트/Ch02-에이전트/01-LangGraph-Agents.ipynb
test/01-LangGraph-Middleware.ipynb

핵심:
``PIIMiddleware`` 로 이메일·신용카드·커스텀 패턴 등 PII를 감지하고
``redact`` / ``mask`` / ``hash`` / ``block`` 전략으로 처리한다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.agents.middleware.pii import PIIDetectionError, PIIMatch
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel

PIIStrategy = Literal["block", "redact", "mask", "hash"]
# 정규식을 사용한 커스텀 PII 유형 (OpenAI API 키 형식)
_DEFAULT_API_KEY_DETECTOR = r"sk-[a-zA-Z0-9]{32}"


# 간단한 도구 정의
@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"It's sunny in {city}!"


def make_pii_middleware(
    pii_type: Literal["email", "credit_card", "ip", "mac_address", "url"] | str,
    *,
    strategy: PIIStrategy = "redact",
    detector: Callable[[str], list[PIIMatch]] | str | None = None,
    apply_to_input: bool = True,
    apply_to_output: bool = False,
    apply_to_tool_results: bool = False,
) -> PIIMiddleware:
    """``PIIMiddleware`` 인스턴스를 생성한다."""
    return PIIMiddleware(
        pii_type,
        strategy=strategy,
        detector=detector,
        apply_to_input=apply_to_input,
        apply_to_output=apply_to_output,
        apply_to_tool_results=apply_to_tool_results,
    )


def make_default_pii_middlewares() -> list[PIIMiddleware]:
    """원본 노트북 기본 3종 PII 미들웨어."""
    return [
        # 사용자 입력에서 이메일 수정
        make_pii_middleware("email", strategy="redact", apply_to_input=True),
        # 신용카드 마스킹 (마지막 4자리 표시)
        make_pii_middleware("credit_card", strategy="mask", apply_to_input=True),
        # 정규식을 사용한 커스텀 PII 유형
        make_pii_middleware(
            "api_key",
            detector=_DEFAULT_API_KEY_DETECTOR,
            strategy="mask",  # 감지 시 마스킹 처리
        ),
    ]


def sanitize_human_input(middleware: PIIMiddleware, content: str) -> str:
    """``before_model`` 로 사용자 입력 1건의 PII 처리 결과를 반환한다 (LLM 없이 검증용)."""
    result = middleware.before_model(
        {"messages": [HumanMessage(content=content)]},
        None,  # type: ignore[arg-type]
    )
    if result is None:
        return content
    return str(result["messages"][0].content)


def sanitize_ai_output(middleware: PIIMiddleware, content: str) -> str:
    """``after_model`` 로 AI 출력 1건의 PII 처리 결과를 반환한다 (LLM 없이 검증용)."""
    result = middleware.after_model(
        {"messages": [AIMessage(content=content)]},
        None,  # type: ignore[arg-type]
    )
    if result is None:
        return content
    return str(result["messages"][0].content)


class MiddlewarePIIAgent(BaseGraph):
    """``PIIMiddleware`` 가 붙은 ``create_agent`` 데모.

    외부에서는 ``g = MiddlewarePIIAgent()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.

    기본은 원본 노트북과 같이 이메일 ``redact`` + 신용카드 ``mask`` + API 키 ``mask`` 3종이다.
    ``middlewares`` 로 목록을 통째로 바꿀 수 있다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        middlewares: Sequence[PIIMiddleware] | None = None,
        tools: list[Any] | None = None,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._llm: BaseChatModel = init_chat_model(model)
        self._middlewares = (
            list(middlewares) if middlewares is not None else make_default_pii_middlewares()
        )
        self._tools = [get_weather] if tools is None else tools
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def middlewares(self) -> list[PIIMiddleware]:
        return self._middlewares

    def _compile_graph(self) -> CompiledStateGraph:
        agent = create_agent(
            model=self._llm,
            tools=self._tools,
            middleware=self._middlewares,
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "MiddlewarePIIAgent",
    "PIIDetectionError",
    "PIIStrategy",
    "_DEFAULT_API_KEY_DETECTOR",
    "get_weather",
    "make_default_pii_middlewares",
    "make_pii_middleware",
    "sanitize_ai_output",
    "sanitize_human_input",
]
