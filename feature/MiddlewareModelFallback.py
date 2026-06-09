"""
참고 문서:
PART02-에이전트/Ch02-에이전트/01-LangGraph-Agents.ipynb
test/01-LangGraph-Middleware.ipynb

핵심:
``ModelFallbackMiddleware`` 로 **primary(Google Gemini)** 호출 실패 시
폴백 모델을 순서대로 시도한다 (``wrap_model_call``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel

# 오류 시 먼저 시도 → 그 다음 이것 (원본 노트북 순서)
_DEFAULT_FALLBACK_MODELS: tuple[str | LangChainChatModel, ...] = (
    LangChainChatModel.ANTHROPIC_CLAUDE_HAIKU_4_5,  # 오류 시 먼저 시도
    LangChainChatModel.OPENAI_GPT_4_1_MINI,  # 그 다음 이것
)


# 간단한 도구 정의
@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"It's sunny in {city}!"


def make_model_fallback_middleware(
    first_fallback: str | BaseChatModel,
    *additional_fallbacks: str | BaseChatModel,
) -> ModelFallbackMiddleware:
    """``ModelFallbackMiddleware`` 인스턴스를 생성한다.

    ``create_agent`` 의 ``model`` 이 1차(primary)이고,
    여기 넘긴 모델들이 오류 시 순서대로 폴백된다.
    """
    return ModelFallbackMiddleware(first_fallback, *additional_fallbacks)


def make_default_fallback_middleware() -> ModelFallbackMiddleware:
    """원본 노트북 기본 폴백 체인 — haiku → gpt-4.1-mini."""
    first, *rest = _DEFAULT_FALLBACK_MODELS
    return make_model_fallback_middleware(first, *rest)


class MiddlewareModelFallbackAgent(BaseGraph):
    """``ModelFallbackMiddleware`` 가 붙은 ``create_agent`` 데모.

    외부에서는 ``g = MiddlewareModelFallbackAgent()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.

    기본 primary 는 ``google_genai:gemini-2.5-flash`` (Google),
    폴백은 ``claude-haiku-4-5`` → ``openai:gpt-4.1-mini`` 순이다.
    ``fallback_models`` 로 폴백 목록을 바꿀 수 있다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.GOOGLE_GENAI_GEMINI_2_5_FLASH,
        *,
        fallback_models: Sequence[str | BaseChatModel] | None = None,
        middleware: ModelFallbackMiddleware | None = make_default_fallback_middleware(),
        tools: list[Any] | None = None,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._llm: BaseChatModel = init_chat_model(model)
        self._primary_model: str | LangChainChatModel = model
        self._fallback_models: tuple[str | BaseChatModel, ...] = (
            tuple(fallback_models)
            if fallback_models is not None
            else _DEFAULT_FALLBACK_MODELS
        )
        self._middleware_override = middleware
        self._tools = [get_weather] if tools is None else tools
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def primary_model(self) -> str | LangChainChatModel:
        return self._primary_model

    @property
    def fallback_models(self) -> tuple[str | BaseChatModel, ...]:
        return self._fallback_models

    def _build_middleware(self) -> ModelFallbackMiddleware:
        if self._middleware_override is not None:
            return self._middleware_override
        first, *rest = self._fallback_models
        return make_model_fallback_middleware(first, *rest)

    def _compile_graph(self) -> CompiledStateGraph:
        agent = create_agent(
            model=self._llm,
            tools=self._tools,
            middleware=[self._build_middleware()],
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "MiddlewareModelFallbackAgent",
    "_DEFAULT_FALLBACK_MODELS",
    "get_weather",
    "make_default_fallback_middleware",
    "make_model_fallback_middleware",
]
