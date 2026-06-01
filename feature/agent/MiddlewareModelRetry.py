"""
참고 문서:
PART02-에이전트/Ch02-에이전트/01-LangGraph-Agents.ipynb

핵심:
``@wrap_model_call`` 로 ``handler(request)`` 호출을 감싸
모델 오류 시 최대 N회까지 재시도한다.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel

_DEFAULT_MAX_RETRIES = 3

# OpenAI 키를 사용하는 경우 gpt-4.1-mini, gpt-4.1 등으로 변경하세요
# 일부러 존재하지 않는 모델명을 사용하여 재시도 로직을 테스트합니다
_DEFAULT_RETRY_TEST_MODEL = "gpt-4.1-mini"


def make_model_retry_middleware(
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> AgentMiddleware:
    """``wrap_model_call`` — ``handler`` 실패 시 최대 ``max_retries`` 회 재시도."""

    @wrap_model_call
    def retry_model(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        for attempt in range(max_retries):
            try:
                raise Exception("test")
                return handler(request)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(
                    f"오류 발생으로 {attempt + 1}/{max_retries} 번째 재시도합니다: {e}"
                )

        raise RuntimeError("unreachable")  # for type checker

    return retry_model


class MiddlewareModelRetryAgent(BaseGraph):
    """``wrap_model_call`` 로 모델 호출 실패 시 재시도하는 ``create_agent`` 데모.

    외부에서는 ``g = MiddlewareModelRetryAgent()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.

    기본 ``model`` 은 존재하지 않는 이름(``gpt-4.1-mini``)으로 재시도 로그를 확인하기 위함입니다.
    정상 응답을 보려면 ``LangChainChatModel.OPENAI_GPT_4O_MINI`` 등 유효한 모델을 넘기세요.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = _DEFAULT_RETRY_TEST_MODEL,
        *,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._llm: BaseChatModel = init_chat_model(model)
        self._max_retries = max_retries
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def _compile_graph(self) -> CompiledStateGraph:
        middleware = make_model_retry_middleware(max_retries=self._max_retries)
        agent = create_agent(
            model=self._llm,
            tools=[],
            middleware=[middleware],
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "MiddlewareModelRetryAgent",
    "_DEFAULT_MAX_RETRIES",
    "_DEFAULT_RETRY_TEST_MODEL",
    "make_model_retry_middleware",
]
