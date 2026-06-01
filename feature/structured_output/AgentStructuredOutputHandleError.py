"""
참고 문서:
/langgraph-v1-tutorial/PART02-에이전트/Ch04-구조화된-출력/05-LangGraph-Structured-Output.ipynb
https://docs.langchain.com/oss/python/langchain/structured-output

핵심:
``ToolStrategy`` 의 ``handle_errors`` 로 구조화 출력 검증 오류 시
자동 재시도·예외 발생·커스텀 메시지/핸들러 등을 제어하고,
결과는 ``structured_response`` 키에 반환된다.
"""

from __future__ import annotations

from typing import Literal, cast

from langchain.agents import create_agent
from langchain.agents.structured_output import (
    MultipleStructuredOutputsError,
    StructuredOutputValidationError,
    ToolStrategy,
)
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel


class ProductRating(BaseModel):
    """제품 평점 정보를 나타내는 스키마

    평점은 1-5 범위로 제한되며, 리뷰 코멘트를 포함합니다.
    """

    rating: int | None = Field(
        description="Rating from 1-5", ge=1, le=5
    )  # 평점 (1-5 범위)
    comment: str = Field(description="Review comment")  # 리뷰 코멘트


# 에이전트 생성 - 기본값 handle_errors=True로 자동 재시도 활성화
_TOOL_STRATEGY_DEFAULT = ToolStrategy(ProductRating)  # 기본값: handle_errors=True

# 커스텀 오류 메시지로 에이전트 생성
_TOOL_STRATEGY_CUSTOM_MESSAGE = ToolStrategy(
    schema=ProductRating,
    # 오류 발생 시 이 메시지로 재시도 요청
    handle_errors="Please provide a valid rating between 1-5 and include a comment.",
)

# ValueError만 처리하는 에이전트 생성
_TOOL_STRATEGY_VALUE_ERROR = ToolStrategy(
    schema=ProductRating,
    handle_errors=ValueError,  # ValueError만 재시도, 다른 예외는 그대로 발생
)

# 여러 예외 유형을 처리하는 에이전트 생성
_TOOL_STRATEGY_MULTIPLE_EXCEPTIONS = ToolStrategy(
    schema=ProductRating,
    # ValueError 및 TypeError 모두 재시도
    handle_errors=(ValueError, TypeError),
)

# 커스텀 오류 핸들러로 에이전트 생성
def custom_error_handler(error: Exception) -> str:
    """오류 유형에 따른 커스텀 오류 메시지 생성

    Args:
        error: 발생한 예외 객체

    Returns:
        모델에게 전달할 오류 메시지
    """
    if isinstance(error, StructuredOutputValidationError):
        # 스키마 검증 오류
        return "There was an issue with the format. Try again."
    elif isinstance(error, MultipleStructuredOutputsError):
        # 여러 구조화된 출력이 반환된 경우
        return "Multiple structured outputs were returned. Pick the most relevant one."
    else:
        # 기타 오류
        return f"Error: {str(error)}"


_TOOL_STRATEGY_CUSTOM_HANDLER = ToolStrategy(
    schema=ProductRating, handle_errors=custom_error_handler
)

# 오류 처리 비활성화 - 모든 오류가 예외로 발생
_TOOL_STRATEGY_DISABLED = ToolStrategy(
    schema=ProductRating, handle_errors=False  # 오류 발생 시 예외 발생
)

_RESPONSE_FORMAT_BY_KIND = {
    "default": _TOOL_STRATEGY_DEFAULT,
    "custom_message": _TOOL_STRATEGY_CUSTOM_MESSAGE,
    "value_error_only": _TOOL_STRATEGY_VALUE_ERROR,
    "multiple_exceptions": _TOOL_STRATEGY_MULTIPLE_EXCEPTIONS,
    "custom_handler": _TOOL_STRATEGY_CUSTOM_HANDLER,
    "disabled": _TOOL_STRATEGY_DISABLED,
}

ErrorStrategy = Literal[
    "default",
    "custom_message",
    "value_error_only",
    "multiple_exceptions",
    "custom_handler",
    "disabled",
]

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that parses product reviews. "
    "Do not make any field or value up."
)


class AgentStructuredOutputHandleErrorAgent(BaseGraph):
    """``ToolStrategy.handle_errors`` 전략별 ``create_agent`` 데모.

    외부에서는 ``g = AgentStructuredOutputHandleErrorAgent(error_strategy="default")`` 뒤
    ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        error_strategy: ErrorStrategy = "default",
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4_1_MINI,
        *,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._error_strategy: ErrorStrategy = error_strategy
        self._response_format = _RESPONSE_FORMAT_BY_KIND[error_strategy]
        self._llm: BaseChatModel = init_chat_model(model)
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def error_strategy(self) -> ErrorStrategy:
        return self._error_strategy

    @property
    def response_format(self) -> ToolStrategy:
        return self._response_format

    def _compile_graph(self) -> CompiledStateGraph:
        agent_kwargs: dict = {
            "model": self._llm,
            "tools": [],
            "response_format": self._response_format,
        }
        if self._error_strategy == "default":
            agent_kwargs["system_prompt"] = _DEFAULT_SYSTEM_PROMPT

        agent = create_agent(**agent_kwargs)
        return cast(CompiledStateGraph, agent)


__all__ = [
    "AgentStructuredOutputHandleErrorAgent",
    "ErrorStrategy",
    "ProductRating",
    "_RESPONSE_FORMAT_BY_KIND",
    "_TOOL_STRATEGY_CUSTOM_HANDLER",
    "_TOOL_STRATEGY_CUSTOM_MESSAGE",
    "_TOOL_STRATEGY_DEFAULT",
    "_TOOL_STRATEGY_DISABLED",
    "_TOOL_STRATEGY_MULTIPLE_EXCEPTIONS",
    "_TOOL_STRATEGY_VALUE_ERROR",
    "custom_error_handler",
]
