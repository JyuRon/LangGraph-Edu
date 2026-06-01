"""
참고 문서:
PART02-에이전트/Ch02-에이전트/01-LangGraph-Agents.ipynb

핵심:
``@dynamic_prompt`` 로 모델 호출 직전에 ``ModelRequest`` 의 런타임 컨텍스트를 읽어
답변 형식·길이에 맞는 시스템 프롬프트 문자열을 동적으로 만든다.
"""

from __future__ import annotations

from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel
from util.model_request import display_model_request

_DEFAULT_PROMPT_TYPE = "default"
_DEFAULT_LANGUAGE = "korean"
_DEFAULT_USER_NAME = "사용자"
_DEFAULT_ANSWER_LENGTH = 20
_BASE_SYSTEM_PROMPT = "You are a helpful assistant.\n"

# 답변 형식에 따라 시스템 프롬프트 생성(동적 프롬프팅)
_PROMPT_FORMAT_BY_TYPE: dict[str, str] = {
    "default": "{language}, user name is {user_name}. [Response format] Answer concisely. Keep the response within {length} characters.",
    "sns": "{language}, user name is {user_name}. [Response format] Answer in SNS style. Keep the response within {length} characters.",
    "article": "{language}, user name is {user_name}. [Response format] Answer in news article style. Keep the response within {length} characters.",
}

_PROMPT_LANGUAGE_BY_TYPE: dict[str, str] = {
    "korean": "only answer in Korean.",
    "english": "only answer in English.",
    "chinese": "only answer in Chinese.",
}


class DynamicPromptContext(TypedDict):
    """``create_agent`` 의 ``context_schema`` — 런타임 ``context`` 로 전달."""

    prompt_type: str
    length: int
    language: str
    user_name: str


@dynamic_prompt  # before_model
def user_role_prompt(request: ModelRequest) -> str:
    """사용자 역할에 따라 시스템 프롬프트 생성"""
    # 답변 형식 설정
    answer_type = (
        request.runtime.context.get("prompt_type", _DEFAULT_PROMPT_TYPE)
        if request.runtime.context
        else _DEFAULT_PROMPT_TYPE
    )
    # 답변 길이 설정
    answer_length = (
        request.runtime.context.get("length", _DEFAULT_ANSWER_LENGTH)
        if request.runtime.context
        else _DEFAULT_ANSWER_LENGTH
    )

    # 답변 언어 설정
    answer_language = (
        request.runtime.context.get("language", _DEFAULT_LANGUAGE)
        if request.runtime.context
        else _DEFAULT_LANGUAGE
    )

    # 사용자 정보
    user_name = (
        request.runtime.context.get("user_name", _DEFAULT_USER_NAME)
        if request.runtime.context
        else _DEFAULT_USER_NAME
    )

    format_hint = _PROMPT_FORMAT_BY_TYPE.get(
        answer_type, _PROMPT_FORMAT_BY_TYPE[_DEFAULT_PROMPT_TYPE]
    )

    print(format_hint.format(length=answer_length, language=_PROMPT_LANGUAGE_BY_TYPE[answer_language], user_name=user_name))

    display_model_request(request)

    return f"{_BASE_SYSTEM_PROMPT} {format_hint.format(length=answer_length, language=_PROMPT_LANGUAGE_BY_TYPE[answer_language], user_name=user_name)}"


class DynamicPromptAgent(BaseGraph):
    """``dynamic_prompt`` 로 답변 형식·길이에 맞는 시스템 프롬프트를 쓰는 ``create_agent`` 데모.

    외부에서는 ``g = DynamicPromptAgent()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O,
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
        # 컨텍스트 스키마와 user_role_prompt 미들웨어를 사용하여 에이전트 생성
        agent = create_agent(
            model=self._llm,
            tools=[],
            middleware=[
                cast(AgentMiddleware[Any, DynamicPromptContext], user_role_prompt)
            ],
            context_schema=DynamicPromptContext,
            system_prompt="only answer in Japaness."
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "DynamicPromptAgent",
    "DynamicPromptContext",
    "_BASE_SYSTEM_PROMPT",
    "_DEFAULT_ANSWER_LENGTH",
    "_DEFAULT_PROMPT_TYPE",
    "user_role_prompt",
]
