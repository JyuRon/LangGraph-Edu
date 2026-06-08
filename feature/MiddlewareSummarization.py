"""
참고 문서:
PART02-에이전트/Ch02-에이전트/01-LangGraph-Agents.ipynb
test/01-LangGraph-Middleware.ipynb

핵심:
``SummarizationMiddleware`` 로 대화 기록이 ``trigger`` 임계값을 넘으면
요약 LLM 호출 후 ``keep`` 정책에 따라 최근 메시지만 남긴다.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, cast

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel

ContextFraction = tuple[Literal["fraction"], float]
ContextTokens = tuple[Literal["tokens"], int]
ContextMessages = tuple[Literal["messages"], int]
ContextSize = ContextFraction | ContextTokens | ContextMessages

TokenCounter = Callable[[Iterable[AnyMessage | list[str] | tuple[str, str] | str | dict[str, Any]]], int]

_DEFAULT_TRIGGER: ContextSize = ("tokens", 4000)  # 4000 토큰에서 요약 트리거
_DEFAULT_KEEP: ContextSize = ("messages", 20)  # 요약 후 최근 20개 메시지 유지
_DEFAULT_TRIM_TOKENS_TO_SUMMARIZE = 4000
_SUMMARY_MESSAGE_PREFIX = "Here is a summary of the conversation to date:"


# 간단한 도구 정의
@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"It's sunny in {city}!"


def make_summarization_middleware(
    summary_model: str | BaseChatModel,
    *,
    # 요약에 사용할 모델 (OpenAI 키를 사용하는 경우 openai:gpt-4.1-mini 등으로 변경 가능)
    trigger: ContextSize | list[ContextSize] | None = _DEFAULT_TRIGGER,
    keep: ContextSize = _DEFAULT_KEEP,
    token_counter: TokenCounter | None = None,
    summary_prompt: str | None = None,
    trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKENS_TO_SUMMARIZE,
) -> SummarizationMiddleware:
    """``SummarizationMiddleware`` 인스턴스를 생성한다."""
    kwargs: dict[str, Any] = {
        "model": summary_model,
        "trigger": trigger,
        "keep": keep,
        "trim_tokens_to_summarize": trim_tokens_to_summarize,
    }
    if token_counter is not None:
        kwargs["token_counter"] = token_counter
    if summary_prompt is not None:
        kwargs["summary_prompt"] = summary_prompt
    return SummarizationMiddleware(**kwargs)


def build_demo_messages(
    num_pairs: int,
    *,
    user_template: str = "Turn {i}: Tell me about topic {i}.",
    ai_template: str = "Topic {i} is interesting. Here is a short explanation.",
) -> list[HumanMessage | AIMessage]:
    """요약 트리거 테스트용 Human/AI 메시지 쌍을 생성한다."""
    messages: list[HumanMessage | AIMessage] = []
    for i in range(1, num_pairs + 1):
        messages.append(HumanMessage(content=user_template.format(i=i)))
        messages.append(AIMessage(content=ai_template.format(i=i)))
    return messages


def messages_contain_summary(messages: Sequence[BaseMessage | dict[str, Any]]) -> bool:
    """요약 미들웨어가 남긴 ``Here is a summary...`` HumanMessage 존재 여부."""
    for msg in messages:
        content = msg.content if isinstance(msg, BaseMessage) else msg.get("content", "")
        if isinstance(content, str) and content.startswith(_SUMMARY_MESSAGE_PREFIX):
            return True
    return False


class MiddlewareSummarizationAgent(BaseGraph):
    """``SummarizationMiddleware`` 가 붙은 ``create_agent`` 데모.

    외부에서는 ``g = MiddlewareSummarizationAgent()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.

    ``trigger`` / ``keep`` / ``summary_model`` 등 ``SummarizationMiddleware`` 옵션을
    생성자에서 그대로 넘길 수 있다. 다턴 대화·요약 확인은 ``checkpointer=True``(기본)와
    ``thread_id`` config 를 함께 쓴다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        summary_model: str | BaseChatModel | None = None,
        trigger: ContextSize | list[ContextSize] | None = _DEFAULT_TRIGGER,
        keep: ContextSize = _DEFAULT_KEEP,
        token_counter: TokenCounter | None = None,
        summary_prompt: str | None = None,
        trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKENS_TO_SUMMARIZE,
        tools: list[Any] | None = None,
        checkpointer: bool = True,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._llm: BaseChatModel = init_chat_model(model)
        # 요약에 사용할 모델 — 미지정 시 에이전트 모델과 동일
        self._summary_model: str | BaseChatModel = summary_model or self._llm
        self._trigger = trigger
        self._keep = keep
        self._token_counter = token_counter
        self._summary_prompt = summary_prompt
        self._trim_tokens_to_summarize = trim_tokens_to_summarize
        self._tools = [get_weather] if tools is None else tools
        self._use_checkpointer = checkpointer
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def summary_model(self) -> str | BaseChatModel:
        return self._summary_model

    @property
    def trigger(self) -> ContextSize | list[ContextSize] | None:
        return self._trigger

    @property
    def keep(self) -> ContextSize:
        return self._keep

    def _compile_graph(self) -> CompiledStateGraph:
        middleware = make_summarization_middleware(
            self._summary_model,
            trigger=self._trigger,
            keep=self._keep,
            token_counter=self._token_counter,
            summary_prompt=self._summary_prompt,
            trim_tokens_to_summarize=self._trim_tokens_to_summarize,
        )
        agent = create_agent(
            model=self._llm,
            tools=self._tools,
            middleware=[middleware],
            checkpointer=MemorySaver() if self._use_checkpointer else None,
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "ContextFraction",
    "ContextMessages",
    "ContextSize",
    "ContextTokens",
    "MiddlewareSummarizationAgent",
    "TokenCounter",
    "_DEFAULT_KEEP",
    "_DEFAULT_TRIGGER",
    "_DEFAULT_TRIM_TOKENS_TO_SUMMARIZE",
    "_SUMMARY_MESSAGE_PREFIX",
    "build_demo_messages",
    "get_weather",
    "make_summarization_middleware",
    "messages_contain_summary",
]
