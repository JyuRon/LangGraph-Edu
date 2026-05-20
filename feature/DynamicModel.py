"""
참고 문서:
PART02-에이전트/Ch02-에이전트/01-LangGraph-Agents.ipynb

핵심:
``@wrap_model_call`` 로 모델 호출 직전에 ``ModelRequest`` 를 가로채
메시지 수(대화 복잡도)에 따라 basic / advanced 모델을 선택한다.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph.state import CompiledStateGraph

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel

_DEFAULT_MESSAGE_THRESHOLD = 10


def make_dynamic_model_middleware(
    *,
    basic_model: BaseChatModel,
    advanced_model: BaseChatModel,
    message_threshold: int = _DEFAULT_MESSAGE_THRESHOLD,
) -> AgentMiddleware:
    """메시지 수에 따라 basic / advanced 모델을 고르는 ``wrap_model_call`` 미들웨어."""

    @wrap_model_call
    def dynamic_model_selection(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """대화 복잡도에 따라 모델 선택"""
        message_count = len(request.state["messages"][-1].content)

        # 긴 대화에는 고급 모델 사용
        if message_count > message_threshold:
            request.model = advanced_model
        else:
            # override 사용(여러 속성을 한번에 변경)
            request =request.override(
                model = basic_model,
                system_message=SystemMessage("한 문장으로 간결하게 답변해줘. emoji 는 무조건 사용해"),
                tool_choice="auto",
            )

        
        print(f"모델 선택: {request.model.model}")

        return handler(request)

    return dynamic_model_selection


class DynamicModelAgent(BaseGraph):
    """``wrap_model_call`` 로 대화 길이에 따라 LLM을 바꾸는 ``create_agent`` 데모.

    외부에서는 ``g = DynamicModelAgent()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        basic_model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        advanced_model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O,
        *,
        message_threshold: int = _DEFAULT_MESSAGE_THRESHOLD,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._basic_llm: BaseChatModel = init_chat_model(basic_model)
        self._advanced_llm: BaseChatModel = init_chat_model(advanced_model)
        self._message_threshold = message_threshold
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def basic_llm(self) -> BaseChatModel:
        return self._basic_llm

    @property
    def advanced_llm(self) -> BaseChatModel:
        return self._advanced_llm

    @property
    def message_threshold(self) -> int:
        return self._message_threshold

    def _compile_graph(self) -> CompiledStateGraph:
        middleware = make_dynamic_model_middleware(
            basic_model=self._basic_llm,
            advanced_model=self._advanced_llm,
            message_threshold=self._message_threshold,
        )
        agent = create_agent(
            model=self._basic_llm,  # 기본 모델
            tools=[],
            middleware=[middleware],
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "DynamicModelAgent",
    "_DEFAULT_MESSAGE_THRESHOLD",
    "make_dynamic_model_middleware",
]
