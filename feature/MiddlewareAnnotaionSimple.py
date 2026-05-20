"""
참고 문서:
PART02-에이전트/Ch02-에이전트/01-LangGraph-Agents.ipynb

핵심:
``@before_model`` / ``@after_model`` 로 모델 호출 전·후에 상태를 로깅하고,
호출 전 마지막 사용자 메시지를 LLM으로 재작성한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import after_model, before_model
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel
from util.messages import depth_colors

# 노드 스타일: 모델 호출 전 로깅
_QUERY_REWRITE_TEMPLATE = (
    "Rewrite the following query to be more understandable. "
    "Do not change the original meaning. Make it one sentence: {query}"
)

_MAGENTA, _RESET = depth_colors[1], depth_colors["reset"]


def make_middleware_annotaion_simple_hooks(
    *,
    rewrite_llm: BaseChatModel,
) -> tuple[AgentMiddleware, AgentMiddleware]:
    """쿼리 재작성·로깅용 ``before_model`` / ``after_model`` 미들웨어 쌍."""

    query_rewrite = PromptTemplate.from_template(_QUERY_REWRITE_TEMPLATE) | rewrite_llm

    @before_model
    def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(
            f"{_MAGENTA}\n\n모델 호출 전 메시지 {len(state['messages'])}개가 있습니다{_RESET}"
        )
        last = state["messages"][-1]
        last_message = last.content if hasattr(last, "content") else str(last)

        rewritten_query = query_rewrite.invoke({"query": last_message})

        return {"messages": [HumanMessage(content=rewritten_query.content)]}

    @after_model
    def log_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(
            f"{_MAGENTA}\n\n모델 호출 후 메시지 {len(state['messages'])}개가 있습니다{_RESET}"
        )

        for i, message in enumerate(state["messages"]):
            print(f"[{i}] {message.content}")

        return None

    return log_before_model, log_after_model


class MiddlewareAnnotaionSimpleAgent(BaseGraph):
    """``before_model`` / ``after_model`` 로 로깅·쿼리 재작성을 하는 ``create_agent`` 데모.

    외부에서는 ``g = MiddlewareAnnotaionSimpleAgent()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        rewrite_model: str | LangChainChatModel | None = None,
        *,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._llm: BaseChatModel = init_chat_model(model)
        # OpenAI 키를 사용하는 경우 gpt-4.1-mini, gpt-4.1 등으로 변경하세요
        rewrite_model_spec = rewrite_model if rewrite_model is not None else model
        self._rewrite_llm: BaseChatModel = init_chat_model(rewrite_model_spec)
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def rewrite_llm(self) -> BaseChatModel:
        return self._rewrite_llm

    def _compile_graph(self) -> CompiledStateGraph:
        before_hook, after_hook = make_middleware_annotaion_simple_hooks(
            rewrite_llm=self._rewrite_llm,
        )
        middleware: Sequence[AgentMiddleware] = [before_hook, after_hook]
        agent = create_agent(
            model=self._llm,
            tools=[],
            middleware=list(middleware),
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "MiddlewareAnnotaionSimpleAgent",
    "_QUERY_REWRITE_TEMPLATE",
    "make_middleware_annotaion_simple_hooks",
]
