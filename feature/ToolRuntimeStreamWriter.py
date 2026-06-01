"""
참고 문서:
PART02-에이전트/Ch02-에이전트/02-LangGraph-Tools.ipynb

핵심:
``ToolRuntime`` 매개변수로 도구 실행 시 ``state``·``context`` 등 런타임 정보에 접근한다.
``runtime: ToolRuntime`` 은 LLM 도구 스키마에 노출되지 않고 자동 주입된다.
"""

from __future__ import annotations

import time
from typing import Any, Literal, cast

from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import RemoveMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import NotRequired

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel

_SYSTEM_PROMPT = "You are a helpful assistant."



# 참고: runtime.stream_writer를 도구 내에서 사용하는 경우,
# 도구는 LangGraph 실행 컨텍스트 내에서 호출되어야 합니다.
# 대용량 데이터 처리를 시뮬레이션하는 도구(예시)
@tool
def get_weather_with_updates(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer
    
    total_items = 50

    # 진행 상황 스트리밍
    for i in range(0, total_items, 10):
        progress = min(i + 10, total_items)
        writer({"stage": "processing", "progress": progress, "total": total_items})
        time.sleep(0.1)  # 작업 시뮬레이션

    # 완료 상태 스트리밍
    writer({"stage": "completed", "messages": total_items})
    writer({"stage": "completed", "messages" : f"In the {city}"})
    writer(f"Acquired data for city: {city}")

    return f"It's always sunny in {city}!"







class ToolRuntimeAgent(BaseGraph):
    """``ToolRuntime`` 으로 ``state``·``context`` 에 접근하는 ``create_agent`` 데모.

    외부에서는 ``g = ToolRuntimeAgent()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
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
        agent = create_agent(
            self._llm,
            tools=[get_weather_with_updates],
            system_prompt=_SYSTEM_PROMPT,
            # checkpointer=InMemorySaver(),
            # context_schema=CustomContext,
            # state_schema=CustomState,
            # store=InMemoryStore(),
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "ToolRuntimeAgent",
]
