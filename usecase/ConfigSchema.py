"""
참고 문서:
/langgraph-v1-tutorial/PART01-LangGraph-기초/Ch01-그래프-생성하기/02-QuickStart-LangGraph-Graph-API.ipynb
(Nodes 심화 — Config 스키마(context_schema)와 configurable)

핵심 :
builder = StateGraph(ConfigSchemaState, context_schema=ConfigSchema)
"""

from __future__ import annotations

from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import NotRequired, TypedDict

from base.base_graph import BaseGraph


class ConfigSchema(TypedDict, total=False):
    """context_schema로 노드에 전달되는 configurable 키의 타입 힌트.

    total=False: invoke 시마다 모든 키를 넣지 않아도 됨. (TypeDict 전용 옵션)
    """

    runtime_value: str
    setting_value: int | str


class ConfigSchemaState(MessagesState, total=False):
    """MessagesState + 노드가 반환하는 예시 키.

    MessagesState만 쓰면 ``updated_key``는 스키마 밖이라 invoke 결과에 남지 않음.
    """

    updated_key: NotRequired[str]


class SimpleConfigSchema(BaseGraph):
    """``context_schema=ConfigSchema`` 로 configurable 타입을 묶은 데모 그래프.

    외부에서는 ``g = SimpleConfigSchema()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        *,
        load_env: bool = True,
        langsmith_project: str | None = "LangChain-V1-Tutorial",
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._graph: CompiledStateGraph = self._compile_graph()

    def _my_node(
        self, state: ConfigSchemaState, config: RunnableConfig
    ) -> dict[str, str]:
        """커스텀 설정 값에 접근하는 방법을 보여주는 노드 함수

        Args:
            state (ConfigSchemaState): 그래프의 현재 상태
            config (RunnableConfig): 런타임 및 사용자 정의 설정을 포함하는 구성 객체

        Returns:
            dict: 업데이트된 상태 딕셔너리
        """
        configurable = config.get("configurable") or {}

        # config["configurable"] 딕셔너리에서 사용자 정의 설정값을 가져옵니다.
        runtime_value = configurable.get("runtime_value", "")
        setting_value = configurable.get("setting_value", "")

        # runtime_value 값이 있으면 해당 값을 출력합니다.
        if runtime_value:
            print(f"runtime_value: {runtime_value}")

        # setting_value 값이 있으면 해당 값을 출력합니다.
        if setting_value:
            print(f"setting_value: {setting_value}")

        # 현재 state 값을 출력합니다.
        print(f"state: {state}")

        # 새로운 키와 값을 포함하는 상태를 반환합니다.
        return {"updated_key": "new_value"}

    def _compile_graph(self) -> CompiledStateGraph:
        builder = StateGraph(ConfigSchemaState, context_schema=ConfigSchema)
        builder.add_node("my_node", self._my_node)
        builder.add_edge(START, "my_node")
        builder.add_edge("my_node", END)
        return cast(CompiledStateGraph, builder.compile())


__all__ = ["ConfigSchema", "ConfigSchemaState", "SimpleConfigSchema"]
