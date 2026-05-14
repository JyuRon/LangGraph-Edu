"""
참고 문서:
/langgraph-v1-tutorial/PART01-LangGraph-기초/Ch01-그래프-생성하기/02-QuickStart-LangGraph-Graph-API.ipynb
(고급 기능 — 재귀 제한 ``recursion_limit``: 무한 루프 방지)

핵심 :
``RunnableConfig`` 에 ``recursion_limit`` 를 두어 super-step 상한을 건다.
분기 루프는 ``Command(..., goto="increment")`` 로 같은 노드에 되돌아가게 만든다.
"""

from __future__ import annotations

from typing import Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from typing_extensions import TypedDict

from base.base_graph import BaseGraph


class LoopState(TypedDict):
    """루프 예제를 위한 상태 정의"""

    counter: int
    history: list[int]


class RecursionLimit(BaseGraph):
    """``Command`` 자기 루프로 super-step을 쌓는 데모 (``recursion_limit`` 과 함께 설명).

    외부에서는 ``g = RecursionLimit()`` 뒤 ``g.invoke(inputs=..., config=...)`` 에서
    ``recursion_limit`` 을 넘기고, LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        *,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._graph: CompiledStateGraph = self._compile_graph()

    def _increment_node(
        self, state: LoopState
    ) -> Command[Literal["increment", "end"]]:
        """카운터를 증가시키는 노드"""
        new_counter = state["counter"] + 1

        print(f"new_counter: {new_counter}")

        if new_counter < 10:  # 의도적으로 높은 목표 설정
            return Command(
                update={
                    "counter": new_counter,
                    "history": state["history"] + [new_counter],
                },
                goto="increment",  # 자기 자신으로 루프
            )
        return Command(update={"counter": new_counter}, goto="end")

    def _end_node(self, state: LoopState):
        """종료 노드"""
        return {"history": state["history"] + [999]}

    def _compile_graph(self) -> CompiledStateGraph:
        builder = StateGraph(LoopState)

        builder.add_node("increment", self._increment_node)
        builder.add_node("end", self._end_node)
        builder.add_edge(START, "increment")
        builder.add_edge("end", END)

        return cast(CompiledStateGraph, builder.compile())


__all__ = ["LoopState", "RecursionLimit"]
