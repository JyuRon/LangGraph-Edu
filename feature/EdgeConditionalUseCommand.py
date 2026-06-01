"""
참고 문서:
/langgraph-v1-tutorial/PART01-LangGraph-기초/Ch01-그래프-생성하기/02-QuickStart-LangGraph-Graph-API.ipynb
(Command · ``goto`` — 조건부 엣지와 동일한 동적 제어, retry 시 ``Command`` 로 decision 복귀)

핵심 :
``decision`` 노드가 ``Command(update=..., goto=...)`` 로 success / failure / retry 분기.
retry는 엣지 없이 ``goto="decision"`` 만으로 루프.


하위 그래프에서 상위 노드로 이동하려면 ``graph=Command.PARENT`` 를 쓰며,
def my_node(state: CommandState) -> Command[Literal["other_subgraph"]]:
    return Command(
        update={"foo": "bar"},
        goto="other_subgraph",  # where `other_subgraph` is a node in the parent graph
        graph=Command.PARENT,
    )
"""

from __future__ import annotations

from typing import Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from typing_extensions import TypedDict

from base.base_graph import BaseGraph


class CommandState(TypedDict):
    """Command 예제를 위한 상태 정의"""

    value: int
    message: str
    path: list[str]


class EdgeConditionalUseCommand(BaseGraph):
    """``Command`` 의 ``goto`` 로 조건 분기·재시도 루프를 만든 데모 그래프.

    외부에서는 ``g = EdgeConditionalUseCommand()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        *,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._graph: CompiledStateGraph = self._compile_graph()

    @staticmethod
    def _extend_path(state: CommandState, *segments: str) -> list[str]:
        """path 필드 갱신용 헬퍼 (기존 경로 뒤에 세그먼트를 이어 붙임)."""
        return [*state["path"], *segments]

    # Command를 반환하는 노드
    def _decision_node(
        self,
        state: CommandState,
    ) -> Command[Literal["success", "failure", "retry"]]:
        """상태를 업데이트하고 결정을 내리는 노드"""
        value = state["value"]

        if value > 100:
            return Command(
                # 상태 업데이트
                update={
                    "message": "값이 너무 큽니다",
                    "path": self._extend_path(state, "decision"),
                },
                # 라우팅
                goto="failure",
            )
        if value > 50:
            return Command(
                update={
                    "message": "처리 성공",
                    "value": value * 2,
                    "path": self._extend_path(state, "decision"),
                },
                goto="success",
            )
        return Command(
            update={
                "message": "재시도 필요",
                "value": value + 30,
                "path": self._extend_path(state, "decision"),
            },
            goto="retry",
        )

    def _success_node(self, state: CommandState):
        """성공 처리 노드"""
        return {
            "message": f"[성공] {state['message']}",
            "path": self._extend_path(state, "success"),
        }

    def _failure_node(self, state: CommandState):
        """실패 처리 노드"""
        return {
            "message": f"[실패] {state['message']}",
            "path": self._extend_path(state, "failure"),
        }

    def _retry_node(self, state: CommandState) -> Command[Literal["decision"]]:
        """재시도 처리 노드 - decision으로 되돌아감"""
        return Command(
            update={"message": "재시도 중...", "path": self._extend_path(state, "retry")},
            goto="decision",  # 다시 결정 노드로
        )

    def _compile_graph(self) -> CompiledStateGraph:
        builder = StateGraph(CommandState)

        # 노드 추가
        builder.add_node("decision", self._decision_node)
        builder.add_node("success", self._success_node)
        builder.add_node("failure", self._failure_node)
        builder.add_node("retry", self._retry_node)

        # 엣지 추가
        builder.add_edge(START, "decision")
        builder.add_edge("success", END)
        builder.add_edge("failure", END)
        # retry는 Command로 decision으로 돌아감

        return cast(CompiledStateGraph, builder.compile())


__all__ = ["CommandState", "EdgeConditionalUseCommand"]
