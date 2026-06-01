"""
참고 문서:
/langgraph-v1-tutorial/PART01-LangGraph-기초/Ch01-그래프-생성하기/02-QuickStart-LangGraph-Graph-API.ipynb
(Nodes 심화 — 노드 캐시 ``cache_policy`` · ``InMemoryCache``)

핵심 :
``add_node(..., cache_policy=CachePolicy(...))`` 후 ``compile(cache=InMemoryCache())``
"""

from __future__ import annotations

import time
from typing import cast

from langgraph.cache.memory import InMemoryCache
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import CachePolicy
from typing_extensions import TypedDict

from base.base_graph import BaseGraph


class CacheState(TypedDict):
    """캐싱 예제를 위한 상태 정의"""

    x: int
    result: int


class NodeCaching(BaseGraph):
    """노드 단위 캐시 정책 데모 그래프.

    외부에서는 ``g = NodeCaching()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
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

    def _expensive_computation(self, state: CacheState) -> dict[str, int]:
        """캐싱이 필요한 무거운 계산 노드"""
        print(f"  무거운 계산 실행 중... (x={state['x']})")
        time.sleep(3)  # 무거운 작업 시뮬레이션
        return {"result": state["x"] * state["x"]}

    def _compile_graph(self) -> CompiledStateGraph:
        builder = StateGraph(CacheState)

        # 캐시 정책과 함께 노드 추가
        builder.add_node(
            "expensive_node",
            self._expensive_computation,
            cache_policy=CachePolicy(
                ttl=None,  # 60초 동안 캐시 유지, None 이면 만료시간 없음
                # key_func=lambda x: hash(x["x"])  # 커스텀 캐시 키 생성 함수
            ),
        )

        builder.add_edge(START, "expensive_node")
        builder.add_edge("expensive_node", END)

        # 캐시와 함께 컴파일
        return cast(CompiledStateGraph, builder.compile(cache=InMemoryCache()))


__all__ = ["CacheState", "NodeCaching"]
