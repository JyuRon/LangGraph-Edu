"""컴파일된 LangGraph 유스케이스 공통 추상 베이스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from .graph_structure_image import GraphStructureImage
from .langchain_project import LangChainProjectSetup
from util.messages import invoke_graph, stream_graph


class BaseGraph(LangChainProjectSetup, GraphStructureImage, ABC):
    """``.env`` / LangSmith + 컴파일 그래프 실행·구조 표시.

    상속 시 MRO: ``LangChainProjectSetup`` → ``GraphStructureImage`` → ``ABC``.

    하위 클래스 ``__init__`` 패턴:

    1. ``super().__init__(load_env=..., langsmith_project=...)``
    2. 그래프에 필요한 의존성 준비 (예: LLM)
    3. ``self._graph = self._compile_graph()``

    - ``invoke()`` → ``util.messages.invoke_graph(graph, inputs=..., config=...)`` 위임 후
      마지막 상태 스냅샷을 반환합니다.
    - ``stream()`` → ``util.messages.stream_graph(graph, inputs=..., config=...)`` 위임합니다.
    """

    _graph: CompiledStateGraph

    def __init__(
        self,
        *,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

    @abstractmethod
    def _compile_graph(self) -> CompiledStateGraph:
        """``StateGraph`` 를 조립해 컴파일된 그래프를 반환한다."""

    @property
    def graph(self) -> CompiledStateGraph:
        return self._graph


    """
    LangGraph의 실행 결과를 스트리밍하여 출력하는 함수입니다.
    (LangGraph v1.0 호환)

    Args:
        graph (CompiledStateGraph): 실행할 컴파일된 LangGraph 객체
        inputs (dict): 그래프에 전달할 입력값 딕셔너리
        config (RunnableConfig, optional): 실행 설정. 기본값은 None
        context (Any, optional): 그래프 실행을 위한 정적 컨텍스트. 기본값은 None
        node_names (List[str], optional): 출력할 노드 이름 목록. 기본값은 빈 리스트
        callback (Callable, optional): 각 청크 처리를 위한 콜백 함수. 기본값은 None
            콜백 함수는 {"node": str, "content": str} 형태의 딕셔너리를 인자로 받습니다.

    Returns:
        None: 함수는 스트리밍 결과를 출력만 하고 반환값은 없습니다.
    """
    def invoke(
        self,
        inputs: dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        context: Any = None,
        node_names: list[str] | None = None,
        callback: Callable[..., Any] | None = None,
        subgraphs: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """``util.messages.invoke_graph`` — ``inputs`` / ``config`` 키워드로 위임."""
        cfg = config or cast(RunnableConfig, {})
        return invoke_graph(
            self._graph,
            inputs=inputs,
            config=cfg,
            context=context,
            node_names=[] if node_names is None else node_names,
            callback=cast(Any, callback),
            subgraphs=subgraphs,
            **kwargs,
        )



    """
    LangGraph의 실행 결과를 스트리밍하여 출력하는 함수입니다.
    (LangGraph v1.0 호환)

    Args:
        graph (CompiledStateGraph): 실행할 컴파일된 LangGraph 객체
        inputs (dict): 그래프에 전달할 입력값 딕셔너리
        config (RunnableConfig, optional): 실행 설정. 기본값은 None
        context (Any, optional): 그래프 실행을 위한 정적 컨텍스트. 기본값은 None
        node_names (List[str], optional): 출력할 노드 이름 목록. 기본값은 빈 리스트
        callback (Callable, optional): 각 청크 처리를 위한 콜백 함수. 기본값은 None
            콜백 함수는 {"node": str, "content": str} 형태의 딕셔너리를 인자로 받습니다.

    Returns:
        None: 함수는 스트리밍 결과를 출력만 하고 반환값은 없습니다.
    """
    def stream(
        self,
        inputs: dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        context: Any = None,
        node_names: list[str] | None = None,
        callback: Callable[..., Any] | None = None,
    ) -> None:
        """``util.messages.stream_graph`` — ``inputs`` / ``config`` 키워드로 위임."""
        stream_graph(
            self._graph,
            inputs=inputs,
            config=config or cast(RunnableConfig, {}),
            context=context,
            node_names=[] if node_names is None else node_names,
            callback=cast(Any, callback),
        )

    def show_graph(self, *, xray: bool = False, ascii: bool = False) -> None:
        """노드·엣지 구조 출력 (Jupyter: PNG 우선)."""
        self._show_graph_structure(xray=xray, ascii=ascii)


__all__ = ["BaseGraph"]
