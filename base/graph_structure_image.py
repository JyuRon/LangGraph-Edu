"""컴파일된 LangGraph의 **노드·엣지 구조**를 이미지(또는 ASCII)로 보여 주는 보조 베이스."""

from __future__ import annotations


class GraphStructureImage:
    """인스턴스에 ``_graph`` (컴파일된 LangGraph)가 있을 때.

    Jupyter(ZMQ)에서는 Mermaid 기반 **PNG 이미지**로, 그 외 환경에서는 **ASCII**로
    그래프 구조를 표시한다. 단독 상속·인스턴스화는 하지 말고 ``BaseGraph`` 와 함께 쓴다.
    """

    def _show_graph_structure(self, *, xray: bool = False, ascii: bool = False) -> None:
        from util.graphs import visualize_graph

        g = getattr(self, "_graph", None)
        if g is None:
            msg = (
                "`_graph`가 설정되지 않았습니다. "
                "`BaseGraph`를 상속한 뒤 `__init__`에서 `self._graph = self._compile_graph()`를 호출하세요."
            )
            raise RuntimeError(msg)
        visualize_graph(g, xray=xray, ascii=ascii)


__all__ = ["GraphStructureImage"]
