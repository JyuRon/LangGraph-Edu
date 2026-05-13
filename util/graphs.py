import random

from langchain_core.runnables.graph import NodeStyles
from langgraph.graph.state import CompiledStateGraph


GRAPH_NODE_STYLES = NodeStyles(
    default=(
        "fill:#45C4B0, fill-opacity:0.3, color:#23260F, stroke:#45C4B0, "
        "stroke-width:1px, font-weight:bold, line-height:1.2"
    ),
    first=(
        "fill:#45C4B0, fill-opacity:0.1, color:#23260F, stroke:#45C4B0, "
        "stroke-width:1px, font-weight:normal, font-style:italic, stroke-dasharray:2,2"
    ),
    last=(
        "fill:#45C4B0, fill-opacity:1, color:#000000, stroke:#45C4B0, "
        "stroke-width:1px, font-weight:normal, font-style:italic, stroke-dasharray:2,2"
    ),
)


def _is_jupyter_zmq_shell() -> bool:
    """Jupyter / VS Code 노트북 등에서 흔한 ZMQ IPython 셸인지."""
    try:
        from IPython.core.getipython import get_ipython

        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__ == "ZMQInteractiveShell"
    except (ImportError, NameError):
        return False


def visualize_graph(graph, xray=False, ascii=False):
    """
    LangGraph 컴파일 그래프의 **노드 구조**를 표시합니다.

    - Jupyter(ZMQ) 셸이면 Mermaid PNG를 ``display`` 하고, 실패 시 ASCII로 대체합니다.
    - 그 외(일반 ``python`` 실행 등)는 IPython ``display`` 없이 ASCII 구조만 출력합니다.

    Args:
        graph: ``CompiledStateGraph`` 인스턴스.
        xray: 내부 상태 노출 여부.
        ascii: True이면 항상 ASCII 다이어그램만 출력합니다.
    """
    if not isinstance(graph, CompiledStateGraph):
        print("CompiledStateGraph 가 아닙니다.")
        return

    g = graph.get_graph(xray=xray)

    if ascii or not _is_jupyter_zmq_shell():
        print(g.draw_ascii())
        return

    try:
        from IPython.display import Image, display

        display(
            Image(
                g.draw_mermaid_png(
                    background_color="white",
                    node_colors=GRAPH_NODE_STYLES,
                )
            )
        )
    except Exception as e:
        print(f"그래프 PNG 시각화 실패 (추가 종속성 필요): {e}")
        print("ASCII로 그래프 표시:")
        try:
            print(g.draw_ascii())
        except Exception as ascii_error:
            print(f"ASCII 표시도 실패: {ascii_error}")


def generate_random_hash():
    return f"{random.randint(0, 0xffffff):06x}"


__all__ = [
    "GRAPH_NODE_STYLES",
    "visualize_graph",
    "generate_random_hash",
]
