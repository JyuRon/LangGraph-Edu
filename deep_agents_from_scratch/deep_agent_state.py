"""State management for deep agents with TODO tracking and a mirrored file system.

This module defines the extended agent state structure that supports:
- Task planning and progress tracking through TODO lists (mirrored to JSON on disk)
- Context offloading through a ``files`` dict in state that mirrors the
  ``AgentFileStore`` disk directory. Disk is the durable store, ``state["files"]``
  is the in-memory view that LangGraph reducers use to propagate updates
  (incl. sub-agent → parent merging during ``task`` delegation).
"""


"""
## State 설계

워크플로우 컨텍스트 저장을 위한 `DeepAgentState` 구조. `AgentState`를 상속받아 세 요소를 가집니다.

*   **`messages`**: 대화 기록 (`add_messages` reducer 사용)
*   **`todos`**: 작업 리스트 (`Todo` 객체 리스트; ``todo_tools``가 JSON 파일과 동기화)
*   **`files`**: 가상 파일 시스템 (파일명-내용 매핑, `file_reducer` 사용).
    ``file_tools`` / ``tavily_search``가 디스크에 영속화하면서 동시에 미러링하므로,
    ``task`` 위임 시 LangGraph reducer가 sub-agent의 갱신을 부모 state로 자동 병합합니다.
"""

from typing import Annotated, Literal, NotRequired

from langchain.agents import AgentState
from typing_extensions import TypedDict


# 복잡한 작업 플로우의 진행 상황 추적을 위한 TODO 항목 구조 정의
class Todo(TypedDict):
    """A structured task item for tracking progress through complex workflows.

    Attributes:
        content: Short, specific description of the task
        status: Current state - pending, in_progress, or completed
    """

    content: str
    status: Literal["pending", "in_progress", "completed"]


# 두 파일 딕셔너리 병합, 오른쪽 값이 우선 적용되는 가상 파일 시스템 업데이트용 reducer 함수
def file_reducer(left, right):
    """Merge two file dictionaries, with right side taking precedence.

    Used as a reducer function for the ``files`` field in agent state, allowing
    incremental updates to the in-memory mirror of the file system. Sub-agent
    updates (returned via ``task`` Command) are merged into the parent state
    automatically through this reducer.

    Args:
        left: Left side dictionary (existing files)
        right: Right side dictionary (new/updated files)

    Returns:
        Merged dictionary with right values overriding left values
    """
    if left is None:
        return right
    elif right is None:
        return left
    else:
        return {**left, **right}


# LangGraph AgentState 상속, TODO 리스트와 가상 파일 시스템 포함한 확장 state 구조 정의
class DeepAgentState(AgentState):
    """Extended agent state that includes task tracking and a virtual file system.

    Inherits from LangGraph's AgentState and adds:
    - todos: List of Todo items for task planning and progress tracking
    - files: Virtual file system mirrored from the disk-backed ``AgentFileStore``
    """

    # 작업 플래닝 및 진행 상황 추적을 위한 Todo 리스트 필드
    todos: NotRequired[list[Todo]]
    # 파일명과 내용 매핑, file_reducer로 병합되는 가상 파일 시스템 필드
    # (디스크의 AgentFileStore가 영속 저장소이며 이 dict는 인메모리 미러)
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
