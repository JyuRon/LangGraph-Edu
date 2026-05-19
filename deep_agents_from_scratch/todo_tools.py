"""TODO management tools for task planning and progress tracking.

This module provides tools for creating and managing structured task lists
that enable agents to plan complex workflows and track progress through
multi-step operations.

TODO lists are persisted as JSON on disk and mirrored in ``DeepAgentState.todos``
during tool calls. Use ``TodoFileStore`` + ``create_todo_tools(store)`` so each
agent/graph instance owns its file path (no process-wide mutable global).
"""

# TODO 리스트 기반 작업 플래닝 및 진행 상황 추적 툴 구현 목적

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from .deep_agent_state import DeepAgentState, Todo
from .prompts import WRITE_TODOS_DESCRIPTION

# 기본 TODO 저장 경로 (패키지 디렉터리 아래 .agent_data/todos.json)
_DEFAULT_TODOS_DIR = Path(__file__).resolve().parent / ".agent_data"
DEFAULT_TODOS_FILE = _DEFAULT_TODOS_DIR / "todos.json"


class TodoFileStore:
    """TODO JSON 파일 경로를 캡슐화. 인스턴스마다 독립된 저장 위치."""

    def __init__(self, todos_file: Path | str = DEFAULT_TODOS_FILE) -> None:
        self._path = Path(todos_file)

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_parent(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def persist(self, todos: list[Todo]) -> Path:
        """TODO 리스트를 JSON 파일로 저장."""
        self._ensure_parent()
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(todos, f, ensure_ascii=False, indent=2)
        return self._path

    def load(self) -> list[Todo]:
        """디스크의 TODO JSON 파일을 읽어 리스트로 반환. 없으면 빈 리스트."""
        if not self._path.exists():
            return []
        with self._path.open(encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError(f"TODO file must contain a JSON array: {self._path}")
        return raw


def format_todos_message(todos: list[Todo]) -> str:
    """TODO 리스트를 에이전트용 포맷 문자열로 변환."""
    if not todos:
        # TODO 리스트가 비어 있을 때 안내 메시지 반환
        return "No todos currently in the list."
    # 현재 TODO 리스트를 번호, 이모지, 상태와 함께 포맷팅하여 문자열로 생성
    result = "Current TODO List:\n"
    for i, todo in enumerate(todos, 1):
        status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}
        emoji = status_emoji.get(todo["status"], "❓")
        result += f"{i}. {emoji} {todo['content']} ({todo['status']})\n"
    return result.strip()


def create_todo_tools(store: TodoFileStore | None = None) -> tuple[BaseTool, BaseTool]:
    """``TodoFileStore``에 바인딩된 ``write_todos`` / ``read_todos`` 툴 쌍을 만든다.

    그래프·에이전트 인스턴스마다 별도 ``TodoFileStore``를 넘기면 경로가 서로 간섭하지 않는다.
    """
    file_store = store or TodoFileStore()

    # write_todos 툴 정의, LLM이 전달한 TODO 리스트를 파일에 저장 및 state 갱신
    @tool(description=WRITE_TODOS_DESCRIPTION, parse_docstring=True)
    def write_todos(
        todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """Create or update the agent's TODO list for task planning and tracking.

        Persists the full list to a JSON file on disk and updates agent state.

        Args:
            todos: List of Todo items with content and status
            tool_call_id: Tool call identifier for message response

        Returns:
            Command to update agent state with new TODO list
        """
        saved_path = file_store.persist(todos)
        # TODO 리스트와 메시지 업데이트를 위한 Command 객체 반환
        return Command(
            update={
                "todos": todos,
                "messages": [
                    ToolMessage(
                        f"Updated todo list ({len(todos)} items) and saved to {saved_path}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    # read_todos 툴 정의, 디스크의 TODO JSON 파일을 읽어 포맷된 문자열로 반환
    @tool(parse_docstring=True)
    def read_todos(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Read the current TODO list from the on-disk JSON file.

        This tool loads todos from the configured file path (not only in-memory state)
        so progress survives across process restarts.

        Args:
            state: Injected agent state (used to sync loaded todos into state)
            tool_call_id: Injected tool call identifier for message tracking

        Returns:
            Command to update agent state with ToolMessage containing formatted TODO list
        """
        todos = file_store.load()
        message_content = format_todos_message(todos)
        if todos:
            message_content = f"{message_content}\n\n(Source: {file_store.path})"

        # Command 객체로 래핑하여 ToolMessage와 함께 반환 (파일 내용을 state에 동기화)
        return Command(
            update={
                "todos": todos,
                "messages": [
                    ToolMessage(message_content, tool_call_id=tool_call_id)
                ],
            }
        )

    return write_todos, read_todos


# 단일 기본 저장소용 모듈 레벨 툴 (노트북·간단 스크립트 호환)
_default_store = TodoFileStore()
write_todos, read_todos = create_todo_tools(_default_store)


__all__ = [
    "DEFAULT_TODOS_FILE",
    "TodoFileStore",
    "create_todo_tools",
    "format_todos_message",
    "read_todos",
    "write_todos",
]
