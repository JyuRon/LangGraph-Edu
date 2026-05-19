"""Filesystem tools for agent context offloading.

This module provides tools for managing files on disk under a configured
directory, enabling context offloading and persistence across agent interactions.
Use ``AgentFileStore`` + ``create_file_tools(store)`` so each agent/graph
instance owns its directory (no process-wide mutable global).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.types import Command

from .prompts import (
    LS_DESCRIPTION,
    READ_FILE_DESCRIPTION,
    WRITE_FILE_DESCRIPTION,
)

# 기본 에이전트 파일 저장 디렉터리 (패키지 디렉터리 아래 .agent_data/files)
_DEFAULT_AGENT_FILES_DIR = Path(__file__).resolve().parent / ".agent_data" / "files"
DEFAULT_AGENT_FILES_DIR = _DEFAULT_AGENT_FILES_DIR


class AgentFileStore:
    """에이전트 파일 디렉터리 경로를 캡슐화. 인스턴스마다 독립된 저장 위치."""

    def __init__(self, agent_files_dir: Path | str = DEFAULT_AGENT_FILES_DIR) -> None:
        self._root = Path(agent_files_dir)

    @property
    def root(self) -> Path:
        return self._root

    def _resolve_path(self, file_path: str) -> Path:
        """루트 내 상대 경로로 해석. 경로 탈출(..)은 거부."""
        root = self._root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        target = (self._root / file_path).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"File path '{file_path}' escapes the agent files directory"
            ) from exc
        return target

    def list_files(self) -> list[str]:
        """저장 디렉터리 아래 파일 상대 경로 목록 (재귀)."""
        self._root.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for path in self._root.rglob("*"):
            if path.is_file():
                paths.append(path.relative_to(self._root).as_posix())
        return sorted(paths)

    def read_text(self, file_path: str) -> str:
        """파일 전체 텍스트 읽기. 없으면 FileNotFoundError."""
        target = self._resolve_path(file_path)
        if not target.is_file():
            raise FileNotFoundError(file_path)
        return target.read_text(encoding="utf-8")

    def write_text(self, file_path: str, content: str) -> Path:
        """파일 쓰기(덮어쓰기). 부모 디렉터리 생성."""
        target = self._resolve_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target


def format_file_content(content: str, offset: int = 0, limit: int = 2000) -> str:
    """파일 내용을 줄 번호와 함께 포맷."""
    if not content:
        return "System reminder: File exists but has empty contents"

    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    result_lines = []
    for i in range(start_idx, end_idx):
        line_content = lines[i][:2000]  # Truncate long lines
        result_lines.append(f"{i + 1:6d}\t{line_content}")

    return "\n".join(result_lines)


def create_file_tools(
    store: AgentFileStore | None = None,
) -> tuple[BaseTool, BaseTool, BaseTool]:
    """``AgentFileStore``에 바인딩된 ``ls`` / ``read_file`` / ``write_file`` 툴 세트를 만든다.

    그래프·에이전트 인스턴스마다 별도 ``AgentFileStore``를 넘기면 경로가 서로 간섭하지 않는다.
    """
    file_store = store or AgentFileStore()

    @tool(description=LS_DESCRIPTION)
    def ls() -> list[str]:
        """List all files in the agent files directory."""
        return file_store.list_files()

    @tool(description=READ_FILE_DESCRIPTION, parse_docstring=True)
    def read_file(
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content from disk with optional offset and limit.

        Args:
            file_path: Path to the file to read (relative to the agent files directory)
            offset: Line number to start reading from (default: 0)
            limit: Maximum number of lines to read (default: 2000)

        Returns:
            Formatted file content with line numbers, or error message if file not found
        """
        try:
            content = file_store.read_text(file_path)
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found"
        except ValueError as exc:
            return f"Error: {exc}"

        return format_file_content(content, offset, limit)

    @tool(description=WRITE_FILE_DESCRIPTION, parse_docstring=True)
    def write_file(
        file_path: str,
        content: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Write content to a file on disk and mirror it into ``state["files"]``.

        디스크가 단일 영속 저장소이며, 동시에 ``state["files"]`` 인메모리 미러를
        갱신해 LangGraph reducer가 sub-agent → 부모 자동 병합을 처리하게 한다.

        Args:
            file_path: Path where the file should be created/updated
            content: Content to write to the file
            tool_call_id: Tool call identifier for message response

        Returns:
            Command updating ``state["files"]`` and a confirming ToolMessage
        """
        try:
            saved_path = file_store.write_text(file_path, content)
        except ValueError as exc:
            return Command(
                update={
                    "messages": [
                        ToolMessage(f"Error: {exc}", tool_call_id=tool_call_id)
                    ],
                }
            )

        # 디스크 + state 미러링: file_reducer가 기존 dict와 자동 병합
        return Command(
            update={
                "files": {file_path: content},
                "messages": [
                    ToolMessage(
                        f"Updated file {file_path} (saved to {saved_path})",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return ls, read_file, write_file


# 단일 기본 저장소용 모듈 레벨 툴 (노트북·간단 스크립트 호환)
_default_store = AgentFileStore()
ls, read_file, write_file = create_file_tools(_default_store)


__all__ = [
    "DEFAULT_AGENT_FILES_DIR",
    "AgentFileStore",
    "create_file_tools",
    "format_file_content",
    "ls",
    "read_file",
    "write_file",
]
