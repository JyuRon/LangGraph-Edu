"""
참고 문서:
/feature/SQLSchemaDiscoveryGraph.py
/feature/SQLQueryGenerationGraph.py
/feature/SQLFullGraph.py

핵심:
SQL 에이전트 LangGraph 공통 상태 스키마 (``messages`` + ``add_messages`` 리듀서).
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# 에이전트의 상태 정의
class SQLAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


__all__ = ["SQLAgentState"]
