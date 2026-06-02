"""
참고 문서:
/feature/HumanIntheLoopMiddlewareUseDB.py
/feature/SQLFullGraph.py

핵심:
스키마 탐색 서브그래프용 노드·도구 헬퍼를 정의하고 조립한다.
(first_tool_call → list_tables_tool → model_get_schema → get_schema_tool)
"""

from __future__ import annotations

from functools import partial

from typing import Any, cast

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from feature.SQLAgentState import SQLAgentState
from util.messages import random_uuid


# 오류 처리 함수
def handle_tool_error(state) -> dict:
    # 오류 정보 조회
    error = state.get("error")
    # 도구 정보 조회
    tool_calls = state["messages"][-1].tool_calls
    # ToolMessage 로 래핑 후 반환
    return {
        "messages": [
            ToolMessage(
                content=f"Here is the error: {repr(error)}\n\nPlease fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


# 오류를 처리하고 에이전트에 오류를 전달하기 위한 ToolNode 생성
def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """오류를 처리하고 에이전트에 오류 내용을 전달하는 ToolNode를 생성합니다.

    오류 발생 시 handle_tool_error 함수를 대체 동작(fallback)으로 실행하여,
    에이전트가 오류 내용을 인지하고 쿼리를 수정할 수 있도록 합니다.
    """
    # 오류 발생 시 대체 동작을 정의하여 ToolNode에 추가
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def setup_sql_toolkit(
    db: SQLDatabase, llm: BaseLanguageModel
) -> dict[str, Any]:
    """``SQLDatabaseToolkit`` 도구 dict를 반환한다."""
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return {t.name: t for t in toolkit.get_tools()}


# Node(Tool Call)
def first_tool_call(state: SQLAgentState) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": f"initial_tool_call_{random_uuid()}",
                    }
                ],
            )
        ]
    }


# ToolNode
def list_tables_tool(sql_tools: dict[str, Any]) -> RunnableWithFallbacks[Any, dict]:
    """테이블 목록 조회 ToolNode (``sql_db_list_tables``)."""
    return create_tool_node_with_fallback([sql_tools["sql_db_list_tables"]])


# Node(Tool Call)
def model_get_schema(
    state: SQLAgentState,
    *,
    llm: BaseLanguageModel,
    sql_tools: dict[str, Any],
) -> dict[str, list[AIMessage]]:
    """질문·테이블 목록을 바탕으로 관련 테이블 스키마 조회용 tool_calls를 생성한다."""
    return {
        "messages": [
            llm.bind_tools([sql_tools["sql_db_schema"]]).invoke(state["messages"])
        ],
    }


# ToolNode
def get_schema_tool(sql_tools: dict[str, Any]) -> RunnableWithFallbacks[Any, dict]:
    """스키마 조회 ToolNode (``sql_db_schema``)."""
    return create_tool_node_with_fallback([sql_tools["sql_db_schema"]])


def compile_schema_discovery_graph(
    *,
    llm: BaseChatModel,
    sql_tools: dict[str, Any],
) -> CompiledStateGraph:
    """스키마 탐색 그래프를 조립한다."""
    subgraph = StateGraph(SQLAgentState)

    # 첫 번째 도구 호출 노드 추가
    subgraph.add_node("first_tool_call", first_tool_call)

    # 테이블 목록 및 스키마 조회 도구 노드 추가
    subgraph.add_node("list_tables_tool", list_tables_tool(sql_tools))
    subgraph.add_node(
        "model_get_schema",
        partial(model_get_schema, llm=llm, sql_tools=sql_tools),
    )
    subgraph.add_node("get_schema_tool", get_schema_tool(sql_tools))

    subgraph.add_edge(START, "first_tool_call")
    subgraph.add_edge("first_tool_call", "list_tables_tool")
    subgraph.add_edge("list_tables_tool", "model_get_schema")
    subgraph.add_edge("model_get_schema", "get_schema_tool")
    subgraph.add_edge("get_schema_tool", END)

    return cast(CompiledStateGraph, subgraph.compile())


__all__ = [
    "compile_schema_discovery_graph",
    "create_tool_node_with_fallback",
    "first_tool_call",
    "get_schema_tool",
    "handle_tool_error",
    "list_tables_tool",
    "model_get_schema",
    "setup_sql_toolkit",
]
