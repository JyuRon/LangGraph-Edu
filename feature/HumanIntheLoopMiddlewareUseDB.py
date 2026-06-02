"""
참고 문서:
/langgraph-v1-tutorial/PART02-에이전트/Ch05-Human-in-the-Loop/02-LangGraph-Human-In-The-Loop.ipynb

핵심:
``HumanInTheLoopMiddleware`` + ``PostgresSaver`` — interrupt 후 ``Command(resume=...)`` 재개.

부모 ``StateGraph`` 에 ``SQLFullGraph`` 노드를 서브그래프로 조립:
``schema_discovery`` → ``query_loop`` (쿼리 생성·점검, 실행 없음) → ``agent`` (HITL ``create_agent``,
SQL 실행은 ``execute_sql`` 만). 체크포인터는 부모 ``compile`` 시만 지정, ``agent`` 는 상속(per-invocation).

PostgreSQL: ``util.postgres_connection.connect_postgres`` / ``postgres_url``.
승인 정책: ``execute_sql`` approve·reject만, ``backup_database`` 전체 결정.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from base.base_graph import BaseGraph
from feature.SQLAgentState import SQLAgentState
from feature.SQLQueryGenerationGraph import compile_sql_query_generation_graph
from feature.SQLSchemaDiscoveryGraph import compile_schema_discovery_graph
from feature.SQLFullGraph import build_db_query_tool, setup_sql_toolkit
from util.chat_model_enums import LangChainChatModel
from util.postgres_connection import create_postgres_checkpointer, postgres_url


def _format_sql_run_result(db: SQLDatabase, command: str) -> str:
    """``SQLDatabase.run`` 결과를 도구 응답 문자열로 변환한다."""
    return str(db.run(command))


def create_db_hitl_tools(db: SQLDatabase, *, load_env: bool = True) -> list[BaseTool]:
    """``SQLDatabase`` 에 바인딩된 DB HITL 도구 목록을 만든다."""

    libpq_conn = postgres_url(load_env=load_env, dialect="libpq")

    @tool
    def execute_sql(query: str) -> str:
        """데이터베이스에서 SQL 쿼리를 실행합니다."""
        try:
            return _format_sql_run_result(db, query)
        except Exception as exc:  # noqa: BLE001 — SQL 오류를 도구 결과로 반환
            return f"SQL execution failed: {exc}"

    # libpq 형태 (postgresql://)
    @tool
    def backup_database() -> str:
        """전체 데이터베이스를 백업합니다."""
        backup_dir = Path("backups")
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_file = backup_dir / f"backup_{timestamp}.sql"
        try:
            completed = subprocess.run(
                ["pg_dump", libpq_conn, "-f", str(out_file)],
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
        except FileNotFoundError:
            return (
                "Backup failed: `pg_dump` not found. "
                "Install PostgreSQL client tools or run backup outside this agent."
            )
        except subprocess.TimeoutExpired:
            return "Backup failed: `pg_dump` timed out after 600s."

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            return f"Backup failed: {stderr or 'pg_dump exited with error'}"

        return f"Database backup completed: {out_file.resolve()}"

    return [execute_sql, backup_database]


class HumanIntheLoopMiddlewareUseDB(BaseGraph):
    """DB HITL 그래프 데모. 그래프 구성·승인 정책·체크포인터는 모듈 docstring 참고.

    외부에서는 ``g = HumanIntheLoopMiddlewareUseDB(db=db)`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다. ``db`` 는 ``connect_postgres`` 로 연결한 ``SQLDatabase``.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        db: SQLDatabase,
        checkpointer: BaseCheckpointSaver | None = None,
        load_env: bool = True,
        langsmith_project: str | None = None,
        description_prefix: str = "Database operation pending approval",
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._db = db
        self._checkpointer: BaseCheckpointSaver = (
            checkpointer
            if checkpointer is not None
            else create_postgres_checkpointer(load_env=load_env)
        )
        self._tools: list[BaseTool] = create_db_hitl_tools(db, load_env=load_env)
        self._llm: BaseChatModel = init_chat_model(model)
        self._sql_tools: dict[str, Any] = setup_sql_toolkit(self._db, self._llm)
        self._db_query_tool = build_db_query_tool(self._db)
        self._description_prefix = description_prefix
        self._schema_discovery_subgraph: CompiledStateGraph = (
            compile_schema_discovery_graph(llm=self._llm, sql_tools=self._sql_tools)
        )
        self._query_loop_subgraph: CompiledStateGraph = (
            compile_sql_query_generation_graph(llm=self._llm, db=self._db)
        )
        self._agent: CompiledStateGraph = self._create_agent()
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def db(self) -> SQLDatabase:
        return self._db

    @property
    def checkpointer(self) -> BaseCheckpointSaver:
        return self._checkpointer

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    def _create_agent(self) -> CompiledStateGraph:
        # 데이터베이스 에이전트 생성
        agent = create_agent(
            model=self._llm,
            tools=self._tools,
            middleware=[
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        # SQL 실행은 승인 또는 거부만 가능 (편집 불가)
                        "execute_sql": {"allowed_decisions": ["approve", "reject"]},
                        # 백업은 모든 결정 허용
                        "backup_database": True,
                    },
                    description_prefix=self._description_prefix,
                ),
            ],
            # checkpointer 는 부모 ``_compile_graph`` 에서만 지정 — 서브그래프가 상속(per-invocation)
        )
        return cast(CompiledStateGraph, agent)

    def _compile_graph(self) -> CompiledStateGraph:
        # schema_discovery → query_loop 서브그래프 → create_agent(HITL)
        builder = StateGraph(SQLAgentState)
        builder.add_node("schema_discovery", self._schema_discovery_subgraph)
        builder.add_node("query_loop", self._query_loop_subgraph)
        builder.add_node("agent", self._agent)

        
        builder.add_edge(START, "schema_discovery")
        builder.add_edge("schema_discovery", "query_loop")
        builder.add_edge("query_loop", "agent")
        # 외부 ``invoke`` 대상 — PostgresSaver 로 thread_id·HITL ``Command(resume=...)`` 유지
        return cast(
            CompiledStateGraph,
            builder.compile(checkpointer=self._checkpointer),
        )


__all__ = [
    "HumanIntheLoopMiddlewareUseDB",
    "create_db_hitl_tools",
]
