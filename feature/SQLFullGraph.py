"""
참고 문서:
/langgraph-v1-tutorial/Appendix/B-Use-Cases/04-LangGraph-SQL-Agent.ipynb

핵심:
``SQLDatabaseToolkit`` + 커스텀 LangGraph 워크플로로 SQL 질의·스키마 조회·쿼리 검증·실행 루프를 구성한다.
DB는 생성자에 ``SQLDatabase`` 를 넘긴다 (연결은 호출 측에서 ``util.postgres_connection.connect_postgres`` 등).
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from base.base_graph import BaseGraph
from feature.SQLAgentState import SQLAgentState
from feature.SQLQueryGenerationGraph import (
    SubmitFinalAnswer,
    handle_fail_query_gen,
    query_gen_node,
    should_continue,
)
from feature.SQLSchemaDiscoveryGraph import (
    create_tool_node_with_fallback,
    first_tool_call,
    get_schema_tool,
    handle_tool_error,
    list_tables_tool,
    model_get_schema,
    setup_sql_toolkit,
)
from util.chat_model_enums import LangChainChatModel



# SQL 쿼리의 일반적인 실수를 점검하기 위한 시스템 메시지 정의
_QUERY_CHECK_SYSTEM = """You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check."""

# SQL 쿼리의 일반적인 실수를 점검하기 위한 시스템 메시지 (한국어)
_QUERY_CHECK_SYSTEM_KOR = """당신은 세밀함에 강한 SQL 전문가입니다.
{dialect} 쿼리에서 다음과 같은 흔한 실수가 있는지 다시 확인하세요.
- NULL 값과 함께 NOT IN 사용
- UNION ALL을 써야 하는데 UNION 사용
- 배타적 범위에 BETWEEN 사용
- 조건절의 데이터 타입 불일치
- 식별자 따옴표 처리 오류
- 함수 인자 개수 오류
- 잘못된 데이터 타입 캐스팅
- 조인에 잘못된 컬럼 사용

위 실수가 있으면 쿼리를 수정하세요. 실수가 없으면 원본 쿼리를 그대로 반환하세요.

점검이 끝나면 적절한 도구를 호출해 쿼리를 실행하세요."""



# Node(Tool Call)
# 쿼리의 정확성을 모델로 점검하기 위한 노드 함수 정의
def correct_query(
    state: SQLAgentState,
    *,
    llm: BaseLanguageModel,
    db: SQLDatabase,
    db_query_tool: Any,
) -> dict[str, list[AIMessage]]:
    """LLM을 통해 쿼리의 정확성을 점검하고 수정된 쿼리를 반환합니다.

    query_check 체인을 통해 마지막 메시지의 쿼리를 검토하고,
    오류가 있으면 수정된 쿼리를 db_query_tool 호출 형태로 반환합니다.
    """

    query_check_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _QUERY_CHECK_SYSTEM.format(dialect=db.dialect)),
            ("placeholder", "{messages}"),
        ]
    )
    # Query Checker 체인 생성
    chain = query_check_prompt | llm.bind_tools(
        [db_query_tool], tool_choice="db_query_tool"
    )

    return {
        "messages": [
            chain.invoke({"messages": [state["messages"][-1]]})
        ]
    }

def build_db_query_tool(db: SQLDatabase):
    """DB 연결을 캡처한 쿼리 실행 도구를 생성한다."""

    @tool
    def db_query_tool(query: str) -> str:
        """
        Run SQL queries against a database and return results
        Returns an error message if the query is incorrect
        If an error is returned, rewrite the query, check, and retry
        """
        result = db.run_no_throw(query)
        if not result:
            return "Error: Query failed. Please rewrite your query and try again."
        return result

    return db_query_tool


class SQLFullGraph(BaseGraph):
    """PostgreSQL 대상 읽기 전용 커스텀 SQL 에이전트 LangGraph.

    외부에서는 ``g = SQLFullGraph(db)`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.

    ``SQLDatabase`` 는 생성자 인자로 받는다.
    """

    def __init__(
        self,
        db: SQLDatabase,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O,
        *,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._llm: BaseLanguageModel = init_chat_model(model)
        self._db = db

        self._sql_tools = setup_sql_toolkit(self._db, self._llm)
        self._db_query_tool = build_db_query_tool(self._db)
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseLanguageModel:
        return self._llm

    @property
    def db(self) -> SQLDatabase:
        return self._db

    def _compile_graph(self) -> CompiledStateGraph:
        # 새로운 그래프 정의
        workflow = StateGraph(SQLAgentState)

        # 첫 번째 도구 호출 노드 추가
        workflow.add_node("first_tool_call", first_tool_call)

        # 테이블 목록 및 스키마 조회 도구 노드 추가
        workflow.add_node("list_tables_tool", list_tables_tool(self._sql_tools))
        workflow.add_node(
            "model_get_schema",
            partial(model_get_schema, llm=self._llm, sql_tools=self._sql_tools),
        )
        workflow.add_node("get_schema_tool", get_schema_tool(self._sql_tools))

        # 쿼리 생성 노드 추가
        workflow.add_node(
            "query_gen",
            partial(query_gen_node, llm=self._llm, db=self._db),
        )

        # 쿼리를 실행하기 전에 모델로 점검하는 노드 추가
        workflow.add_node(
            "correct_query",
            partial(
                correct_query,
                llm=self._llm,
                db=self._db,
                db_query_tool=self._db_query_tool,
            ),
        )

        # 쿼리를 실행하기 위한 노드 추가
        workflow.add_node(
            "execute_query",
            create_tool_node_with_fallback([self._db_query_tool]),
        )

        # query_gen 단계 결과(SubmitFinalAnswer tool_call)를 AIMessage로 교체하는 노드 추가
        workflow.add_node("handle_fail_query_gen", handle_fail_query_gen)

        # 노드 간의 엣지 지정
        workflow.add_edge(START, "first_tool_call")
        workflow.add_edge("first_tool_call", "list_tables_tool")
        workflow.add_edge("list_tables_tool", "model_get_schema")
        workflow.add_edge("model_get_schema", "get_schema_tool")
        workflow.add_edge("get_schema_tool", "query_gen")
        workflow.add_conditional_edges(
            "query_gen",
            should_continue,
        )
        
        workflow.add_edge("correct_query", "execute_query")
        workflow.add_edge("execute_query", "query_gen")
        workflow.add_edge("handle_fail_query_gen", END)

        # 실행 가능한 워크플로우로 컴파일
        return workflow.compile(checkpointer=MemorySaver())


__all__ = [
    "SQLFullGraph",
    "SQLAgentState",
    "SubmitFinalAnswer",
    "build_db_query_tool",
    "correct_query",
    "create_tool_node_with_fallback",
    "first_tool_call",
    "get_schema_tool",
    "handle_fail_query_gen",
    "handle_tool_error",
    "list_tables_tool",
    "model_get_schema",
    "query_gen_node",
    "setup_sql_toolkit",
    "should_continue",
]
