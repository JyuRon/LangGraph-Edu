"""
참고 문서:
/langgraph-v1-tutorial/Appendix/B-Use-Cases/04-LangGraph-SQL-Agent.ipynb

핵심:
``SQLDatabaseToolkit`` + 커스텀 LangGraph 워크플로로 SQL 질의·스키마 조회·쿼리 검증·실행 루프를 구성한다.
DB는 생성자에 ``SQLDatabase`` 를 넘긴다 (연결은 호출 측에서 ``util.postgres_connection.connect_postgres`` 등).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel
from util.messages import random_uuid




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


# 쿼리 생성 지시 프롬프트 정의
_QUERY_GEN_INSTRUCTION = """You are a SQL expert with a strong attention to detail.

You can define SQL queries, analyze queries results and interpretate query results to response an answer.

Read the messages bellow and identify the user question, table schemas, query statement and query result, or error if they exist.

1. If there's not any query result that make sense to answer the question, create a syntactically correct {dialect} query to answer the user question. DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

2. If you create a query, response ONLY the query statement. For example, "SELECT id, name FROM pets;"

3. If a query was already executed, but there was an error. Response with the same error message you found. For example: "Error: Pets table doesn't exist"

4. If a query was already executed successfully interpretate the response and answer the question following this pattern: Answer: <<question answer>>. For example: "Answer: There three cats registered as adopted"
"""

# 쿼리 생성 지시 프롬프트 (한국어)
_QUERY_GEN_INSTRUCTION_KOR = """당신은 세밀함에 강한 SQL 전문가입니다.

SQL 쿼리를 작성하고, 쿼리 결과를 분석·해석해 사용자 질문에 답할 수 있습니다.

아래 메시지에서 사용자 질문, 테이블 스키마, 쿼리 문장, 쿼리 결과(또는 오류)를 파악하세요.

1. 질문에 답할 만한 쿼리 결과가 없으면, 사용자 질문에 맞는 문법적으로 올바른 {dialect} 쿼리를 작성하세요. 데이터베이스에 DML(INSERT, UPDATE, DELETE, DROP 등)은 실행하지 마세요.

2. 쿼리를 새로 작성한 경우, 쿼리 문장만 응답하세요. 예: "SELECT id, name FROM pets;"

3. 쿼리가 이미 실행됐지만 오류가 있으면, 발견한 오류 메시지를 그대로 응답하세요. 예: "Error: Pets table doesn't exist"

4. 쿼리가 성공적으로 실행됐으면 결과를 해석해 다음 형식으로 답하세요: Answer: <<질문에 대한 답>>. 예: "Answer: 등록된 입양 고양이는 세 마리입니다"
"""


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


# 최종 답변 제출을 위한 도구 스키마 정의
class SubmitFinalAnswer(BaseModel):
    """쿼리 결과를 기반으로 사용자에게 최종 답변을 제출합니다."""

    final_answer: str = Field(..., description="The final answer to the user")


# 에이전트의 상태 정의
class SQLAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class SQLAgentGraph(BaseGraph):
    """PostgreSQL 대상 커스텀 SQL 에이전트 LangGraph.

    외부에서는 ``g = SQLAgentGraph(db)`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
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

        self._sql_tools = self._setup_sql_toolkit()
        self._db_query_tool = self._build_db_query_tool()
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseLanguageModel:
        return self._llm

    @property
    def db(self) -> SQLDatabase:
        return self._db

    def _build_db_query_tool(self):
        """현재 인스턴스 DB 연결을 캡처한 쿼리 실행 도구를 생성한다."""

        @tool
        def db_query_tool(query: str) -> str:
            """
            Run SQL queries against a database and return results
            Returns an error message if the query is incorrect
            If an error is returned, rewrite the query, check, and retry
            """
            result = self._db.run_no_throw(query)
            if not result:
                return "Error: Query failed. Please rewrite your query and try again."
            return result

        return db_query_tool

    def _setup_sql_toolkit(self) -> dict[str, Any]:
        """``SQLDatabaseToolkit`` 도구 dict를 반환한다."""
        toolkit = SQLDatabaseToolkit(db=self._db, llm=self._llm)
        return {t.name: t for t in toolkit.get_tools()}

    # Node(Tool Call)
    def _first_tool_call(self, state: SQLAgentState) -> dict[str, list[AIMessage]]:
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
    def _list_tables_tool(self) -> RunnableWithFallbacks[Any, dict]:
        """테이블 목록 조회 ToolNode (``sql_db_list_tables``)."""
        return create_tool_node_with_fallback([self._sql_tools["sql_db_list_tables"]])


    # Node(Tool Call)
    def _model_get_schema(self, state: SQLAgentState) -> dict[str, list[AIMessage]]:
        """질문·테이블 목록을 바탕으로 관련 테이블 스키마 조회용 tool_calls를 생성한다."""
        return {
            "messages": [
                self._llm.bind_tools([self._sql_tools["sql_db_schema"]]).invoke(
                    state["messages"]
                )
            ],
        }

    # ToolNode
    def _get_schema_tool(self) -> RunnableWithFallbacks[Any, dict]:
        """스키마 조회 ToolNode (``sql_db_schema``)."""
        return create_tool_node_with_fallback([self._sql_tools["sql_db_schema"]])

    # Node (쿼리 생성 노드 함수 정의)
    def _query_gen_node(self, state: SQLAgentState) -> dict[str, list[AnyMessage]]:
        query_gen_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _QUERY_GEN_INSTRUCTION.format(dialect=self._db.dialect)),
                ("placeholder", "{messages}"),
            ]
        )

        # SubmitFinalAnswer 도구만 바인딩 (_query_gen_node 도구가 아닌 노드 함수)
        chain = query_gen_prompt | self._llm.bind_tools([SubmitFinalAnswer])

        message = chain.invoke(state)

        # LLM이 잘못된 도구를 호출할 경우 오류 메시지를 반환
        tool_messages = []
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc["name"] != "SubmitFinalAnswer":
                    tool_messages.append(
                        ToolMessage(
                            content=(
                                f"Error: The wrong tool was called: {tc['name']}. "
                                "Please fix your mistakes. Remember to only call "
                                "SubmitFinalAnswer to submit the final answer. "
                                "Generated queries should be outputted WITHOUT a tool call."
                            ),
                            tool_call_id=tc["id"],
                        )
                    )
        return {"messages": [message] + tool_messages}



    # 조건부 에지 정의: 다음 노드로의 라우팅 결정
    def _should_continue(
        self, state: SQLAgentState
    ) -> Literal[END, "handle_fail_query_gen", "correct_query", "query_gen"]:
        messages = state["messages"]
        last_message = messages[-1]

        # query_gen에서 나온 SubmitFinalAnswer tool_call을
        # handle_fail_query_gen 노드에서 AIMessage로 교체
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tc in last_message.tool_calls:
                if tc["name"] == "SubmitFinalAnswer":
                    return "handle_fail_query_gen"

        # 텍스트 응답이 "Answer:"로 시작하면 종료
        if isinstance(last_message.content, str) and last_message.content.startswith(
            "Answer:"
        ):
            return END

        # 텍스트 응답이 "Error:"로 시작하면 쿼리 재생성
        if isinstance(last_message.content, str) and last_message.content.startswith(
            "Error:"
        ):
            return "query_gen"

        # 그 외의 경우 쿼리 점검 노드로 이동
        return "correct_query"

    # Node(Tool Call)
    # 쿼리의 정확성을 모델로 점검하기 위한 노드 함수 정의
    def _correct_query(self, state: SQLAgentState) -> dict[str, list[AIMessage]]:
        """LLM을 통해 쿼리의 정확성을 점검하고 수정된 쿼리를 반환합니다.

        query_check 체인을 통해 마지막 메시지의 쿼리를 검토하고,
        오류가 있으면 수정된 쿼리를 db_query_tool 호출 형태로 반환합니다.
        """

        query_check_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _QUERY_CHECK_SYSTEM.format(dialect=self._db.dialect)),
                ("placeholder", "{messages}"),
            ]
        )
        # Query Checker 체인 생성
        chain = query_check_prompt | self._llm.bind_tools(
            [self._db_query_tool], tool_choice="db_query_tool"
        )

        return {
            "messages": [
                chain.invoke({"messages": [state["messages"][-1]]})
            ]
        }


    # Node: query_gen 단계에서 나온 SubmitFinalAnswer tool_call을 AIMessage로 답변
    def _handle_fail_query_gen(
        self, state: SQLAgentState
    ) -> dict[str, list[AIMessage]]:
        """query_gen에서 나온 SubmitFinalAnswer tool_call 메시지를 AIMessage로 교체한다.

        add_messages 리듀서는 같은 id의 메시지를 반환하면 기존 메시지를 덮어쓴다.
        엣지 함수에서 직접 state를 변경하면 리듀서를 거치지 않으므로 이 노드에서 처리한다.
        """
        last_message = state["messages"][-1]
        for tc in last_message.tool_calls:
            if tc["name"] == "SubmitFinalAnswer":
                final_answer = str(tc.get("args", {}).get("final_answer", "")).strip()
                return {
                    "messages": [
                        AIMessage(
                            # 같은 id 로 반환해야 add_messages 리듀서가 기존 메시지를 교체한다
                            id=last_message.id,
                            content=(
                                f"Answer: {final_answer}"
                                if final_answer
                                else "Error: final_answer is missing."
                            ),
                        )
                    ]
                }
        return {"messages": []}


    def _compile_graph(self) -> CompiledStateGraph:
        # 새로운 그래프 정의
        workflow = StateGraph(SQLAgentState)

        # 첫 번째 도구 호출 노드 추가
        workflow.add_node("first_tool_call", self._first_tool_call)

        # 테이블 목록 및 스키마 조회 도구 노드 추가
        workflow.add_node("list_tables_tool", self._list_tables_tool())
        workflow.add_node("model_get_schema", self._model_get_schema)
        workflow.add_node("get_schema_tool", self._get_schema_tool())

        # 쿼리 생성 노드 추가
        workflow.add_node("query_gen", self._query_gen_node)

        
        # 쿼리를 실행하기 전에 모델로 점검하는 노드 추가
        workflow.add_node("correct_query", self._correct_query)

        # 쿼리를 실행하기 위한 노드 추가
        workflow.add_node(
            "execute_query",
            create_tool_node_with_fallback([self._db_query_tool]),
        )

        # query_gen 단계 결과(SubmitFinalAnswer tool_call)를 AIMessage로 교체하는 노드 추가
        workflow.add_node("handle_fail_query_gen", self._handle_fail_query_gen)

        # 노드 간의 엣지 지정
        workflow.add_edge(START, "first_tool_call")
        workflow.add_edge("first_tool_call", "list_tables_tool")
        workflow.add_edge("list_tables_tool", "model_get_schema")
        workflow.add_edge("model_get_schema", "get_schema_tool")
        workflow.add_edge("get_schema_tool", "query_gen")
        workflow.add_conditional_edges(
            "query_gen",
            self._should_continue,
        )
        
        workflow.add_edge("correct_query", "execute_query")
        workflow.add_edge("execute_query", "query_gen")
        workflow.add_edge("handle_fail_query_gen", END)

        # 실행 가능한 워크플로우로 컴파일
        return workflow.compile(checkpointer=MemorySaver())


__all__ = [
    "SQLAgentGraph",
    "SQLAgentState",
    "SubmitFinalAnswer",
    "create_tool_node_with_fallback",
    "handle_tool_error",
]
