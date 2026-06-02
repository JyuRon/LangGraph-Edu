"""
참고 문서:
/feature/HumanIntheLoopMiddlewareUseDB.py
/feature/SQLFullGraph.py

핵심:
SQL 쿼리 생성·점검 서브그래프용 상태·노드·조건부 엣지를 정의하고 조립한다.
SQL 실행은 그래프 밖 ``agent`` 의 ``execute_sql`` (HITL) 로만 수행한다.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Literal, cast

from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from feature.SQLAgentState import SQLAgentState

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

# SQL 쿼리 점검 — 실행·도구 호출 없이 텍스트(SQL 또는 재작성 안내)만 반환
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

If there are any of the above mistakes, rewrite the query.
If there are no mistakes, reproduce the original query.

Respond with plain text only. Do not use tool calls. Do not execute the query.
- If the query is valid (possibly after your rewrite), output ONLY the SQL statement.
- If the query cannot be validated or must be regenerated, output:
  Error: Please rewrite your query and try again.
  followed by one short line explaining what to fix."""



# SQL 쿼리 점검 — 실행·도구 호출 없이 텍스트만 반환 (한국어)
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

일반 텍스트로만 응답하세요. 도구 호출을 사용하지 마세요. 쿼리를 실행하지 마세요.
- 쿼리가 유효하면(수정했다면 수정본 기준) SQL 문장만 출력하세요.
- 쿼리를 검증할 수 없거나 다시 작성해야 하면 다음을 출력하세요:
  Error: Please rewrite your query and try again.
  그리고 수정할 내용을 한 줄로 간단히 설명하세요."""




# 최종 답변 제출을 위한 도구 스키마 정의
class SubmitFinalAnswer(BaseModel):
    """쿼리 결과를 기반으로 사용자에게 최종 답변을 제출합니다."""

    final_answer: str = Field(..., description="The final answer to the user")


# Node (쿼리 생성 노드 함수 정의)
def query_gen_node(
    state: SQLAgentState,
    *,
    llm: BaseLanguageModel,
    db: SQLDatabase,
) -> dict[str, list[AnyMessage]]:
    query_gen_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _QUERY_GEN_INSTRUCTION.format(dialect=db.dialect)),
            ("placeholder", "{messages}"),
        ]
    )

    # SubmitFinalAnswer 도구만 바인딩 (query_gen_node 도구가 아닌 노드 함수)
    chain = query_gen_prompt | llm.bind_tools([SubmitFinalAnswer])

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
def should_continue(
    state: SQLAgentState,
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


# Node: query_gen 단계에서 나온 SubmitFinalAnswer tool_call을 AIMessage로 답변
def handle_fail_query_gen(
    state: SQLAgentState,
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


def correct_query(
    state: SQLAgentState,
    *,
    llm: BaseChatModel,
    db: SQLDatabase,
) -> dict[str, list[AIMessage]]:
    """LLM으로 쿼리를 점검하고 SQL 문장 또는 재작성 안내 텍스트만 반환한다 (도구 호출 없음).

    ``SQLFullGraph.correct_query`` 는 ``db_query_tool`` tool_call 을 쓰지만,
    HITL 그래프에서는 실행을 ``agent`` 의 ``execute_sql`` 에만 맡긴다.
    """
    query_check_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _QUERY_CHECK_SYSTEM.format(dialect=db.dialect)),
            ("placeholder", "{messages}"),
        ]
    )
    chain = query_check_prompt | llm
    response = chain.invoke({"messages": [state["messages"][-1]]})

    if not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))

    return {"messages": [response]}


def compile_sql_query_generation_graph(
    *,
    llm: BaseChatModel,
    db: SQLDatabase,
) -> CompiledStateGraph:
    """SQL 쿼리 생성·점검 그래프를 조립한다.

    SQL 실행은 그래프 밖 ``agent`` 의 ``execute_sql`` (HITL) 로만 수행한다.
    """
    subgraph = StateGraph(SQLAgentState)

    # 쿼리 생성 노드 추가
    subgraph.add_node(
        "query_gen",
        partial(query_gen_node, llm=llm, db=db),
    )

    # 쿼리를 실행하기 전에 모델로 점검하는 노드 추가 (SQL 텍스트 또는 재작성 안내만 반환)
    subgraph.add_node(
        "correct_query",
        partial(correct_query, llm=llm, db=db),
    )

    # query_gen 단계 결과(SubmitFinalAnswer tool_call)를 AIMessage로 교체하는 노드 추가
    subgraph.add_node("handle_fail_query_gen", handle_fail_query_gen)

    # 노드 간의 엣지 지정
    subgraph.add_edge(START, "query_gen")
    subgraph.add_conditional_edges(
        "query_gen",
        should_continue,
    )
    # 점검 후 실행은 agent(HITL execute_sql)에 위임
    subgraph.add_edge("correct_query", END)
    subgraph.add_edge("handle_fail_query_gen", END)

    return cast(CompiledStateGraph, subgraph.compile())


__all__ = [
    "SubmitFinalAnswer",
    "compile_sql_query_generation_graph",
    "correct_query",
    "handle_fail_query_gen",
    "query_gen_node",
    "should_continue",
]
