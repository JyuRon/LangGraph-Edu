"""
참고 문서:
note/Runtime_ToolRuntime.md
PART02-에이전트/Ch03-Runtime/04-LangGraph-Runtime.ipynb

핵심:
``context_schema`` 로 PostgreSQL 연결·권한·요청 설정을 주입하고,
``ToolRuntime``(도구)과 ``Runtime``(미들웨어)에서 ``context`` 에 접근한다.

``create_sql_agent_tool`` 은 ``SQLAgentGraph`` 를 래핑해 자연어 SQL 조회 도구를 반환하는
팩토리 함수다 (``task_tool._create_task_tool`` 과 같은 팩토리 패턴).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import before_agent
from langchain.agents.middleware.types import AgentState
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from base.base_graph import BaseGraph
from feature.SQLAgentGraph import SQLAgentGraph
from util.chat_model_enums import LangChainChatModel


# ``create_agent`` invoke 시 ``context=`` 로 전달하는 런타임 컨텍스트
@dataclass
class DatabaseContext:
    """PostgreSQL 연결·권한·요청 설정을 한 객체로 주입한다."""

    # DB
    user_id: str

    # Auth (check_permissions)
    permissions: list[str] = field(default_factory=lambda: ["read"])

    # Request (logging_after_user_tool) · 입력 길이 (check_permissions)
    verbose: bool = True
    timeout: int = 30
    max_tokens: int = 4096  # 최신 HumanMessage 토큰 상한


def _latest_human_message(state: AgentState) -> HumanMessage | None:
    """상태에서 가장 최근 HumanMessage. 없으면 None."""
    messages = state.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    return None


def _user_message_lower(state: AgentState) -> str:
    """가장 최근 HumanMessage 본문을 소문자 문자열로 반환."""
    human = _latest_human_message(state)
    if human is None:
        return ""
    if isinstance(human.content, str):
        return human.content.lower()
    return str(human.content).lower()


def _message_token_count(state: AgentState) -> int:
    """가장 최근 HumanMessage의 대략적 토큰 수 (모델별 정확한 카운트는 아님)."""
    human = _latest_human_message(state)
    if human is None:
        return 0
    return count_tokens_approximately([human])



# 모든 도구 응답 완료 후 logging_after_user_tool 호출을 강제하는 시스템 프롬프트.
#
# ⚠️ 코드 레벨 강제 여부
# - 현재(프롬프트 방식): LLM이 지시를 따르지 않을 경우 호출이 누락될 수 있다.
# - 코드로 100% 보장하려면 두 가지 대안이 있다.
#   1) logging_after_user_tool 로직을 @tool 에서 @after_agent 미들웨어로 이동
#      → 에이전트 루프 종료 시점에 한 번 실행되므로 "도구별 로그"는 불가능.
#   2) create_agent 대신 StateGraph 로 직접 구성하고, ToolNode 이후에
#      항상 logging 노드를 실행하는 엣지를 추가
#      → 세밀한 제어 가능하지만 그래프를 직접 조립해야 한다.
_AUDIT_LOG_INSTRUCTION = """\
After every tool response is received and before producing your final answer, \
you MUST call `logging_after_user_tool` exactly once.
Pass the name of the tool you just called as `tool_name` \
and its arguments as `tool_args`.
Do not skip this step even if the tool returned an error.\
"""

# 권한·입력 길이 검사 미들웨어 (max_tokens 초과·delete/remove 요청 차단)
@before_agent(can_jump_to=["end"])
def check_permissions(
    state: AgentState, runtime: Runtime[DatabaseContext]
) -> dict[str, Any] | None:
    """Check if user has required permissions"""
    permissions = runtime.context.permissions
    max_tokens = runtime.context.max_tokens

    # 사용자 입력 토큰 상한 (context.max_tokens)
    token_count = _message_token_count(state)
    if token_count > max_tokens:
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"Your message is too long. "
                        f"Maximum is {max_tokens} tokens "
                        f"(estimated {token_count})."
                    ),
                ),
            ],
            "jump_to": "end",
        }

    # 메시지에서 요청된 작업 확인
    content = _user_message_lower(state)
    if content:
        # 관리자 작업 요청 시 권한 확인 (메시지에 delete, remove 문구가 존재하면, 데모용)
        if "delete" in content or "remove" in content:
            if "admin" not in permissions:
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "You don't have permission to perform "
                                "this action."
                            ),
                        ),
                    ],
                    "jump_to": "end",
                }

    return None

# 개인적으로 middleware 영역이라고 생각
# 다른 도구 호출 직후 감사 로그를 남기는 데모 도구 (실제 DB·파일 I/O 없음)
@tool(parse_docstring=True)
def logging_after_user_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    runtime: ToolRuntime[DatabaseContext],
) -> str:
    """Record an audit log entry after another user-facing tool was invoked.

    데모: ``DatabaseContext`` 의 ``user_id``·``verbose``·``timeout`` 을 이용해
    어떤 도구가 어떤 인자로 호출됐는지 감사 로그 형태로 남긴다.

    Args:
        tool_name: 직전(또는 보고 대상) 사용자 도구 이름. 예: ``query_database``.
        tool_args: 해당 도구에 전달된 인자 dict (민감 값은 데모에서 그대로 출력하지 말 것).

    Returns:
        감사 로그가 기록되었음을 알리는 요약 문자열.
    """
    ctx = runtime.context
    log_line = (
        f"[audit] user={ctx.user_id} tool={tool_name!r} "
        f"args={tool_args!r} max_wait={ctx.timeout}s"
    )
    if ctx.verbose:
        print(log_line)
    return (
        f"Audit logged for user={ctx.user_id}: "
        f"tool={tool_name}, timeout_budget={ctx.timeout}s"
    )


def create_sql_agent_tool(
    db: SQLDatabase,
    model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O,
) -> BaseTool:
    """``SQLAgentGraph`` 인스턴스를 캡처해 자연어 SQL 조회 도구로 래핑한다.

    ``task_tool._create_task_tool`` 과 같은 팩토리 패턴:
    ``db``·``model`` 로 ``SQLAgentGraph`` 를 미리 생성한 뒤,
    반환된 도구가 호출될 때마다 해당 그래프로 위임한다.

    ``DatabaseContext.user_id`` 를 LangGraph ``thread_id`` 로 활용해
    사용자별 체크포인터 대화 이력을 격리한다.

    Args:
        db: 연결된 ``SQLDatabase`` 인스턴스.
        model: ``SQLAgentGraph`` 에서 사용할 LLM 모델 이름 또는 열거형.

    Returns:
        ``query_database_with_agent`` 도구 인스턴스.
    """
    # SQLAgentGraph 미리 생성 — 이후 도구 호출마다 재사용
    # 환경 변수는 부모 에이전트가 이미 로드했으므로 load_env=False
    sql_graph = SQLAgentGraph(db=db, model=model, load_env=False)

    @tool(parse_docstring=True)
    def query_database_with_agent(
        question: str,
        runtime: ToolRuntime[DatabaseContext],
    ) -> str:
        """Run a natural-language question against the database via the SQL agent.

        ``SQLAgentGraph`` 를 통해 자연어 질문을 SQL 쿼리로 변환·실행하고
        해석된 자연어 답변을 반환한다.

        Args:
            question: 데이터베이스에 대한 자연어 질문. 예: "가장 많이 팔린 상품은?".

        Returns:
            쿼리 결과를 해석한 자연어 답변 문자열.
        """
        ctx = runtime.context
        # user_id 를 thread_id 로 사용해 사용자별 체크포인터 대화 이력 격리
        config = cast(RunnableConfig, {"configurable": {"thread_id": ctx.user_id}})
        result = sql_graph.invoke({"messages": [("user", question)]}, config=config)
        if result is None:
            return "SQL 에이전트가 결과를 반환하지 않았습니다."
        messages = result.get("messages", [])
        if not messages:
            return "SQL 에이전트가 결과를 반환하지 않았습니다."
        last = messages[-1]
        return last.content if hasattr(last, "content") else str(last)

    return query_database_with_agent


class RuntimeDBConnectionAgent(BaseGraph):
    """PostgreSQL ``context`` 주입·권한 미들웨어·요청 설정 도구를 쓰는 ``create_agent`` 데모.

    외부에서는 ``g = RuntimeDBConnectionAgent(db=db)`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.

    DB 연결은 ``util.postgres_connection.connect_postgres`` 후 ``DatabaseContext`` 를 ``invoke`` 에 넘긴다.
    ``db`` 를 넘기면 ``SQLAgentGraph`` 기반 ``query_database_with_agent`` 도구가 자동으로 추가된다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        db: SQLDatabase | None = None,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._model = model
        self._llm: BaseChatModel = init_chat_model(model)
        self._db = db
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def db(self) -> SQLDatabase | None:
        return self._db

    def _compile_graph(self) -> CompiledStateGraph:
        tools: list[BaseTool] = [logging_after_user_tool]

        # db 가 주입된 경우 SQLAgentGraph 기반 SQL 조회 도구를 추가한다
        if self._db is not None:
            tools.append(create_sql_agent_tool(self._db, model=self._model))

        agent = create_agent(
            model=self._llm,
            tools=tools,
            middleware=[check_permissions],
            context_schema=DatabaseContext,
            # 프롬프트로 LLM에게 호출 순서를 지시한다.
            # 코드 레벨 강제가 필요하면 위 _AUDIT_LOG_INSTRUCTION 주석 참고.
            system_prompt=_AUDIT_LOG_INSTRUCTION,
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "DatabaseContext",
    "RuntimeDBConnectionAgent",
    "check_permissions",
    "create_sql_agent_tool",
    "logging_after_user_tool",
]
