"""
참고 문서:
/deep_agents_from_scratch/notebooks_original/01-DeepAgents-Basic.ipynb
(LangChain ``create_agent`` — ReAct 계산기 에이전트, ``CalcState`` + ``ops`` 누적)
"""

from __future__ import annotations

from typing import Annotated, Literal, cast

from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel

_DEFAULT_RECURSION_LIMIT = 20

# 시스템 프롬프트 정의
SYSTEM_PROMPT = """You are a helpful arithmetic assistant who is an expert at using a calculator.
Return all text as plain text without Markdown math delimiters.
"""


# 리듀서: 두 리스트를 안전하게 병합하는 함수, None 입력 시 빈 리스트로 처리
def reduce_list(left: list | None, right: list | None) -> list:
    """두 리스트를 안전하게 병합, 입력값이 None일 경우 빈 리스트로 처리

    Args:
        left (list | None): 병합할 첫 번째 리스트 또는 None
        right (list | None): 병합할 두 번째 리스트 또는 None

    Returns:
        list: 두 입력 리스트의 모든 요소를 포함하는 새 리스트, None은 빈 리스트로 간주
    """
    if not left:
        left = []
    if not right:
        right = []
    return left + right


# ops 라는 연산 기록용 키를 추가합니다. 이때 reducer 를 사용하여 연산 값이 누적되도록 합니다.
class CalcState(AgentState):
    """그래프 상태 클래스"""

    ops: Annotated[list[str], reduce_list]


@tool
def calculator(
    operation: Literal["add", "subtract", "multiply", "divide"],
    a: int | float,
    b: int | float,
    state: Annotated[CalcState, InjectedState],  # not sent to LLM
    tool_call_id: Annotated[str, InjectedToolCallId],  # not sent to LLM
) -> Command:
    """Define a two-input calculator tool.

    Arg:
        operation (str): The operation to perform ('add', 'subtract', 'multiply', 'divide').
        a (float or int): The first number.
        b (float or int): The second number.

    Returns:
        result (float or int): the result of the operation
    Example
        Divide: result   = a / b
        Subtract: result = a - b
    """
    if operation == "divide" and b == 0:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "Division by zero is not allowed.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    # Perform calculation
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        result = a / b

    ops = [f"({operation}, {a}, {b}),"]
    return Command(
        update={
            "ops": ops,
            "messages": [ToolMessage(f"{result}", tool_call_id=tool_call_id)],
        }
    )


class SampleAgnetState(BaseGraph):
    """``create_agent`` ReAct 계산기 에이전트 (``CalcState`` + 연산 기록 ``ops``).

    외부에서는 ``a = SampleAgnetState()`` 뒤 ``a.invoke(...)`` / ``a.stream(...)``,
    LangGraph **노드 구조**는 ``a.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        temperature: float = 0.0,
        recursion_limit: int = _DEFAULT_RECURSION_LIMIT,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        # LLM 인스턴스 생성
        self._llm: BaseChatModel = init_chat_model(model, temperature=temperature)
        # 도구 목록에 Tool 추가
        self._tools: list[BaseTool] = [calculator]
        self._recursion_limit = recursion_limit
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    def _compile_graph(self) -> CompiledStateGraph:
        # ReAct Agent 생성
        agent = create_agent(
            self._llm,
            self._tools,
            system_prompt=SYSTEM_PROMPT,
            state_schema=CalcState,
        ).with_config(
            {"recursion_limit": self._recursion_limit}
        )  # 에이전트 반복 실행 최대 횟수 제한
        return cast(CompiledStateGraph, agent)


__all__ = [
    "CalcState",
    "SYSTEM_PROMPT",
    "SampleAgnetState",
    "calculator",
    "reduce_list",
]
