"""
참고 문서:
/deep_agents_from_scratch/notebooks/02-deep_agent_todo_graph.ipynb
(``DeepAgentState`` + ``write_todos`` / ``read_todos`` + 모킹 ``web_search`` ReAct 에이전트)
"""

from __future__ import annotations

from typing import cast

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph
from util.messages_in_jupyter import show_prompt
from base.base_graph import BaseGraph
from deep_agents_from_scratch.deep_agent_state import DeepAgentState
from deep_agents_from_scratch.prompts import TODO_USAGE_INSTRUCTIONS, TODO_USAGE_INSTRUCTIONS_KOR
from deep_agents_from_scratch.todo_tools import (
    DEFAULT_TODOS_FILE,
    TodoFileStore,
    create_todo_tools,
)
from util.chat_model_enums import LangChainChatModel

_DEFAULT_RECURSION_LIMIT = 20

# 단일 웹 검색 호출만 허용하는 간단한 리서치 지침 문자열
SIMPLE_RESEARCH_INSTRUCTIONS = """IMPORTANT: Just make a single call to the web_search tool and use the result provided by the tool to answer the user's question. Answer in Korean."""

SIMPLE_RESEARCH_INSTRUCTIONS_KOR = """중요: web_search 도구를 한 번만 호출하고, 도구가 제공한 결과를 사용하여 사용자 질문에 답하십시오. 답변은 한국어로 작성하십시오."""

SYSTEM_PROMPT = TODO_USAGE_INSTRUCTIONS + "\n\n" + "=" * 80 + "\n\n" + SIMPLE_RESEARCH_INSTRUCTIONS
SYSTEM_PROMPT_KOR = (
    TODO_USAGE_INSTRUCTIONS_KOR + "\n\n" + "=" * 80 + "\n\n" + SIMPLE_RESEARCH_INSTRUCTIONS_KOR
)

# 웹 검색 툴 모킹: 실제 검색 대신 고정된 결과 반환
@tool(parse_docstring=True)
def web_search(
    query: str,
):
    """Search the web for information on a specific topic.

    This tool performs web searches and returns relevant results
    for the given query. Use this when you need to gather information from
    the internet about any topic.

    Args:
        query: The search query string. Be specific and clear about what
               information you're looking for.

    Returns:
        Search results from search engine.

    Example:
        web_search("machine learning applications in healthcare")
    """

    # 웹 검색 결과 모킹, 실제 검색 대신 고정된 결과 반환
    search_result = """모델 컨텍스트 프로토콜(MCP)은 Anthropic이 개발한 개방형 표준 프로토콜로,
AI 모델과 도구, 데이터베이스, 기타 서비스와 같은 외부 시스템 간의 원활한 통합을 가능하게 합니다.
이는 표준화된 통신 계층 역할을 하여 AI 모델이 다양한 출처의 데이터에 일관되고 효율적인 방식으로 접근하고 활용할 수 있도록 합니다.
본질적으로 MCP는 데이터 교환을 위한 통합 언어를 제공함으로써 AI 어시스턴트를 외부 서비스에 연결하는 과정을 단순화합니다."""

    return search_result


class DeepAgentTodoGraph(BaseGraph):
    """``create_agent`` TODO·리서치 ReAct 에이전트 (``DeepAgentState`` + ``web_search`` 모킹).

    외부에서는 ``g = DeepAgentTodoGraph()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        temperature: float = 0.0,
        recursion_limit: int = _DEFAULT_RECURSION_LIMIT,
        todos_file: str | None = None,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        # 인스턴스별 TODO 저장소 (글로벌 경로 변경 없이 경로 격리)
        self._todo_store = TodoFileStore(
            todos_file if todos_file is not None else DEFAULT_TODOS_FILE
        )
        write_todos_tool, read_todos_tool = create_todo_tools(self._todo_store)

        # LLM 모델 초기화
        self._llm: BaseChatModel = init_chat_model(model, temperature=temperature)
        # 에이전트에 사용할 툴 리스트 정의, TODO 관리 및 웹 검색 포함
        self._tools: list[BaseTool] = [
            write_todos_tool,
            web_search,
            read_todos_tool,
        ]
        self._recursion_limit = recursion_limit
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    @property
    def todo_store(self) -> TodoFileStore:
        return self._todo_store

    def _compile_graph(self) -> CompiledStateGraph:
        # create_agent 함수로 에이전트 생성, 시스템 프롬프트에 TODO 사용 지침 및 리서치 지침 포함
        agent = create_agent(
            self._llm,
            self._tools,
            system_prompt=SYSTEM_PROMPT,
            state_schema=DeepAgentState,
        ).with_config(
            {"recursion_limit": self._recursion_limit}
        )  # 에이전트 반복 실행 최대 횟수 제한
        return cast(CompiledStateGraph, agent)

   

__all__ = [
    "DeepAgentTodoGraph",
    "SIMPLE_RESEARCH_INSTRUCTIONS",
    "SIMPLE_RESEARCH_INSTRUCTIONS_KOR",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_KOR",
    "web_search",
]
