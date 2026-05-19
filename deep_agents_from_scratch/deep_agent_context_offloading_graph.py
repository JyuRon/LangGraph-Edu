"""
참고 문서:
/deep_agents_from_scratch/notebooks/03-deep_agent_context_offloading.ipynb
(``DeepAgentState`` + 디스크 파일 ``ls`` / ``read_file`` / ``write_file`` + 모킹 ``web_search`` ReAct 에이전트)
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph

from base.base_graph import BaseGraph
from deep_agents_from_scratch.deep_agent_state import DeepAgentState
from deep_agents_from_scratch.file_tools import (
    DEFAULT_AGENT_FILES_DIR,
    AgentFileStore,
    create_file_tools,
)
from deep_agents_from_scratch.prompts import (
    FILE_USAGE_INSTRUCTIONS,
    FILE_USAGE_INSTRUCTIONS_KOR,
    SIMPLE_RESEARCH_INSTRUCTIONS,
    SIMPLE_RESEARCH_INSTRUCTIONS_KOR,
)
from util.chat_model_enums import LangChainChatModel

_DEFAULT_RECURSION_LIMIT = 20

SYSTEM_PROMPT = (
    FILE_USAGE_INSTRUCTIONS + "\n\n" + "=" * 80 + "\n\n" + SIMPLE_RESEARCH_INSTRUCTIONS
)
SYSTEM_PROMPT_KOR = (
    FILE_USAGE_INSTRUCTIONS_KOR
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SIMPLE_RESEARCH_INSTRUCTIONS_KOR
)


# 웹 검색 도구 정의, 실제 검색 대신 모의 결과 반환
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

    # 웹 검색 모의 결과 데이터
    search_result = """모델 컨텍스트 프로토콜(MCP)은 Anthropic이 개발한 개방형 표준 프로토콜로,
AI 모델과 도구, 데이터베이스, 기타 서비스와 같은 외부 시스템 간의 원활한 통합을 가능하게 합니다.
이는 표준화된 통신 계층 역할을 하여 AI 모델이 다양한 출처의 데이터에 일관되고 효율적인 방식으로 접근하고 활용할 수 있도록 합니다.
본질적으로 MCP는 데이터 교환을 위한 통합 언어를 제공함으로써 AI 어시스턴트를 외부 서비스에 연결하는 과정을 단순화합니다."""
    return search_result


class DeepAgentContextOffloadingGraph(BaseGraph):
    """``create_agent`` Context Offloading ReAct 에이전트 (``DeepAgentState`` + 디스크 파일 + ``web_search`` 모킹).

    외부에서는 ``g = DeepAgentContextOffloadingGraph()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4_1_MINI,
        *,
        temperature: float = 0.0,
        recursion_limit: int = _DEFAULT_RECURSION_LIMIT,
        system_prompt: str = SYSTEM_PROMPT,
        agent_files_dir: str | Path | None = None,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        # 인스턴스별 파일 저장소 (글로벌 경로 변경 없이 경로 격리)
        self._file_store = AgentFileStore(
            agent_files_dir if agent_files_dir is not None else DEFAULT_AGENT_FILES_DIR
        )
        ls_tool, read_file_tool, write_file_tool = create_file_tools(self._file_store)

        # LLM 모델 초기화
        self._llm: BaseChatModel = init_chat_model(model, temperature=temperature)
        # 파일 도구 및 웹 검색 도구 리스트 생성, 에이전트에 전달
        self._tools: list[BaseTool] = [
            ls_tool,
            read_file_tool,
            write_file_tool,
            web_search,
        ]
        self._system_prompt = system_prompt
        self._recursion_limit = recursion_limit
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def file_store(self) -> AgentFileStore:
        return self._file_store

    def _compile_graph(self) -> CompiledStateGraph:
        # 시스템 프롬프트와 상태 스키마 지정, 에이전트 생성
        agent = create_agent(
            self._llm,
            self._tools,
            system_prompt=self._system_prompt,
            state_schema=DeepAgentState,
        ).with_config(
            {"recursion_limit": self._recursion_limit}
        )  # 에이전트 반복 실행 최대 횟수 제한
        return cast(CompiledStateGraph, agent)


__all__ = [
    "DeepAgentContextOffloadingGraph",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_KOR",
    "web_search",
]
