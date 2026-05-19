"""
참고 문서:
/deep_agents_from_scratch/notebooks/04-deep_agent_sub_agent_delegation_graph.ipynb
(``DeepAgentState`` + ``task`` 위임 + 연구 Sub-agent ``web_search`` 모킹 Supervisor ReAct 에이전트)
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph

from base.base_graph import BaseGraph
from deep_agents_from_scratch.deep_agent_state import DeepAgentState
from deep_agents_from_scratch.prompts import (
    SUBAGENT_USAGE_INSTRUCTIONS,
    SUBAGENT_USAGE_INSTRUCTIONS_KOR,
)
from deep_agents_from_scratch.task_tool import SubAgent, _create_task_tool
from util.chat_model_enums import LangChainChatModel

_DEFAULT_RECURSION_LIMIT = 20
# 동시 연구 작업 단위 최대 개수 제한 설정
_DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS = 3
# 연구자 에이전트 반복 횟수 최대값 설정
_DEFAULT_MAX_RESEARCHER_ITERATIONS = 3

# 연구용 Sub-agent 프롬프트 문자열 정의, 단일 웹 검색만 허용
RESEARCH_SUB_AGENT_PROMPT = """You are a researcher. Research the topic provided to you. IMPORTANT: Just make a single call to the web_search tool and use the result provided by the tool to answer the provided topic."""

# 연구용 Sub-agent 구성 정보 딕셔너리 생성, 이름/설명/프롬프트/도구 지정
RESEARCH_SUB_AGENT: SubAgent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "prompt": RESEARCH_SUB_AGENT_PROMPT,
    "tools": ["web_search"],
}


# 웹 검색 도구 모킹 함수 정의, 실제 검색 대신 고정된 결과 반환
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
        Search results from the search engine.

    Example:
        web_search("machine learning applications in healthcare")
    """
    # 웹 검색 결과 모킹 데이터 정의, 실제 검색 대신 고정된 결과 반환
    search_result = """The Model Context Protocol (MCP) is an open standard protocol developed
by Anthropic to enable seamless integration between AI models and external systems like
tools, databases, and other services. It acts as a standardized communication layer,
allowing AI models to access and utilize data from various sources in a consistent and
efficient manner. Essentially, MCP simplifies the process of connecting AI assistants
to external services by providing a unified language for data exchange. """
    return search_result




class DeepAgentSubAgentDelegationGraph(BaseGraph):
    """``create_agent`` Sub-agent 위임 Supervisor ReAct 에이전트 (``task`` + 연구 Sub-agent).

    외부에서는 ``g = DeepAgentSubAgentDelegationGraph()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        max_concurrent_research_units: int = _DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS,
        max_researcher_iterations: int = _DEFAULT_MAX_RESEARCHER_ITERATIONS,
        temperature: float = 0.0,
        recursion_limit: int = _DEFAULT_RECURSION_LIMIT,
        system_prompt: str | None = None,
        subagents: list[SubAgent] | None = None,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._max_concurrent_research_units = max_concurrent_research_units
        self._max_researcher_iterations = max_researcher_iterations
        self._subagents = subagents if subagents is not None else [RESEARCH_SUB_AGENT]

        # 위에서 초기화된 기본 LLM 사용, 모델 및 온도 설정
        self._llm: BaseChatModel = init_chat_model(model, temperature=temperature)

        # Sub-agent에 할당할 도구 리스트 정의, 웹 검색 도구 포함
        self._sub_agent_tools: list[BaseTool] = [web_search]

        # Supervisor Agent가 사용할 도구 리스트, task 도구 포함
        self._system_prompt = (
            system_prompt
            if system_prompt is not None
            else SUBAGENT_USAGE_INSTRUCTIONS.format(
                max_concurrent_research_units=max_concurrent_research_units,
                max_researcher_iterations=max_researcher_iterations,
                date=datetime.now().strftime("%a %b %-d, %Y"),
            )
        )
        # Sub-agent에게 작업 위임을 위한 task 도구 생성, 컨텍스트 격리 구현
        task_tool = _create_task_tool(
            self._sub_agent_tools, self._subagents, self._llm, DeepAgentState
        )
        # Supervisor Agent가 사용할 도구 리스트, task 도구 포함
        self._tools: list[BaseTool] = [task_tool]
        self._recursion_limit = recursion_limit
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def sub_agent_tools(self) -> list[BaseTool]:
        return list(self._sub_agent_tools)

    @property
    def subagents(self) -> list[SubAgent]:
        return list(self._subagents)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    def _compile_graph(self) -> CompiledStateGraph:
        # Supervisor Agent 생성, 시스템 프롬프트 및 상태 스키마 지정
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
    "DeepAgentSubAgentDelegationGraph",
    "RESEARCH_SUB_AGENT",
    "RESEARCH_SUB_AGENT_PROMPT",
    "SUBAGENT_USAGE_INSTRUCTIONS_KOR",
    "web_search",
]
