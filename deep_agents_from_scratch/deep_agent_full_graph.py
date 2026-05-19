"""
참고 문서:
/deep_agents_from_scratch/notebooks/05-deep_agent_full_graph.ipynb
(``DeepAgentState`` + TODO + 파일 시스템 + Tavily ``tavily_search`` + ``think_tool`` +
 Research Sub-agent ``task`` 위임 Supervisor ReAct 에이전트)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
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
    RESEARCHER_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS_KOR,
    SUBAGENT_USAGE_INSTRUCTIONS,
    SUBAGENT_USAGE_INSTRUCTIONS_KOR,
    TODO_USAGE_INSTRUCTIONS,
    TODO_USAGE_INSTRUCTIONS_KOR,
)
from deep_agents_from_scratch.research_tools import (
    create_tavily_search_tool,
    get_current_time,
    think_tool,
    tavily_search,
)
from deep_agents_from_scratch.task_tool import SubAgent, _create_task_tool
from deep_agents_from_scratch.todo_tools import (
    DEFAULT_TODOS_FILE,
    TodoFileStore,
    create_todo_tools,
)
from util.chat_model_enums import LangChainChatModel

_DEFAULT_RECURSION_LIMIT = 100
# 리서치 서브에이전트 동시 실행 최대 개수 및 반복 횟수 제한 설정
_DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS = 3
_DEFAULT_MAX_RESEARCHER_ITERATIONS = 3

# 리서치 서브에이전트 기본 설명 (이름·도구는 ``_make_research_sub_agent`` 참고)
RESEARCH_SUB_AGENT_DESCRIPTION = (
    "Delegate research to the sub-agent researcher. "
    "Only give this researcher one topic at a time."
)


def _make_research_sub_agent(
    *,
    researcher_instructions: str = RESEARCHER_INSTRUCTIONS,
) -> SubAgent:
    """리서치 서브에이전트 구성 정보 (이름, 설명, 프롬프트, 도구 목록)."""
    return {
        "name": "research-agent",
        "description": RESEARCH_SUB_AGENT_DESCRIPTION,
        "prompt": researcher_instructions.format(date=get_current_time()),
        "tools": ["tavily_search", "think_tool"],
    }


def _build_subagent_instructions(
    max_concurrent_research_units: int,
    max_researcher_iterations: int,
    *,
    template: str = SUBAGENT_USAGE_INSTRUCTIONS,
) -> str:
    # 서브에이전트 프롬프트 생성 (최대 동시 리서치 개수, 반복 횟수, 날짜 포함)
    return template.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
        date=datetime.now().strftime("%a %b %-d, %Y"),
    )


def _build_full_system_prompt(
    subagent_instructions: str,
    *,
    todo_instructions: str = TODO_USAGE_INSTRUCTIONS,
    file_instructions: str = FILE_USAGE_INSTRUCTIONS,
) -> str:
    # TODO 관리, 파일 시스템 사용, 서브에이전트 위임 관련 프롬프트를 하나의 문자열로 통합
    return (
        "# TODO MANAGEMENT\n"
        + todo_instructions
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + "# FILE SYSTEM USAGE\n"
        + file_instructions
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + "# SUB-AGENT DELEGATION\n"
        + subagent_instructions
    )


SYSTEM_PROMPT = _build_full_system_prompt(
    _build_subagent_instructions(
        _DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS,
        _DEFAULT_MAX_RESEARCHER_ITERATIONS,
    )
)
SYSTEM_PROMPT_KOR = _build_full_system_prompt(
    _build_subagent_instructions(
        _DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS,
        _DEFAULT_MAX_RESEARCHER_ITERATIONS,
        template=SUBAGENT_USAGE_INSTRUCTIONS_KOR,
    ),
    todo_instructions=TODO_USAGE_INSTRUCTIONS_KOR,
    file_instructions=FILE_USAGE_INSTRUCTIONS_KOR,
)

RESEARCH_SUB_AGENT = _make_research_sub_agent()


class DeepAgentFullGraph(BaseGraph):
    """Full deep agent: TODO, files, Tavily search, and sub-agent ``task`` delegation."""

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        max_concurrent_research_units: int = _DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS,
        max_researcher_iterations: int = _DEFAULT_MAX_RESEARCHER_ITERATIONS,
        temperature: float = 0.0,
        recursion_limit: int = _DEFAULT_RECURSION_LIMIT,
        system_prompt: str | None = None,
        researcher_instructions: str = RESEARCHER_INSTRUCTIONS,
        subagent_instructions_template: str = SUBAGENT_USAGE_INSTRUCTIONS,
        todo_instructions: str = TODO_USAGE_INSTRUCTIONS,
        file_instructions: str = FILE_USAGE_INSTRUCTIONS,
        subagents: list[SubAgent] | None = None,
        agent_files_dir: str | Path | None = None,
        todos_file: str | Path | None = None,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._max_concurrent_research_units = max_concurrent_research_units
        self._max_researcher_iterations = max_researcher_iterations

        self._file_store = AgentFileStore(
            agent_files_dir if agent_files_dir is not None else DEFAULT_AGENT_FILES_DIR
        )
        self._todo_store = TodoFileStore(
            todos_file if todos_file is not None else DEFAULT_TODOS_FILE
        )
        ls_tool, read_file_tool, write_file_tool = create_file_tools(self._file_store)
        write_todos_tool, read_todos_tool = create_todo_tools(self._todo_store)
        tavily_search_tool = create_tavily_search_tool(self._file_store)

        self._llm: BaseChatModel = init_chat_model(model, temperature=temperature)
        self._sub_agent_tools: list[BaseTool] = [tavily_search_tool, think_tool]
        self._built_in_tools: list[BaseTool] = [
            ls_tool,
            read_file_tool,
            write_file_tool,
            write_todos_tool,
            read_todos_tool,
            think_tool,
        ]

        self._subagents = (
            subagents
            if subagents is not None
            else [_make_research_sub_agent(researcher_instructions=researcher_instructions)]
        )

        subagent_instructions = _build_subagent_instructions(
            max_concurrent_research_units,
            max_researcher_iterations,
            template=subagent_instructions_template,
        )
        self._system_prompt = (
            system_prompt
            if system_prompt is not None
            else _build_full_system_prompt(
                subagent_instructions,
                todo_instructions=todo_instructions,
                file_instructions=file_instructions,
            )
        )

        task_tool = _create_task_tool(
            self._sub_agent_tools,
            self._subagents,
            self._llm,
            DeepAgentState,
        )
        delegation_tools = [task_tool]
        self._tools: list[BaseTool] = (
            self._sub_agent_tools + self._built_in_tools + delegation_tools
        )

        self._recursion_limit = recursion_limit
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def sub_agent_tools(self) -> list[BaseTool]:
        return list(self._sub_agent_tools)

    @property
    def built_in_tools(self) -> list[BaseTool]:
        return list(self._built_in_tools)

    @property
    def subagents(self) -> list[SubAgent]:
        return list(self._subagents)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    @property
    def file_store(self) -> AgentFileStore:
        return self._file_store

    @property
    def todo_store(self) -> TodoFileStore:
        return self._todo_store

    def _compile_graph(self) -> CompiledStateGraph:
        agent = create_agent(
            self._llm,
            self._tools,
            system_prompt=self._system_prompt,
            state_schema=DeepAgentState,
        ).with_config({"recursion_limit": self._recursion_limit})
        return cast(CompiledStateGraph, agent)


__all__ = [
    "DeepAgentFullGraph",
    "RESEARCH_SUB_AGENT",
    "RESEARCH_SUB_AGENT_DESCRIPTION",
    "RESEARCHER_INSTRUCTIONS_KOR",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_KOR",
    "_build_full_system_prompt",
    "_build_subagent_instructions",
    "_make_research_sub_agent",
    "get_current_time",
    "think_tool",
    "tavily_search",
]
