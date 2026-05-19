"""Task delegation tools for context isolation through sub-agents.

This module provides the core infrastructure for creating and managing sub-agents
with isolated contexts. Sub-agents prevent context clash by operating with clean
context windows containing only their specific task description.

Files are tracked through ``state["files"]`` (mirrored from disk by ``file_tools``
and ``tavily_search``). When a sub-agent updates files, LangGraph's
``file_reducer`` automatically merges its ``state["files"]`` back into the
parent's state, so the parent always sees the latest file inventory without any
disk snapshot comparison inside this module.
"""

from __future__ import annotations

from typing import Annotated, NotRequired

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import TypedDict

from deep_agents_from_scratch.deep_agent_state import DeepAgentState
from deep_agents_from_scratch.prompts import TASK_DESCRIPTION_PREFIX


class SubAgent(TypedDict):
    """Configuration for a specialized sub-agent."""

    """
        - `name`: 에이전트 식별자(`main agent`에서 호출 시 사용)
        - `description`: 역할 설명(`main agent`에서 호출 시 사용)
        - `prompt`: 전용 시스템 프롬프트(`sub-agent` 작업 지시)
        - `tools`: 사용 가능한 도구 목록(`sub-agent` 작업 도구)
    """

    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]


def _create_task_tool(tools, subagents: list[SubAgent], model, state_schema):
    """Create a task delegation tool that enables context isolation through sub-agents.

    This function implements the core pattern for spawning specialized sub-agents with
    isolated contexts, preventing context clash and confusion in complex multi-step tasks.

    Files are tracked entirely via ``state["files"]``. The sub-agent inherits the
    parent's current ``files`` mirror, updates it through ``write_file`` /
    ``tavily_search``, and returns it; LangGraph's ``file_reducer`` then merges
    those updates back into the parent state. The parent receives the list of
    files the sub-agent touched as part of the ``ToolMessage`` so it knows which
    files to ``read_file()`` (in offset/limit chunks) when forming its answer.

    Args:
        tools: List of available tools that can be assigned to sub-agents
        subagents: List of specialized sub-agent configurations
        model: The language model to use for all agents
        state_schema: The state schema (typically DeepAgentState)

    Returns:
        A 'task' tool that can delegate work to specialized sub-agents
    """
    # Sub-agent 레지스트리 딕셔너리 생성, 이름을 키로 하여 에이전트 인스턴스 저장
    agents = {}

    # 도구 이름별 매핑 딕셔너리 생성, Sub-agent별 도구 할당에 활용
    tools_by_name = {}
    for tool_ in tools:
        if not isinstance(tool_, BaseTool):
            tool_ = tool(tool_)
        tools_by_name[tool_.name] = tool_

    # Sub-agent 구성 정보 기반으로 특화 에이전트 생성 및 레지스트리에 등록
    for _agent in subagents:
        if "tools" in _agent:
            # Sub-agent에 지정된 도구만 할당
            _tools = [tools_by_name[t] for t in _agent["tools"]]
        else:
            # 도구 미지정 시 전체 도구 할당
            _tools = tools
        agents[_agent["name"]] = create_agent(  # updated 1.0
            model,
            system_prompt=_agent["prompt"],
            tools=_tools,
            state_schema=state_schema,
        )

    # 사용 가능한 Sub-agent 목록을 도구 설명에 활용하기 위한 문자열 리스트 생성
    other_agents_string = [
        f"- {_agent['name']}: {_agent['description']}" for _agent in subagents
    ]

    @tool(description=TASK_DESCRIPTION_PREFIX.format(other_agents=other_agents_string))
    def task(
        description: str,
        subagent_type: str,
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Delegate a task to a specialized sub-agent with **strict isolation**.

        Isolation guarantees:
        - Sub-agent never sees the parent's message history (fresh ``HumanMessage`` only).
        - Sub-agent never sees the parent's ``state["files"]`` either — it starts from an
          empty file map, so its job is purely to **produce new files** (e.g. via
          ``tavily_search``).
        - Sub-agent's natural-language answer is **discarded**; the parent only receives
          the list of files the sub-agent produced. The parent must then call
          ``read_file()`` in offset/limit chunks to actually decide.

        This is the "sub-agent makes files, parent reads & judges" pattern: the parent
        agent stays in control of all reasoning, while the sub-agent is a pure
        file-producer that doesn't leak its own conclusions back into the parent's
        context window.
        """
        # 요청된 Sub-agent 타입이 레지스트리에 존재하는지 검증, 미존재 시 에러 반환
        if subagent_type not in agents:
            return f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in agents]}"

        # 요청된 Sub-agent 인스턴스 가져오기
        sub_agent = agents[subagent_type]

        # 격리된 새 컨텍스트: 작업 설명만 포함. 부모의 messages/files는 일절 전달하지 않는다.
        # (todos는 LLM 가시성을 위해 그대로 넘겨주지만, sub-agent가 갱신하지 않는 한 부모 todo가 그대로)
        new_state = {
            "messages": [HumanMessage(content=description)],
            "todos": state.get("todos", []),
            "files": {},
        }

        # 격리된 환경에서 Sub-agent 실행 및 결과 획득
        result = sub_agent.invoke(new_state)

        # Sub-agent가 새로 만든 파일들 — 부모 state["files"]에 file_reducer로 자동 병합된다.
        produced_files: dict[str, str] = result.get("files", {}) or {}
        produced_names = sorted(produced_files.keys())

        # ⚠️ Sub-agent의 자연어 답변(result["messages"][-1].content)은 의도적으로 폐기한다.
        # 부모가 받는 것은 오직 "어떤 파일이 만들어졌는가" 리스트뿐. 추론은 부모가 직접 한다.
        if produced_names:
            artifacts_block = "\n".join(f"  - {name}" for name in produced_names)
            body = (
                f"Sub-agent '{subagent_type}' completed and produced the following file(s):\n"
                f"{artifacts_block}\n\n"
                f"These files are now in state[\"files\"]. You MUST call read_file() on the "
                f"relevant files in offset/limit chunks (e.g. offset=0, limit=400 → next chunk "
                f"only if needed) and base your reasoning entirely on the file contents. "
                f"The sub-agent's own analysis is intentionally not shared with you."
            )
        else:
            body = (
                f"Sub-agent '{subagent_type}' completed without producing any files. "
                f"There are no new artifacts to read; consider refining the task description "
                f"or trying a different sub-agent."
            )

        # 작업 결과를 Command 객체로 래핑하여 부모 에이전트에 반환
        # - files: file_reducer가 부모 state["files"]와 자동 병합
        # - messages: ToolMessage 한 건 추가 (sub-agent 답변은 포함되지 않음)
        update: dict = {
            "messages": [ToolMessage(body, tool_call_id=tool_call_id)],
        }
        if produced_files:
            update["files"] = produced_files
        return Command(update=update)

    return task
