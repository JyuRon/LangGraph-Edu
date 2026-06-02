"""
참고 문서:
/langgraph-v1-tutorial/PART02-에이전트/Ch05-Human-in-the-Loop/02-LangGraph-Human-In-The-Loop.ipynb

핵심:
``HumanInTheLoopMiddleware`` 로 위험한 도구 호출 전 사람 승인을 받고,
``InMemorySaver`` 체크포인터로 interrupt 후 ``Command(resume=...)`` 재개가 가능하다.

edit 결정 시 ``EditAwareHumanInTheLoopMiddleware`` 로 "사람이 수정한 최종 행동"임을
모델에 알려, 원본 요청으로 되돌아가 다시 interrupt 되는 루프를 막는다.
(구현: ``feature.EditAwareHumanInTheLoopMiddleware``)
"""

from __future__ import annotations

import os
from typing import cast

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from base.base_graph import BaseGraph
from feature.EditAwareHumanInTheLoopMiddleware import (
    DEFAULT_EDIT_NOTICE,
    EditAwareHumanInTheLoopMiddleware,
)
from util.chat_model_enums import LangChainChatModel


# 도구 정의
@tool
def write_file(filename: str, content: str) -> str:
    """파일에 내용을 작성합니다."""
    with open(filename, "w") as f:
        f.write(content)
    return f"File {filename} written successfully"


@tool
def read_file(filename: str) -> str:
    """파일에서 내용을 읽어옵니다."""
    try:
        with open(filename, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"File {filename} not found"


@tool
def delete_file(filename: str) -> str:
    """파일을 삭제합니다."""
    try:
        os.remove(filename)
        return f"File {filename} deleted successfully"
    except FileNotFoundError:
        return f"File {filename} not found"


class HumanIntheLoopMiddlewareBasic(BaseGraph):
    """``HumanInTheLoopMiddleware`` 로 파일 도구 실행 전 사람 승인을 받는 ``create_agent`` 데모.

    ``write_file``·``delete_file`` 은 interrupt, ``read_file`` 은 자동 실행.
    체크포인터가 필요하므로 ``InMemorySaver`` 를 사용한다.
    외부에서는 ``g = HumanIntheLoopMiddlewareBasic()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        load_env: bool = True,
        langsmith_project: str | None = None,
        description_prefix: str = "Tool execution pending approval",
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._tools: list[BaseTool] = [write_file, read_file, delete_file]
        self._llm: BaseChatModel = init_chat_model(model)
        self._description_prefix = description_prefix
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    def _compile_graph(self) -> CompiledStateGraph:
        # HITL 미들웨어와 함께 에이전트 생성
        agent = create_agent(
            model=self._llm,
            tools=self._tools,
            middleware=[
                # 기본 HumanInTheLoopMiddleware 대신, edit 후 재시도 루프를 막는
                # EditAwareHumanInTheLoopMiddleware 를 사용한다.
                EditAwareHumanInTheLoopMiddleware(
                    interrupt_on={
                        "write_file": True,  # 모든 결정(approve, edit, reject) 허용
                        "delete_file": True,  # 모든 결정 허용
                        "read_file": False,  # 안전한 작업, 승인 불필요
                    },
                    description_prefix=self._description_prefix,
                ),
            ],
            checkpointer=InMemorySaver(),  # 체크포인터 필수
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "DEFAULT_EDIT_NOTICE",
    "EditAwareHumanInTheLoopMiddleware",
    "HumanIntheLoopMiddlewareBasic",
    "delete_file",
    "read_file",
    "write_file",
]
