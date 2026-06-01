"""
참고 문서:
PART02-에이전트/Ch02-에이전트/01-LangGraph-Agents.ipynb

핵심:
``create_agent`` 의 ``response_format`` 에 Pydantic 스키마를 넘겨
이메일에서 발신자·주소를 구조화된 ``structured_response`` 로 받는다.
"""

from __future__ import annotations

from typing import cast

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel

_SYSTEM_PROMPT = "Extract useful information from the email."


class ResponseFormat(BaseModel):
    """에이전트 응답 스키마"""

    email_sender: str = Field(description="이메일 발신자")
    email_sender_address: str = Field(description="발신자 주소")


class ResponseFormatAgent(BaseGraph):
    """``response_format`` 으로 이메일 발신자 정보를 구조화 출력하는 ``create_agent`` 데모.

    외부에서는 ``g = ResponseFormatAgent()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._llm: BaseChatModel = init_chat_model(model)
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    def _compile_graph(self) -> CompiledStateGraph:
        agent = create_agent(
            model=self._llm,
            system_prompt=_SYSTEM_PROMPT,
            tools=[],
            response_format=ResponseFormat,
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "ResponseFormat",
    "ResponseFormatAgent",
    "_SYSTEM_PROMPT",
]
