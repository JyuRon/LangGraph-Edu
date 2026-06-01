"""
참고 문서:
/langgraph-v1-tutorial/PART02-에이전트/Ch04-구조화된-출력/05-LangGraph-Structured-Output.ipynb
https://docs.langchain.com/oss/python/langchain/structured-output

핵심:
``create_agent`` 의 ``response_format`` 에 Pydantic / dataclass / TypedDict 스키마를 넘기면
모델에 따라 ``ProviderStrategy`` 또는 ``ToolStrategy`` 가 자동 선택되고,
결과는 ``structured_response`` 키에 반환된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel

SchemaKind = Literal["pydantic", "dataclass", "typeddict"]


class ContactInfoPydantic(BaseModel):
    """사람의 연락처 정보를 나타내는 스키마

    이름, 이메일, 전화번호를 구조화된 형태로 추출합니다.
    """

    name: str = Field(description="The name of the person")  # 이름
    email: str = Field(description="The email address of the person")  # 이메일 주소
    phone: str = Field(description="The phone number of the person")  # 전화번호


@dataclass
class ContactInfoDataclass:
    """사람의 연락처 정보를 나타내는 스키마"""

    name: str  # 이름 (The name of the person)
    email: str  # 이메일 주소 (The email address of the person)
    phone: str  # 전화번호 (The phone number of the person)


class ContactInfoTypedDict(TypedDict):
    """사람의 연락처 정보를 나타내는 스키마"""

    name: str  # 이름 (The name of the person)
    email: str  # 이메일 주소 (The email address of the person)
    phone: str  # 전화번호 (The phone number of the person)


_SCHEMA_BY_KIND: dict[
    SchemaKind,
    type[ContactInfoPydantic | ContactInfoDataclass | ContactInfoTypedDict],
] = {
    "pydantic": ContactInfoPydantic,
    "dataclass": ContactInfoDataclass,
    "typeddict": ContactInfoTypedDict,
}


class AgentStructuredOutputProviderStrategyAgent(BaseGraph):
    """``response_format`` 스키마 종류별 ProviderStrategy 자동 선택 ``create_agent`` 데모.

    외부에서는 ``g = AgentStructuredOutputProviderStrategyAgent(schema_kind="pydantic")`` 뒤
    ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        schema_kind: SchemaKind = "pydantic",
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4_1_MINI,
        *,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._schema_kind: SchemaKind = schema_kind
        self._response_format = _SCHEMA_BY_KIND[schema_kind]
        self._llm: BaseChatModel = init_chat_model(model)
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def schema_kind(self) -> SchemaKind:
        return self._schema_kind

    @property
    def response_format(
        self,
    ) -> type[ContactInfoPydantic | ContactInfoDataclass | ContactInfoTypedDict]:
        return self._response_format

    def _compile_graph(self) -> CompiledStateGraph:
        # 에이전트 생성 - response_format에 스키마 타입 전달 시 ProviderStrategy 자동 선택
        agent = create_agent(
            model=self._llm,
            tools=[],
            response_format=self._response_format,
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "AgentStructuredOutputProviderStrategyAgent",
    "ContactInfoDataclass",
    "ContactInfoPydantic",
    "ContactInfoTypedDict",
    "SchemaKind",
    "_SCHEMA_BY_KIND",
]
