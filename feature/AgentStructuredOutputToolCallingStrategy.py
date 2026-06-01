"""
참고 문서:
/langgraph-v1-tutorial/PART02-에이전트/Ch04-구조화된-출력/05-LangGraph-Structured-Output.ipynb
https://docs.langchain.com/oss/python/langchain/structured-output

핵심:
``create_agent`` 의 ``response_format`` 에 ``ToolStrategy(스키마)`` 를 명시하면
모델 네이티브 지원 여부와 관계없이 도구 호출 방식으로 구조화된 출력을 강제하고,
결과는 ``structured_response`` 키에 반환된다.
"""

from __future__ import annotations

from typing import Literal, Union, cast

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel




class ProductReview(BaseModel):
    """제품 리뷰 분석 결과를 나타내는 스키마

    긍정/부정 감정 분석과 핵심 포인트를 추출합니다.
    """

    rating: int | None = Field(
        description="The rating of the product", ge=1, le=5
    )  # 평점 (1-5)
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the review"
    )  # 감정 분석 결과
    key_points: list[str] = Field(
        description="The key points of the review. Lowercase, 1-3 words each."
    )  # 핵심 포인트


class CustomerComplaint(BaseModel):
    """고객 불만 정보를 나타내는 스키마

    문제 유형, 심각도, 설명을 구조화된 형태로 추출합니다.
    """

    issue_type: Literal["product", "service", "shipping", "billing"] = Field(
        description="The type of issue"
    )  # 문제 유형
    severity: Literal["low", "medium", "high"] = Field(
        description="The severity of the complaint"
    )  # 심각도
    description: str = Field(
        description="Brief description of the complaint"
    )  # 불만 내용


class MeetingAction(BaseModel):
    """회의에서 추출된 액션 아이템을 나타내는 스키마

    담당자, 작업 내용, 우선순위를 구조화합니다.
    """

    task: str = Field(description="The specific task to be completed")  # 작업 내용
    assignee: str = Field(description="Person responsible for the task")  # 담당자
    priority: Literal["low", "medium", "high"] = Field(
        description="Priority level"
    )  # 우선순위


# Union 스키마 별칭 — ``ToolStrategy(FeedbackUnion)`` 형태로 사용
FeedbackUnion = Union[ProductReview, CustomerComplaint]

# [단일 스키마] 에이전트 생성 - ToolStrategy 명시적 사용
_TOOL_STRATEGY_SINGLE = ToolStrategy(ProductReview)

# [Union 스키마] 에이전트 생성 - Union 타입으로 여러 스키마 지원
_TOOL_STRATEGY_UNION = ToolStrategy(FeedbackUnion)

# [커스텀 도구 메시지] 에이전트 생성 - 커스텀 도구 메시지 설정
_TOOL_STRATEGY_MEETING_ACTION = ToolStrategy(
    schema=MeetingAction,
    tool_message_content="Action item captured and added to meeting notes!",
)

_SCHEMA_BY_KIND = {
    "product_review": ProductReview,
    "feedback_union": FeedbackUnion,
    "meeting_action": MeetingAction,
}

_RESPONSE_FORMAT_BY_KIND = {
    "product_review": _TOOL_STRATEGY_SINGLE,
    "feedback_union": _TOOL_STRATEGY_UNION,
    "meeting_action": _TOOL_STRATEGY_MEETING_ACTION,
}


SchemaKind = Literal["product_review", "feedback_union", "meeting_action"]


class AgentStructuredOutputToolCallingStrategyAgent(BaseGraph):
    """``ToolStrategy`` 로 구조화 출력을 강제하는 ``create_agent`` 데모.

    외부에서는 ``g = AgentStructuredOutputToolCallingStrategyAgent(schema_kind="product_review")`` 뒤
    ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        schema_kind: SchemaKind = "product_review",
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4_1_MINI,
        *,
        load_env: bool = True,
        langsmith_project: str | None = None,
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._schema_kind: SchemaKind = schema_kind
        self._schema = _SCHEMA_BY_KIND[schema_kind]
        self._llm: BaseChatModel = init_chat_model(model)
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def schema_kind(self) -> SchemaKind:
        return self._schema_kind

    @property
    def schema(self):
        return self._schema

    def _compile_graph(self) -> CompiledStateGraph:
        agent = create_agent(
            model=self._llm,
            tools=[],
            response_format=_RESPONSE_FORMAT_BY_KIND[self._schema_kind],
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "AgentStructuredOutputToolCallingStrategyAgent",
    "CustomerComplaint",
    "FeedbackUnion",
    "MeetingAction",
    "ProductReview",
    "SchemaKind",
    "_RESPONSE_FORMAT_BY_KIND",
    "_SCHEMA_BY_KIND",
    "_TOOL_STRATEGY_MEETING_ACTION",
    "_TOOL_STRATEGY_SINGLE",
    "_TOOL_STRATEGY_UNION",
]
