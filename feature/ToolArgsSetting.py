"""
참고 문서:
/langgraph-v1-tutorial/PART01-LangGraph-기초/Ch01-그래프-생성하기/01-QuickStart-LangGraph-Tutorial.ipynb
/langgraph-v1-tutorial/PART02-에이전트/Ch02-에이전트/02-LangGraph-Tools.ipynb
"""

from __future__ import annotations

from typing import Annotated, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool, tool
from langchain_tavily import TavilySearch
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel
from deep_agents_from_scratch.file_tools import read_file


@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"


@tool("web_search")  # 커스텀 이름
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"



@tool(
    "calculator",
    description="Performs arithmetic calculations. Use this for any math problems.",
)
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))


class WeatherInput(BaseModel):
    """Input for weather queries."""

    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius", description="Temperature unit preference"
    )
    include_forecast: bool = Field(default=False, description="Include 5-day forecast")


@tool(args_schema=WeatherInput)
def get_weather(
    location: str, units: str = "celsius", include_forecast: bool = False
) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"현재 {location} 지역의 날씨는 {temp} {units[0].upper()} 도"
    if include_forecast:
        result += "\n다음 5일 날씨: 맑음"
    return result






class ToolNodeState(TypedDict):
    """도구 호출 그래프 상태.

    messages: 대화 메시지 리스트
    - add_messages 리듀서로 새 메시지가 누적됨 (덮어쓰기 아님)
    """

    messages: Annotated[list[AnyMessage], add_messages]


class ToolArgsSetting(BaseGraph):
    """챗봇 + ToolNode 루프(START → chatbot ⇄ tools).

    외부에서는 ``g = ToolArgsSetting()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
    LangGraph **노드 구조**는 ``g.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        tavily_max_results: int = 2,
        load_env: bool = True,
        langsmith_project: str | None = "LangChain-V1-Tutorial",
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._search_tool = TavilySearch(max_results=tavily_max_results)
        self._tools: list[BaseTool] = [self._search_tool, search_database, search, calc, get_weather, read_file]
        self._llm: BaseChatModel = init_chat_model(model)
        self._llm_with_tools = self._llm.bind_tools(self._tools)
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    def _chatbot(self, state: ToolNodeState) -> dict[str, list[AnyMessage]]:
        response = self._llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def _compile_graph(self) -> CompiledStateGraph:
        builder = StateGraph(ToolNodeState)
        builder.add_node("chatbot", self._chatbot)
        builder.add_node("tools", ToolNode(tools=self._tools))

        builder.add_conditional_edges("chatbot", tools_condition)
        builder.add_edge("tools", "chatbot")
        builder.add_edge(START, "chatbot")

        return builder.compile()


__all__ = [
    "ToolNodeState",
    "ToolArgsSetting",
    "WeatherInput",
    "calc",
    "get_weather",
    "search",
    "search_database",
]
