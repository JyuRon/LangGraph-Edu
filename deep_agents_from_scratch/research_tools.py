"""Research Tools.

This module provides search and content processing utilities for the research agent,
including web search capabilities and content summarization tools.
"""

from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing_extensions import Annotated, Literal

from .prompts import SUMMARIZE_WEB_SEARCH
from .deep_agent_state import DeepAgentState


def get_current_time() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%b %-d, %Y %H:%M:%S (%A)")


tavily_client = TavilyClient()


def run_web_search(
    search_query: str,
    max_results: int = 2,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> dict:
    """Perform search using Tavily API for a single query.

    Args:
        search_query: Search query to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content

    Returns:
        Search results dictionary
    """
    result = tavily_client.search(
        search_query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    return result


# 요약을 위한 LLM 초기화
summarization_model = init_chat_model(
    model="anthropic:claude-haiku-4-5", temperature=0.0
)


class Summary(BaseModel):
    """Schema for webpage content summarization."""

    filename: str = Field(description="Name of the file to store.")
    summary: str = Field(description="Key learnings from the webpage.")


# tavily search -> List[raw_content] -> List[Summary]
def summarize_webpage_contents(webpage_contents: list[str]) -> list[Summary]:
    """Summarize multiple webpage contents in parallel using batch processing.

    LangChain의 batch() 메서드를 활용하여 여러 웹페이지 콘텐츠를 병렬로 요약

    Args:
        webpage_contents: List of raw webpage contents to summarize

    Returns:
        List of Summary objects with filename and summary for each content
    """
    if not webpage_contents:
        return []

    # 구조화된 출력 모델 설정
    structured_model = summarization_model.with_structured_output(Summary)

    # 각 콘텐츠에 대한 메시지 리스트 생성 (batch 입력 형식)
    batch_inputs = [
        [
            HumanMessage(
                content=SUMMARIZE_WEB_SEARCH.format(
                    webpage_content=content, date=get_current_time()
                )
            )
        ]
        for content in webpage_contents
    ]

    try:
        # batch() 메서드로 병렬 처리 실행
        summaries = structured_model.batch(batch_inputs)
        return summaries

    except Exception as e:
        # 실패시 기본 요약 객체 리스트 반환
        print(f"Batch processing failed: {e}, falling back to sequential processing")
        return [
            Summary(
                filename=f"search_result_{i}.md",
                summary=(content[:1000] + "..." if len(content) > 1000 else content),
            )
            for i, content in enumerate(webpage_contents)
        ]


def process_search_results(results: dict) -> list[dict]:
    """Process search results by summarizing content in parallel using batch processing.

    LangChain의 batch() 메서드를 활용하여 여러 검색 결과를 병렬로 요약합니다.
    순차 처리 대비 처리 시간을 대폭 단축할 수 있습니다.

    Args:
        results: Tavily search results dictionary

    Returns:
        List of processed results with summaries
    """
    search_results = results.get("results", [])

    if not search_results:
        return []

    # 모든 raw_content를 리스트로 추출 (batch 입력용)
    raw_contents = [result.get("raw_content", "") for result in search_results]

    # batch 방식으로 모든 콘텐츠를 병렬 요약 처리 (summarize_webpage_contents 함수 활용)
    summary_objects = summarize_webpage_contents(raw_contents)

    # 요약 결과와 원본 검색 결과를 결합하여 최종 결과 생성
    processed_results = []
    for result, summary_obj in zip(search_results, summary_objects):
        processed_results.append(
            {
                "url": result["url"],
                "title": result["title"],
                "summary": summary_obj.summary,
                "filename": summary_obj.filename,
                "raw_content": result.get("raw_content", ""),
            }
        )

    return processed_results


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_results: Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> Command:
    """웹 검색을 수행하고 상세한 결과를 파일에 저장하면서 최소한의 컨텍스트만 반환합니다.

    웹 검색을 수행하고 전체 콘텐츠를 파일에 저장하여 컨텍스트 오프로딩을 수행합니다.
    에이전트가 다음 단계를 결정하는 데 도움이 되는 필수 정보만 반환합니다.

    Args:
        query: 실행할 검색 쿼리
        state: 파일 저장을 위한 주입된 에이전트 상태
        tool_call_id: 주입된 도구 호출 식별자
        max_results: 반환할 최대 결과 수 (기본값: 1)
        topic: 토픽 필터 - 'general', 'news', 또는 'finance' (기본값: 'general')

    Returns:
        전체 결과를 파일에 저장하고 최소한의 요약을 제공하는 Command
    """
    # 검색 실행
    search_results = run_web_search(
        query,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )

    # 결과 처리 및 요약
    processed_results = process_search_results(search_results)

    # 각 결과를 파일에 저장하고 요약 준비
    files = state.get("files", {})
    saved_files = []
    summaries = []

    for i, result in enumerate(processed_results):
        # 요약에서 AI가 생성한 파일명 사용
        filename = result["filename"]

        # 전체 상세 정보를 포함한 파일 콘텐츠 생성
        file_content = f"""
            # Search Result: {result['title']}

            **URL:** {result['url']}
            **Query:** {query}
            **Date:** {get_current_time()}

            ## Summary
            {result['summary']}

            ## Raw Content
            {result['raw_content'] if result['raw_content'] else 'No raw content available'}
        """

        files[filename] = file_content
        saved_files.append(filename)
        summaries.append(f"- {filename}: {result['summary']}...")

    # 도구 메시지를 위한 최소한의 요약 생성 - 수집된 내용에 집중
    # chr(10) = \n
    summary_text = f"""
        🔍 Found {len(processed_results)} result(s) for '{query}':

        {chr(10).join(summaries)}

        Files: {', '.join(saved_files)}
        💡 Use read_file() to access full details when needed.
    """

    return Command(
        update={
            "files": files,
            "messages": [ToolMessage(summary_text, tool_call_id=tool_call_id)],
        }
    )


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?
    - How complex is the question: Have I reached the number of search limits?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"
