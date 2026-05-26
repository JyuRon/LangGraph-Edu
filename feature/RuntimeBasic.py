"""
참고 문서:
note/Runtime_ToolRuntime.md

핵심:
``Runtime``(미들웨어)과 ``ToolRuntime``(도구)로 ``context``·``store``에 접근하고,
``@dynamic_prompt``·``@before_model`` 미들웨어로 프롬프트·사용량을 제어한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model, dynamic_prompt
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel

_FREE_TIER_USAGE_LIMIT = 10


# 사용자 정보를 담는 Context 스키마
@dataclass
class UserContext:
    user_id: str
    user_name: str
    user_tier: str  # "free", "premium", "enterprise"
    language: str  # "ko", "en"


# 데이터베이스 검색 도구
@tool
def search_database(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """Search the database. Access level depends on user tier."""
    user_tier = runtime.context.user_tier

    # 사용자 등급에 따라 다른 결과 제공
    if user_tier == "enterprise":
        return f"Full database search results for: {query} (Enterprise access)"
    elif user_tier == "premium":
        return f"Premium search results for: {query}"
    else:
        return f"Basic search results for: {query} (Limited to 10 results)"


# 사용자 검색 기록을 가져오는 도구
@tool
def get_user_history(runtime: ToolRuntime[UserContext]) -> str:
    """Get user's search history from store."""
    user_id = runtime.context.user_id

    if runtime.store:
        if history := runtime.store.get(("history",), user_id):
            return f"Recent searches: {history.value['searches']}"

    return "No search history found"


# 검색 기록을 저장하는 도구
@tool
def save_search(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """Save search query to user history."""
    user_id = runtime.context.user_id

    if runtime.store:
        # 기존 히스토리 가져오기
        existing = runtime.store.get(("history",), user_id)
        searches = existing.value["searches"] if existing else []

        # 새 검색어 추가 (최근 5개만 유지)
        searches.append(query)
        runtime.store.put(("history",), user_id, {"searches": searches[-5:]})

        return f"Saved search: {query}"

    return "Store not available"


# 동적 프롬프트 - 사용자 언어에 따라 변경
@dynamic_prompt
def multilingual_prompt(request: ModelRequest) -> str:
    ctx = request.runtime.context
    if ctx is None:
        return "You are a helpful assistant."

    user_name = ctx.user_name
    language = ctx.language
    user_tier = ctx.user_tier

    if language == "ko":
        prompt = f"당신은 도움이 되는 어시스턴트입니다. 사용자를 '{user_name}'님으로 호칭하세요."
        if user_tier == "enterprise":
            prompt += (
                " 이 사용자는 엔터프라이즈 회원이므로 모든 기능에 액세스할 수 있습니다."
            )
    else:
        prompt = f"You are a helpful assistant. Address the user as {user_name}."
        if user_tier == "enterprise":
            prompt += " This is an enterprise user with full access."

    return prompt


# 사용량 추적 미들웨어
# (조기 종료 권한 부여)
@before_model(can_jump_to=["end"]) 
def track_usage(state: AgentState, runtime: Runtime[UserContext]) -> dict[str, Any] | None:
    """Track API usage for billing.

    무료 등급 한도 초과 시 ``jump_to: \"end\"`` 로 모델 호출 없이 에이전트를 종료한다.
    (``ModelCallLimitMiddleware`` 와 동일한 ``before_model`` 점프 패턴)
    """
    user_id = runtime.context.user_id
    user_tier = runtime.context.user_tier

    print(f"[Usage Tracker] User: {user_id}, Tier: {user_tier}")

    # 무료 사용자의 경우 사용량 제한 확인
    if user_tier == "free":
        if runtime.store:
            usage = runtime.store.get(("usage",), user_id)
            count = usage.value["count"] if usage else 0

            if count >= _FREE_TIER_USAGE_LIMIT:
                print("[Usage Tracker] Free tier limit reached!")
                # @before_model(can_jump_to=["end"]) — 모델·도구 호출 없이 종료
                return {
                    "jump_to": "end",
                    "messages": [
                        AIMessage(
                            content=(
                                f"무료 플랜 모델 호출 한도({_FREE_TIER_USAGE_LIMIT}회)에 "
                                f"도달했습니다. (현재 사용량: {count}회)"
                            )
                        ),
                    ],
                }

            # 사용량 업데이트
            runtime.store.put(("usage",), user_id, {"count": count + 1})

    return None


def _make_seeded_store() -> InMemoryStore:
    """Store 생성 및 초기 데이터 설정"""
    store = InMemoryStore()
    store.put(
        ("history",), "user_001", {"searches": ["Python tutorial", "LangChain guide"]}
    )
    store.put(("usage",), "user_002", {"count": 5})
    # 한도 초과 시 jump_to end 데모용 (track_usage 가 모델 호출 전에 종료)
    store.put(("usage",), "user_limit", {"count": _FREE_TIER_USAGE_LIMIT})
    return store


class RuntimeBasicAgent(BaseGraph):
    """``Runtime``·``ToolRuntime``·``store``·미들웨어를 함께 쓰는 ``create_agent`` 데모.

    외부에서는 ``g = RuntimeBasicAgent()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
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
        # 에이전트 생성
        agent = create_agent(
            model=self._llm,
            tools=[search_database, get_user_history, save_search],
            middleware=[
                cast(AgentMiddleware[Any, UserContext], multilingual_prompt),
                cast(AgentMiddleware[AgentState, UserContext], track_usage),
            ],
            context_schema=UserContext,
            store=_make_seeded_store(),
        )
        return cast(CompiledStateGraph, agent)


__all__ = [
    "FREE_TIER_USAGE_LIMIT",
    "RuntimeBasicAgent",
    "UserContext",
    "get_user_history",
    "multilingual_prompt",
    "save_search",
    "search_database",
    "track_usage",
]

FREE_TIER_USAGE_LIMIT = _FREE_TIER_USAGE_LIMIT
