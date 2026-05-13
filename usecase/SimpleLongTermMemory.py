"""
참고 문서:
/langgraph-v1-tutorial/PART01-LangGraph-기초/Ch01-그래프-생성하기/01-QuickStart-LangGraph-Tutorial.ipynb
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config, get_store
from langgraph.graph import START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel


# Pydantic 모델 정의
class MemoryItem(BaseModel):
    """개별 메모리 아이템"""

    key: str = Field(description="메모리 키 (예: user_name, preference, fact)")
    value: str = Field(description="메모리 값")
    category: str = Field(
        description="카테고리 (personal_info, preference, interest, relationship, fact, etc.)"
    )
    importance: int = Field(description="중요도 (1-5, 5가 가장 중요)", ge=1, le=5)
    confidence: float = Field(description="추출 신뢰도 (0.0-1.0)", ge=0.0, le=1.0)


class ExtractedMemories(BaseModel):
    """추출된 메모리 컬렉션"""

    memories: List[MemoryItem] = Field(description="추출된 메모리 아이템 리스트")
    summary: str = Field(description="대화 내용 요약")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(), description="추출 시간"
    )


# 기본 시스템 프롬프트
DEFAULT_SYSTEM_PROMPT = """You are an expert memory extraction assistant. Your task is to extract important information from user conversations and convert them into structured key-value pairs for long-term memory storage.

Extract ALL relevant information from the conversation, including:
- Personal information (name, age, location, occupation, etc.)
- Preferences and interests
- Relationships and social connections
- Important facts or events mentioned
- Opinions and beliefs
- Goals and aspirations
- Any other notable information

For each piece of information:
1. Create a concise, searchable key
2. Store the complete value
3. Categorize appropriately
4. Assess importance (1-5 scale)
5. Evaluate extraction confidence (0.0-1.0)"""


def create_memory_extractor(
    model_name: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
    system_prompt: Optional[str] = None,
) -> Runnable:
    """메모리 추출기를 생성합니다.

    Args:
        model_name: 사용할 언어 모델 식별자. 기본값 ``OPENAI_GPT_4O_MINI``.
        system_prompt: 시스템 프롬프트. None일 경우 기본 프롬프트 사용

    Returns:
        메모리 추출 체인
    """
    # Output Parser 생성
    memory_parser = PydanticOutputParser(pydantic_object=ExtractedMemories)

    # 시스템 프롬프트 설정
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    # 전체 프롬프트 템플릿 구성
    template = f"""{system_prompt}

User Input: {{input}}

{{format_instructions}}

Remember to:
- Extract multiple memory items if the conversation contains various pieces of information
- Use clear, consistent key naming conventions
- Preserve context in values when necessary
- Be comprehensive but avoid redundancy
"""

    # 프롬프트 생성
    prompt = ChatPromptTemplate.from_template(
        template,
        partial_variables={
            "format_instructions": memory_parser.get_format_instructions()
        },
    )

    # 모델 설정
    model = init_chat_model(model_name)

    # 메모리 추출 체인 생성
    memory_extractor = prompt | model | memory_parser

    return memory_extractor


class SimpleLongTermMemory(BaseGraph):
    """Store + Checkpointer 기반 장기 메모리 그래프(START → call_model).

    ``config`` 의 ``configurable`` 에 ``user_id`` 와 (체크포인터용) ``thread_id`` 가 필요합니다.

    외부에서는 ``m = SimpleLongTermMemory()`` 뒤 ``m.invoke(...)`` / ``m.stream(...)``,
    LangGraph **노드 구조**는 ``m.show_graph()`` 로 본다.
    """

    def __init__(
        self,
        model: str | LangChainChatModel = LangChainChatModel.OPENAI_GPT_4O_MINI,
        *,
        system_prompt: str | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
        load_env: bool = True,
        langsmith_project: str | None = "LangChain-V1-Tutorial",
    ) -> None:
        super().__init__(load_env=load_env, langsmith_project=langsmith_project)

        self._llm: BaseChatModel = init_chat_model(model)
        self._memory_extractor: Runnable = create_memory_extractor(
            model_name=model,
            system_prompt=system_prompt,
        )
        # 메모리 체크포인터 생성
        # 실제 프로덕션에서는 PostgresSaver 사용 권장
        self._checkpointer: BaseCheckpointSaver = (
            checkpointer if checkpointer is not None else InMemorySaver()
        )
        self._store: BaseStore = store if store is not None else InMemoryStore()
        self._graph: CompiledStateGraph = self._compile_graph()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def store(self) -> BaseStore:
        return self._store

    @property
    def checkpointer(self) -> BaseCheckpointSaver:
        return self._checkpointer

    def _call_model(self, state: MessagesState) -> dict[str, list[AnyMessage]]:
        """LLM 모델을 호출하고 사용자 메모리를 관리합니다.

        Args:
            state (MessagesState): 메시지를 포함하는 현재 상태
            config (RunnableConfig): 실행 가능 구성 (그래프 실행 시 ``get_config()``)
            store (BaseStore): 메모리 저장소 (그래프 실행 시 ``get_store()``)
        """
        config = get_config()
        store = get_store()

        # 마지막 메시지에서 user_id 추출
        configurable = config.get("configurable") or {}
        user_id = configurable["user_id"]
        namespace = ("memories", user_id)

        print(namespace)

        # 유저의 메모리 검색(의미검색)
        memories = store.search(namespace, query=str(state["messages"][-1].content))
        info = "\n".join([f"{memory.key}: {memory.value}" for memory in memories])
        system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

        # 사용자가 기억 요청 시 메모리 저장
        last_message = state["messages"][-1]
        if "remember" in str(last_message.content).lower():
            result = self._memory_extractor.invoke(
                {"input": str(state["messages"][-1].content)}
            )
            for memory in result.memories:
                print(memory)
                print("-" * 100)
                store.put(namespace, str(uuid.uuid4()), {memory.key: memory.value})

        # LLM 호출
        response = self._llm.invoke(
            [{"role": "system", "content": system_msg}] + state["messages"]
        )
        return {"messages": [response]}

    def _compile_graph(self) -> CompiledStateGraph:
        # 그래프 빌드
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", self._call_model)
        builder.add_edge(START, "call_model")

        # 그래프 컴파일
        return builder.compile(
            checkpointer=self._checkpointer,
            store=self._store,
        )


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "ExtractedMemories",
    "MemoryItem",
    "SimpleLongTermMemory",
    "create_memory_extractor",
]
