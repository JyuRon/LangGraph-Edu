"""
참고 문서:
/langgraph-v1-tutorial/PART01-LangGraph-기초/Ch01-그래프-생성하기/02-QuickStart-LangGraph-Graph-API.ipynb
(Send · 동적 병렬 라우팅 — 농담 생성 예제)
"""

from __future__ import annotations

import random
from typing import Annotated, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from base.base_graph import BaseGraph
from util.chat_model_enums import LangChainChatModel

# 목표 농담 개수 (라우팅·출력에서 동일 기준으로 사용)
TARGET_JOKE_COUNT = 3


# State 정의
class JokeGeneratorState(TypedDict):
    """농담 생성 상태"""

    jokes: Annotated[list[AnyMessage], add_messages]
    current_subject: str
    attempt_count: int


class SingleJokeState(TypedDict):
    """개별 농담 생성 상태"""

    subject: str
    joke_number: int


# ChatGPT 모델 초기화
class SendParallel(BaseGraph):
    """``Send`` 로 부족한 농담만큼 ``generate_joke`` 를 동적 병렬 호출하는 예제 그래프.

    외부에서는 ``g = SendParallel()`` 뒤 ``g.invoke(...)`` / ``g.stream(...)``,
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

    # node
    def _initialize_state(self, state: JokeGeneratorState) -> dict:
        """초기 상태 설정"""
        subjects = ["프로그래머", "AI", "파이썬", "자바스크립트", "데이터베이스"]
        selected_subject = random.choice(subjects)

        print(f"선택된 주제: {selected_subject}")
        print("=" * 50)

        return {
            "current_subject": selected_subject,
            "jokes": [],
            "attempt_count": 0,
        }

    # node
    def _update_attempt_count(self, state: JokeGeneratorState) -> dict:
        """시도 횟수 업데이트"""
        attempt_count = state.get("attempt_count", 0) + 1
        current_joke_count = len(state.get("jokes", []))

        print(f"\n현재 농담 개수: {current_joke_count}/{TARGET_JOKE_COUNT}")
        print(f"시도 횟수: {attempt_count}")

        return {"attempt_count": attempt_count}

    # route
    def _route_based_on_count(self, state: JokeGeneratorState) -> list[Send]:
        """농담 개수에 따라 Send로 라우팅"""
        current_joke_count = len(state.get("jokes", []))

        if current_joke_count < TARGET_JOKE_COUNT:
            # 3개 미만이면 부족한 만큼 generate_joke로 Send
            sends: list[Send] = []
            for i in range(current_joke_count + 1, TARGET_JOKE_COUNT + 1):
                sends.append(
                    Send(
                        "generate_joke",
                        {"subject": state["current_subject"], "joke_number": i},
                    )
                )
            print(f"{len(sends)}개의 농담을 추가로 생성합니다...")
            return sends
        else:
            # 3개 이상이면 finalize로
            print(
                f"농담 {TARGET_JOKE_COUNT}개 생성 완료! 최종 정리 단계로 이동합니다."
            )
            return [Send("finalize", state)]

    def _generate_single_joke(self, state: SingleJokeState) -> dict:
        """LLM을 사용하여 개별 농담 생성"""
        messages = [
            SystemMessage(
                content="당신은 재미있는 IT 농담을 만드는 코미디언입니다. 짧고 재치있는 농담을 한국어로 만들어주세요."
            ),
            HumanMessage(
                content=f"{state['subject']}에 대한 재미있는 농담을 하나만 만들어주세요. (농담 #{state['joke_number']})"
            ),
        ]

        response = self._llm.invoke(messages)
        joke = str(response.content).strip()

        print(f"농담 #{state['joke_number']}: {joke}")

        return {"jokes": [joke]}

    # node
    def _finalize_jokes(self, state: JokeGeneratorState) -> dict:
        """최종 농담 정리"""
        print("\n" + "=" * 50)
        print(f"최종 농담 컬렉션 ({state['current_subject']} 주제)")
        print("=" * 50)

        jokes = state.get("jokes", [])
        for i, joke in enumerate(jokes, 1):
            print(f"\n농담 {i}: {joke}")

        summary = (
            f"\n\n총 {len(jokes)}개의 농담이 생성되었습니다. "
            f"(시도 횟수: {state.get('attempt_count', 1)}회)"
        )

        return {"jokes": jokes + [summary]}

    def _compile_graph(self) -> CompiledStateGraph:
        # StateGraph 생성
        builder = StateGraph(JokeGeneratorState)

        # 노드 추가
        builder.add_node("initialize", self._initialize_state)
        builder.add_node("generate_joke", self._generate_single_joke)
        builder.add_node("update_count", self._update_attempt_count)
        builder.add_node("finalize", self._finalize_jokes)

        # 엣지 추가
        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "update_count")

        # update_count 노드 이후 conditional_edges로 Send 처리
        builder.add_conditional_edges(
            "update_count",
            self._route_based_on_count,  # Send 리스트를 반환하는 라우팅 함수
            ["generate_joke", "finalize"],  # 가능한 목적지 노드들
        )

        # generate_joke 완료 후 다시 update_count로
        builder.add_edge("generate_joke", "update_count")

        # finalize 후 종료
        builder.add_edge("finalize", END)

        return cast(CompiledStateGraph, builder.compile())


__all__ = ["JokeGeneratorState", "SendParallel", "SingleJokeState", "TARGET_JOKE_COUNT"]
