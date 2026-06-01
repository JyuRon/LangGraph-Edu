"""
참고 문서:
/langgraph-v1-tutorial/PART02-에이전트/Ch05-Human-in-the-Loop/02-LangGraph-Human-In-The-Loop.ipynb

핵심:
``HumanInTheLoopMiddleware`` 의 edit 결정 후, 원본 사용자 요청이 messages 에 남아
모델이 같은 도구를 재호출하며 interrupt 가 반복되는 루프를 막는다.
edit 된 도구 실행 결과 ``ToolMessage`` 에 "사람이 수정한 최종 행동" 안내를 덧붙인다.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import Decision, InterruptOnConfig
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolCall, ToolMessage
from langgraph.types import Command


# edit 된 도구 호출 결과에 덧붙일 안내. 모델이 "사람이 의도적으로 바꾼 최종 행동"임을
# 인지하도록 영어로 작성한다(대화·도구 결과가 영어라 모델 준수도가 높다).
DEFAULT_EDIT_NOTICE = (
    "[human-in-the-loop notice] A human reviewer intentionally edited this tool "
    "call's arguments before execution, so they may differ from the original user "
    "request. This edited execution is the human-approved final action. Do not retry "
    "or re-issue the original request — treat this result as the completed task."
)


class EditAwareHumanInTheLoopMiddleware(HumanInTheLoopMiddleware):
    """edit 후 모델이 **원본 요청으로 되돌아가 다시 interrupt** 되는 루프를 막는 미들웨어.

    표준 ``HumanInTheLoopMiddleware`` 는 edit 시 ``AIMessage`` 의 tool_calls 만
    수정된 인수로 교체한다. 하지만 사용자의 **원본 요청 메시지는 그대로 남기** 때문에,
    예) ``original.txt`` 요청을 ``modified.txt`` 로 edit 하면 모델은
    "아직 original.txt 를 처리하지 않았다"고 판단해 같은 도구를 다시 호출하고
    interrupt 가 반복된다.

    이를 막기 위해 edit 된 도구 호출 id 를 기록해 두었다가(``_process_decision``),
    해당 도구가 실제 실행된 직후(``wrap_tool_call``) 결과 ``ToolMessage`` 에
    "사람이 의도적으로 수정한 최종 행동" 안내를 덧붙인다. 모델은 이 안내를 보고
    편집된 행동을 최종 의도로 받아들여 원본 요청을 재시도하지 않는다.
    """

    def __init__(
        self,
        interrupt_on: dict[str, bool | InterruptOnConfig],
        *,
        description_prefix: str = "Tool execution requires approval",
        edit_notice: str = DEFAULT_EDIT_NOTICE,
    ) -> None:
        super().__init__(interrupt_on, description_prefix=description_prefix)
        self._edit_notice = edit_notice
        # tool_call_id -> True (edit 된 호출만 표시). 실행 후 pop 으로 소비한다.
        self._edited_call_ids: dict[str, bool] = {}

    def _process_decision(
        self,
        decision: Decision,
        tool_call: ToolCall,
        config: InterruptOnConfig,
    ) -> tuple[ToolCall | None, ToolMessage | None]:
        """기본 동작은 그대로 두되, edit 결정이면 해당 호출 id 를 기록한다."""
        revised_tool_call, tool_message = super()._process_decision(
            decision, tool_call, config
        )
        if decision["type"] == "edit" and "edit" in config["allowed_decisions"]:
            # edit 된 호출은 원본 tool_call 의 id 를 그대로 유지한다(상위 구현 참고).
            call_id = tool_call.get("id")
            if call_id is not None:
                self._edited_call_ids[call_id] = True
        return revised_tool_call, tool_message

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """도구 실행 후, edit 된 호출이면 결과에 안내를 덧붙여 재시도 루프를 끊는다."""
        result = handler(request)
        call_id = request.tool_call.get("id")
        # edit 된 호출만 1회 소비. 일반 승인/거부 호출은 영향받지 않는다.
        if call_id and self._edited_call_ids.pop(call_id, False):
            if isinstance(result, ToolMessage) and isinstance(result.content, str):
                result.content = f"{result.content}\n\n{self._edit_notice}"
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> ToolMessage | Command:
        """비동기 경로에서도 동일하게 edit 안내를 덧붙인다."""
        result = await handler(request)
        call_id = request.tool_call.get("id")
        if call_id and self._edited_call_ids.pop(call_id, False):
            if isinstance(result, ToolMessage) and isinstance(result.content, str):
                result.content = f"{result.content}\n\n{self._edit_notice}"
        return result


__all__ = [
    "DEFAULT_EDIT_NOTICE",
    "EditAwareHumanInTheLoopMiddleware",
]
