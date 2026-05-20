"""``ModelRequest`` 디버깅용 출력 유틸."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from util.messages import depth_colors, display_message_tree


def _object_to_display_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return repr(obj)


def _format_callable_for_display(value: Any) -> str:
    if value is None:
        return "None"
    name = getattr(value, "__name__", None)
    if name:
        return f"<{type(value).__name__}: {name}>"
    return f"<{type(value).__name__}>"


def _format_runtime_for_display(runtime: Any) -> dict[str, Any]:
    if runtime is None:
        return {
            "context": {},
            "store": None,
            "stream_writer": None,
            "previous": None,
            "execution_info": None,
        }

    ctx = getattr(runtime, "context", None)
    execution_info_value = _object_to_display_dict(
        getattr(runtime, "execution_info", None)
    )

    store = getattr(runtime, "store", None)
    return {
        "context": dict(ctx) if ctx else {},
        "store": None if store is None else repr(store),
        "stream_writer": _format_callable_for_display(
            getattr(runtime, "stream_writer", None)
        ),
        "previous": getattr(runtime, "previous", None),
        "execution_info": execution_info_value,
    }


def display_model_request(request: Any) -> None:
    """``ModelRequest`` 필드를 트리 형태로 출력합니다 (디버깅용)."""
    c, r = depth_colors[1], depth_colors["reset"]
    print(f"\n{c}═══ ModelRequest ═══{r}")

    model = request.model
    model_label = getattr(model, "model_name", None) or getattr(model, "model", str(model))

    tools = request.tools or []
    tool_names = [getattr(t, "name", str(t)) for t in tools]

    response_format = getattr(request, "response_format", None)
    if response_format is not None:
        response_format_value: Any = repr(response_format)
    else:
        response_format_value = None

    summary: dict[str, Any] = {
        "model": model_label,
        "system_prompt": request.system_prompt,
        "system_message": getattr(request, "system_message", None),
        "messages": request.messages or [],
        "tool_choice": request.tool_choice,
        "tools": tool_names,
        "response_format": response_format_value,
        "state": getattr(request, "state", None) or {"messages": []},
        "runtime": _format_runtime_for_display(getattr(request, "runtime", None)),
        "model_settings": request.model_settings or {},
    }

    display_message_tree(summary)


__all__ = ["display_model_request"]
