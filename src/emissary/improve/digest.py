"""What the improver is told about failures: rendered by code from the event log.

Built from events rather than messages because the events hold what a
transcript drops — tool errors, rejected calls and the reason a run stopped —
and those are usually the cause.
"""

from collections.abc import Mapping

from ..harness.state import RunResult


def failure_digest(failures: Mapping[str, tuple[RunResult, ...]], *, max_chars: int) -> str:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    sections = [
        _render(name, index, run)
        for name, runs in sorted(failures.items())
        for index, run in enumerate(runs, 1)
    ]
    text = "\n\n".join(sections) or "No failing runs were reported."
    if len(text) <= max_chars:
        return text
    marker = f"\n\n[{len(text) - max_chars} more characters of failures omitted]"
    return text[: max_chars - len(marker)] + marker


def _render(name: str, index: int, run: RunResult) -> str:
    lines = [f"## {name}, failed run {index}", f"stop reason: {run.stop_reason.value}"]
    for event in run.events:
        if event.kind == "user_message":
            task = " ".join(block["text"] for block in event.data["content"])
            lines.append(f"task: {task}")
        elif event.kind == "tool_call_rejected":
            lines.append(f"tool call rejected: {event.data['reason']}")
        elif event.kind == "tool_call_completed" and event.data["status"] != "success":
            result = event.data["result"]
            lines.append(f"tool {event.data['tool']} {event.data['status']}: {result['summary']}")
        elif event.kind == "tool_call_completed":
            content = event.data["result"].get("content")
            lines.append(f"tool {event.data['tool']} returned: {content!r}")
    if run.output is not None:
        answer = run.output.text if run.output.text is not None else repr(run.output.value)
        lines.append(f"final answer: {answer}")
    return "\n".join(lines)


__all__ = ["failure_digest"]
