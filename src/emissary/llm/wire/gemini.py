"""The native Gemini `generateContent` adapter, for Gemini and Vertex.

Exists because the OpenAI-compatibility layer drops `thought_signature`, and
Gemini 3+ rejects a multi-turn tool call whose parts have lost it (ADR-0020).
Everything else here follows from that: the wire must round-trip *parts*, not
just text and calls.

One adapter serves two providers. They differ in credential and address — an
API key versus ADC with a project and region — never in protocol, which is
precisely what the provider table is for.
"""

import base64
import uuid
from typing import Any

from ..credentials import GoogleADC
from ..decision import (
    FinalOutput,
    ModelResult,
    ModelSettings,
    ReasoningState,
    Refusal,
    ToolCall,
    ToolCalls,
    ToolDefinition,
    Usage,
)
from ..errors import ProviderError, retryable_status
from ..messages import AssistantMessage, Message, ToolMessage, UserMessage
from ..provider import MAX_TOKENS, Spec
from .thinking import thinking_kwargs

WIRE = "gemini"
"""Tags reasoning state this wire issues, so no other wire replays it."""

BLOCKED_FINISH_REASONS = frozenset(
    {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}
)
"""Stops where the model was prevented from answering rather than choosing to.

These arrive as a 200 with no usable parts. Reporting that as an empty
completion would let a caller treat a content block as a real answer.
"""


def _client(spec: Spec):
    from google import genai

    credential = spec.provider.credential
    if isinstance(credential, GoogleADC):
        project = credential.project()
        if not project:
            raise ProviderError(f"{spec}: {credential.describe()} is not configured")
        # `enterprise` replaced `vertexai`, which the SDK keeps only as a
        # legacy alias. ADC is resolved by the SDK, so no key is passed.
        return genai.Client(enterprise=True, project=project, location=credential.location())
    return genai.Client(api_key=credential.token())


def _encode_signature(value: Any) -> Any:
    """Thought signatures arrive as bytes and must reach the event log as JSON.

    Base64 because the log is JSON-native (ADR-0011) and a signature is opaque
    — it is never inspected, only handed back exactly as it was issued.
    """
    return base64.b64encode(value).decode() if isinstance(value, bytes) else value


def _part_data(part) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if getattr(part, "text", None) is not None:
        data["text"] = part.text
    if getattr(part, "thought", None):
        data["thought"] = True
    signature = getattr(part, "thought_signature", None)
    if signature is not None:
        data["thought_signature"] = _encode_signature(signature)
    call = getattr(part, "function_call", None)
    if call is not None:
        data["function_call"] = {
            "id": call.id,
            "name": call.name,
            "args": dict(call.args or {}),
        }
    return data


def _gemini_contents(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, UserMessage):
            contents.append(
                {
                    "role": "user",
                    "parts": [{"text": block.text} for block in message.content],
                }
            )
        elif isinstance(message, AssistantMessage):
            # Signed parts come back first and unmodified. When this wire issued
            # the reasoning it already holds the full part list, including the
            # calls; rebuilding them from `tool_calls` would drop the signatures
            # and provoke the 400 this adapter exists to avoid.
            if message.reasoning is not None and message.reasoning.wire == WIRE:
                parts = [dict(block) for block in message.reasoning.blocks]
            else:
                parts = []
                if message.text:
                    parts.append({"text": message.text})
                parts.extend(
                    {"function_call": {"id": call.id, "name": call.name, "args": call.arguments}}
                    for call in message.tool_calls
                )
            contents.append({"role": "model", "parts": parts})
        elif isinstance(message, ToolMessage):
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "id": message.call_id,
                                "name": message.tool_name,
                                "response": {"output": message.content},
                            }
                        }
                    ],
                }
            )
    return contents


def call_model(
    spec: Spec,
    *,
    system: str,
    messages: tuple[Message, ...],
    tools: tuple[ToolDefinition, ...] = (),
    settings: ModelSettings | None = None,
) -> ModelResult:
    """Execute one provider-neutral conversational model turn."""
    from google.genai import errors

    configured = settings or ModelSettings()
    config: dict[str, Any] = {
        "system_instruction": system,
        "max_output_tokens": configured.max_output_tokens or MAX_TOKENS,
    }
    if tools and configured.tool_choice != "none":
        config["tools"] = [
            {
                "function_declarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        # `parameters_json_schema` takes raw JSON Schema;
                        # `parameters` expects the SDK's own Schema type.
                        "parameters_json_schema": tool.input_schema,
                    }
                    for tool in tools
                ]
            }
        ]
        if configured.tool_choice == "required":
            config["tool_config"] = {"function_calling_config": {"mode": "ANY"}}
    config.update(thinking_kwargs(spec, configured.thinking))

    try:
        response = _client(spec).models.generate_content(
            model=spec.model, contents=_gemini_contents(messages), config=config
        )
    except errors.APIError as exc:
        raise ProviderError(
            f"{spec}: {exc.code} {exc.message}", retryable=retryable_status(exc.code)
        ) from exc

    return _normalize(spec, response)


def _normalize(spec: Spec, response) -> ModelResult:
    if not response.candidates:
        raise ProviderError(f"{spec}: no candidate returned")
    candidate = response.candidates[0]
    finish = getattr(candidate, "finish_reason", None)
    finish_reason = getattr(finish, "value", finish)

    parts = list(getattr(candidate.content, "parts", None) or []) if candidate.content else []
    blocks = tuple(_part_data(part) for part in parts)

    if finish_reason in BLOCKED_FINISH_REASONS:
        decision = Refusal(f"the model was stopped by {finish_reason}")
    else:
        calls = tuple(
            ToolCall(
                # The Developer API may omit the id; the harness pairs every
                # result to its call by id, so a blank one loses the pairing.
                block["function_call"]["id"] or f"gemini-{uuid.uuid4().hex[:12]}",
                block["function_call"]["name"],
                block["function_call"]["args"],
            )
            for block in blocks
            if "function_call" in block
        )
        text = "".join(
            block["text"] for block in blocks if block.get("text") and not block.get("thought")
        )
        if calls:
            decision = ToolCalls(calls, text=text or None)
        elif text:
            decision = FinalOutput(text=text)
        else:
            raise ProviderError(f"{spec}: response contained no usable decision")

    thinking = (
        "\n".join(block["text"] for block in blocks if block.get("thought") and block.get("text"))
        or None
    )
    # Every part is kept, not only the signed ones: Gemini attaches signatures
    # to function-call parts too, and the API verifies the sequence as a whole.
    reasoning = (
        ReasoningState(wire=WIRE, blocks=blocks)
        if any("thought_signature" in block for block in blocks)
        else None
    )

    usage = response.usage_metadata
    return ModelResult(
        decision=decision,
        provider=spec.name,
        model=getattr(response, "model_version", None) or spec.model,
        usage=Usage(
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            cached_input_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
        ),
        finish_reason=finish_reason,
        thinking=thinking,
        reasoning=reasoning,
    )


__all__ = ["WIRE", "call_model"]
