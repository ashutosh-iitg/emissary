"""Per-provider translation of the neutral thinking setting (ADR-0019).

The provider table names a dialect; this module owns what that name means on
the wire. Keeping the payload shapes here is what lets `provider.py` stay free
of SDK vocabulary — `extra_body` is an OpenAI-SDK concept and has no business
in the neutral layer.

A dialect returns request kwargs to merge, or raises. It never silently drops a
setting: every explicit value is a promise about cost or disclosure, and a
caller who asked not to be billed for reasoning must not be billed for it.
"""

from collections.abc import Callable
from typing import Any

from ..decision import Thinking
from ..errors import CapabilityError
from ..provider import Spec


def _unsupported(spec: Spec, setting: Thinking) -> CapabilityError:
    return CapabilityError(f"{spec}: this provider cannot express thinking={setting!r}")


def _none(spec: Spec, setting: Thinking) -> dict[str, Any]:
    raise _unsupported(spec, setting)


def _anthropic(spec: Spec, setting: Thinking) -> dict[str, Any]:
    if setting == "off":
        return {"thinking": {"type": "disabled"}}
    if setting == "on":
        return {"thinking": {"type": "adaptive"}}
    return {"thinking": {"type": "adaptive", "display": "summarized"}}


def _deepseek(spec: Spec, setting: Thinking) -> dict[str, Any]:
    enabled = "disabled" if setting == "off" else "enabled"
    return {"extra_body": {"thinking": {"type": enabled}}}


def _gemini(spec: Spec, setting: Thinking) -> dict[str, Any]:
    """`ThinkingLevel` has no OFF member, so disabling is a zero budget.

    That is the documented switch for 2.5-generation models; Gemini 3+ prefers
    `thinking_level` and may not honour a budget of zero. Named here rather
    than hidden, because a silently-ignored `off` is the failure ADR-0019
    exists to prevent.
    """
    if setting == "off":
        return {"thinking_config": {"thinking_budget": 0}}
    return {"thinking_config": {"include_thoughts": setting == "visible"}}


def _effort(spec: Spec, setting: Thinking) -> dict[str, Any]:
    """Models that always reason (Kimi K3), tuned by effort rather than toggled.

    `on` and `visible` are already true of the model, so they send nothing.
    `off` is not expressible and must not be quietly accepted.
    """
    if setting == "off":
        raise _unsupported(spec, setting)
    return {}


DIALECTS: dict[str, Callable[[Spec, Thinking], dict[str, Any]]] = {
    "none": _none,
    "anthropic": _anthropic,
    "deepseek": _deepseek,
    "effort": _effort,
    "gemini": _gemini,
}


def thinking_kwargs(spec: Spec, setting: Thinking) -> dict[str, Any]:
    """Request kwargs for `setting`, or `{}` when nothing should be sent."""
    if setting == "default":
        return {}
    dialect = DIALECTS.get(spec.provider.thinking_dialect)
    if dialect is None:
        raise CapabilityError(
            f"{spec}: unknown thinking dialect {spec.provider.thinking_dialect!r}"
        )
    return dialect(spec, setting)


__all__ = ["DIALECTS", "thinking_kwargs"]
