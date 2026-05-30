"""Versioned prompts for the calibration agent.

All prompt text lives here (no inline prompt strings elsewhere). The knob
specification is generated from the TunableParams schema so the prompt can
never drift from the actual tunable set or bounds.
"""

from __future__ import annotations

import json
import re

from sensorforge.agent.llm import Message
from sensorforge.agent.state import TunableParams, _bounds

SYSTEM_PROMPT_V1 = (
    "You are a camera ISP calibration engineer. A simulated camera's output must "
    "match a real one. You adjust a small set of ISP parameters and reason from "
    "image-difference metrics. Lower SSIM means structural mismatch; higher "
    "deltaE2000 means color mismatch (target: under 3). Be concrete and change "
    "only one or two parameters at a time."
)


def _knob_spec() -> str:
    lines = []
    for name, field in TunableParams.model_fields.items():
        lo, hi = _bounds(field)
        lines.append(f"- {name} [{lo:.3g}, {hi:.3g}]: {field.description}")
    return "\n".join(lines)


def _history_block(history: list, k: int = 3) -> str:
    if not history:
        return "(no prior attempts)"
    recent = history[-k:]
    return "\n".join(
        f"  iter {a.iteration}: deltaE2000={a.metrics.get('deltaE2000', float('nan')):.3g}, "
        f"params={a.params.model_dump()}"
        for a in recent
    )


def build_diagnosis_messages(current: TunableParams, metrics: dict, history: list) -> list[Message]:
    user = (
        f"Current metrics: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}\n"
        f"Current parameters: {current.model_dump()}\n"
        f"Last attempts:\n{_history_block(history)}\n\n"
        "What is the single most likely cause of the remaining gap, and which 1-2 "
        "parameters should change to reduce it? Answer in 2-3 sentences."
    )
    return [Message("system", SYSTEM_PROMPT_V1), Message("user", user)]


def build_proposal_messages(current: TunableParams, diagnosis: str) -> list[Message]:
    user = (
        f"Diagnosis: {diagnosis}\n\n"
        f"Tunable parameters and their valid ranges:\n{_knob_spec()}\n\n"
        f"Current values: {current.model_dump()}\n\n"
        "Propose new values as a JSON object. Include only the parameters you want "
        "to change. Respond with JSON only, no prose."
    )
    return [Message("system", SYSTEM_PROMPT_V1), Message("user", user)]


def parse_proposal(text: str) -> dict:
    """Extract a JSON object from an LLM reply, tolerating code fences and prose.
    Returns an empty dict if nothing parseable is found, so the caller can fall
    back to the current parameters without crashing.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
