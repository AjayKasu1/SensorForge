"""Proposers: how new parameters are chosen each round.

Two implementations behind one interface. ``LLMProposer`` is the project's
method (an LLM diagnoses and proposes from metrics and history). ``Heuristic
Proposer`` is a deterministic gray-world auto-calibration that closes the loop
from the actual sim/real channel statistics, with no foreknowledge of the
target. The heuristic needs no LLM, so it powers the reproducible demo and is
the graceful fallback when no model is configured.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from sensorforge.agent.llm import LLMClient
from sensorforge.agent.state import Attempt, TunableParams, clamp_proposal
from sensorforge.agent.tools import propose_param_update


class Proposer(Protocol):
    def propose(
        self,
        current: TunableParams,
        sim: NDArray,
        real: NDArray,
        metrics: dict,
        history: list[Attempt],
    ) -> tuple[TunableParams, str]: ...


class LLMProposer:
    """Delegates to the LLM diagnose-then-propose flow; ignores the images."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def propose(self, current, sim, real, metrics, history):
        return propose_param_update(self.llm, current, metrics, history)


class HeuristicProposer:
    """Gray-world balance: scale exposure toward the real luminance and the AWB
    gains toward the real per-channel color ratios. ``damping`` < the encoding
    gamma keeps the iteration stable (output ratios understate the raw change
    needed, so we converge geometrically rather than overshoot).
    """

    def __init__(self, damping: float = 1.4):
        self.damping = damping

    def propose(self, current, sim, real, metrics, history):
        sim_m = sim.reshape(-1, 3).mean(axis=0)
        real_m = real.reshape(-1, 3).mean(axis=0)
        sim_luma = max(sim_m.mean(), 1e-6)
        real_luma = max(real_m.mean(), 1e-6)

        level = (real_luma / sim_luma) ** self.damping
        raw = current.model_dump()
        raw["exposure_ms"] = current.exposure_ms * level
        for c, knob in enumerate(("awb_gain_r", "awb_gain_g", "awb_gain_b")):
            # Color ratio with the overall level divided out, so AWB only
            # corrects hue and exposure handles level.
            color = (max(real_m[c], 1e-6) / max(sim_m[c], 1e-6)) / (real_luma / sim_luma)
            raw[knob] = getattr(current, knob) * color**self.damping

        new = clamp_proposal(raw, current)
        reasoning = (
            f"gray-world balance: exposure x{level:.2f}; real R/G/B means "
            f"{real_m[0]:.0f}/{real_m[1]:.0f}/{real_m[2]:.0f} vs sim "
            f"{sim_m[0]:.0f}/{sim_m[1]:.0f}/{sim_m[2]:.0f}"
        )
        return new, reasoning
