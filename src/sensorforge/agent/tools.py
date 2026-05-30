"""Tool functions the graph nodes call.

Plain functions, not a registry: the graph is small and over-abstracting the
tools would hide the data flow. The scene render is constant across a
calibration run (only ISP params change), so render_sim takes a pre-rendered
linear frame and re-runs just the ISP each iteration.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from sensorforge.agent.llm import LLMClient, Message
from sensorforge.agent.prompts import (
    build_diagnosis_messages,
    build_proposal_messages,
    parse_proposal,
)
from sensorforge.agent.state import Assumption, Attempt, TunableParams, clamp_proposal
from sensorforge.capture.webcam import Webcam
from sensorforge.isp.params import ISPParams
from sensorforge.isp.pipeline import forward
from sensorforge.metrics.image_metrics import delta_e2000, psnr, ssim

# A hosted LLM can throttle us (free-tier rate limits) or blip transiently. The
# loop should ride that out, not die, so generate calls get bounded backoff.
LLM_RETRY_ATTEMPTS = 5
LLM_RETRY_BASE_DELAY_S = 20.0


def _generate_with_retry(llm: LLMClient, messages: list[Message]) -> str:
    for attempt in range(LLM_RETRY_ATTEMPTS):
        try:
            return llm.generate(messages)
        except Exception as e:  # any provider error is worth one bounded retry
            if attempt == LLM_RETRY_ATTEMPTS - 1:
                raise
            delay = LLM_RETRY_BASE_DELAY_S * (attempt + 1)
            logger.warning("LLM call failed ({}); retrying in {:.0f}s", str(e)[:90], delay)
            time.sleep(delay)
    raise RuntimeError("unreachable")  # loop either returns or raises


def render_sim(
    linear_rgb: NDArray, params: ISPParams, rng: np.random.Generator | None = None
) -> NDArray[np.uint8]:
    return forward(linear_rgb, params, rng)


def capture_real(
    real_source: str,
    *,
    linear_rgb: NDArray | None = None,
    hidden_params: ISPParams | None = None,
    webcam_index: int = 0,
    frames: int = 30,
    rng: np.random.Generator | None = None,
) -> NDArray[np.uint8]:
    """Produce the reference frame the sim is calibrated against.

    'sim' renders the scene through hidden ground-truth params (reproducible, no
    hardware). 'webcam' averages real frames to suppress temporal noise.
    """
    if real_source == "sim":
        # Average several noisy renders, mirroring the webcam path, so the
        # reference has a low temporal-noise floor.
        stack = np.stack([forward(linear_rgb, hidden_params, rng) for _ in range(frames)])
        return np.round(stack.mean(axis=0)).astype(np.uint8)
    if real_source == "webcam":
        with Webcam(webcam_index) as cam:
            cam.lock_exposure_and_white_balance()
            stack = cam.grab_frames(frames)
        return (np.clip(stack.mean(axis=0), 0, 1) * 255 + 0.5).astype(np.uint8)
    raise ValueError(f"unknown real_source {real_source!r}; use 'sim' or 'webcam'")


def compute_metrics(sim: NDArray, real: NDArray) -> dict[str, float]:
    return {
        "SSIM": ssim(sim, real),
        "PSNR": psnr(sim, real),
        "deltaE2000": delta_e2000(sim, real),
    }


def propose_param_update(
    llm: LLMClient, current: TunableParams, metrics: dict, history: list[Attempt]
) -> tuple[TunableParams, str]:
    """Two-step LLM call: diagnose the gap, then propose new values. The
    proposal is parsed and clamped to physical bounds before returning.
    """
    diagnosis = _generate_with_retry(llm, build_diagnosis_messages(current, metrics, history))
    raw = _generate_with_retry(llm, build_proposal_messages(current, diagnosis))
    proposed = clamp_proposal(parse_proposal(raw), current)
    return proposed, diagnosis


def write_assumption(run_dir: str | Path, assumption: Assumption) -> Path:
    """Append an assumption to runs/<ts>/assumptions.md, the run deliverable."""
    path = Path(run_dir) / "assumptions.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a") as f:
        if new:
            f.write("# Calibration assumptions\n\n")
        f.write(
            f"- {assumption.timestamp} | {assumption.parameter} = "
            f"{assumption.value:.4g} | {assumption.justification}\n"
        )
    return path
