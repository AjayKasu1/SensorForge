"""LangGraph calibration loop.

Two nodes over a pydantic AgentState: ``measure`` renders the sim with the
current params and scores it, then sets a stop reason if the run is done;
``propose`` asks the LLM for new values and logs the changes as assumptions.
A conditional edge ends the run when a stop reason is set.

Per-run context (the constant scene render, the reference frame, the LLM, the
run directory) is held in CalibrationContext and closed over by the nodes, so
AgentState carries only serializable data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from langgraph.graph import END, StateGraph
from loguru import logger
from numpy.typing import NDArray

from sensorforge.agent.llm import LLMClient
from sensorforge.agent.state import AgentState, Assumption, Attempt, TunableParams
from sensorforge.agent.tools import (
    compute_metrics,
    propose_param_update,
    render_sim,
    write_assumption,
)
from sensorforge.isp.params import ISPParams

STALL_ROUNDS = 3


@dataclass
class CalibrationContext:
    linear_rgb: NDArray
    real: NDArray
    base_params: ISPParams
    llm: LLMClient
    run_dir: str
    rng: np.random.Generator
    n_average: int = 1  # sim frames averaged per measurement to cut noise

    def now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def render_avg(self, params: ISPParams) -> NDArray:
        if self.n_average == 1:
            return render_sim(self.linear_rgb, params, self.rng)
        stack = np.stack(
            [render_sim(self.linear_rgb, params, self.rng) for _ in range(self.n_average)]
        )
        return np.round(stack.mean(axis=0)).astype(np.uint8)


def _stop_reason(state: AgentState, best: Attempt) -> str | None:
    if best.metrics["deltaE2000"] <= state.tolerance_de:
        return "within_tolerance"
    if state.iteration >= state.max_iters - 1:
        return "budget_exhausted"
    if state.iteration - best.iteration >= STALL_ROUNDS:
        return "stalled"
    return None


def _make_measure(ctx: CalibrationContext):
    def measure(state: AgentState) -> dict:
        params = state.current.apply_to(ctx.base_params)
        sim = ctx.render_avg(params)
        metrics = compute_metrics(sim, ctx.real)
        attempt = Attempt(iteration=state.iteration, params=state.current, metrics=metrics)
        history = [*state.history, attempt]
        best = state.best
        if best is None or metrics["deltaE2000"] < best.metrics["deltaE2000"]:
            best = attempt
        reason = _stop_reason(state, best)
        logger.info(
            "iter {} deltaE2000={:.3g} (best {:.3g}){}",
            state.iteration,
            metrics["deltaE2000"],
            best.metrics["deltaE2000"],
            f" -> stop: {reason}" if reason else "",
        )
        return {"history": history, "best": best, "stop_reason": reason}

    return measure


def _make_propose(ctx: CalibrationContext):
    def propose(state: AgentState) -> dict:
        current = state.current
        last_metrics = state.history[-1].metrics
        new_params, diagnosis = propose_param_update(ctx.llm, current, last_metrics, state.history)
        assumptions = list(state.assumptions)
        for name in TunableParams.model_fields:
            if getattr(current, name) != getattr(new_params, name):
                a = Assumption(
                    timestamp=ctx.now(),
                    parameter=name,
                    value=getattr(new_params, name),
                    justification=diagnosis,
                )
                write_assumption(ctx.run_dir, a)
                assumptions.append(a)
        return {
            "current": new_params,
            "assumptions": assumptions,
            "iteration": state.iteration + 1,
        }

    return propose


def build_graph(ctx: CalibrationContext):
    builder = StateGraph(AgentState)
    builder.add_node("measure", _make_measure(ctx))
    builder.add_node("propose", _make_propose(ctx))
    builder.set_entry_point("measure")
    builder.add_conditional_edges(
        "measure",
        lambda s: "stop" if s.stop_reason else "continue",
        {"continue": "propose", "stop": END},
    )
    builder.add_edge("propose", "measure")
    return builder.compile()


def run_calibration(ctx: CalibrationContext, initial: AgentState) -> AgentState:
    """Run the loop to a stop condition and return the final state."""
    graph = build_graph(ctx)
    config = {"recursion_limit": 2 * initial.max_iters + 10}
    final = graph.invoke(initial, config)
    return final if isinstance(final, AgentState) else AgentState(**final)
