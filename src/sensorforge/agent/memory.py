"""Run records and cross-run learnings.

Each calibration writes its full state to ``<run_dir>/state.json`` and appends a
one-line summary to a shared ``learnings.jsonl``. ``best_prior`` reads that log
to warm-start a new run from the best past calibration for the same target.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sensorforge.agent.state import AgentState, TunableParams

DEFAULT_LEARNINGS = Path("runs/learnings.jsonl")


def record_run(
    state: AgentState, run_dir: str | Path, learnings_path: str | Path = DEFAULT_LEARNINGS
) -> Path:
    """Write the full run state and append a learnings summary line."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "state.json").write_text(state.model_dump_json(indent=2))

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "target": state.target,
        "real_source": state.real_source,
        "iterations": len(state.history),
        "best_deltaE": state.best.metrics["deltaE2000"] if state.best else None,
        "stop_reason": state.stop_reason,
        "best_params": state.best.params.model_dump() if state.best else None,
    }
    learnings_path = Path(learnings_path)
    learnings_path.parent.mkdir(parents=True, exist_ok=True)
    with learnings_path.open("a") as f:
        f.write(json.dumps(summary) + "\n")
    return run_dir / "state.json"


def load_learnings(learnings_path: str | Path = DEFAULT_LEARNINGS) -> list[dict]:
    path = Path(learnings_path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def best_prior(
    target: str, learnings_path: str | Path = DEFAULT_LEARNINGS
) -> TunableParams | None:
    """Best past tunable params for ``target`` (lowest ΔE), or None if no usable
    prior run exists.
    """
    candidates = [
        r
        for r in load_learnings(learnings_path)
        if r["target"] == target and r.get("best_params") and r.get("best_deltaE") is not None
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda r: r["best_deltaE"])
    return TunableParams(**best["best_params"])
