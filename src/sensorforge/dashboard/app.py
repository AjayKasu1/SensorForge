"""Streamlit dashboard for inspecting a calibration run.

Run with: ``uv run streamlit run src/sensorforge/dashboard/app.py`` (or
``make dashboard``). The data loading (``load_run``, ``list_runs``) is pure and
unit-tested; the Streamlit view under ``main`` is a thin shell.
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS_DIR = Path("runs")


def list_runs(runs_dir: Path = RUNS_DIR) -> list[Path]:
    """Run directories that have a state.json, newest first."""
    if not runs_dir.exists():
        return []
    runs = [p for p in runs_dir.iterdir() if p.is_dir() and (p / "state.json").exists()]
    return sorted(runs, reverse=True)


def load_run(run_dir: str | Path) -> dict:
    """Load a run's state plus the display-ready convergence series and paths."""
    run_dir = Path(run_dir)
    state = json.loads((run_dir / "state.json").read_text())
    history = state["history"]
    best = state.get("best")
    return {
        "state": state,
        "iterations": [h["iteration"] for h in history],
        "deltaE": [h["metrics"]["deltaE2000"] for h in history],
        "ssim": [h["metrics"]["SSIM"] for h in history],
        "comparison_png": run_dir / "comparison.png",
        "assumptions_md": run_dir / "assumptions.md",
        "best_params": best["params"] if best else None,
        "best_deltaE": best["metrics"]["deltaE2000"] if best else None,
    }


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="SensorForge", layout="wide")
    st.title("SensorForge calibration dashboard")

    runs = list_runs()
    if not runs:
        st.info("No runs yet. Run `make demo` or `sensorforge calibrate`.")
        return

    selected = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name)
    run = load_run(selected)
    state = run["state"]

    c1, c2, c3 = st.columns(3)
    c1.metric("best ΔE2000", f"{run['best_deltaE']:.2f}")
    c2.metric("iterations", len(state["history"]))
    c3.metric("stop reason", state["stop_reason"])
    st.caption(f"target: {state['target']}  |  real source: {state['real_source']}")

    st.subheader("Convergence")
    st.line_chart({"ΔE2000": run["deltaE"], "SSIM": run["ssim"]})

    if run["comparison_png"].exists():
        st.subheader("Sim vs real")
        st.image(str(run["comparison_png"]))

    if run["best_params"]:
        st.subheader("Calibrated parameters")
        st.table(run["best_params"])

    if run["assumptions_md"].exists():
        st.subheader("Assumptions log")
        st.markdown(run["assumptions_md"].read_text())


if __name__ == "__main__":
    main()
