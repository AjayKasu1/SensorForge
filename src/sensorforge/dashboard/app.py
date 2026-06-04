"""Streamlit dashboard for inspecting a calibration run.

Run with: ``uv run streamlit run src/sensorforge/dashboard/app.py`` (or
``make dashboard``). The data loading (``load_run``, ``list_runs``) is pure and
unit-tested; the Streamlit view under ``main`` is a thin shell.
"""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

DATA_DIR = Path("data")
RUNS_DIR = Path("runs")
UPLOADS_DIR = DATA_DIR / "uploads"


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


def save_uploaded_image(uploaded: BinaryIO, upload_dir: Path = UPLOADS_DIR) -> Path:
    """Persist an uploaded reference image and return its local path."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    name = Path(getattr(uploaded, "name", "reference.png")).name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = upload_dir / f"{stamp}_{name}"
    data = uploaded.getbuffer() if hasattr(uploaded, "getbuffer") else uploaded.read()
    path.write_bytes(bytes(data))
    return path


def run_image_calibration(
    image_path: str | Path,
    *,
    proposer: str = "heuristic",
    llm: str | None = None,
    model: str | None = None,
    target: str = "uniform",
    scene: str = "checkerboard",
    max_iters: int = 8,
    tolerance: float = 3.0,
    avg: int = 8,
    seed: int = 0,
    gif: str | None = None,
) -> Path | None:
    """Run calibration against an uploaded image and return the newest run."""
    from sensorforge.cli import _cmd_calibrate

    before = set(list_runs())
    _cmd_calibrate(
        Namespace(
            target=target,
            real_source="image",
            real_npy=None,
            real_image=str(image_path),
            scene=scene,
            max_iters=max_iters,
            tolerance=tolerance,
            avg=avg,
            index=0,
            frames=30,
            proposer=proposer,
            llm=llm,
            model=model,
            seed=seed,
            warm_start=False,
            gif=gif,
        )
    )
    created = [p for p in list_runs() if p not in before]
    return created[0] if created else (list_runs()[0] if list_runs() else None)


def _render_run_view(st, selected: Path) -> None:
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


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="SensorForge", layout="wide")
    st.title("SensorForge calibration dashboard")

    inspect_tab, upload_tab = st.tabs(["Runs", "Image calibration"])

    with inspect_tab:
        runs = list_runs()
        if not runs:
            st.info("No runs yet. Run `make demo`, `sensorforge calibrate`, or upload an image.")
        else:
            selected = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name)
            _render_run_view(st, selected)

    with upload_tab:
        uploaded = st.file_uploader("Reference image", type=["png", "jpg", "jpeg"])
        c1, c2, c3, c4 = st.columns(4)
        target = c1.selectbox("Target", ["image_pattern", "uniform", "checkerboard"])
        proposer = c2.selectbox("Proposer", ["heuristic", "llm"])
        llm = c3.selectbox("LLM", ["gemini", "anthropic", "openai", "ollama"])
        model = c4.text_input("Model override", value="")

        c4, c5, c6 = st.columns(3)
        max_iters = c4.number_input("Max iterations", min_value=1, max_value=50, value=8)
        avg = c5.number_input("Frames averaged", min_value=1, max_value=64, value=8)
        tolerance = c6.number_input("Target ΔE2000", min_value=0.1, max_value=50.0, value=3.0)

        if uploaded is not None:
            st.image(uploaded, caption=uploaded.name)

        if st.button("Run calibration", disabled=uploaded is None):
            image_path = save_uploaded_image(uploaded)
            gif_path = f"docs/upload_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
            with st.spinner("Calibrating..."):
                run_dir = run_image_calibration(
                    image_path,
                    proposer=proposer,
                    llm=llm if proposer == "llm" else None,
                    model=model or None,
                    target=target,
                    max_iters=int(max_iters),
                    tolerance=float(tolerance),
                    avg=int(avg),
                    gif=gif_path,
                )
            if run_dir is None:
                st.error("Calibration did not create a run.")
            else:
                st.success(f"Created run {run_dir.name}")
                _render_run_view(st, run_dir)


if __name__ == "__main__":
    main()
