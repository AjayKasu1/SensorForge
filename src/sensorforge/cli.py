"""Command-line entry point: ``sensorforge render | capture | calibrate``.

argparse keeps it dependency-free; if the command set grows past a handful we
can revisit typer.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from sensorforge.agent.graph import CalibrationContext, run_calibration
from sensorforge.agent.llm import make_llm_client
from sensorforge.agent.memory import best_prior, record_run
from sensorforge.agent.proposers import HeuristicProposer, LLMProposer
from sensorforge.agent.state import AgentState, TunableParams
from sensorforge.agent.tools import capture_real
from sensorforge.capture.targets import checkerboard, uniform_field
from sensorforge.capture.webcam import Webcam
from sensorforge.isp.params import ISPParams
from sensorforge.metrics.report import write_convergence_gif, write_report
from sensorforge.sim.camera import SimCamera
from sensorforge.sim.renderer import SimRenderer

SCENES_DIR = Path("scenes")
DATA_DIR = Path("data")
RUNS_DIR = Path("runs")

# The hidden "real camera look" the agent recovers in sim-as-real mode: a near-
# neutral white balance and different tone vs the uncalibrated defaults, so the
# starting gap is a large color cast (baseline deltaE2000 > 16 on the uniform
# target).
SIM_REAL_PRESET = TunableParams(
    exposure_ms=14.0,
    black_level=0.06,
    awb_gain_r=1.05,
    awb_gain_g=1.0,
    awb_gain_b=1.1,
    gamma=1.8,
)

UNIFORM_LEVEL = 0.5  # mid-gray scene radiance for the uniform calibration target


def _scene_linear(target: str, scene: str) -> NDArray[np.float32]:
    """Linear RGB the ISP calibrates on. The uniform target is a flat field (the
    ISP's spatial effects still apply); other targets render the MJCF scene.
    """
    cam = SimCamera.from_scene(_resolve_scene(scene))
    if target == "uniform":
        h, w = cam.intrinsics.height_px, cam.intrinsics.width_px
        return np.full((h, w, 3), UNIFORM_LEVEL, dtype=np.float32)
    with SimRenderer(cam) as r:
        return r.render()


def _resolve_scene(scene: str) -> Path:
    """Accept either a bare scene name or a path to an .xml."""
    candidate = Path(scene)
    if candidate.suffix == ".xml" and candidate.exists():
        return candidate
    return SCENES_DIR / f"{scene}.xml"


def _reference_target(name: str) -> NDArray[np.float32]:
    return uniform_field() if name == "uniform" else checkerboard()


def _default_out(kind: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DATA_DIR / f"{kind}_{stamp}.npy"


def _cmd_render(args: argparse.Namespace) -> int:
    scene_path = _resolve_scene(args.scene)
    cam = SimCamera.from_scene(scene_path)
    with SimRenderer(cam) as r:
        frames = np.stack([r.render() for _ in range(args.frames)])

    out = Path(args.out) if args.out else _default_out("render")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, frames)
    logger.info("rendered {} frame(s) {} -> {}", args.frames, frames.shape, out)
    return 0


def _cmd_capture(args: argparse.Namespace) -> int:
    with Webcam(index=args.index) as cam:
        cam.lock_exposure_and_white_balance()
        frames = cam.grab_frames(args.frames)

    out = Path(args.out) if args.out else _default_out("capture")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, frames)
    # Save the reference target next to the capture so calibration has the
    # ground-truth pattern that was physically in front of the lens.
    ref_path = out.with_name(out.stem + f"_target_{args.target}.npy")
    np.save(ref_path, _reference_target(args.target))
    logger.info(
        "captured {} frame(s) {} of target '{}' -> {}",
        args.frames,
        frames.shape,
        args.target,
        out,
    )
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    if args.proposer == "heuristic":
        proposer = HeuristicProposer()
    else:
        proposer = LLMProposer(make_llm_client(args.llm))
    linear = _scene_linear(args.target, args.scene)
    base = ISPParams()
    rng = np.random.default_rng(args.seed)

    if args.real_source == "sim":
        hidden = SIM_REAL_PRESET.apply_to(base)
        real = capture_real(
            "sim", linear_rgb=linear, hidden_params=hidden, frames=args.avg, rng=rng
        )
    else:
        real = capture_real("webcam", webcam_index=args.index, frames=args.frames, rng=rng)

    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    start = (
        best_prior(args.target, RUNS_DIR / "learnings.jsonl") if args.warm_start else None
    ) or (TunableParams())
    ctx = CalibrationContext(
        linear_rgb=linear,
        real=real,
        base_params=base,
        proposer=proposer,
        run_dir=str(run_dir),
        rng=rng,
        n_average=args.avg,
    )
    state0 = AgentState(
        target=args.target,
        real_source=args.real_source,
        current=start,
        max_iters=args.max_iters,
        tolerance_de=args.tolerance,
        run_dir=str(run_dir),
    )
    final = run_calibration(ctx, state0)

    best_isp = final.best.params.apply_to(base)
    write_report(run_dir, ctx.render_avg(best_isp), real, final.best.metrics, best_isp)
    record_run(final, run_dir, RUNS_DIR / "learnings.jsonl")

    if args.gif:
        frames = [ctx.render_avg(a.params.apply_to(base)) for a in final.history]
        write_convergence_gif(args.gif, frames, real)
        logger.info("wrote convergence gif to {}", args.gif)
    logger.info(
        "calibrate done: stop={} best deltaE2000={:.3g} report={}/report.md",
        final.stop_reason,
        final.best.metrics["deltaE2000"],
        run_dir,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sensorforge")
    sub = parser.add_subparsers(dest="command", required=True)

    p_render = sub.add_parser("render", help="render a sim scene to a .npy frame stack")
    p_render.add_argument("--scene", default="checkerboard", help="scene name or path to .xml")
    p_render.add_argument("--frames", type=int, default=1)
    p_render.add_argument("--out", default=None, help="output .npy path")
    p_render.set_defaults(func=_cmd_render)

    p_capture = sub.add_parser("capture", help="capture frames from a webcam")
    p_capture.add_argument("--target", choices=["uniform", "checkerboard"], default="uniform")
    p_capture.add_argument("--frames", type=int, default=50)
    p_capture.add_argument("--index", type=int, default=0, help="webcam device index")
    p_capture.add_argument("--out", default=None, help="output .npy path")
    p_capture.set_defaults(func=_cmd_capture)

    p_cal = sub.add_parser("calibrate", help="run the LLM calibration loop")
    p_cal.add_argument("--target", choices=["uniform", "checkerboard"], default="uniform")
    p_cal.add_argument("--real-source", choices=["sim", "webcam"], default="sim")
    p_cal.add_argument("--scene", default="checkerboard", help="scene name or path to .xml")
    p_cal.add_argument("--max-iters", type=int, default=20)
    p_cal.add_argument("--tolerance", type=float, default=3.0, help="target deltaE2000")
    p_cal.add_argument("--avg", type=int, default=16, help="frames averaged per measurement")
    p_cal.add_argument("--index", type=int, default=0, help="webcam index (webcam source)")
    p_cal.add_argument("--frames", type=int, default=30, help="webcam frames (webcam source)")
    p_cal.add_argument(
        "--proposer",
        choices=["llm", "heuristic"],
        default="llm",
        help="llm (needs Ollama/API) or heuristic (no LLM, reproducible)",
    )
    p_cal.add_argument("--llm", default=None, help="ollama|openai|anthropic (else SENSORFORGE_LLM)")
    p_cal.add_argument("--seed", type=int, default=0)
    p_cal.add_argument("--warm-start", action="store_true", help="seed from best prior run")
    p_cal.add_argument("--gif", default=None, help="write a convergence GIF to this path")
    p_cal.set_defaults(func=_cmd_calibrate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
