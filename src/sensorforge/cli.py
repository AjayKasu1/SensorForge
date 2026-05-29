"""Command-line entry point: ``sensorforge render`` and ``sensorforge capture``.

Phase 1 surface is deliberately small. argparse keeps it dependency-free; if
the command set grows past a handful we can revisit typer.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from sensorforge.capture.targets import checkerboard, uniform_field
from sensorforge.capture.webcam import Webcam
from sensorforge.sim.camera import SimCamera
from sensorforge.sim.renderer import SimRenderer

SCENES_DIR = Path("scenes")
DATA_DIR = Path("data")


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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
