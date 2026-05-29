# SensorForge

Calibrate a simulated MuJoCo camera against a real webcam, automatically, with an LLM-driven
loop that proposes parameter updates and logs every assumption it makes.

> Status: bootstrap. Phase 0 complete. Phases 1–5 land in subsequent commits.

## What it is

A Python framework that:
1. Renders a scene through a configurable ISP pipeline in MuJoCo,
2. Captures the same scene from a real webcam,
3. Asks an LLM to diagnose the gap and propose parameter updates,
4. Iterates until SSIM / ΔE2000 / EMVA-1288 metrics fall within tolerance,
5. Produces a per-run report and a documented assumptions log.

## Why it exists

Sim-to-real calibration is usually done by hand. This project explores whether an agent can
drive the loop while staying auditable — every parameter change is logged with a justification.

## Quickstart

```bash
uv sync
uv run pytest
```

End-to-end demo lands in Phase 5:

```bash
make demo   # TODO (Phase 5)
```

## Demo

TODO — 90-second screen recording (Phase 5).

## Results

TODO — table of SSIM / PSNR / ΔE2000 / EMVA-1288 numbers on the reference webcam (Phase 4–5).

## Architecture

See [docs/architecture.md](docs/architecture.md).

## Limitations

See `LIMITATIONS.md` (lands in Phase 5).
