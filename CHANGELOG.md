# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/). This is a
portfolio project; the history is grouped by the build phases.

## [0.1.0] - 2026-05-30

First end-to-end version: render, ISP, metrics, and an agentic calibration loop.

### Added

- **Simulation** (`sim/`): MuJoCo camera wrapper with physical intrinsics, a
  checkerboard MJCF scene with a controllable light, and an offscreen renderer
  to float32 linear RGB.
- **ISP pipeline** (`isp/`): vignetting and Brown-Conrady distortion, RGGB
  mosaic, physics-grounded shot/read/dark noise, black level, AWB, bilinear
  demosaic, CCM, gamma, and 8-bit quantize, composed into a forward pass.
- **Capture** (`capture/`): webcam capture with best-effort exposure/WB lock,
  and checkerboard/uniform target generators.
- **Metrics** (`metrics/`): SSIM, PSNR, CIEDE2000, and an EMVA-1288 subset
  (temporal dark noise, photon-transfer gain, PRNU, DSNU), plus a per-run
  markdown report with plots.
- **Agent** (`agent/`): a LangGraph calibration loop, an LLM adapter
  (Ollama / OpenAI / Anthropic), a heuristic offline proposer, run records with
  warm-start, and an assumptions log.
- **CLI**: `sensorforge render | capture | calibrate`, plus a Streamlit
  dashboard and `make demo` / `make dashboard`.
- Six ADRs documenting the key decisions, and docs for the architecture, ISP
  pipeline, and EMVA-1288 subset.
- **More reference sources**: calibrate against a hidden sim preset, a live
  webcam, a captured `.npy` stack, or an image file (`--real-source image`).
- **Gemini provider** added to the LLM adapter, with a `--model` override flag.
- **Fault tolerance**: LLM calls retry with backoff through rate limits, and the
  loop stops gracefully (preserving the best result) when a proposer gives up,
  both found and fixed against live API quota walls.
- **Packaging**: PyPI metadata, MIT LICENSE, scenes shipped as package data, and
  a GitHub Actions CI workflow (ruff + pytest coverage gate, headless MuJoCo).

### Known limitations

See [LIMITATIONS.md](LIMITATIONS.md).
