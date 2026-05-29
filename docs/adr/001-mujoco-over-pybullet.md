# ADR 001: MuJoCo over PyBullet for simulation

- **Date:** 2026-05-29
- **Status:** Accepted

## Context

We need a simulator that can render a controllable camera into a scene with configurable
geometry and lighting, runs on a CPU laptop, and has clean Python bindings. The main
candidates are MuJoCo (DeepMind bindings), PyBullet, Isaac Sim, and Gazebo.

## Decision

Use MuJoCo via the `mujoco` Python package (the DeepMind bindings, not legacy `mujoco-py`).

## Consequences

- Free, MPL-2.0, no license friction for a public portfolio repo.
- CPU rendering is fast enough for our resolutions (<= 640×480), so no GPU dependency.
- MJCF scenes are concise and diff-friendly compared to URDF + world files.
- We give up PyBullet's larger contact-physics community, which is irrelevant here since
  we are not simulating dynamics, only rendering.
- Isaac Sim is off the table: GPU-bound, heavy install, overkill for a single-camera scene.
