# ADR 006: parameter tuning constraints

- **Date:** 2026-05-30
- **Status:** Accepted

## Context

The ISP exposes many parameters, and several of them move the output the same
way. Output brightness alone is set by the sim light intensity, `exposure_ms`,
and `full_well_e`. If the agent can tune all of them it will thrash between
equivalent solutions and never converge cleanly.

## Decision

Let the agent tune only the six-knob color and exposure chain: `exposure_ms`,
`black_level`, `awb_gain_r/g/b`, and `gamma`. Everything else is fixed.

## Consequences

- The six knobs are mutually non-degenerate: one linear level (`exposure_ms`),
  one offset (`black_level`), three per-channel scales (AWB), and one curve
  (`gamma`). The search is well conditioned.
- Fixed and why: the sim light (scene illumination, not a camera property);
  `full_well_e` (a sensor spec, and fixing it removes the brightness
  degeneracy); `read_noise_e` and `dark_current_e_per_s` (noise, which does not
  move ΔE on an averaged target); the optics (geometric, second-order on a
  uniform field); and the CCM, held at identity.
- CCM stays fixed because on a single uniform patch the AWB gains already match
  the three channel means; the CCM is only identifiable against a multi-patch
  ColorChecker, which is the v2 extension.
- Proposals are clamped to each knob's physical bounds (read from the
  `TunableParams` schema) before they reach the pipeline, so a bad LLM response
  is bounded, not fatal.
