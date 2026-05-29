# ADR 004: ISP radiometric model and stage ordering

- **Date:** 2026-05-29
- **Status:** Accepted

## Context

The ISP turns the renderer's RGB output into something comparable to a real
webcam frame. Two things must be pinned down: what the renderer's float values
physically mean, and in what order the pipeline stages run. Both are easy to
get subtly wrong in ways that only show up as a calibration that never
converges.

## Decision

**Radiance convention.** MuJoCo's float32 output is treated as approximate
linear *scene radiance* in [0, 1]. We do not linearize or inverse-correct the
sim input. The pipeline's gamma stage and 8-bit quantize stage own the encoding
into display-referred sRGB, which is what makes the sim output comparable to
the real webcam (whose own ISP already gamma-encodes).

**Stage order** (linear radiance in, uint8 sRGB out):

1. vignetting (cos⁴) → 2. distortion (Brown-Conrady) → 3. Bayer mosaic (RGGB)
→ 4. integration → 5. dark current → 6. shot noise → 7. read noise →
8. normalize + black-level pedestal → 9. AWB gains → 10. demosaic (bilinear)
→ 11. CCM → 12. gamma → 13. 8-bit quantize.

The Bayer mosaic runs **before** the noise stages (3 before 4-7), which
reverses the order sketched in the original build plan.

## Consequences

- Shot/read/dark noise act on the single-channel post-CFA raw, matching real
  hardware: one photodiode per pixel sees one color. Applying noise to three
  full-resolution color planes (mosaic-last) would inject 3x the samples at the
  wrong magnitude and the wrong spatial correlation.
- Optics run on linear radiance, where vignetting is a true multiplicative
  falloff and distortion is a clean geometric remap. Vignetting uses the
  undistorted (ideal) image radius, then distortion warps the result.
- AWB gains are applied on the raw before demosaic (white balance on raw),
  the standard ISP location.
- Gamma is the only nonlinearity between scene and display, so the sim-vs-real
  gap is concentrated where the calibration agent can reason about it.
- Forward pass only for v1: no inverse ISP. We do not recover scene radiance
  from a real frame; we push sim radiance forward and compare.
