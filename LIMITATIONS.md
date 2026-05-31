# Limitations

An honest list of what SensorForge does not do, or does only in a limited way.

## Calibration scope

- **Single uniform target for v1.** The headline calibration matches a flat
  field, where AWB gains plus exposure are enough to align the three channel
  means. A multi-patch ColorChecker, which is what makes the 3x3 CCM
  identifiable, is deferred to v2. The CCM is held at identity (ADR 006).
- **Radiometric only, no geometric registration.** The loop tunes color and
  exposure on spatially-aligned images. A structured target (e.g. a
  checkerboard) needs corner detection, a homography estimate, and a warp to
  align sim and real before radiometry can be matched. Without it the loop
  stalls on the spatial mismatch: a checkerboard PNG against the checkerboard
  scene plateaus around deltaE2000 41 (the stall detector stops it). Geometric
  calibration is v2; uniform/flat targets are the v1 scope.
- **Single illuminant.** White balance assumes one light. Mixed-illuminant
  scenes are out of scope.
- **Fixed sensor and optics during calibration.** Only the six color/exposure
  knobs are tuned. Full-well, read noise, dark current, vignetting, and
  distortion are fixed, so the agent matches appearance, not the full sensor
  characterization (the EMVA estimators measure those separately).

## Reproducible numbers vs the LLM

- The results table uses the **heuristic proposer** so anyone can reproduce it
  without a model or keys. The project's method is the **LLM-driven loop**, but
  its exact numbers depend on the model; a small local model (llama3.2) does not
  always converge as cleanly as the heuristic. The LLM path is implemented and
  tested, just not the source of the committed numbers.
- **No committed real-webcam results.** The webcam path works, but webcam
  output is device- and lighting-specific and not reproducible, so it is not in
  the results table.

## Physical modeling

- **Linearity is an assumption.** MuJoCo's framebuffer is treated as
  approximately-linear scene radiance (ADR 004); this is not radiometrically
  validated against a real sensor.
- **Gamma is a pure power law**, a close approximation of the piecewise
  IEC 61966-2-1 sRGB curve; the two differ slightly in deep shadows.
- **Distortion coefficients are corner-normalized**, not OpenCV's
  focal-normalized convention, so they are not interchangeable with
  `cv2.calibrateCamera` outputs.
- **EMVA-1288 subset only.** No spectral sensitivity, absolute responsivity,
  linearity-error metric, or defect-pixel characterization (ADR 003); these
  need lab-grade radiometry.
- **Forward ISP only.** There is no inverse pipeline; we push sim radiance
  forward and compare, we do not recover scene radiance from a real frame.

## Tooling

- **No CI.** This is a fast-moving local portfolio project; tests and lint are
  run locally via `make`.
- A harmless third-party deprecation warning from `langgraph` surfaces once in
  the pytest output.
