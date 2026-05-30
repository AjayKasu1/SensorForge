# ADR 003: EMVA-1288 implemented subset

- **Date:** 2026-05-30
- **Status:** Accepted

## Context

EMVA-1288 is a broad standard for characterizing image sensors, and a faithful
full implementation assumes a radiometric lab bench (calibrated light source,
monochromator, integrating sphere) that neither the sim nor a consumer webcam
provides. We want enough of the standard to demonstrate real understanding of
sensor characterization and to drive calibration, not a checkbox of every
metric.

## Decision

Implement four metrics by the book: temporal dark noise, photon-transfer system
gain K (the SNR-curve deliverable), PRNU, and DSNU. Everything requiring
absolute radiometric calibration or a controlled illuminant is out of scope.

## Consequences

- Temporal noise uses the two-image difference method (FPN cancels in the
  difference), and gain K comes from a linear fit of temporal variance vs mean
  signal across an illumination sweep. These are the core photon-transfer
  techniques and prove the method end to end.
- PRNU and DSNU cover the spatial fixed-pattern story (light and dark), so both
  noise axes (temporal and spatial) are represented.
- Explicitly excluded: spectral sensitivity / quantum efficiency, absolute
  responsivity in W, the linearity-error metric, dark current vs temperature,
  defect-pixel characterization, and the full dynamic-range / absolute-
  sensitivity-threshold report. Each needs lab-grade radiometry we do not have.
- Units are reported in DN with electron equivalents via K, not in absolute
  photometric units, since we have no calibrated source.
- Reference: EMVA-1288 Standard for Characterization of Image Sensors and
  Cameras, Release 3.1 (2016).
