# EMVA-1288 implemented subset

What this project measures from the standard, how, and what it deliberately
leaves out. Scope rationale is in [ADR 003](adr/003-emva1288-scope.md). Code is
in [`metrics/emva1288.py`](../src/sensorforge/metrics/emva1288.py); reference is
EMVA-1288 Release 3.1.

## Implemented

| Metric | What it captures | Method | Unit |
|---|---|---|---|
| Temporal dark noise | Read + dark temporal noise floor | two-image difference on dark frames | DN |
| System gain K | DN per electron (the SNR-curve deliverable) | slope of temporal variance vs mean over an illumination sweep | DN/e- |
| DSNU | Dark fixed-pattern non-uniformity | spatial std of the temporal-mean dark image | DN |
| PRNU | Light-response fixed-pattern non-uniformity | spatial std of the dark-subtracted mean response, over its mean | % |

### The two-image difference method

Temporal noise must be separated from fixed-pattern (spatial) noise. For two
nominally-identical frames `a` and `b`, the fixed pattern is common to both and
cancels in `a - b`, so the spatial variance of the difference is twice the
temporal variance:

    sigma_temporal^2 = var(a - b) / 2

This is used for temporal dark noise and for every point on the photon-transfer
curve. For the spatial metrics (DSNU, PRNU) we instead average many frames to
suppress temporal noise, then subtract the residual temporal variance
(`var / L`) so the result reflects fixed pattern alone.

### System gain and the photon-transfer curve

Across an illumination sweep, temporal variance is linear in mean signal:

    sigma_temporal^2 = K * (mean - mean_dark) + sigma_dark^2

The slope is the system gain `K` (DN/e-). The SNR curve,
`SNR = (mean - mean_dark) / sigma_temporal`, comes from the same sweep.

## Validation

Each estimator is checked by injection and recovery: a synthetic linear sensor
with known gain, read noise, PRNU, and DSNU generates frame stacks, the
estimators measure them, and the recovered values match the injected
ground truth within 5% (`tests/test_emva1288.py`). The synthetic sensor is
intentionally decoupled from the project's ISP so the tests validate the
estimators, not the pipeline.

## Not implemented, and why

| Excluded | Why |
|---|---|
| Spectral sensitivity / quantum efficiency | Needs a monochromator and calibrated source |
| Absolute responsivity (W, photons) | Needs a radiometrically calibrated illuminant |
| Linearity error metric | Out of scope for v1; gain fit already assumes linearity |
| Dark current vs temperature | No temperature control on a webcam |
| Defect-pixel characterization | Not needed to drive sim-to-real calibration |
| Dynamic range / absolute sensitivity threshold | Derived quantities we do not report in v1 |

All values are reported in DN with electron equivalents via `K`, never in
absolute photometric units, because we have no calibrated source.
