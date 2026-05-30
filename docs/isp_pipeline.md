# ISP pipeline

Stage-by-stage reference for the forward image-signal-processing pipeline:
linear scene radiance in, 8-bit sRGB out. See
[ADR 004](adr/004-isp-ordering.md) for why the stages run in this order and how
the radiance convention is defined.

Notation: the input is linear RGB in [0, 1] (approximate scene radiance). Image
coordinates `(x, y)` are centered on the image center and normalized by the
half-diagonal, so the corner is at radius `r = 1`. Parameters live in
[`isp/params.py`](../src/sensorforge/isp/params.py); units are repeated here.

## 1. Vignetting (`lens.py`)

Multiplicative cos⁴ illumination falloff, using the identity
`cos⁴θ = (1 + r²)⁻²` for `r = tanθ`:

    gain(r) = 1 - s · (1 - (1 + r²)⁻²)
    out = in · gain

`s = vignette_strength` (dimensionless, [0, 1]). The corner is treated as a 45°
field angle; `s` absorbs the lens-specific deviation. Cosine-fourth law:
Kingslake, *Optics in Photography* (1992).

## 2. Distortion (`lens.py`)

Brown-Conrady radial + tangential. With `r² = x² + y²`:

    x_d = x(1 + k1 r² + k2 r⁴ + k3 r⁶) + 2 p1 x y + p2 (r² + 2x²)
    y_d = y(1 + k1 r² + k2 r⁴ + k3 r⁶) + p1 (r² + 2y²) + 2 p2 x y

Each output pixel samples the input at `(x_d, y_d)` (bilinear). `k1, k2, k3`
radial and `p1, p2` tangential coefficients, all dimensionless, in
corner-normalized coordinates (not OpenCV's focal-normalized convention).
Brown (1966), *Decentering Distortion of Lenses*.

## 3. Bayer mosaic (`bayer.py`)

Collapse RGB to a single-channel raw via the RGGB color filter array:

    raw[y, x] = RGB[y, x, C(y, x)],   C = [[R, G], [G, B]] over the 2x2 tile

Bayer (1976), US Patent 3,971,065.

## 4. Integration / exposure (`noise.py`)

Map linear raw to mean photoelectrons:

    μ_e = raw · N_full · (t / t_ref)

`N_full = full_well_e` (electrons), `t = exposure_ms` (ms),
`t_ref = EXPOSURE_REF_MS = 10 ms` (the exposure at which `raw = 1` fills the
well).

## 5. Dark current (`noise.py`)

Thermally generated electrons accumulated over the exposure:

    μ_dark = D · (t / 1000),   total mean μ = μ_e + μ_dark

`D = dark_current_e_per_s` (electrons/second).

## 6. Shot noise (`noise.py`)

Collected charge is Poisson; one draw captures photon and dark shot noise:

    n ~ Poisson(μ),   Var(n) = E(n) = μ

Healey & Kondepudy (1994), *Radiometric CCD Camera Calibration and Noise
Estimation*; EMVA-1288 linear sensor model.

## 7. Well saturation (`noise.py`)

    n = min(n, N_full)

Hard highlight clip at the full-well capacity.

## 8. Read noise (`noise.py`)

Signal-independent Gaussian added at readout:

    y = n + 𝒩(0, σ_r²)

`σ_r = read_noise_e` (electrons RMS).

## 9. Black level + normalize (`color.py`)

    DN = clip(y / N_full + b, 0, 1)

`b = black_level` (normalized [0, 1]). The pedestal is added and **not**
subtracted: its residual is exactly the output dark floor we must reproduce to
match a real camera, so it stays a meaningful calibration knob (see the
`to_digital` docstring).

## 10. AWB gains (`color.py`)

White balance on the raw, per CFA position:

    DN'[y, x] = DN[y, x] · g_{C(y,x)}

`g_r = awb_gain_r`, `g_g = awb_gain_g`, `g_b = awb_gain_b` (dimensionless). No
clip here; highlights may exceed 1 and are clamped at gamma/quantize.

## 11. Demosaic, bilinear (`demosaic.py`)

Per-channel bilinear interpolation by convolving the sparse color planes:

    G kernel  = [[0,1,0],[1,4,1],[0,1,0]] / 4
    R,B kernel = [[1,2,1],[2,4,2],[1,2,1]] / 4

Bilinear is chosen to expose false-color/zipper artifacts on high-frequency
edges. Malvar, He & Cutler (2004), *High-Quality Linear Interpolation for
Demosaicing*, is the upgrade path if v1 cannot hit the color target.

## 12. Color correction matrix (`color.py`)

    out_i = Σ_j M_ij · in_j

`M = ccm`, a 3x3 dimensionless matrix (identity by default).

## 13. Gamma (`color.py`)

Linear-to-display encoding by a pure power law:

    out = clip(in, 0, 1) ^ (1 / γ)

`γ = gamma` (dimensionless, default 2.2). Approximates the piecewise
IEC 61966-2-1 sRGB curve; the difference is confined to deep shadows.

## 14. 8-bit quantize (`color.py`)

    u = clip(round(out · 255), 0, 255)   as uint8
