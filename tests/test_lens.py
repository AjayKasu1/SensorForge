import numpy as np

from sensorforge.isp.lens import apply_distortion, apply_vignetting


def test_vignetting_darkens_corners_keeps_center():
    img = np.ones((101, 101, 3), dtype=np.float32)
    out = apply_vignetting(img, strength=0.5)
    center = out[50, 50, 0]
    corner = out[0, 0, 0]
    assert center == np.float32(1.0)  # gain == 1 at the optical center
    assert corner < center
    # cos^4 corner gain at r=1 is 0.25, so gain = 1 - 0.5*(1-0.25) = 0.625.
    assert np.isclose(corner, 0.625)


def test_vignetting_strength_zero_is_identity():
    img = np.random.default_rng(0).random((32, 48, 3)).astype(np.float32)
    out = apply_vignetting(img, strength=0.0)
    assert np.allclose(out, img)


def test_vignetting_is_radially_monotonic():
    img = np.ones((81, 81, 3), dtype=np.float32)
    out = apply_vignetting(img, strength=0.6)[..., 0]
    # Walk from center to the right edge along the middle row: non-increasing.
    profile = out[40, 40:]
    assert np.all(np.diff(profile) <= 1e-7)


def test_distortion_zero_coeffs_is_identity():
    img = np.random.default_rng(1).random((40, 60, 3)).astype(np.float32)
    out = apply_distortion(img, 0.0, 0.0, 0.0, 0.0, 0.0)
    # Identity map under bilinear remap reproduces the input closely.
    assert np.allclose(out, img, atol=1e-5)


def test_distortion_keeps_center_fixed():
    # A bright dot at the exact center must stay at the center: r=0 -> no shift.
    img = np.zeros((51, 51, 3), dtype=np.float32)
    img[25, 25] = 1.0
    out = apply_distortion(img, 0.3, 0.0, 0.0, 0.0, 0.0)
    assert out[25, 25, 0] == img[25, 25, 0]
    # And the peak is still at the center.
    peak = np.unravel_index(np.argmax(out[..., 0]), out[..., 0].shape)
    assert peak == (25, 25)
