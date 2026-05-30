import numpy as np
import pytest

from sensorforge.metrics.image_metrics import delta_e2000, psnr, ssim


def _rng_img(seed, shape=(32, 32, 3)):
    return (np.random.default_rng(seed).random(shape) * 255).astype(np.uint8)


@pytest.mark.filterwarnings("ignore:divide by zero")
def test_identical_images_are_perfect():
    img = _rng_img(0)
    assert ssim(img, img) == 1.0
    assert np.isinf(psnr(img, img))
    assert delta_e2000(img, img) == 0.0


def test_degradation_moves_metrics_the_right_way():
    img = _rng_img(1)
    noisy = np.clip(img.astype(int) + 40, 0, 255).astype(np.uint8)
    assert ssim(img, noisy) < 1.0
    assert psnr(img, noisy) < 60.0
    assert delta_e2000(img, noisy) > 0.0


def test_more_degradation_is_monotonic():
    img = _rng_img(2)
    small = np.clip(img.astype(int) + 10, 0, 255).astype(np.uint8)
    large = np.clip(img.astype(int) + 60, 0, 255).astype(np.uint8)
    assert delta_e2000(img, large) > delta_e2000(img, small)
    assert ssim(img, large) < ssim(img, small)


def test_delta_e2000_matches_known_color_pair():
    # Two flat sRGB colors; cross-check our end-to-end value against a direct
    # Lab-space CIEDE2000 (guards the sRGB->Lab path and the method choice).
    import colour

    c1 = np.full((4, 4, 3), [0.50, 0.20, 0.70])
    c2 = np.full((4, 4, 3), [0.55, 0.22, 0.68])
    lab1 = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(c1[0, 0]))
    lab2 = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(c2[0, 0]))
    expected = float(colour.delta_E(lab1, lab2, method="CIE 2000"))
    assert np.isclose(delta_e2000(c1, c2), expected, rtol=1e-6)


def test_accepts_float_and_uint8_equivalently():
    u8 = _rng_img(3)
    f = u8.astype(np.float64) / 255.0
    assert np.isclose(delta_e2000(u8, u8), delta_e2000(f, f))
    assert np.isclose(ssim(u8, u8), ssim(f, f))
