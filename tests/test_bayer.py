import numpy as np

from sensorforge.isp.bayer import mosaic_rggb


def test_mosaic_shape_and_dtype():
    rgb = np.zeros((4, 6, 3), dtype=np.float32)
    raw = mosaic_rggb(rgb)
    assert raw.shape == (4, 6)
    assert raw.dtype == np.float32


def test_each_position_carries_its_channel():
    # Distinct per-channel constants so we can read back which channel landed.
    rgb = np.zeros((4, 4, 3), dtype=np.float32)
    rgb[..., 0] = 0.1  # R
    rgb[..., 1] = 0.5  # G
    rgb[..., 2] = 0.9  # B
    raw = mosaic_rggb(rgb)
    assert raw[0, 0] == np.float32(0.1)  # R site
    assert raw[0, 1] == np.float32(0.5)  # G site
    assert raw[1, 0] == np.float32(0.5)  # G site
    assert raw[1, 1] == np.float32(0.9)  # B site


def test_green_sites_are_half_the_pixels():
    rgb = np.dstack(
        [np.zeros((8, 8), np.float32), np.ones((8, 8), np.float32), np.zeros((8, 8), np.float32)]
    )
    raw = mosaic_rggb(rgb)
    # Only the two green sites per 2x2 block carry the 1.0; that's half of all pixels.
    assert raw.sum() == np.float32(8 * 8 / 2)
