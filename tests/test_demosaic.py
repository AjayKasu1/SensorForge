import numpy as np

from sensorforge.isp.bayer import mosaic_rggb
from sensorforge.isp.demosaic import demosaic_bilinear


def test_output_is_full_rgb():
    raw = np.zeros((8, 8), dtype=np.float32)
    out = demosaic_bilinear(raw)
    assert out.shape == (8, 8, 3)


def test_constant_color_reconstructs_exactly_in_interior():
    # mosaic then demosaic of a flat color must return that color (DC preserved).
    rgb = np.empty((16, 16, 3), dtype=np.float64)
    rgb[...] = (0.2, 0.5, 0.8)
    out = demosaic_bilinear(mosaic_rggb(rgb))
    interior = out[2:-2, 2:-2]
    assert np.allclose(interior, (0.2, 0.5, 0.8), atol=1e-6)


def test_smooth_gradient_recovered_with_small_error():
    h, w = 32, 32
    ramp = np.linspace(0.1, 0.9, w)[None, :].repeat(h, axis=0)
    rgb = np.dstack([ramp, ramp, ramp])
    out = demosaic_bilinear(mosaic_rggb(rgb))
    # A smooth low-frequency signal is reconstructed well.
    assert np.abs(out[2:-2, 2:-2] - rgb[2:-2, 2:-2]).mean() < 0.01


def test_high_frequency_edge_produces_color_artifacts():
    # A luminance-only vertical stripe pattern (no color) should come back gray
    # under a perfect demosaic; bilinear introduces nonzero chroma (false color).
    h, w = 32, 32
    stripes = np.tile([0.0, 1.0], w // 2)[None, :].repeat(h, axis=0)
    rgb = np.dstack([stripes, stripes, stripes])
    out = demosaic_bilinear(mosaic_rggb(rgb))
    chroma = np.abs(out[..., 0] - out[..., 2])  # R-B difference = false color
    assert chroma.max() > 0.05
