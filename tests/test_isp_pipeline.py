from pathlib import Path

import numpy as np

from sensorforge.isp.params import ISPParams
from sensorforge.isp.pipeline import forward
from sensorforge.sim.camera import SimCamera
from sensorforge.sim.renderer import SimRenderer

SCENE = Path(__file__).parent.parent / "scenes" / "checkerboard.xml"


def _gray(level=0.5, h=64, w=64):
    return np.full((h, w, 3), level, dtype=np.float32)


def test_forward_shape_and_dtype():
    out = forward(_gray(), ISPParams(), rng=np.random.default_rng(0))
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_forward_is_deterministic_under_seed():
    a = forward(_gray(), ISPParams(), rng=np.random.default_rng(1))
    b = forward(_gray(), ISPParams(), rng=np.random.default_rng(1))
    assert np.array_equal(a, b)


def test_brighter_scene_gives_brighter_output():
    p = ISPParams()
    dim = forward(_gray(0.2), p, rng=np.random.default_rng(2)).mean()
    bright = forward(_gray(0.8), p, rng=np.random.default_rng(2)).mean()
    assert bright > dim


def test_flat_gray_output_is_roughly_uniform_interior():
    # Defaults have vignetting, so check only the central region for flatness.
    out = forward(_gray(0.5, 128, 128), ISPParams(), rng=np.random.default_rng(3))
    center = out[48:80, 48:80].astype(np.float64)
    # Far from clipped overall.
    assert 20 < center.mean() < 235
    # Spatial flatness is per-channel: the cross-channel spread comes from AWB
    # gains, not from non-uniformity, so measure each channel's own std.
    per_channel_std = center.reshape(-1, 3).std(axis=0)
    assert per_channel_std.max() < 10


def test_end_to_end_from_rendered_scene():
    # Real integration: render the MuJoCo scene, push it through the ISP.
    cam = SimCamera.from_scene(SCENE)
    with SimRenderer(cam) as r:
        linear = r.render()
    out = forward(linear, ISPParams(), rng=np.random.default_rng(4))
    assert out.shape == linear.shape
    assert out.dtype == np.uint8
    assert out.max() > out.min()  # the frame has content
