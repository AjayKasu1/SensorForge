from pathlib import Path

import numpy as np

from sensorforge.sim.camera import SimCamera
from sensorforge.sim.renderer import SimRenderer

SCENE = Path(__file__).parent.parent / "scenes" / "checkerboard.xml"


def test_render_shape_dtype_range():
    cam = SimCamera.from_scene(SCENE)
    with SimRenderer(cam) as r:
        img = r.render()
    assert img.shape == (480, 640, 3)
    assert img.dtype == np.float32
    assert img.min() >= 0.0 and img.max() <= 1.0
    # The checkerboard has bright squares, so the frame is not all black.
    assert img.max() > 0.5


def test_light_intensity_drives_brightness():
    # The exposure proxy must actually flow through to rendered intensity:
    # more light -> brighter mean over the lit target region.
    cam = SimCamera.from_scene(SCENE)
    with SimRenderer(cam) as r:
        cam.set_light_intensity(0.2)
        dim = r.render().mean()
        cam.set_light_intensity(1.0)
        bright = r.render().mean()
    assert bright > dim
