import math
from pathlib import Path

import numpy as np
import pytest

from sensorforge.sim.camera import CameraIntrinsics, SimCamera

SCENE = Path(__file__).parent.parent / "scenes" / "checkerboard.xml"


def test_intrinsics_pixel_focal_and_fov():
    # 36mm focal on a 36x24mm full-frame sensor at 1000x1000:
    # fx = 36 * 1000 / 36 = 1000 px; fovx = 2*atan(36/72) = 53.13 deg.
    intr = CameraIntrinsics(
        focal_length_mm=36.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        width_px=1000,
        height_px=1000,
    )
    assert intr.fx_px == pytest.approx(1000.0)
    assert intr.fovx_deg == pytest.approx(math.degrees(2 * math.atan(0.5)))


def test_intrinsics_rejects_nonphysical():
    with pytest.raises(ValueError):
        CameraIntrinsics(
            focal_length_mm=0.0,
            sensor_width_mm=36.0,
            sensor_height_mm=24.0,
            width_px=100,
            height_px=100,
        )


def test_intrinsics_read_from_scene():
    cam = SimCamera.from_scene(SCENE)
    intr = cam.intrinsics
    assert (intr.width_px, intr.height_px) == (640, 480)
    # Scene declares 3.6mm focal on a 3.58x2.69mm sensor.
    assert intr.focal_length_mm == pytest.approx(3.6, abs=1e-3)
    assert intr.fx_px == pytest.approx(3.6 * 640 / 3.58, rel=1e-3)


def test_light_intensity_and_position_round_trip():
    cam = SimCamera.from_scene(SCENE)
    # Setters mutate the model; reading back confirms the write took.
    cam.position = np.array([0.0, -0.6, 0.3])
    assert cam.position[1] == pytest.approx(-0.6)

    cam.set_light_intensity(0.5)
    lid = 0  # 'key' is the only light
    assert np.allclose(cam.model.light_diffuse[lid], 0.5)
