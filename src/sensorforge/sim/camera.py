"""MuJoCo camera wrapper and its intrinsics.

MuJoCo stores camera intrinsics in SI units (meters) on the model. We surface
them in millimetres (the unit lens and sensor datasheets actually use) and
derive the pixel-space focal lengths and fields of view that downstream
calibration and corner detection care about.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

MM_PER_M = 1000.0


class CameraIntrinsics(BaseModel):
    """Pinhole intrinsics in physical units.

    A single ``focal_length_mm`` is the physical lens focal length; the
    horizontal/vertical pixel focal lengths differ only because the pixel
    pitch differs between axes (non-square sensor / resolution).
    """

    focal_length_mm: float = Field(gt=0, description="Lens focal length, mm")
    sensor_width_mm: float = Field(gt=0, description="Sensor width, mm")
    sensor_height_mm: float = Field(gt=0, description="Sensor height, mm")
    width_px: int = Field(gt=0, description="Horizontal resolution, pixels")
    height_px: int = Field(gt=0, description="Vertical resolution, pixels")

    @property
    def fx_px(self) -> float:
        """Horizontal focal length in pixels: f / pixel_pitch_x."""
        return self.focal_length_mm * self.width_px / self.sensor_width_mm

    @property
    def fy_px(self) -> float:
        return self.focal_length_mm * self.height_px / self.sensor_height_mm

    @property
    def fovx_deg(self) -> float:
        return math.degrees(2 * math.atan(self.sensor_width_mm / (2 * self.focal_length_mm)))

    @property
    def fovy_deg(self) -> float:
        return math.degrees(2 * math.atan(self.sensor_height_mm / (2 * self.focal_length_mm)))

    @classmethod
    def from_mujoco(cls, model: mujoco.MjModel, cam_id: int) -> CameraIntrinsics:
        sensor_w, sensor_h = model.cam_sensorsize[cam_id]
        # cam_intrinsic is [focal_x, focal_y, principal_x, principal_y] in metres.
        # We carry the x focal length; in our scenes fx == fy by construction.
        focal_x = model.cam_intrinsic[cam_id, 0]
        res_w, res_h = model.cam_resolution[cam_id]
        if sensor_w <= 0 or focal_x <= 0:
            raise ValueError(
                f"camera {cam_id} has no physical intrinsics set "
                "(sensorsize/focal); add them to the MJCF <camera>"
            )
        return cls(
            focal_length_mm=float(focal_x * MM_PER_M),
            sensor_width_mm=float(sensor_w * MM_PER_M),
            sensor_height_mm=float(sensor_h * MM_PER_M),
            width_px=int(res_w),
            height_px=int(res_h),
        )


class SimCamera:
    """A named camera inside a loaded MuJoCo model, plus the scene state it
    renders. Owns the ``MjData`` so callers render a consistent snapshot.
    """

    def __init__(self, model: mujoco.MjModel, camera_name: str = "cam"):
        self.model = model
        self.data = mujoco.MjData(model)
        self.camera_name = camera_name
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.cam_id < 0:
            raise ValueError(f"no camera named {camera_name!r} in model")
        mujoco.mj_forward(self.model, self.data)

    @classmethod
    def from_scene(cls, path: str | Path, camera_name: str = "cam") -> SimCamera:
        return cls(mujoco.MjModel.from_xml_path(str(path)), camera_name)

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics.from_mujoco(self.model, self.cam_id)

    @property
    def position(self) -> NDArray[np.float64]:
        """World position of a fixed camera, metres."""
        return self.model.cam_pos[self.cam_id].copy()

    @position.setter
    def position(self, xyz: NDArray[np.float64]) -> None:
        self.model.cam_pos[self.cam_id] = xyz
        mujoco.mj_forward(self.model, self.data)

    def set_light_intensity(self, intensity: float, light_name: str = "key") -> None:
        """Set a light's diffuse intensity. This is the Phase 1 exposure proxy:
        MuJoCo has no shutter, so brightening the key light stands in for a
        longer exposure. Moves to a scene/lighting module once a second caller
        needs programmatic light control (Phase 4).
        """
        lid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, light_name)
        if lid < 0:
            raise ValueError(f"no light named {light_name!r} in model")
        self.model.light_diffuse[lid] = (intensity, intensity, intensity)
        mujoco.mj_forward(self.model, self.data)
