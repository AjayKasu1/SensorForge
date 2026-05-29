"""Render a SimCamera's scene to a numpy array.

Output convention: float32 RGB in [0, 1], treated as approximately-linear
scene intensity.

On the linearity claim (Engineer B): MuJoCo computes Lambertian shading in
linear space and writes to a plain (non-sRGB) framebuffer, so dividing the
8-bit output by 255 yields values that are close to linear radiance, not
gamma-encoded. We take that as the ISP's input convention. It is an
*assumption*, not a measurement; the ISP's explicit gamma stage in Phase 2 is
where we own the linear-to-display transform, and where this gets revisited.
"""

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import NDArray

from sensorforge.sim.camera import SimCamera


class SimRenderer:
    """Offscreen renderer bound to one camera. Holds a GL context, so close it
    (or use it as a context manager) when done.
    """

    def __init__(self, camera: SimCamera, height: int | None = None, width: int | None = None):
        intr = camera.intrinsics
        self.height = height or intr.height_px
        self.width = width or intr.width_px
        self.camera = camera
        self._renderer = mujoco.Renderer(camera.model, height=self.height, width=self.width)

    def render(self) -> NDArray[np.float32]:
        """Render the current scene state to float32 RGB in [0, 1]."""
        self._renderer.update_scene(self.camera.data, camera=self.camera.camera_name)
        rgb_u8 = self._renderer.render()  # (H, W, 3) uint8
        return rgb_u8.astype(np.float32) / 255.0

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> SimRenderer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
