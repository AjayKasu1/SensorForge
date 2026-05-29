"""Webcam capture via OpenCV VideoCapture.

Frames are returned as float32 RGB in [0, 1] to share one numeric convention
with the renderer. Note the asymmetry this hides: real webcam output is
gamma-encoded sRGB after the camera's own ISP, whereas the sim render is
approximately linear. Comparing the two directly is apples-to-oranges until
the ISP's gamma stage (Phase 2) maps sim intensity into display space. That
gap is the whole point of the calibration loop, not a bug to paper over here.
"""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

# OpenCV inherits V4L2's convention for this property: 0.25 requests manual
# exposure, 0.75 auto. Several backends (notably macOS AVFoundation) ignore it
# entirely, so we set best-effort and report what actually stuck.
CAP_AUTO_EXPOSURE_MANUAL = 0.25


class Webcam:
    """Thin wrapper over a single VideoCapture device."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"could not open webcam at index {index}; "
                "check it is connected and not in use by another app"
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def lock_exposure_and_white_balance(self) -> dict[str, float]:
        """Disable auto-exposure and auto-WB so repeated captures are
        comparable. Returns the property values the driver reports *after* the
        request, since many cameras silently refuse.
        """
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, CAP_AUTO_EXPOSURE_MANUAL)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        achieved = {
            "auto_exposure": self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "auto_wb": self.cap.get(cv2.CAP_PROP_AUTO_WB),
        }
        logger.debug("requested exposure/WB lock; driver reports {}", achieved)
        return achieved

    def grab(self) -> NDArray[np.float32]:
        """Capture one frame as float32 RGB in [0, 1]."""
        ok, frame_bgr = self.cap.read()
        if not ok:
            raise RuntimeError("frame grab failed; device may have disconnected")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb.astype(np.float32) / 255.0

    def grab_frames(self, n: int) -> NDArray[np.float32]:
        """Capture ``n`` frames, stacked as (n, H, W, 3). Useful for temporal
        noise statistics where we need many frames of a static scene.
        """
        return np.stack([self.grab() for _ in range(n)])

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> Webcam:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
