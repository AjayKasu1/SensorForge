"""Webcam tests run against a mocked VideoCapture; CI has no camera."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sensorforge.capture.webcam import Webcam


def _fake_capture(frame_bgr):
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.return_value = (True, frame_bgr)
    return cap


def test_grab_converts_bgr_to_rgb_and_normalizes():
    # Distinct channel values so a BGR<->RGB swap is detectable.
    frame_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    frame_bgr[..., 0] = 255  # blue channel in BGR
    with patch("cv2.VideoCapture", return_value=_fake_capture(frame_bgr)):
        cam = Webcam()
        img = cam.grab()
    assert img.shape == (4, 4, 3)
    assert img.dtype == np.float32
    # Blue should land in the last (R,G,B) channel after conversion.
    assert img[0, 0, 2] == pytest.approx(1.0)
    assert img[0, 0, 0] == pytest.approx(0.0)


def test_open_failure_raises():
    cap = MagicMock()
    cap.isOpened.return_value = False
    with patch("cv2.VideoCapture", return_value=cap), pytest.raises(RuntimeError):
        Webcam(index=9)


def test_grab_frames_stacks():
    frame_bgr = np.full((2, 3, 3), 100, dtype=np.uint8)
    with patch("cv2.VideoCapture", return_value=_fake_capture(frame_bgr)):
        cam = Webcam()
        frames = cam.grab_frames(5)
    assert frames.shape == (5, 2, 3, 3)


def test_failed_read_raises():
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.return_value = (False, None)
    with patch("cv2.VideoCapture", return_value=cap):
        cam = Webcam()
        with pytest.raises(RuntimeError):
            cam.grab()
