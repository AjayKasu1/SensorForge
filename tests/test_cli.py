from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from sensorforge.cli import main

SCENE = Path(__file__).parent.parent / "scenes" / "checkerboard.xml"


def test_render_command_writes_frame_stack(tmp_path):
    out = tmp_path / "r.npy"
    rc = main(["render", "--scene", str(SCENE), "--frames", "2", "--out", str(out)])
    assert rc == 0
    frames = np.load(out)
    assert frames.shape == (2, 480, 640, 3)
    assert frames.dtype == np.float32


def test_capture_command_writes_frames_and_reference(tmp_path):
    fake_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.return_value = (True, fake_frame)

    out = tmp_path / "c.npy"
    with patch("cv2.VideoCapture", return_value=cap):
        rc = main(["capture", "--target", "uniform", "--frames", "3", "--out", str(out)])
    assert rc == 0
    assert np.load(out).shape == (3, 480, 640, 3)
    # The reference target is saved alongside for calibration provenance.
    ref = out.with_name("c_target_uniform.npy")
    assert ref.exists()
