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


class _ConvergingLLM:
    """Proposes the hidden preset directly so the loop converges; no network."""

    def __init__(self, proposal_json):
        self._proposal = proposal_json

    def generate(self, messages):
        if "JSON only" in messages[-1].content:
            return self._proposal
        return "match the reference white balance and tone"


def test_calibrate_sim_runs_and_writes_artifacts(tmp_path, monkeypatch):
    import json

    from sensorforge import cli

    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "runs")
    proposal = json.dumps(cli.SIM_REAL_PRESET.model_dump())
    monkeypatch.setattr(cli, "make_llm_client", lambda *a, **k: _ConvergingLLM(proposal))

    rc = main(
        [
            "calibrate",
            "--real-source",
            "sim",
            "--scene",
            str(SCENE),
            "--max-iters",
            "6",
            "--avg",
            "8",
        ]
    )
    assert rc == 0

    runs = list((tmp_path / "runs").glob("*/"))
    assert len(runs) == 1
    rd = runs[0]
    assert (rd / "report.md").exists()
    assert (rd / "assumptions.md").exists()
    assert (rd / "state.json").exists()
    # The agent recovered the preset, so the gap closed under tolerance.
    state = json.loads((rd / "state.json").read_text())
    assert state["stop_reason"] == "within_tolerance"
    assert state["best"]["metrics"]["deltaE2000"] < 3.0
