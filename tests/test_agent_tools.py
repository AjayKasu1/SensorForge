import numpy as np
import pytest

from sensorforge.agent.state import Assumption, TunableParams
from sensorforge.agent.tools import (
    capture_real,
    compute_metrics,
    propose_param_update,
    render_sim,
    write_assumption,
)
from sensorforge.isp.params import ISPParams


class FakeLLM:
    """Returns canned outputs in order; records the prompts it saw."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.seen = []

    def generate(self, messages):
        self.seen.append(messages)
        return self.outputs.pop(0)


def _linear(level=0.5):
    return np.full((16, 16, 3), level, dtype=np.float32)


def test_render_sim_returns_uint8_frame():
    out = render_sim(_linear(), ISPParams(), rng=np.random.default_rng(0))
    assert out.shape == (16, 16, 3) and out.dtype == np.uint8


def test_capture_real_sim_matches_forward():
    lin = _linear()
    hidden = ISPParams()
    a = capture_real("sim", linear_rgb=lin, hidden_params=hidden, rng=np.random.default_rng(1))
    b = render_sim(lin, hidden, rng=np.random.default_rng(1))
    assert np.array_equal(a, b)


@pytest.mark.filterwarnings("ignore:divide by zero")
def test_compute_metrics_keys_and_identity():
    img = (np.random.default_rng(2).random((16, 16, 3)) * 255).astype(np.uint8)
    m = compute_metrics(img, img)
    assert set(m) == {"SSIM", "PSNR", "deltaE2000"}
    assert m["SSIM"] == 1.0 and m["deltaE2000"] == 0.0


def test_propose_param_update_parses_clamps_and_keeps_rest():
    llm = FakeLLM(["white balance too red", '{"awb_gain_r": 1.0, "gamma": 99}'])
    current = TunableParams()
    new, diagnosis = propose_param_update(llm, current, {"deltaE2000": 8.0}, history=[])
    assert diagnosis == "white balance too red"
    assert new.awb_gain_r == 1.0  # applied
    assert new.gamma == 3.5  # clamped to bound
    assert new.black_level == current.black_level  # untouched
    assert len(llm.seen) == 2  # diagnosis then proposal


def test_write_assumption_appends(tmp_path):
    a1 = Assumption(timestamp="t0", parameter="gamma", value=2.4, justification="too dark")
    a2 = Assumption(timestamp="t1", parameter="awb_gain_r", value=1.2, justification="too red")
    write_assumption(tmp_path, a1)
    path = write_assumption(tmp_path, a2)
    text = path.read_text()
    assert text.startswith("# Calibration assumptions")
    assert "gamma = 2.4" in text and "awb_gain_r = 1.2" in text
    assert text.count("# Calibration assumptions") == 1  # header written once
