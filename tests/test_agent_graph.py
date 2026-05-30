import numpy as np

from sensorforge.agent.graph import CalibrationContext, run_calibration
from sensorforge.agent.proposers import LLMProposer
from sensorforge.agent.state import AgentState, TunableParams
from sensorforge.agent.tools import capture_real
from sensorforge.isp.params import ISPParams


class ScriptedLLM:
    """Returns a fixed diagnosis, and proposals from a list (then no-change)."""

    def __init__(self, proposals):
        self.proposals = list(proposals)

    def generate(self, messages):
        if "JSON only" in messages[-1].content:
            return self.proposals.pop(0) if self.proposals else "{}"
        return "the red channel is too strong; lower awb_gain_r"


def _low_noise_base():
    # High full well + low read noise so renders are near-deterministic and the
    # deltaE is dominated by the parameter gap, not noise.
    p = ISPParams()
    p.sensor.full_well_e = 1_000_000.0
    p.sensor.read_noise_e = 0.5
    p.sensor.dark_current_e_per_s = 0.0
    return p


def _context(llm, run_dir, hidden_awb_r=1.0):
    base = _low_noise_base()
    linear = np.full((48, 48, 3), 0.5, dtype=np.float32)
    hidden = base.model_copy(deep=True)
    hidden.color.awb_gain_r = hidden_awb_r
    rng = np.random.default_rng(0)
    real = capture_real("sim", linear_rgb=linear, hidden_params=hidden, rng=rng)
    return CalibrationContext(
        linear_rgb=linear,
        real=real,
        base_params=base,
        proposer=LLMProposer(llm),
        run_dir=str(run_dir),
        rng=rng,
    )


def test_loop_converges_and_logs_assumptions(tmp_path):
    # Real is rendered with awb_gain_r=1.0; agent starts at 1.6 and is scripted
    # to walk toward 1.0.
    llm = ScriptedLLM([f'{{"awb_gain_r": {v}}}' for v in (1.4, 1.2, 1.05, 1.0)])
    ctx = _context(llm, tmp_path, hidden_awb_r=1.0)
    final = run_calibration(ctx, AgentState(max_iters=20, tolerance_de=3.0))

    assert final.stop_reason == "within_tolerance"
    assert final.best.metrics["deltaE2000"] <= 3.0
    # deltaE improved from first to best.
    assert final.history[0].metrics["deltaE2000"] > final.best.metrics["deltaE2000"]
    # Every accepted change was logged.
    assert (tmp_path / "assumptions.md").exists()
    assert "awb_gain_r" in (tmp_path / "assumptions.md").read_text()


def test_loop_stops_when_stalled(tmp_path):
    # LLM never changes anything; best is set at iter 0 and never improves.
    ctx = _context(ScriptedLLM([]), tmp_path, hidden_awb_r=1.0)
    final = run_calibration(ctx, AgentState(max_iters=20, tolerance_de=0.001))
    assert final.stop_reason == "stalled"
    assert final.iteration - final.best.iteration >= 3


class _DeadLLM:
    """Always fails, like an exhausted API quota."""

    def generate(self, messages):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")


def test_loop_stops_gracefully_when_proposer_fails(tmp_path, monkeypatch):
    # No real backoff sleeps during the test.
    monkeypatch.setattr("sensorforge.agent.tools.time.sleep", lambda s: None)
    from sensorforge.agent.proposers import LLMProposer

    ctx = _context(_DeadLLM(), tmp_path, hidden_awb_r=1.0)
    ctx.proposer = LLMProposer(_DeadLLM())
    final = run_calibration(ctx, AgentState(max_iters=20, tolerance_de=0.1))
    # It did not crash; it ended with the baseline as best.
    assert final.stop_reason == "proposer_unavailable"
    assert final.best is not None
    assert len(final.history) == 1  # measured once, proposer failed, stopped


def test_loop_stops_immediately_when_already_matched(tmp_path):
    # Hidden params equal the starting params, so iteration 0 is within tolerance.
    ctx = _context(ScriptedLLM([]), tmp_path, hidden_awb_r=TunableParams().awb_gain_r)
    final = run_calibration(ctx, AgentState(max_iters=20, tolerance_de=3.0))
    assert final.stop_reason == "within_tolerance"
    assert len(final.history) == 1
