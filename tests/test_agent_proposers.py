import numpy as np

from sensorforge.agent.graph import CalibrationContext, run_calibration
from sensorforge.agent.proposers import HeuristicProposer
from sensorforge.agent.state import AgentState, TunableParams
from sensorforge.agent.tools import capture_real
from sensorforge.isp.params import ISPParams

# The hidden target the heuristic must recover, matching the CLI preset.
HIDDEN = TunableParams(
    exposure_ms=14.0, black_level=0.06, awb_gain_r=1.05, awb_gain_g=1.0, awb_gain_b=1.1, gamma=1.8
)


def test_heuristic_lowers_gain_when_sim_channel_too_bright():
    # sim is much redder than real -> propose a lower red gain.
    sim = np.zeros((8, 8, 3), dtype=np.uint8)
    sim[..., 0] = 220  # strong red
    sim[..., 1] = 120
    sim[..., 2] = 120
    real = np.full((8, 8, 3), 120, dtype=np.uint8)  # neutral
    new, reason = HeuristicProposer().propose(TunableParams(), sim, real, {}, [])
    assert new.awb_gain_r < TunableParams().awb_gain_r
    assert "gray-world" in reason


def test_heuristic_converges_on_uniform_sim_as_real():
    base = ISPParams()
    linear = np.full((96, 96, 3), 0.5, dtype=np.float32)
    rng = np.random.default_rng(0)
    real = capture_real(
        "sim", linear_rgb=linear, hidden_params=HIDDEN.apply_to(base), frames=16, rng=rng
    )
    ctx = CalibrationContext(
        linear_rgb=linear,
        real=real,
        base_params=base,
        proposer=HeuristicProposer(),
        run_dir=str("/tmp/sf_heur_test"),
        rng=rng,
        n_average=16,
    )
    final = run_calibration(ctx, AgentState(max_iters=20, tolerance_de=3.0))
    assert final.stop_reason == "within_tolerance"
    assert final.best.metrics["deltaE2000"] < 3.0
    # baseline was a large color cast
    assert final.history[0].metrics["deltaE2000"] > 10.0
