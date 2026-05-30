import pytest
from pydantic import ValidationError

from sensorforge.agent.state import AgentState, TunableParams, clamp_proposal
from sensorforge.isp.params import ISPParams


def test_apply_to_sets_tunable_and_preserves_fixed():
    base = ISPParams()
    base.optics.vignette_strength = 0.33  # a fixed param
    t = TunableParams(exposure_ms=20.0, awb_gain_r=2.0)
    out = t.apply_to(base)
    assert out.sensor.exposure_ms == 20.0
    assert out.color.awb_gain_r == 2.0
    assert out.optics.vignette_strength == 0.33  # untouched
    # base is not mutated
    assert base.sensor.exposure_ms == 10.0


def test_from_isp_round_trip():
    p = ISPParams()
    p.color.gamma = 2.4
    assert TunableParams.from_isp(p).gamma == 2.4


def test_clamp_proposal_bounds_and_filters():
    base = TunableParams()
    raw = {
        "gamma": 99.0,  # above le=3.5 -> clamps to 3.5
        "awb_gain_r": -5.0,  # below gt=0 -> clamps just above 0
        "exposure_ms": 30.0,  # in range
        "bogus_key": 1.0,  # ignored
    }
    out = clamp_proposal(raw, base)
    assert out.gamma == 3.5
    assert out.awb_gain_r > 0.0  # clamped just above the exclusive lower bound
    assert out.exposure_ms == 30.0
    assert out.black_level == base.black_level  # untouched
    assert not hasattr(out, "bogus_key")


def test_clamped_proposal_always_validates():
    # Even an absurd proposal must produce a valid (constructible) object.
    out = clamp_proposal({"black_level": 10.0, "exposure_ms": -1.0}, TunableParams())
    assert 0 <= out.black_level < 0.5
    assert 0 < out.exposure_ms <= 100


def test_direct_construction_still_rejects_out_of_bounds():
    with pytest.raises(ValidationError):
        TunableParams(gamma=0.5)  # gamma must be > 1


def test_agent_state_defaults():
    s = AgentState()
    assert s.iteration == 0 and s.max_iters == 20 and s.tolerance_de == 3.0
    assert s.real_source == "sim"
