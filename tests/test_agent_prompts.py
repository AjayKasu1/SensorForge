from sensorforge.agent.prompts import (
    build_diagnosis_messages,
    build_proposal_messages,
    parse_proposal,
)
from sensorforge.agent.state import Attempt, TunableParams


def test_diagnosis_messages_carry_metrics_and_history():
    history = [Attempt(iteration=0, params=TunableParams(), metrics={"deltaE2000": 9.1})]
    msgs = build_diagnosis_messages(TunableParams(), {"deltaE2000": 7.5, "SSIM": 0.9}, history)
    assert msgs[0].role == "system"
    assert "7.5" in msgs[1].content
    assert "iter 0" in msgs[1].content  # history included


def test_proposal_messages_list_knobs_and_bounds():
    msgs = build_proposal_messages(TunableParams(), "white balance is too red")
    body = msgs[1].content
    # Every tunable knob name appears with its range.
    for knob in ("exposure_ms", "black_level", "awb_gain_r", "gamma"):
        assert knob in body
    assert "JSON only" in body


def test_parse_plain_json():
    assert parse_proposal('{"awb_gain_r": 1.2}') == {"awb_gain_r": 1.2}


def test_parse_fenced_json_with_prose():
    text = 'Here is my proposal:\n```json\n{"gamma": 2.4, "black_level": 0.05}\n```\nDone.'
    assert parse_proposal(text) == {"gamma": 2.4, "black_level": 0.05}


def test_parse_garbage_returns_empty():
    assert parse_proposal("I cannot help with that.") == {}
    assert parse_proposal("{not valid json}") == {}
