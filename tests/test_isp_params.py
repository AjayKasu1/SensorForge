import pytest
from pydantic import ValidationError

from sensorforge.isp.params import IDENTITY_CCM, ISPParams


def test_defaults_construct_and_nest():
    p = ISPParams()
    assert p.sensor.full_well_e == 10000.0
    assert p.color.ccm == IDENTITY_CCM
    assert p.optics.vignette_strength == 0.4


def test_round_trip_through_dump():
    p = ISPParams()
    again = ISPParams(**p.model_dump())
    assert again == p


def test_json_schema_exposes_nested_fields():
    # The agent relies on this in Phase 4 to know what it may tune.
    schema = ISPParams.model_json_schema()
    assert "optics" in schema["properties"]
    assert "$defs" in schema and "SensorParams" in schema["$defs"]


@pytest.mark.parametrize(
    "section,field,bad",
    [
        ("sensor", "full_well_e", 0.0),
        ("sensor", "exposure_ms", -1.0),
        ("color", "awb_gain_r", 0.0),
        ("color", "gamma", 0.0),
        ("optics", "vignette_strength", 1.5),
        ("sensor", "black_level", 1.0),
    ],
)
def test_bounds_reject_nonphysical(section, field, bad):
    base = ISPParams().model_dump()
    base[section][field] = bad
    with pytest.raises(ValidationError):
        ISPParams(**base)
