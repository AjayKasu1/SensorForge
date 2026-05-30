"""Agent state: the six tunable knobs and the calibration run record.

Only the color/exposure chain is tunable (ADR 006). Numpy frames are kept out
of this state so it stays JSON-serializable for the run log; images live in the
graph context instead.
"""

from __future__ import annotations

import math

from annotated_types import Ge, Gt, Le, Lt
from pydantic import BaseModel, Field

from sensorforge.isp.params import ISPParams


class TunableParams(BaseModel):
    """The knobs the agent may change, with physical bounds. Bounds double as
    the clamp range for LLM proposals (see ``clamp_proposal``).
    """

    exposure_ms: float = Field(10.0, gt=0, le=100, description="integration time; ms")
    black_level: float = Field(0.02, ge=0, lt=0.5, description="optical-black pedestal; [0,1]")
    awb_gain_r: float = Field(1.6, gt=0, le=4, description="white-balance gain, red")
    awb_gain_g: float = Field(1.0, gt=0, le=4, description="white-balance gain, green")
    awb_gain_b: float = Field(1.9, gt=0, le=4, description="white-balance gain, blue")
    gamma: float = Field(2.2, gt=1, le=3.5, description="encoding gamma")

    @classmethod
    def from_isp(cls, p: ISPParams) -> TunableParams:
        return cls(
            exposure_ms=p.sensor.exposure_ms,
            black_level=p.sensor.black_level,
            awb_gain_r=p.color.awb_gain_r,
            awb_gain_g=p.color.awb_gain_g,
            awb_gain_b=p.color.awb_gain_b,
            gamma=p.color.gamma,
        )

    def apply_to(self, base: ISPParams) -> ISPParams:
        """Return a copy of ``base`` with the tunable knobs overwritten; fixed
        params (full well, noise, optics, CCM) are left untouched.
        """
        p = base.model_copy(deep=True)
        p.sensor.exposure_ms = self.exposure_ms
        p.sensor.black_level = self.black_level
        p.color.awb_gain_r = self.awb_gain_r
        p.color.awb_gain_g = self.awb_gain_g
        p.color.awb_gain_b = self.awb_gain_b
        p.color.gamma = self.gamma
        return p


def _bounds(field) -> tuple[float, float]:
    """Inclusive (lo, hi) clamp range from a field's annotated-types metadata.
    Exclusive bounds (gt/lt) step one float inward so a clamped value validates.
    """
    lo, hi = -math.inf, math.inf
    for m in field.metadata:
        if isinstance(m, Gt):
            lo = max(lo, math.nextafter(m.gt, math.inf))
        elif isinstance(m, Ge):
            lo = max(lo, m.ge)
        elif isinstance(m, Lt):
            hi = min(hi, math.nextafter(m.lt, -math.inf))
        elif isinstance(m, Le):
            hi = min(hi, m.le)
    return lo, hi


def clamp_proposal(raw: dict, base: TunableParams) -> TunableParams:
    """Build a TunableParams from a messy LLM proposal: clamp known keys to
    their bounds, ignore unknown keys, keep ``base`` for anything missing.
    """
    values = base.model_dump()
    for name, field in TunableParams.model_fields.items():
        if name not in raw:
            continue
        lo, hi = _bounds(field)
        values[name] = min(max(float(raw[name]), lo), hi)
    return TunableParams(**values)


class Attempt(BaseModel):
    iteration: int
    params: TunableParams
    metrics: dict[str, float]


class Assumption(BaseModel):
    timestamp: str
    parameter: str
    value: float
    justification: str


class AgentState(BaseModel):
    target: str = "uniform"
    real_source: str = "sim"  # "sim" (hidden-params) or "webcam"
    current: TunableParams = Field(default_factory=TunableParams)
    history: list[Attempt] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    iteration: int = 0
    max_iters: int = 20
    tolerance_de: float = 3.0
    best: Attempt | None = None
    stop_reason: str | None = None
    run_dir: str | None = None
