"""All ISP parameters as a nested pydantic model.

Every field carries its unit in the description; the calibration agent reads
``ISPParams.model_json_schema()`` to know what it may tune and within what
bounds. Defaults are representative of a low-end USB webcam, not a calibrated
target: the whole point of the project is to move them until sim matches real.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Row-major 3x3. Tuples keep it hashable and immutable so a params object can
# be logged/compared without surprise mutation.
Mat3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
IDENTITY_CCM: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class OpticsParams(BaseModel):
    vignette_strength: float = Field(
        0.4, ge=0, le=1, description="cos^4 falloff weight: 0=none, 1=full corner darkening"
    )
    radial_k1: float = Field(-0.15, description="Brown-Conrady radial distortion k1; dimensionless")
    radial_k2: float = Field(0.04, description="radial distortion k2; dimensionless")
    radial_k3: float = Field(0.0, description="radial distortion k3; dimensionless")
    tangential_p1: float = Field(0.0, description="tangential distortion p1; dimensionless")
    tangential_p2: float = Field(0.0, description="tangential distortion p2; dimensionless")


class SensorParams(BaseModel):
    full_well_e: float = Field(10000.0, gt=0, description="full-well capacity; electrons")
    exposure_ms: float = Field(10.0, gt=0, description="integration time; milliseconds")
    dark_current_e_per_s: float = Field(50.0, ge=0, description="dark current; electrons/second")
    read_noise_e: float = Field(3.0, ge=0, description="RMS read noise; electrons")
    black_level: float = Field(
        0.02, ge=0, lt=1, description="optical-black pedestal; normalized [0,1]"
    )


class ColorParams(BaseModel):
    awb_gain_r: float = Field(1.6, gt=0, description="white-balance gain, red; dimensionless")
    awb_gain_g: float = Field(1.0, gt=0, description="white-balance gain, green; dimensionless")
    awb_gain_b: float = Field(1.9, gt=0, description="white-balance gain, blue; dimensionless")
    ccm: Mat3 = Field(IDENTITY_CCM, description="3x3 color-correction matrix; dimensionless")
    gamma: float = Field(
        2.2, gt=0, description="encoding gamma; output = signal ** (1/gamma); dimensionless"
    )


class ISPParams(BaseModel):
    optics: OpticsParams = Field(default_factory=OpticsParams)
    sensor: SensorParams = Field(default_factory=SensorParams)
    color: ColorParams = Field(default_factory=ColorParams)
