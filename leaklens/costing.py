"""
Leak rate estimation and cost attribution.

This module turns "there is a noise over there" into "that fitting costs you
Rs. 74,000 a year", which is the output that actually makes a plant manager act.
It is also the part of the project judges will remember.

TWO WAYS TO GET LEAK RATE, use whichever your data supports:

  CALIBRATED  - fit measured ultrasonic amplitude against known flow from your
                orifice-plate rig. Preferred. This is what calibrate() does.

  PHYSICAL    - compute theoretical choked-flow through an equivalent orifice.
                Useful as a sanity check on the calibration, and it gives you a
                defensible equation for the report.

IMPORTANT HONESTY NOTE FOR YOUR REPORT
The energy constants below are industry rules of thumb. Cite a primary source
(the US DOE Compressed Air Systems Sourcebook is the standard reference) and
state your assumed electricity tariff explicitly. Do not present these numbers
as measured when they are estimated - a judge who knows the field will ask, and
"we assumed X, sourced from Y" is a much better answer than a confident wrong one.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict

import numpy as np

# --- assumptions, all overridable -----------------------------------------
KW_PER_CFM_AT_7BAR = 0.20        # compressor shaft power per CFM of free air
HOURS_PER_YEAR = 8760.0
DEFAULT_TARIFF_INR_PER_KWH = 8.0
DEFAULT_DUTY_CYCLE = 1.0         # fraction of the year the line is pressurised

LPM_PER_CFM = 28.3168


@dataclass
class LeakEstimate:
    flow_lpm: float
    flow_cfm: float
    equivalent_diameter_mm: float
    power_kw: float
    annual_kwh: float
    annual_cost_inr: float
    confidence: str               # "calibrated" | "physical" | "uncalibrated"
    assumptions: dict

    def summary(self) -> str:
        return (f"{self.equivalent_diameter_mm:.1f} mm equivalent leak, "
                f"{self.flow_lpm:.1f} L/min, "
                f"Rs. {self.annual_cost_inr:,.0f}/year")

    def to_json(self, **kw):
        return json.dumps(asdict(self), indent=2, **kw)


# ==========================================================================
# physical model
# ==========================================================================

def choked_flow_lpm(diameter_mm, gauge_bar=7.0, cd=0.62, temp_k=293.0):
    """
    Free-air flow through a sharp-edged orifice, choked (which it is for any
    gauge pressure above about 0.9 bar).

    Standard compressible choked-flow relation for air:
        m_dot = Cd * A * P0 * sqrt(gamma / (R * T0)) * (2/(gamma+1))^((gamma+1)/(2(gamma-1)))

    Returned in litres/minute of FREE AIR at 1 atm, which is how compressor
    capacity and leak surveys are always quoted.
    """
    gamma = 1.4
    R = 287.05                       # J/(kg K) for air
    p0 = (gauge_bar + 1.01325) * 1e5  # absolute, Pa
    area = math.pi * (diameter_mm * 1e-3) ** 2 / 4.0

    k = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    m_dot = cd * area * p0 * math.sqrt(gamma / (R * temp_k)) * k   # kg/s

    rho_free = 1.204                 # kg/m3 at 20 C, 1 atm
    m3_per_s = m_dot / rho_free
    return m3_per_s * 1000.0 * 60.0


def diameter_from_flow_mm(flow_lpm, gauge_bar=7.0, cd=0.62):
    """Invert choked_flow_lpm. Flow scales with area, so d ~ sqrt(flow)."""
    ref = choked_flow_lpm(1.0, gauge_bar, cd)
    if ref <= 0:
        return float("nan")
    return math.sqrt(max(flow_lpm, 0.0) / ref)


# ==========================================================================
# amplitude -> flow calibration
# ==========================================================================

class AmplitudeCalibration:
    """
    Fits log(flow) = a * log(amplitude_at_1m) + b.

    Amplitude is first range-corrected back to a nominal 1 m using spherical
    spreading plus atmospheric absorption, so readings taken at different
    distances become comparable. This correction is the reason the ToF sensor
    earns its place in the BOM.
    """

    ABSORPTION_DB_PER_M = 1.2        # ~40 kHz, 20 C, 50% RH

    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b

    # --- range correction ---------------------------------------------------
    @classmethod
    def to_1m(cls, amplitude, distance_m):
        d = max(float(distance_m), 0.3)
        spreading = d ** 2                                   # 1/r^2 undone
        absorption = 10 ** (cls.ABSORPTION_DB_PER_M * d / 20.0)
        return float(amplitude) * spreading * absorption

    # --- fitting ------------------------------------------------------------
    def fit(self, amplitudes, distances, flows_lpm):
        amps1m = np.array([self.to_1m(a, d)
                           for a, d in zip(amplitudes, distances)])
        x = np.log(np.clip(amps1m, 1e-12, None))
        y = np.log(np.clip(np.asarray(flows_lpm, dtype=float), 1e-6, None))
        if x.size < 2:
            raise ValueError("need at least two calibration points")
        self.a, self.b = np.polyfit(x, y, 1)
        resid = y - (self.a * x + self.b)
        return dict(a=float(self.a), b=float(self.b),
                    rms_log_error=float(np.sqrt(np.mean(resid ** 2))),
                    n_points=int(x.size))

    def predict_lpm(self, amplitude, distance_m):
        if self.a is None:
            raise RuntimeError("calibration not fitted")
        amp1m = self.to_1m(amplitude, distance_m)
        return float(math.exp(self.a * math.log(max(amp1m, 1e-12)) + self.b))

    # --- persistence --------------------------------------------------------
    def save(self, path):
        with open(path, "w") as f:
            json.dump(dict(a=self.a, b=self.b), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["a"], d["b"])


# ==========================================================================
# top level
# ==========================================================================

def estimate(amplitude, distance_m, calibration: AmplitudeCalibration | None = None,
             gauge_bar=7.0, tariff=DEFAULT_TARIFF_INR_PER_KWH,
             duty_cycle=DEFAULT_DUTY_CYCLE,
             kw_per_cfm=KW_PER_CFM_AT_7BAR) -> LeakEstimate:
    """
    Amplitude + distance -> full cost estimate.

    Falls back gracefully if no calibration exists yet, so the UI works from
    day one and simply reports lower confidence.
    """
    if calibration is not None and calibration.a is not None:
        flow_lpm = calibration.predict_lpm(amplitude, distance_m)
        confidence = "calibrated"
    else:
        # crude placeholder so the pipeline runs before calibration day.
        # DO NOT ship a demo relying on this branch.
        amp1m = AmplitudeCalibration.to_1m(amplitude, distance_m)
        flow_lpm = 12.0 * max(amp1m, 0.0)
        confidence = "uncalibrated"

    flow_cfm = flow_lpm / LPM_PER_CFM
    power_kw = flow_cfm * kw_per_cfm
    annual_kwh = power_kw * HOURS_PER_YEAR * duty_cycle
    cost = annual_kwh * tariff

    return LeakEstimate(
        flow_lpm=float(flow_lpm),
        flow_cfm=float(flow_cfm),
        equivalent_diameter_mm=float(diameter_from_flow_mm(flow_lpm, gauge_bar)),
        power_kw=float(power_kw),
        annual_kwh=float(annual_kwh),
        annual_cost_inr=float(cost),
        confidence=confidence,
        assumptions=dict(gauge_bar=gauge_bar, tariff_inr_per_kwh=tariff,
                         duty_cycle=duty_cycle, kw_per_cfm=kw_per_cfm,
                         distance_m=float(distance_m)),
    )


def blowdown_flow_lpm(volume_l, p_start_bar, p_end_bar, seconds, temp_k=293.0):
    """
    GROUND TRUTH measurement for your calibration rig, no flow meter needed.

    Seal a vessel of known volume at a known pressure, open the orifice, time the
    decay to a second pressure. Ideal gas law gives you the free air released.

        V_free = V_tank * (P_start - P_end) / P_atm

    Do this for each orifice size and you have real labels for the regressor.
    Cheap, rigorous, and it makes your results defensible in the report.
    """
    p_atm = 1.01325
    free_litres = volume_l * (p_start_bar - p_end_bar) / p_atm
    return free_litres / (seconds / 60.0)
