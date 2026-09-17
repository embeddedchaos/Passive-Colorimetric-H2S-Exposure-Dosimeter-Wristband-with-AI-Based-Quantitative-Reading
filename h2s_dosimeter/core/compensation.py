"""
compensation.py
Corrects the sample strip's color-change reading for temperature/humidity
driven drift that is NOT caused by H2S exposure, per the PS requirement:
"Temperature and humidity affect reaction speed too, so the strip design
needs to account for that, either through a sealed reference cell or a
compensation method in the app."

Two independent mechanisms are implemented; use whichever the hardware
supports (or both, taking the smaller correction so we never over-correct):

1. Reference-cell subtraction (primary, hardware-driven):
   A second patch of the identical chemistry, sealed from gas but exposed to
   the same ambient temperature/humidity as the sample strip. Any color
   change it shows is drift, not dose, and gets subtracted from the sample.

2. Temp/humidity model correction (software-only fallback, if a badge has
   no sealed cell, or as a cross-check): an Arrhenius-style rate adjustment
   using logged temperature and a linear humidity term, both either entered
   manually by the safety officer or pulled from a printed thermochromic /
   humidity-indicator patch read by the same photo.
"""

import numpy as np


def reference_cell_compensation(deltaE_sample: float, deltaE_control: float,
                                 alpha: float = 1.0) -> float:
    """
    Subtract control-cell drift from the sample reading.

    alpha : how much of the control cell's drift is attributable to the
            sample (1.0 = full 1:1 subtraction, the default assumption that
            both patches sit under identical conditions). Tune alpha from
            side-by-side lab trials if the two patches don't drift identically.
    """
    corrected = deltaE_sample - alpha * max(deltaE_control, 0.0)
    return max(corrected, 0.0)


def arrhenius_rate_factor(temp_celsius: float, ref_temp_celsius: float = 25.0,
                           activation_energy_k: float = 5000.0) -> float:
    """
    Relative reaction-rate multiplier vs. a reference temperature, using a
    simplified Arrhenius form. activation_energy_k is Ea/R in Kelvin
    (a placeholder until your chemistry's actual activation energy is
    measured from lab trials — swap this constant once you have it).
    """
    T = temp_celsius + 273.15
    T_ref = ref_temp_celsius + 273.15
    return float(np.exp(activation_energy_k * (1.0 / T_ref - 1.0 / T)))


def humidity_rate_factor(relative_humidity_pct: float, ref_rh_pct: float = 50.0,
                          humidity_sensitivity: float = 0.01) -> float:
    """
    Simple linear humidity sensitivity multiplier. humidity_sensitivity is
    the fractional rate change per % RH away from the reference; fit this
    from controlled-humidity lab trials.
    """
    return float(1.0 + humidity_sensitivity * (relative_humidity_pct - ref_rh_pct))


def temp_humidity_compensation(deltaE_sample: float, temp_celsius: float,
                                relative_humidity_pct: float,
                                ref_temp_celsius: float = 25.0,
                                ref_rh_pct: float = 50.0,
                                activation_energy_k: float = 5000.0,
                                humidity_sensitivity: float = 0.01) -> float:
    """
    Normalize a measured deltaE back to what it would have read at reference
    conditions (25 C, 50% RH), by dividing out the combined rate multiplier.
    Use this only when no sealed reference cell is available on the badge.
    """
    rate = (arrhenius_rate_factor(temp_celsius, ref_temp_celsius, activation_energy_k)
            * humidity_rate_factor(relative_humidity_pct, ref_rh_pct, humidity_sensitivity))
    rate = max(rate, 1e-3)
    return deltaE_sample / rate


def combined_compensation(deltaE_sample: float, deltaE_control: float = None,
                           temp_celsius: float = None, relative_humidity_pct: float = None,
                           alpha: float = 1.0) -> float:
    """
    Applies reference-cell compensation if a control-cell reading is given;
    otherwise falls back to the temp/humidity model if env data is given;
    otherwise returns the raw sample reading uncompensated (and the caller
    should widen the reported uncertainty band accordingly).
    """
    if deltaE_control is not None:
        return reference_cell_compensation(deltaE_sample, deltaE_control, alpha)
    if temp_celsius is not None and relative_humidity_pct is not None:
        return temp_humidity_compensation(deltaE_sample, temp_celsius, relative_humidity_pct)
    return deltaE_sample
