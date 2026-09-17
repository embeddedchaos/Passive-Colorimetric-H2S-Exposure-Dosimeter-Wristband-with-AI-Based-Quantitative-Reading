"""
calibration.py
Maps corrected strip color change (Delta E from the unexposed baseline) to
an estimated cumulative H2S dose (ppm*hr), and back.

The problem statement explicitly flags that colorimetric reactions don't
scale linearly at very low concentrations or over long durations. Lead/metal
sulfide staining reactions are classic diffusion + saturation-limited
processes, so we model dose -> color with a Hill-type saturating curve
rather than a straight line:

    deltaE(dose) = deltaE_max * dose^n / (K^n + dose^n)

This is monotonic, so it can be inverted numerically (bisection) to go from
a measured deltaE back to an estimated dose. Parameters (deltaE_max, K, n)
are fit from your lab-controlled exposure trials (known concentration x
known duration -> photograph -> measured deltaE) and are stored in a JSON
file so the model can be swapped/re-fit as real chemistry data comes in,
without touching the pipeline code.
"""

import json
import numpy as np
from scipy.optimize import curve_fit, brentq


def hill_model(dose, deltaE_max, K, n):
    dose = np.asarray(dose, dtype=np.float64)
    return deltaE_max * np.power(dose, n) / (np.power(K, n) + np.power(dose, n))


class CalibrationModel:
    def __init__(self, deltaE_max: float, K: float, n: float,
                 fit_residual_std: float = 0.0, dose_unit: str = "ppm*hr"):
        self.deltaE_max = deltaE_max
        self.K = K
        self.n = n
        self.fit_residual_std = fit_residual_std  # residual scatter of the lab fit
        self.dose_unit = dose_unit

   
    def forward(self, dose):
        return hill_model(dose, self.deltaE_max, self.K, self.n)

    
    def invert(self, deltaE: float, dose_search_max: float = 1e5):
        deltaE = float(np.clip(deltaE, 0.0, self.deltaE_max * 0.999))
        if deltaE <= 1e-9:
            return 0.0

        f = lambda d: hill_model(d, self.deltaE_max, self.K, self.n) - deltaE
        f_lo = f(1e-6)
        f_hi = f(dose_search_max)

        if f_lo >= 0:
            # Even a near-zero dose already meets/exceeds this deltaE (can
            # happen once noise is subtracted down near the measurement
            # floor) -> report as ~0, not as a runaway high dose.
            return 0.0
        if f_hi <= 0:
            # deltaE is outside what any dose up to dose_search_max could
            # produce -> genuinely saturated at the top of the curve.
            return dose_search_max

        return brentq(f, 1e-6, dose_search_max)

    
    @classmethod
    def fit_from_trials(cls, doses: np.ndarray, deltaEs: np.ndarray, dose_unit="ppm*hr"):
        """
        doses, deltaEs: arrays of (known cumulative dose, measured deltaE)
        pairs from controlled lab exposure trials at known concentration and
        duration (the PS's "controlled, lab-simulated H2S exposure").
        """
        doses = np.asarray(doses, dtype=np.float64)
        deltaEs = np.asarray(deltaEs, dtype=np.float64)

        # sane initial guesses
        p0 = [deltaEs.max() * 1.1, np.median(doses), 1.0]
        bounds = ([1e-3, 1e-3, 0.1], [200, 1e5, 5])

        popt, _ = curve_fit(hill_model, doses, deltaEs, p0=p0, bounds=bounds, maxfev=20000)
        deltaE_max, K, n = popt

        residuals = deltaEs - hill_model(doses, *popt)
        residual_std = float(np.std(residuals))

        return cls(deltaE_max, K, n, fit_residual_std=residual_std, dose_unit=dose_unit)

    
    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "deltaE_max": self.deltaE_max,
                "K": self.K,
                "n": self.n,
                "fit_residual_std": self.fit_residual_std,
                "dose_unit": self.dose_unit,
            }, f, indent=2)

    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            d = json.load(f)
        d = {k: v for k, v in d.items() if not k.startswith("_")}
        return cls(**d)

    def dose_uncertainty(self, deltaE: float, deltaE_measurement_noise: float) -> tuple:
        """
        Propagate uncertainty from two sources into a dose range:
          1. fit_residual_std   - how well the Hill curve fit the lab trials
          2. deltaE_measurement_noise - patch color noise + CCM fit residual
             for *this specific photo*, passed in by the pipeline
        Returns (dose_low, dose_est, dose_high) — a band, not a single number,
        per the PS requirement that this be presented as an estimate.
        """
        total_deltaE_noise = float(np.sqrt(self.fit_residual_std**2 + deltaE_measurement_noise**2))
        dose_est = self.invert(deltaE)
        dose_low = self.invert(max(deltaE - total_deltaE_noise, 1e-6))
        dose_high = self.invert(deltaE + total_deltaE_noise)
        return dose_low, dose_est, dose_high
