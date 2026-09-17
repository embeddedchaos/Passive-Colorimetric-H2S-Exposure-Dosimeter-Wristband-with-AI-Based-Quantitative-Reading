"""
pipeline.py
Orchestrates the full read: one photograph of the wristband card in, one
structured occupational-exposure report out.

    photo -> extract patches -> fit CCM from reference swatches
          -> correct sample/control/expiry colors -> deltaE vs fresh baseline
          -> environmental (reference-cell) compensation
          -> dose estimate with uncertainty band (calibration model)
          -> expiry validity check
          -> JSON record tagged with worker ID + shift
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import numpy as np

from . import color_utils as cu
from . import compensation as comp
from .calibration import CalibrationModel
from .expiry import evaluate_expiry, ExpiryResult


@dataclass
class DoseReport:
    worker_id: str
    shift: str
    timestamp: str
    dose_estimate_ppm_hr: Optional[float]
    dose_low_ppm_hr: Optional[float]
    dose_high_ppm_hr: Optional[float]
    ccm_fit_residual: float
    deltaE_sample_raw: float
    deltaE_sample_compensated: float
    deltaE_control: float
    expiry_status: str
    expiry_valid: bool
    expiry_message: str
    reading_reliable: bool
    warnings: list


class DosimeterPipeline:
    # Above this CCM fit residual (in RGB units), the lighting correction
    # itself can't be trusted enough to certify a dose figure -> reject the
    # reading and ask for a retake rather than report a number.
    CCM_RESIDUAL_REJECT_THRESHOLD = 10.0

    def __init__(self, layout_config_path: str, dose_calibration_path: str,
                 expiry_calibration_path: str):
        with open(layout_config_path) as f:
            self.layout = json.load(f)
        self.dose_model = CalibrationModel.from_json(dose_calibration_path)
        with open(expiry_calibration_path) as f:
            self.expiry_cfg = json.load(f)

        self.sample_fresh_rgb = np.array(self.layout["sample_strip_fresh_rgb"], dtype=np.float64)
        self.expiry_fresh_rgb = np.array(self.layout["expiry_patch_fresh_rgb"], dtype=np.float64)

    # -----------------------------------------------------------------
    def _extract_and_correct(self, image_rgb: np.ndarray):
        warnings = []

        # 1. Reference swatches -> fit lighting correction matrix
        measured, true = [], []
        for sw in self.layout["reference_swatches"]:
            mean_rgb, _ = cu.extract_patch_rgb(image_rgb, tuple(sw["bbox_norm"]))
            measured.append(mean_rgb)
            true.append(sw["true_rgb"])
        measured = np.array(measured)
        true = np.array(true, dtype=np.float64)

        M = cu.fit_color_correction_matrix(measured, true)
        ccm_residual = cu.ccm_fit_residual(measured, true, M)
        if ccm_residual > 12.0:
            warnings.append(
                f"High lighting-correction residual ({ccm_residual:.1f}). "
                "Retake photo: check glare, shadow, or non-flat card angle.")

        # 2. Sample strip
        sample_mean_rgb, sample_std_rgb = cu.extract_patch_rgb(
            image_rgb, tuple(self.layout["sample_strip"]["bbox_norm"]))
        sample_corrected_rgb = cu.apply_color_correction(sample_mean_rgb, M)
        sample_lab = cu.rgb_to_lab(sample_corrected_rgb)
        sample_fresh_lab = cu.rgb_to_lab(self.sample_fresh_rgb)
        deltaE_sample = cu.delta_e_cie76(sample_lab, sample_fresh_lab)

        # 3. Control (sealed reference) cell
        control_mean_rgb, control_std_rgb = cu.extract_patch_rgb(
            image_rgb, tuple(self.layout["control_cell"]["bbox_norm"]))
        control_corrected_rgb = cu.apply_color_correction(control_mean_rgb, M)
        control_lab = cu.rgb_to_lab(control_corrected_rgb)
        deltaE_control = cu.delta_e_cie76(control_lab, sample_fresh_lab)

        # 4. Expiry patch
        expiry_mean_rgb, _ = cu.extract_patch_rgb(
            image_rgb, tuple(self.layout["expiry_patch"]["bbox_norm"]))
        expiry_corrected_rgb = cu.apply_color_correction(expiry_mean_rgb, M)
        expiry_lab = cu.rgb_to_lab(expiry_corrected_rgb)
        expiry_fresh_lab = cu.rgb_to_lab(self.expiry_fresh_rgb)
        deltaE_expiry = cu.delta_e_cie76(expiry_lab, expiry_fresh_lab)

        # patch-noise contribution to measurement uncertainty (propagate std->Lab roughly via RGB norm)
        patch_noise = float(np.linalg.norm(sample_std_rgb)) * 0.5  # empirical scale factor RGB->deltaE

        return {
            "deltaE_sample": deltaE_sample,
            "deltaE_control": deltaE_control,
            "deltaE_expiry": deltaE_expiry,
            "ccm_residual": ccm_residual,
            "patch_noise": patch_noise,
            "warnings": warnings,
        }

    # -----------------------------------------------------------------
    def run(self, image_rgb: np.ndarray, worker_id: str, shift: str,
            manufacture_date: datetime, temp_celsius: Optional[float] = None,
            relative_humidity_pct: Optional[float] = None) -> DoseReport:

        extracted = self._extract_and_correct(image_rgb)
        warnings = list(extracted["warnings"])

        # Environmental compensation: prefer sealed reference cell; fall back
        # to temp/humidity model if env readings were supplied instead.
        deltaE_compensated = comp.combined_compensation(
            deltaE_sample=extracted["deltaE_sample"],
            deltaE_control=extracted["deltaE_control"],
            temp_celsius=temp_celsius,
            relative_humidity_pct=relative_humidity_pct,
        )
        if extracted["deltaE_control"] is None and temp_celsius is None:
            warnings.append("No control-cell or temp/humidity data available; "
                             "dose estimate uncertainty widened to reflect uncompensated drift risk.")

        # Dose estimate with uncertainty band
        measurement_noise = extracted["patch_noise"] + extracted["ccm_residual"] * 0.3
        dose_low, dose_est, dose_high = self.dose_model.dose_uncertainty(
            deltaE_compensated, measurement_noise)

        # Expiry check
        expiry_result: ExpiryResult = evaluate_expiry(
            manufacture_date=manufacture_date,
            shelf_life_days=self.expiry_cfg["shelf_life_days"],
            deltaE_expiry_patch=extracted["deltaE_expiry"],
            expiry_deltaE_threshold=self.expiry_cfg["expiry_deltaE_threshold"],
        )

        if not expiry_result.valid:
            warnings.append("EXPIRY CHECK FAILED — dose estimate below may not be reliable; "
                             "replace badge and flag this reading for review.")

        reading_reliable = extracted["ccm_residual"] <= self.CCM_RESIDUAL_REJECT_THRESHOLD
        if not reading_reliable:
            warnings.append("READING REJECTED — lighting correction too unreliable to certify "
                             "a dose figure. Retake photo before logging.")

        return DoseReport(
            worker_id=worker_id,
            shift=shift,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            dose_estimate_ppm_hr=round(dose_est, 2) if reading_reliable else None,
            dose_low_ppm_hr=round(dose_low, 2) if reading_reliable else None,
            dose_high_ppm_hr=round(dose_high, 2) if reading_reliable else None,
            ccm_fit_residual=round(extracted["ccm_residual"], 2),
            deltaE_sample_raw=round(extracted["deltaE_sample"], 2),
            deltaE_sample_compensated=round(deltaE_compensated, 2),
            deltaE_control=round(extracted["deltaE_control"], 2),
            expiry_status=expiry_result.status,
            expiry_valid=expiry_result.valid,
            expiry_message=expiry_result.message,
            reading_reliable=reading_reliable,
            warnings=warnings,
        )


def save_report(report: DoseReport, path: str, append: bool = True):
    """Append this reading to a local JSON-lines record store — the
    stand-in for the DGMS/OISD-format occupational health record log."""
    mode = "a" if append else "w"
    with open(path, mode) as f:
        f.write(json.dumps(asdict(report)) + "\n")


def load_all_reports(jsonl_path: str) -> list:
    """Read every logged record back out, oldest first. Returns [] if the
    log doesn't exist yet (fresh install / no readings taken)."""
    import os
    if not os.path.exists(jsonl_path):
        return []
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def export_records_csv(jsonl_path: str, csv_path: str):
    """Flatten the JSON-lines record log into a CSV for DGMS/OISD-style
    reporting or import into a spreadsheet / occupational health system."""
    import csv
    records = load_all_reports(jsonl_path)
    if not records:
        return 0
    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            r = dict(r)
            r["warnings"] = "; ".join(r.get("warnings", []))
            writer.writerow(r)
    return len(records)
