"""
visualization.py
Draws the detected patch regions on top of the captured photo, so a worker
or safety officer (and SIH judges) can see exactly what the app is reading
before trusting the number it outputs. Also renders a simple dose gauge.
"""

import numpy as np
import cv2


def draw_patch_overlay(image_rgb: np.ndarray, layout: dict) -> np.ndarray:
    img = image_rgb.copy()
    h, w = img.shape[:2]

    def box(bbox_norm, color, label):
        x0, y0, x1, y1 = bbox_norm
        p0 = (int(x0 * w), int(y0 * h))
        p1 = (int(x1 * w), int(y1 * h))
        cv2.rectangle(img, p0, p1, color, 3)
        cv2.putText(img, label, (p0[0], max(p0[1] - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    for sw in layout["reference_swatches"]:
        box(sw["bbox_norm"], (0, 140, 255), sw["name"][:4])

    box(layout["sample_strip"]["bbox_norm"], (0, 200, 0), "SAMPLE")
    box(layout["control_cell"]["bbox_norm"], (200, 0, 200), "CONTROL")
    box(layout["expiry_patch"]["bbox_norm"], (0, 0, 220), "EXPIRY")

    return img


def dose_severity_band(dose_ppm_hr: float, caution_ppm_hr: float = 8.0,
                        alarm_ppm_hr: float = 80.0) -> str:
    """
    Rough banding of cumulative dose into severity tiers for a quick
    color-coded readout. Defaults are placeholders derived loosely from an
    8-hour-shift TLV-style ceiling (~1 ppm TWA -> ~8 ppm*hr/shift caution,
    ~10x that as an alarm tier) — replace with your validated DGMS/OISD
    thresholds once confirmed.
    """
    if dose_ppm_hr is None:
        return "unknown"
    if dose_ppm_hr < caution_ppm_hr:
        return "normal"
    if dose_ppm_hr < alarm_ppm_hr:
        return "caution"
    return "alarm"
