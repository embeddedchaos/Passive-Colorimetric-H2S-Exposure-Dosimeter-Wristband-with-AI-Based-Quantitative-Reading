"""
generate_test_card.py
Since there's no physical wristband/photo yet, this generates a synthetic
"photograph" of the reading card at a KNOWN ground-truth dose, under a
randomized simulated lighting condition. This lets us validate that the
pipeline (CCM correction -> deltaE -> calibration inversion) round-trips
back to something close to the known dose, BEFORE any real chemistry exists.

Once you have real strip photos from lab trials, this file becomes
unnecessary — swap synthetic images for real ones and everything downstream
(pipeline.py) is unchanged.
"""

import json
import numpy as np

from core.calibration import hill_model


def _blend(fresh_rgb, dark_rgb, fraction):
    fraction = np.clip(fraction, 0, 1)
    return fresh_rgb * (1 - fraction) + dark_rgb * fraction


def generate_card(layout_config: dict, dose_calibration: dict,
                   true_dose_ppm_hr: float, true_control_drift_deltaE: float = 0.0,
                   days_since_manufacture: float = 0, expiry_calibration: dict = None,
                   lighting_gain=(1.0, 1.0, 1.0), lighting_tint=(0, 0, 0),
                   noise_std: float = 3.0, image_size=(600, 900), seed=None) -> np.ndarray:
    """
    Returns an HxWx3 uint8 RGB image simulating a photographed reading card.

    lighting_gain/tint : simulate a colored / over- or under-exposed light
                          source, applied uniformly to the whole photographed
                          scene (as real ambient lighting would).
    noise_std           : sensor/JPEG noise added per-pixel.
    """
    rng = np.random.default_rng(seed)
    h, w = image_size
    img = np.full((h, w, 3), 235, dtype=np.float64)  # neutral card background

    def paint_bbox(bbox_norm, rgb):
        x0, y0, x1, y1 = bbox_norm
        px0, py0, px1, py1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
        img[py0:py1, px0:px1] = rgb

    # Reference swatches painted at their TRUE printed colors (this is what
    # makes them useful as a lighting reference — they're manufactured to a
    # known spec, unlike the reactive strip).
    for sw in layout_config["reference_swatches"]:
        paint_bbox(sw["bbox_norm"], np.array(sw["true_rgb"], dtype=np.float64))

    # Sample strip: color computed forward from the known ground-truth dose
    # using the same Hill model the pipeline will later try to invert.
    deltaE_max = dose_calibration["deltaE_max"]
    K = dose_calibration["K"]
    n = dose_calibration["n"]
    deltaE_sample = float(hill_model(true_dose_ppm_hr, deltaE_max, K, n))
    fresh_rgb = np.array(layout_config["sample_strip_fresh_rgb"], dtype=np.float64)
    dark_rgb = np.array([45, 40, 35], dtype=np.float64)  # fully-reacted dark color
    sample_rgb = _blend(fresh_rgb, dark_rgb, deltaE_sample / deltaE_max)
    paint_bbox(layout_config["sample_strip"]["bbox_norm"], sample_rgb)

    # Control cell: same chemistry, but should NOT respond to dose — only to
    # ambient drift, represented here directly as a small deltaE fraction.
    control_rgb = _blend(fresh_rgb, dark_rgb, true_control_drift_deltaE / deltaE_max)
    paint_bbox(layout_config["control_cell"]["bbox_norm"], control_rgb)

    # Expiry patch: ages with time only (reuse Hill model with "days" as the
    # forcing variable and the expiry threshold as its own scale).
    if expiry_calibration is not None:
        exp_threshold = expiry_calibration["expiry_deltaE_threshold"]
        shelf_life = expiry_calibration["shelf_life_days"]
        deltaE_expiry = float(hill_model(days_since_manufacture, exp_threshold * 1.05,
                                          shelf_life * 0.6, 1.5))
        expiry_fresh_rgb = np.array(layout_config["expiry_patch_fresh_rgb"], dtype=np.float64)
        expiry_rgb = _blend(expiry_fresh_rgb, dark_rgb, deltaE_expiry / (exp_threshold * 1.05))
        paint_bbox(layout_config["expiry_patch"]["bbox_norm"], expiry_rgb)

    # Apply simulated lighting: per-channel gain + tint offset, then noise
    gain = np.array(lighting_gain, dtype=np.float64)
    tint = np.array(lighting_tint, dtype=np.float64)
    img = img * gain + tint
    img = img + rng.normal(0, noise_std, img.shape)
    img = np.clip(img, 0, 255)

    return img.astype(np.uint8)


def load_configs(base_dir="calib_data"):
    with open(f"{base_dir}/layout_config.json") as f:
        layout = json.load(f)
    with open(f"{base_dir}/dose_calibration.json") as f:
        dose_cal = json.load(f)
    with open(f"{base_dir}/expiry_calibration.json") as f:
        expiry_cal = json.load(f)
    return layout, dose_cal, expiry_cal
