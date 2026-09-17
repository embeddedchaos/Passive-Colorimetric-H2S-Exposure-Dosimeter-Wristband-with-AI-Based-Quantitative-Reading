"""
color_utils.py
Low-level color science: patch extraction from an image, RGB<->Lab conversion,
and lighting self-calibration via a Color Correction Matrix (CCM) fitted
against the printed reference scale that sits in every photo.

This is the piece that solves "the app corrects for whatever lighting the
photo was taken in by calibrating against the reference" from the problem
statement.
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patch_rgb(image_rgb: np.ndarray, bbox_norm: tuple, trim_pct: float = 0.15):
    """
    Extract the mean (and std) sRGB color of a rectangular patch.

    image_rgb   : HxWx3 uint8 array, RGB order
    bbox_norm   : (x0, y0, x1, y1) in normalized [0,1] image coordinates
    trim_pct    : fraction of extreme pixels (by luminance) discarded on each
                  end before averaging, to reject specular glare / shadow edge
                  pixels without needing a hand-tuned threshold.

    Returns (mean_rgb: np.ndarray[3], std_rgb: np.ndarray[3])
    """
    h, w = image_rgb.shape[:2]
    x0, y0, x1, y1 = bbox_norm
    px0, py0, px1, py1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    patch = image_rgb[py0:py1, px0:px1].reshape(-1, 3).astype(np.float64)

    if patch.shape[0] == 0:
        raise ValueError(f"Empty patch for bbox {bbox_norm} on image size {w}x{h}")

    # Trim by luminance to drop glare highlights / shadowed edges
    lum = patch.mean(axis=1)
    order = np.argsort(lum)
    n = len(order)
    lo = int(n * trim_pct)
    hi = int(n * (1 - trim_pct))
    keep = order[lo:hi] if hi > lo else order
    trimmed = patch[keep]

    return trimmed.mean(axis=0), trimmed.std(axis=0)


# ---------------------------------------------------------------------------
# Color space conversion
# ---------------------------------------------------------------------------

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert a single sRGB triple (0-255 floats) to CIE Lab (OpenCV convention)."""
    px = np.clip(rgb, 0, 255).astype(np.uint8).reshape(1, 1, 3)
    lab = cv2.cvtColor(px, cv2.COLOR_RGB2LAB).astype(np.float64).reshape(3)
    # OpenCV Lab: L in [0,255] -> rescale to [0,100]; a,b shifted by 128
    L = lab[0] * 100.0 / 255.0
    a = lab[1] - 128.0
    b = lab[2] - 128.0
    return np.array([L, a, b])


def delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Simple perceptual color distance (Euclidean in Lab). Good enough given
    the strip only ever moves along roughly one darkening trajectory."""
    return float(np.linalg.norm(lab1 - lab2))


# ---------------------------------------------------------------------------
# Lighting self-calibration (Color Correction Matrix)
# ---------------------------------------------------------------------------

def fit_color_correction_matrix(measured_rgbs: np.ndarray, true_rgbs: np.ndarray) -> np.ndarray:
    """
    Fit a per-channel (diagonal) affine correction: true_ch ≈ a_ch * measured_ch + b_ch,
    fitted independently for R, G, B against the printed reference swatches visible in
    the same photo as the sensing strip. This is what cancels out "whatever lighting
    the photo was taken in".

    A per-channel model (rather than a full 3x3 cross-channel matrix) is deliberately
    used: with a small, mostly-neutral reference scale (a handful of gray swatches plus
    one or two chromatic ones, as fits on a wristband), the gray swatches are nearly
    collinear in RGB space, which makes a full cross-channel fit ill-conditioned and
    prone to large spurious cross-term corrections. A diagonal fit only needs each
    channel's own values to be varied across the swatch set, which a simple gray-scale
    ramp already provides, and correctly handles the two dominant real-world lighting
    artifacts (color-temperature tint and exposure gain).

    measured_rgbs : Nx3 array, the swatch colors as the camera actually saw them
    true_rgbs     : Nx3 array, the known/printed ground-truth colors of those
                    same swatches (calibrated once at manufacturing time)

    Returns M: 3x4 array [[aR,0,0,bR],[0,aG,0,bG],[0,0,aB,bB]] so it composes with the
    same apply_color_correction() call used elsewhere.
    """
    n = measured_rgbs.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 reference swatches to fit lighting correction; "
                          f"got {n}.")

    M = np.zeros((3, 4))
    for ch in range(3):
        x = measured_rgbs[:, ch]
        y = true_rgbs[:, ch]
        x_mean, y_mean = x.mean(), y.mean()
        denom = np.sum((x - x_mean) ** 2)
        a = np.sum((x - x_mean) * (y - y_mean)) / denom if denom > 1e-9 else 1.0
        b = y_mean - a * x_mean
        M[ch, ch] = a
        M[ch, 3] = b
    return M


def apply_color_correction(rgb: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a fitted 3x4 CCM to a single measured RGB triple."""
    homo = np.append(rgb, 1.0)
    corrected = M @ homo
    return np.clip(corrected, 0, 255)


def ccm_fit_residual(measured_rgbs: np.ndarray, true_rgbs: np.ndarray, M: np.ndarray) -> float:
    """RMS error (in RGB units) of the CCM fit — used downstream as one
    component of the reported dose uncertainty. A high residual means the
    lighting was too far off (or the photo angle/glare too bad) to trust."""
    n = measured_rgbs.shape[0]
    A = np.hstack([measured_rgbs, np.ones((n, 1))])
    pred = (A @ M.T)
    return float(np.sqrt(np.mean((pred - true_rgbs) ** 2)))
