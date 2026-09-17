"""
expiry.py
Validates badge shelf-life using the SEPARATE expiry-indicator patch (not
the H2S-sensing strip). This patch changes color purely as a function of
time/temperature/humidity since manufacture, independent of any gas
exposure, so a badge that has been sitting in a hot warehouse past its
useful life shows it here even if the printed date says it's still valid.

The app cross-checks the physical patch reading against the printed
manufacture date + declared shelf life. Disagreement between the two is
itself useful information (badge was stored badly) and is flagged rather
than silently resolved.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np


@dataclass
class ExpiryResult:
    valid: bool
    status: str                 # "valid" | "near_expiry" | "expired" | "flagged_mismatch"
    days_remaining_by_date: float
    patch_reading_valid: bool
    message: str


def expiry_by_date(manufacture_date: datetime, shelf_life_days: int,
                    check_date: datetime = None) -> tuple:
    """Simple calendar-based check: is today within the declared shelf life."""
    check_date = check_date or datetime.now()
    expiry_date = manufacture_date + timedelta(days=shelf_life_days)
    days_remaining = (expiry_date - check_date).total_seconds() / 86400.0
    return days_remaining, days_remaining > 0


def expiry_by_patch(deltaE_expiry_patch: float, expiry_deltaE_threshold: float,
                     near_expiry_fraction: float = 0.85) -> tuple:
    """
    Physical check: how far the expiry patch has drifted from its fresh
    state, compared to the threshold deltaE that marks 'expired' (calibrated
    once from accelerated-aging lab trials, same way as the dose curve).

    Returns (patch_valid: bool, patch_near_expiry: bool, fraction_used: float)
    """
    fraction_used = deltaE_expiry_patch / expiry_deltaE_threshold if expiry_deltaE_threshold > 0 else 1.0
    patch_valid = fraction_used < 1.0
    patch_near_expiry = fraction_used >= near_expiry_fraction
    return patch_valid, patch_near_expiry, float(np.clip(fraction_used, 0, 2))


def evaluate_expiry(manufacture_date: datetime, shelf_life_days: int,
                     deltaE_expiry_patch: float, expiry_deltaE_threshold: float,
                     check_date: datetime = None) -> ExpiryResult:
    days_remaining, date_valid = expiry_by_date(manufacture_date, shelf_life_days, check_date)
    patch_valid, patch_near_expiry, fraction_used = expiry_by_patch(
        deltaE_expiry_patch, expiry_deltaE_threshold)

    if date_valid and patch_valid and not patch_near_expiry:
        return ExpiryResult(True, "valid", days_remaining, patch_valid,
                             f"Badge valid. Expiry patch at {fraction_used*100:.0f}% of threshold, "
                             f"{days_remaining:.0f} days remaining by manufacture date.")

    if date_valid and patch_valid and patch_near_expiry:
        return ExpiryResult(True, "near_expiry", days_remaining, patch_valid,
                             f"Badge usable but nearing end of life "
                             f"({fraction_used*100:.0f}% of expiry threshold). Consider replacing soon.")

    if date_valid != patch_valid:
        # The printed date and the physical patch disagree -> storage/handling issue.
        return ExpiryResult(False, "flagged_mismatch", days_remaining, patch_valid,
                             "Mismatch between printed expiry date and physical expiry-patch reading. "
                             "Badge likely stored outside recommended conditions. Do NOT use for a dose reading; "
                             "flag for inspection.")

    return ExpiryResult(False, "expired", days_remaining, patch_valid,
                         "Badge expired by both date and physical patch check. Do not use.")
