"""
Minimal sanity tests — run with: python3 -m tests.test_pipeline (from project root)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from core.calibration import CalibrationModel, hill_model
from core.expiry import evaluate_expiry


def test_calibration_round_trip():
    model = CalibrationModel(deltaE_max=60, K=220, n=0.85, fit_residual_std=1.0)
    for true_dose in [5, 50, 200, 450]:
        deltaE = model.forward(true_dose)
        recovered = model.invert(deltaE)
        rel_err = abs(recovered - true_dose) / true_dose
        assert rel_err < 0.01, f"Round-trip failed for dose={true_dose}: recovered={recovered}"
    print("test_calibration_round_trip: PASS")


def test_calibration_fit_from_synthetic_trials():
    true_params = (55.0, 180.0, 1.1)
    doses = [5, 20, 50, 100, 200, 400, 800]
    deltaEs = [hill_model(d, *true_params) for d in doses]
    fitted = CalibrationModel.fit_from_trials(doses, deltaEs)
    assert abs(fitted.deltaE_max - true_params[0]) < 2.0
    assert abs(fitted.K - true_params[1]) < 20.0
    print("test_calibration_fit_from_synthetic_trials: PASS")


def test_expiry_valid_fresh_badge():
    result = evaluate_expiry(
        manufacture_date=datetime.now() - timedelta(days=5),
        shelf_life_days=90,
        deltaE_expiry_patch=2.0,
        expiry_deltaE_threshold=30.0,
    )
    assert result.valid and result.status == "valid"
    print("test_expiry_valid_fresh_badge: PASS")


def test_expiry_expired_by_both():
    result = evaluate_expiry(
        manufacture_date=datetime.now() - timedelta(days=120),
        shelf_life_days=90,
        deltaE_expiry_patch=35.0,
        expiry_deltaE_threshold=30.0,
    )
    assert not result.valid and result.status == "expired"
    print("test_expiry_expired_by_both: PASS")


def test_expiry_flags_mismatch():
    # Date says valid, but physical patch says degraded -> should flag, not silently pass
    result = evaluate_expiry(
        manufacture_date=datetime.now() - timedelta(days=5),
        shelf_life_days=90,
        deltaE_expiry_patch=35.0,
        expiry_deltaE_threshold=30.0,
    )
    assert not result.valid and result.status == "flagged_mismatch"
    print("test_expiry_flags_mismatch: PASS")


if __name__ == "__main__":
    test_calibration_round_trip()
    test_calibration_fit_from_synthetic_trials()
    test_expiry_valid_fresh_badge()
    test_expiry_expired_by_both()
    test_expiry_flags_mismatch()
    print("\nAll tests passed.")
