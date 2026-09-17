"""
run_demo.py
End-to-end validation run: generates synthetic cards at KNOWN ground-truth
doses under several different simulated lighting conditions, runs them
through the real pipeline, and reports how close the recovered dose
estimate is to ground truth. This is the software analogue of the PS's
"tested against a controlled, lab-simulated H2S exposure" requirement,
standing in until real strip photos from actual lab trials are available.
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import DosimeterPipeline, save_report
from synthetic.generate_test_card import generate_card, load_configs

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_DIR = os.path.join(BASE, "calib_data")


LIGHTING_CONDITIONS = {
    "neutral_daylight":   dict(lighting_gain=(1.00, 1.00, 1.00), lighting_tint=(0, 0, 0)),
    "warm_indoor_bulb":   dict(lighting_gain=(1.15, 1.00, 0.75), lighting_tint=(5, 0, -8)),
    "cool_fluorescent":   dict(lighting_gain=(0.90, 1.00, 1.15), lighting_tint=(-4, 2, 6)),
    "dim_underexposed":   dict(lighting_gain=(0.65, 0.65, 0.65), lighting_tint=(0, 0, 0)),
    "bright_overexposed": dict(lighting_gain=(1.35, 1.35, 1.35), lighting_tint=(10, 10, 10)),
}

TRUE_DOSES_PPM_HR = [10, 50, 120, 250, 500]


def main():
    layout, dose_cal, expiry_cal = load_configs(CALIB_DIR)
    pipeline = DosimeterPipeline(
        layout_config_path=os.path.join(CALIB_DIR, "layout_config.json"),
        dose_calibration_path=os.path.join(CALIB_DIR, "dose_calibration.json"),
        expiry_calibration_path=os.path.join(CALIB_DIR, "expiry_calibration.json"),
    )

    print(f"{'Lighting':<20}{'True Dose':>12}{'Est. Dose':>12}{'Range':>18}{'% Error':>10}  Warnings")
    print("-" * 100)

    results = []
    for true_dose in TRUE_DOSES_PPM_HR:
        for cond_name, cond_params in LIGHTING_CONDITIONS.items():
            img = generate_card(
                layout_config=layout,
                dose_calibration=dose_cal,
                true_dose_ppm_hr=true_dose,
                true_control_drift_deltaE=0.8,   # small realistic ambient drift
                days_since_manufacture=20,
                expiry_calibration=expiry_cal,
                noise_std=2.0,
                seed=42,
                **cond_params,
            )

            report = pipeline.run(
                image_rgb=img,
                worker_id="W-0042",
                shift="Shift-A_2026-09-15",
                manufacture_date=datetime.now() - timedelta(days=20),
            )

            if report.reading_reliable:
                pct_error = 100.0 * (report.dose_estimate_ppm_hr - true_dose) / true_dose
                est_str = f"{report.dose_estimate_ppm_hr:>12.1f}"
                range_str = f"[{report.dose_low_ppm_hr:>6.1f}, {report.dose_high_ppm_hr:>6.1f}]"
                err_str = f"{pct_error:>9.1f}%"
            else:
                est_str = f"{'REJECTED':>12}"
                range_str = f"{'--':>18}"
                err_str = f"{'n/a':>9}"

            results.append((cond_name, true_dose, report))
            print(f"{cond_name:<20}{true_dose:>12.1f}{est_str}   {range_str}{err_str}  "
                  f"{'; '.join(report.warnings) if report.warnings else '-'}")

    save_report(report, os.path.join(BASE, "sample_report_log.jsonl"), append=False)

    print("\nLast report full JSON record:")
    import json
    from dataclasses import asdict
    print(json.dumps(asdict(report), indent=2))

    # Save one example synthetic card image for visual inspection
    from PIL import Image
    example_img = generate_card(layout, dose_cal, true_dose_ppm_hr=250,
                                 true_control_drift_deltaE=1.5, days_since_manufacture=20,
                                 expiry_calibration=expiry_cal, seed=1,
                                 **LIGHTING_CONDITIONS["warm_indoor_bulb"])
    Image.fromarray(example_img).save(os.path.join(BASE, "example_synthetic_card.png"))
    print("\nSaved example synthetic card image -> example_synthetic_card.png")


if __name__ == "__main__":
    main()
