This is the software half of the solution of Problem statement : 'Passive Colorimetric H2S Exposure-Dosimeter Wristband with AI-Based Quantitative Reading'.

## Why it's structured this way — mapped to the software gaps identified

Gap identified in existing solutions - how we tackle them:
1. No cumulative dose model (spot-reading only) - `core/calibration.py` — Hill/saturation curve fit on (dose, ΔE) pairs, inverted per reading
2. No live per-photo self-calibration against a co-photographed reference  `core/color_utils.py` — CCM fit fresh from the reference swatches in *every* photo, not a stored calibration |
3. No expiry check in the capture flow  `core/expiry.py` — separate patch, cross-checked against the printed manufacture date 
4. No temp/humidity compensation  `core/compensation.py` — sealed reference-cell subtraction (primary) or Arrhenius+humidity model (fallback) |
5. No occupational-record output  `core/pipeline.py: save_report()` — structured JSON record keyed to worker ID + shift 
6. No uncertainty handling (single deterministic number)  Every dose is returned as `(low, estimate, high)`, and bad photos are **rejected outright** rather than given a number

## Layout


core:
  color_utils.py    patch extraction, Lab conversion, lighting self-calibration (CCM)
  calibration.py    dose <-> color-change model (Hill curve), fit + invert + uncertainty
  compensation.py   temp/humidity drift correction (reference-cell or Arrhenius model)
  expiry.py         shelf-life validity from the separate expiry patch
  pipeline.py        orchestrates all of the above into one DoseReport

synthetic:
  generate_test_card.py   fabricates a photo of the card at a KNOWN dose under
                           simulated lighting — stand-in until real strip photos exist

calib_data:
  layout_config.json       patch positions + reference swatch true colors (edit once
                            the physical card layout is final — nothing else changes)
  dose_calibration.json    PLACEHOLDER Hill-curve params — replace with real lab-fit data
  expiry_calibration.json  PLACEHOLDER expiry threshold — replace with accelerated-aging data

demo:
  run_demo.py       runs the full pipeline across a dose sweep x lighting sweep,
                     reports estimate vs. known ground truth


## Running the UI

```bash
pip install -r requirements.txt
cd h2s_dosimeter
streamlit run app/streamlit_app.py
```

Opens a browser app with two tabs:

- 'New Reading' — enter worker ID / shift / manufacture date, either
  capture a photo (webcam/phone camera via browser, or upload one) or
  'generate a simulated photo' at a known dose (useful for demoing the
  full pipeline as actual band is not available yet — pick a true dose and a
  lighting condition and it renders a synthetic card through the same
  forward model the calibration curve was fit against). The result panel
  shows the detected-region overlay, dose estimate with its uncertainty
  band and a color-coded severity tier, expiry verdict, and any quality
  warnings. Bad photos are rejected rather than given a number. "Log this
  reading" writes it to `records_log.jsonl`.
- 'Records / Export' — table of everything logged so far, with a CSV
  download button (`h2s_dosimeter_records.csv`) formatted for DGMS/OISD-style
  occupational exposure reporting.

`app/streamlit_app.py` calls straight into `core/pipeline.py` — it holds no
dosimetry logic of its own, so swapping the UI framework later (React
Native, Flutter, etc.) only means re-wiring this file, not the core.

## Running the demo (command-line validation, no UI)

```bash
pip install -r requirements.txt
cd h2s_dosimeter
python3 demo/run_demo.py
```

This generates synthetic cards at five known doses (10–500 ppm·hr) under
five simulated lighting conditions, runs each through the real pipeline, and
prints estimated dose vs. ground truth. One condition (`bright_overexposed`)
is deliberately included to show the rejection path working — clipped
reference swatches make the lighting correction untrustworthy, so the
pipeline refuses to output a dose number rather than guessing.

## Swapping in real data (once lab trials exist)

1. Calibration curve: run controlled exposures at known ppm × known
   hours, photograph the strip each time, extract ΔE with
   `color_utils`/`pipeline`'s internal steps, then call
   `CalibrationModel.fit_from_trials(doses, deltaEs).to_json("calib_data/dose_calibration.json")`.
2. Expiry threshold: same idea via accelerated-aging trials at your
   target shelf life; update `calib_data/expiry_calibration.json`.
3. Layout: once the physical card/wristband artwork is finalized, update
   the normalized bounding boxes in `calib_data/layout_config.json` to match.
4. Swap `synthetic/generate_test_card.py` output for real camera photos —
   `pipeline.run()` takes any HxWx3 RGB numpy array, so this is a drop-in
   replacement.

## What's left to complete the software side

- Real calibration data — run the controlled lab H2S exposures the
      PS asks for, fit `dose_calibration.json` and `expiry_calibration.json`
      from actual measurements (steps are in the README section above).
- Real card layout — once the physical wristband/reference-scale
      artwork is finalized, update `calib_data/layout_config.json` bounding
      boxes and reference swatch colors to match the printed design exactly.
- Auto-crop / card detection — right now the pipeline assumes the
      photo is already reasonably cropped to the card. For a production
      app, add a marker/fiducial-based auto-detection step (e.g. ArUco
      markers printed at the card corners) so the wearer doesn't have to
      frame it perfectly by hand.
-Mobile packaging — Streamlit is the fastest path to a demoable
      prototype (works in any phone browser, no app-store step) but for a
      deployed field tool you'd eventually package this as a native/PWA
      app; `core/` doesn't need to change, only `app/`.
- Backend sync — `records_log.jsonl` is local-file storage for the
      prototype. A real deployment needs this pushed to a central
      database per site for DGMS/OISD roll-up reporting across workers.
- Threshold validation — `dose_severity_band()` in
      `core/visualization.py` uses placeholder caution/alarm cutoffs;
      confirm against actual DGMS/OISD cumulative-exposure guidance before
      presenting these as real safety thresholds.

## Known placeholder limitations (by design, until real chemistry data exists)

- `dose_calibration.json` and `expiry_calibration.json` contain assumed
  curve shapes, not measured ones — replace before citing any accuracy
  numbers in the SIH report.
- The reference-cell compensation uses `alpha=1.0` (full 1:1 subtraction);
  the demo run shows this can over-correct low-dose readings when the
  control cell's own measurement noise is comparable to a small real signal
  — worth tuning `alpha` down, or enlarging the control-cell patch area,
  once real noise levels are known from actual photos.
- CCM rejection threshold (`DosimeterPipeline.CCM_RESIDUAL_REJECT_THRESHOLD = 10.0`)
  is a reasonable starting guess, not a validated cutoff.
