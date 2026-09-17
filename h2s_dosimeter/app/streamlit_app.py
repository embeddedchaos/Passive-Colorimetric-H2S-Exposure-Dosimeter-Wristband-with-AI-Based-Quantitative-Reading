"""
streamlit_app.py
The UI layer on top of core/pipeline.py.

Run with:
    cd h2s_dosimeter
    streamlit run app/streamlit_app.py

Three things happen here:
  1. Capture/upload a photo of the wristband card (or generate a synthetic
     demo photo — useful for showing the app off before physical badges exist).
  2. Run it through DosimeterPipeline and show the dose band, expiry status,
     detected-region overlay, and any quality warnings.
  3. Log accepted readings to a local record store and export to CSV for
     DGMS/OISD-style reporting.
"""

import sys
import os
from datetime import datetime, date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
import streamlit as st

from core.pipeline import DosimeterPipeline, save_report, load_all_reports, export_records_csv
from core.visualization import draw_patch_overlay, dose_severity_band
from synthetic.generate_test_card import generate_card, load_configs

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_DIR = os.path.join(BASE, "calib_data")
LOG_PATH = os.path.join(BASE, "records_log.jsonl")

st.set_page_config(page_title="H2S Wristband Dosimeter", layout="wide")


@st.cache_resource
def get_pipeline():
    return DosimeterPipeline(
        layout_config_path=os.path.join(CALIB_DIR, "layout_config.json"),
        dose_calibration_path=os.path.join(CALIB_DIR, "dose_calibration.json"),
        expiry_calibration_path=os.path.join(CALIB_DIR, "expiry_calibration.json"),
    )


pipeline = get_pipeline()
layout_cfg, dose_cal_cfg, expiry_cal_cfg = load_configs(CALIB_DIR)

SEVERITY_COLOR = {"normal": "#1a9c4b", "caution": "#e0a300", "alarm": "#c62828", "unknown": "#888888"}

st.title("H2S Wristband Dosimeter")
st.caption("Passive colorimetric dosimeter reader — cumulative dose + expiry check from one photo")

tab_read, tab_records = st.tabs(["New Reading", "Records / Export"])

# ===========================================================================
# TAB 1 — New Reading
# ===========================================================================
with tab_read:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("1. Shift details")
        worker_id = st.text_input("Worker ID", value="W-0042")
        shift = st.text_input("Shift", value=f"Shift-A_{date.today().isoformat()}")
        manufacture_date = st.date_input("Badge manufacture date", value=date.today())

        st.subheader("2. Environmental compensation")
        has_ref_cell = st.checkbox(
            "This badge has a sealed reference/control cell (recommended)", value=True)
        temp_c, rh_pct = None, None
        if not has_ref_cell:
            st.info("No control cell on this badge — falling back to manual "
                    "temperature/humidity compensation.")
            temp_c = st.number_input("Ambient temperature (°C)", value=30.0)
            rh_pct = st.number_input("Relative humidity (%)", value=60.0)

        st.subheader("3. Badge photo")
        st.caption("Frame the card so the reference color scale, sample strip, "
                   "control cell, and expiry patch are all visible and evenly lit.")
        source = st.radio("Image source", ["Camera / upload", "Generate synthetic demo photo"],
                           horizontal=True)

        image_rgb = None
        if source == "Camera / upload":
            cam_file = st.camera_input("Capture badge photo")
            upload_file = st.file_uploader("...or upload a photo", type=["jpg", "jpeg", "png"])
            chosen = cam_file or upload_file
            if chosen is not None:
                image_rgb = np.array(Image.open(chosen).convert("RGB"))
        else:
            st.caption("No physical badge yet? Generate a simulated photo at a known "
                       "dose to demo the pipeline end-to-end.")
            demo_dose = st.slider("Simulated true dose (ppm·hr)", 0, 600, 150)
            demo_lighting = st.selectbox(
                "Simulated lighting", ["neutral_daylight", "warm_indoor_bulb",
                                        "cool_fluorescent", "dim_underexposed", "bright_overexposed"])
            lighting_presets = {
                "neutral_daylight":   dict(lighting_gain=(1.00, 1.00, 1.00), lighting_tint=(0, 0, 0)),
                "warm_indoor_bulb":   dict(lighting_gain=(1.15, 1.00, 0.75), lighting_tint=(5, 0, -8)),
                "cool_fluorescent":   dict(lighting_gain=(0.90, 1.00, 1.15), lighting_tint=(-4, 2, 6)),
                "dim_underexposed":   dict(lighting_gain=(0.65, 0.65, 0.65), lighting_tint=(0, 0, 0)),
                "bright_overexposed": dict(lighting_gain=(1.35, 1.35, 1.35), lighting_tint=(10, 10, 10)),
            }
            if st.button("Generate synthetic photo"):
                image_rgb = generate_card(
                    layout_cfg, dose_cal_cfg, true_dose_ppm_hr=demo_dose,
                    true_control_drift_deltaE=0.8, days_since_manufacture=20,
                    expiry_calibration=expiry_cal_cfg, seed=None,
                    **lighting_presets[demo_lighting])
                st.session_state["synthetic_image"] = image_rgb
                st.session_state["synthetic_true_dose"] = demo_dose
            image_rgb = st.session_state.get("synthetic_image", None)

    with right:
        st.subheader("Result")
        if image_rgb is None:
            st.info("Capture, upload, or generate a photo to get a reading.")
        else:
            overlay = draw_patch_overlay(image_rgb, layout_cfg)
            st.image(overlay, caption="Detected regions", use_container_width=True)

            if "synthetic_true_dose" in st.session_state and source == "Generate synthetic demo photo":
                st.caption(f"(Synthetic ground truth: {st.session_state['synthetic_true_dose']} ppm·hr — "
                           f"for validation only, not shown in a real deployment)")

            report = pipeline.run(
                image_rgb=image_rgb,
                worker_id=worker_id,
                shift=shift,
                manufacture_date=datetime.combine(manufacture_date, datetime.min.time()),
                temp_celsius=temp_c,
                relative_humidity_pct=rh_pct,
            )

            if report.reading_reliable:
                severity = dose_severity_band(report.dose_estimate_ppm_hr)
                color = SEVERITY_COLOR[severity]
                st.markdown(
                    f"""
                    <div style="padding:16px;border-radius:10px;background:{color}22;
                                border:2px solid {color};">
                        <div style="font-size:14px;color:#555;">Estimated cumulative dose</div>
                        <div style="font-size:34px;font-weight:700;color:{color};">
                            {report.dose_estimate_ppm_hr} ppm·hr
                        </div>
                        <div style="font-size:14px;color:#555;">
                            Range: {report.dose_low_ppm_hr} – {report.dose_high_ppm_hr} ppm·hr
                            &nbsp;|&nbsp; Severity: <b>{severity.upper()}</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Reading rejected — could not certify a dose from this photo. "
                          "Retake under more even, non-glare lighting.")

            st.markdown("")
            exp_color = {"valid": "#1a9c4b", "near_expiry": "#e0a300",
                         "expired": "#c62828", "flagged_mismatch": "#c62828"}[report.expiry_status]
            st.markdown(
                f"**Expiry status:** <span style='color:{exp_color};font-weight:700;'>"
                f"{report.expiry_status.replace('_', ' ').upper()}</span> — {report.expiry_message}",
                unsafe_allow_html=True)

            if report.warnings:
                for w in report.warnings:
                    st.warning(w)

            with st.expander("Raw diagnostic data"):
                st.json({
                    "ccm_fit_residual": report.ccm_fit_residual,
                    "deltaE_sample_raw": report.deltaE_sample_raw,
                    "deltaE_sample_compensated": report.deltaE_sample_compensated,
                    "deltaE_control": report.deltaE_control,
                })

            if st.button("Log this reading", type="primary", disabled=not report.reading_reliable):
                save_report(report, LOG_PATH, append=True)
                st.success(f"Logged reading for {worker_id} / {shift}.")

# ===========================================================================
# TAB 2 — Records / Export
# ===========================================================================
with tab_records:
    st.subheader("Logged readings")
    records = load_all_reports(LOG_PATH)
    if not records:
        st.info("No readings logged yet. Take a reading in the 'New Reading' tab and click "
                "'Log this reading'.")
    else:
        import pandas as pd
        df = pd.DataFrame(records)
        df["warnings"] = df["warnings"].apply(lambda w: "; ".join(w) if isinstance(w, list) else w)
        st.dataframe(df, use_container_width=True)

        csv_path = os.path.join(BASE, "records_export.csv")
        export_records_csv(LOG_PATH, csv_path)
        with open(csv_path, "rb") as f:
            st.download_button("Download CSV (DGMS / OISD-style export)", f,
                                file_name="h2s_dosimeter_records.csv", mime="text/csv")
