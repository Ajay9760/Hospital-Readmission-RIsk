import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from icd9_utils import map_icd9

# ============================================================
# CONFIG — fixed to match your actual Colab exports
# ============================================================
USE_SCALER = True  # Set to True because you used StandardScaler
MODEL_PATH = "random_forest_model.joblib"
FEATURE_COLUMNS_PATH = "model_columns.json"
THRESHOLD_PATH = "threshold.json"
SCALER_PATH = "scaler.joblib"
SCALED_COLUMNS_PATH = "scaled_columns.json"

st.set_page_config(page_title="Readmission Risk Predictor", page_icon="🏥", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH) as f:
        feature_columns = json.load(f)
    with open(THRESHOLD_PATH) as f:
        threshold = json.load(f)["threshold"]

    scaler, scaled_columns = None, []
    if USE_SCALER:
        scaler = joblib.load(SCALER_PATH)
        with open(SCALED_COLUMNS_PATH) as f:
            scaled_columns = json.load(f)

    return model, feature_columns, threshold, scaler, scaled_columns


try:
    model, feature_columns, default_threshold, scaler, scaled_columns = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Model artifacts not found. Run the export cell in your notebook first, "
        "then place random_forest_model.joblib, model_columns.json, and threshold.json "
        f"in this folder.\n\nMissing: {e.filename}"
    )
    st.stop()

st.title("🏥 30-Day Readmission Risk Predictor")
st.caption(
    "Estimates a diabetic inpatient's risk of readmission within 30 days of discharge, "
    "based on the Diabetes 130-US Hospitals dataset. Built as a learning project — "
    "not a validated clinical tool."
)

with st.form("patient_form"):
    st.subheader("Patient & encounter details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.selectbox(
            "Age group",
            ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
             '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
            index=6,
        )
        time_in_hospital = st.number_input("Time in hospital (days)", min_value=1, max_value=14, value=4)
        num_lab_procedures = st.number_input("Number of lab procedures", min_value=0, max_value=150, value=43)
        num_procedures = st.number_input("Number of procedures", min_value=0, max_value=10, value=1)
        num_medications = st.number_input("Number of medications", min_value=0, max_value=100, value=16)

    with col2:
        number_diagnoses = st.number_input("Number of diagnoses", min_value=1, max_value=20, value=7)
        number_inpatient = st.number_input("Prior inpatient visits (past year)", min_value=0, max_value=30, value=0)
        max_glu_serum = st.selectbox("Max glucose serum test result", ["Not_Measured", "Norm", ">200", ">300"])
        a1c_result = st.selectbox("A1C test result", ["Not_Measured", "Norm", ">7", ">8"])
        diabetes_med = st.selectbox("On a diabetes medication?", ["No", "Yes"])

    st.subheader("Diagnosis codes (ICD-9)")
    st.caption("e.g. 250.83 (diabetes), 428 (heart failure), 486 (pneumonia). Leave blank if unknown.")
    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        diag_1 = st.text_input("Primary diagnosis (diag_1)", value="250.83")
    with dcol2:
        diag_2 = st.text_input("Secondary diagnosis (diag_2)", value="")
    with dcol3:
        diag_3 = st.text_input("Additional diagnosis (diag_3)", value="")

    st.subheader("Operating threshold")
    threshold = st.slider(
        "Flag as high-risk if predicted probability is at or above:",
        min_value=0.0, max_value=1.0, value=float(default_threshold), step=0.01,
        help="Lower = flags more patients (higher recall, lower precision). "
             "Default comes from the notebook's lift-curve analysis.",
    )

    submitted = st.form_submit_button("Predict risk")

if submitted:
    raw_row = {
        "age": age,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "max_glu_serum": max_glu_serum,
        "A1Cresult": a1c_result,
        "diabetesMed": diabetes_med,
        "diag_1_category": map_icd9(diag_1),
        "diag_2_category": map_icd9(diag_2),
        "diag_3_category": map_icd9(diag_3),
    }

    row_df = pd.DataFrame([raw_row])

    categorical_cols = ["age", "max_glu_serum", "A1Cresult", "diabetesMed",
                         "diag_1_category", "diag_2_category", "diag_3_category"]
    encoded = pd.get_dummies(row_df, columns=categorical_cols)

    # Align to the exact column set/order the model was trained on.
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    if USE_SCALER and scaler is not None and scaled_columns:
        encoded[scaled_columns] = scaler.transform(encoded[scaled_columns])

    prob = model.predict_proba(encoded)[0, 1]
    is_high_risk = prob >= threshold

    st.divider()
    st.subheader("Result")

    m1, m2 = st.columns(2)
    m1.metric("Predicted readmission probability", f"{prob:.1%}")
    m2.metric("Flag", "🔴 High risk" if is_high_risk else "🟢 Not flagged")

    st.progress(min(float(prob), 1.0))

    if hasattr(model, "feature_importances_"):
        st.subheader("What the model weighs most heavily overall")
        importances = pd.Series(model.feature_importances_, index=feature_columns)
        top_importances = importances.sort_values(ascending=False).head(8)
        st.bar_chart(top_importances)
        st.caption(
            "These are the model's overall top drivers across all patients, not a "
            "per-patient explanation (that would need SHAP values — a natural next "
            "step if you extend this project)."
        )

st.divider()
st.caption(
    "Diabetes 130-US Hospitals dataset (Strack et al., 2014), via UCI ML Repository (CC BY 4.0). "
    "This tool is a portfolio/learning project and is not validated for clinical use."
)