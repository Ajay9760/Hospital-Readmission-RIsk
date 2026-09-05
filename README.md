# 🏥 Hospital Readmission Risk Predictor

Predicting which diabetic inpatients are at high risk of being **readmitted within 30 days of discharge** — and surfacing *why* — using 10 years of real clinical encounter data from 130 US hospitals.

**🔗 Live app:** https://hospital-readmission-risk-auyqrxqyrqfaywmh3rrgmn.streamlit.app/

---

## What this is

Hospital readmissions within 30 days are a major cost and quality problem for healthcare systems — and a direct financial penalty for hospitals under CMS's Hospital Readmissions Reduction Program in the US. Care teams can't call every discharged patient for follow-up, so the real question isn't just "will this patient be readmitted?" — it's **"given limited staff capacity, which patients should we prioritize?"**

This project builds and deploys a model that answers that question, end to end:
- Cleans and engineers features from a real, messy clinical dataset
- Trains and compares two model families
- Tunes the decision threshold against a realistic operational capacity constraint (not just a default 50% cutoff)
- Ships as a live, interactive risk calculator

## Try it

Open the [live app](https://hospital-readmission-risk-auyqrxqyrqfaywmh3rrgmn.streamlit.app/), fill in a patient's encounter details (age, length of stay, lab results, diagnosis codes, etc.), and get:
- A predicted readmission probability
- A high-risk flag, based on a tunable threshold slider
- The model's top overall risk drivers

## Dataset

[Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008) (Strack et al., 2014), via the UCI Machine Learning Repository (CC BY 4.0).

- 101,766 inpatient encounters across 130 hospitals
- Every encounter includes a diabetes diagnosis, a 1–14 day stay, lab tests, and medications
- Target: whether the patient was readmitted within 30 days (`<30`) vs. later or never (`>30` / `NO`)
- ~11% positive class — meaningfully imbalanced

## Methodology

**1. Missingness — decided per-column, not defaulted:**
| Column | Missing | Decision & why |
|---|---|---|
| `weight` | 96.9% | Dropped — too sparse to impute without introducing artificial bias |
| `max_glu_serum`, `A1Cresult` | 94.7% / 83.3% | Kept, filled as `"Not_Measured"` — in clinical workflows, *not ordering* a test is itself informative, not just missing data |
| `medical_specialty` | 49.1% | Filled as `"Missing"` — admitting specialty is a proxy for patient acuity |
| `payer_code` | 39.6% | Filled as `"Missing"` — insurance type proxies socioeconomic access to care |
| `race`, `diag_1/2/3` | <2% each | Rows dropped — negligible data loss, and avoids unstable, low-support dummy variables downstream |

Encounters discharged to hospice or deceased (discharge codes 11, 13, 14, 19, 20, 21) were also excluded — those patients cannot be meaningfully "readmitted."

**2. Feature engineering:** Raw ICD-9 diagnosis codes (hundreds of distinct values) were bucketed into 8 broad clinical categories (Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other) rather than one-hot-encoding every individual code.

**3. Avoiding a subtle leakage bug:** ~17,000 patients in this dataset have more than one encounter. A random row-level train/test split would let the same patient appear in both sets, inflating test performance. The split here is done with `GroupShuffleSplit`, grouped by `patient_nbr`, so no patient appears in both train and test.

**4. Modeling:** Logistic Regression (interpretable baseline, continuous features standardized post-split to avoid leakage) and Random Forest (captures non-linear effects), both trained with `class_weight='balanced'` given the class imbalance.

**5. Threshold tuning:** Rather than using the default 50% cutoff, the decision threshold was tuned against a realistic operational constraint — most hospital care-management programs can only intensively follow up with roughly the top 10–20% highest-risk discharges. The precision/recall tradeoff was mapped across a full range of flagging capacities (5–50%) before picking an operating point.

| Top % flagged | Threshold | Precision | Recall |
|---|---|---|---|
| 5% | 0.610 | 0.235 | 0.103 |
| 10% | 0.568 | 0.218 | 0.190 |
| 15% | 0.542 | 0.206 | 0.269 |
| **20%** | **0.521** | **0.190** | **0.332** |
| 25% | 0.503 | 0.183 | 0.399 |
| 30% | 0.489 | 0.173 | 0.453 |
| 40% | 0.464 | 0.160 | 0.557 |
| 50% | 0.440 | 0.150 | 0.652 |

**Operating point chosen: top 20%** (threshold 0.521) — assuming a care team can realistically manage intensive follow-up for about 1 in 5 discharged patients. This is a business-capacity assumption, not something the math picks on its own.

## Results

| Model | Precision (readmit) | Recall (readmit) | F1 | Accuracy | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.17 | 0.49 | 0.25 | 0.66 | — |
| Random Forest (default 0.5 threshold) | 0.18 | 0.41 | 0.25 | 0.72 | 0.631 |
| Random Forest (tuned, top-20% threshold) | 0.19 | 0.33 | 0.24 | 0.76 | — |

An ROC-AUC of ~0.63 is consistent with published results on this dataset (Strack et al., 2014) — 30-day readmission is driven substantially by factors this dataset doesn't capture (see Limitations).

**Consensus risk drivers** (features both Logistic Regression and Random Forest independently rank among their strongest predictors):
- **Increase risk:** number of prior inpatient visits, age 70–90, being on a diabetes medication, no A1C test measured during the stay
- **Decreases risk:** a primary diagnosis in the respiratory category

Two independent model families agreeing on the same drivers is a stronger claim than either model's feature ranking alone.

## Limitations

- Socioeconomic factors that plausibly drive readmission most — income, housing stability, post-discharge support, medication adherence — aren't in this dataset at all.
- Precision on the positive class is low (~0.19 at the chosen threshold): of the patients flagged high-risk, roughly 1 in 5 will actually be readmitted. This is a **screening tool to prioritize limited follow-up capacity**, not a diagnostic instrument, and shouldn't be read as "this specific patient will be readmitted."
- The dataset spans 1999–2008; clinical practice and hospital systems have changed since then.
- This is a portfolio/learning project and has not been clinically validated.

## Repository structure

```
├── Hospital_Readmission_Risk.ipynb   # Full analysis: EDA, cleaning, feature engineering,
│                                      # modeling, threshold tuning, evaluation
├── app.py                            # Streamlit app
├── icd9_utils.py                     # ICD-9 -> clinical category mapping (shared by
│                                      # notebook and app — keep these in sync)
├── random_forest_model.joblib        # Trained RF model (used by the app)
├── logistic_regression_model.joblib  # Trained LR model (baseline/interpretation)
├── scaler.joblib                     # StandardScaler fitted on training data only
├── model_columns.json                # Exact feature column order the model expects
├── scaled_columns.json               # Which columns get standardized at inference
├── threshold.json                    # Chosen operating threshold (0.521 = top 20%)
├── outputs_readmit_by_los.png        # EDA chart: readmission rate by length of stay
└── requirements.txt
```

## Running it locally

```bash
git clone https://github.com/Ajay9760/Hospital-Readmission-RIsk.git
cd Hospital-Readmission-RIsk
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL it prints (usually `http://localhost:8501`).

To reproduce the analysis from scratch, open `Hospital_Readmission_Risk.ipynb` — it downloads/expects the raw UCI dataset (`diabetic_data.csv`) and runs the full pipeline end to end, including re-exporting the model artifacts.

## Tech stack

Python · pandas · scikit-learn · Streamlit · matplotlib/seaborn

## Data source & citation

Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J., & Clore, J.N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International. Data shared via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008) (CC BY 4.0).

---

*This is a personal/portfolio project built for learning purposes and is not a validated clinical decision-making tool.*
