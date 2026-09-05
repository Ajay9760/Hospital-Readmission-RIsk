# ============================================================
# Add this as a NEW cell at the end of your notebook and run it.
# It saves your actual trained model + everything needed to
# preprocess a new patient the same way at inference time.
# ============================================================
import joblib
import json

# 1. Save the trained model
joblib.dump(rf, "rf_model.joblib")

# 2. Save the exact column order/names your model was trained on
#    (needed to align a new patient's one-hot-encoded row later)
feature_columns = list(X_train.columns)
with open("feature_columns.json", "w") as f:
    json.dump(feature_columns, f)

# 3. Save which columns were standardized (if you used StandardScaler)
#    and the fitted scaler itself. SKIP this block if you didn't scale.
# joblib.dump(scaler, "scaler.joblib")
# with open("scaled_columns.json", "w") as f:
#     json.dump(list(continuous_features), f)  # whatever list you scaled

# 4. Save the chosen operating threshold (from your lift-curve analysis)
with open("threshold.json", "w") as f:
    json.dump({"threshold": 0.30}, f)  # <-- replace with your chosen value

print("Saved: rf_model.joblib, feature_columns.json, threshold.json")
print("Download these files from the Colab file browser (left sidebar).")
