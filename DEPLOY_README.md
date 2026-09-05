# Deploying the Readmission Risk Predictor

## What's in this folder
- `app.py` — the Streamlit app
- `icd9_utils.py` — the ICD-9 categorization function (must match your notebook's `map_icd9` exactly)
- `export_snippet.py` — code to add to your notebook to save your real trained model
- `requirements.txt` — dependencies for local run or cloud deployment

## Important: this app needs YOUR model, not a placeholder
This was built and smoke-tested end-to-end using a reproduction of your pipeline,
but the artifacts it needs (`rf_model.joblib`, `feature_columns.json`,
`threshold.json`) are not included — they must come from your actual notebook,
so the deployed model is the one you actually trained and can explain, not a
stand-in.

## Step 1 — export your real model
1. Open your notebook.
2. Paste the contents of `export_snippet.py` as a new cell at the end and run it.
3. **Before running:** if your final `rf` was trained on features you standardized
   with `StandardScaler`, uncomment the scaler-saving lines in the snippet and
   set `USE_SCALER = True` in `app.py`. If not, leave both as-is.
4. Update the `threshold.json` value to whatever cutoff your lift-curve analysis
   pointed to (don't leave it at the placeholder 0.30).
5. Download `rf_model.joblib`, `feature_columns.json`, and `threshold.json`
   (and `scaler.joblib` / `scaled_columns.json` if applicable) from Colab's file
   browser, and place them in this same folder.

## Step 2 — check `icd9_utils.py` matches your notebook
If you changed the `map_icd9` function at any point (e.g. added a category),
copy your final version into `icd9_utils.py` so the app categorizes new
diagnosis codes exactly the way your training data was categorized.

## Step 3 — run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the URL it prints (usually http://localhost:8501).

## Step 4 — deploy it for free (Streamlit Community Cloud)
1. Push this folder (with your real artifacts added) to a public GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click "New app", pick the repo, branch, and `app.py` as the entry point.
4. Deploy. You'll get a public URL you can link from your resume/portfolio.

## A note for your write-up / interview
Being able to say "I deployed this as a live demo" is good, but be ready to
explain the one design decision every reviewer of a deployed ML app checks:
how a new patient's raw inputs get turned into the exact same encoded format
the model was trained on. That's the `pd.get_dummies` + `reindex(columns=...,
fill_value=0)` step in `app.py` — know why `reindex` (not just `get_dummies`)
is necessary here (a single new row won't naturally produce every dummy column
your training set had).
