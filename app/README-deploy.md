# Deploying the demo

The app (`app/app.py`) runs entirely from `app/artifacts/` — a cached,
stratified 20K-row sample of the *test month* with precomputed calibrated
scores, per-row SHAP top-contributors, drift series, the frozen policy table,
and investigation narratives. No Kaggle download, no model, and no network
access are needed at runtime.

## Run locally

```bash
pip install -r app/requirements-app.txt
make evaluate          # once, to produce app/artifacts/ (requires the data)
streamlit run app/app.py
```

## Streamlit Community Cloud

1. Push this repository to GitHub **including** `app/artifacts/` (see the
   licensing note below — the folder is gitignored by default).
2. On share.streamlit.io: New app → pick the repo/branch → main file
   `app/app.py` → advanced settings → requirements file
   `app/requirements-app.txt` → deploy.

## Hugging Face Spaces

1. Create a Space (SDK: Streamlit).
2. Push `app/app.py`, `app/requirements-app.txt` (rename/copy to
   `requirements.txt` at the Space root), and `app/artifacts/`.
3. The Space builds and serves automatically.

## Licensing note — read before deploying publicly

`app/artifacts/sample.parquet` and `shap_top.parquet` are **derived from the
IEEE-CIS competition data**, which is licensed "subject to the Competition
Rules" and must not be redistributed outside the competition. A public
deployment that ships these artifacts is a redistribution decision that the
repository owner must make consciously — options, in decreasing order of
caution:

- keep the demo local (default; artifacts are gitignored),
- deploy to a **private** Space / app for interview demonstrations,
- replace the cached sample with a synthetic sample before any public deploy
  (the pipeline's smoke-test generator shows the shape such data needs).

The model file, metrics JSONs, and figures aggregate the data heavily and are
lower-risk, but the row-level sample is the sensitive artifact.
