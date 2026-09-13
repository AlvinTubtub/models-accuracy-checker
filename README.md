# PSE Model Accuracy Recheck (Tier 2)

An **independent** codebase for rechecking whether ForecastPH's three models
(Lag-Informed Regression, ARIMA, LSTM) actually beat the Naive benchmark
(`Close(t+1) = Close(t)`) over the Jan 2, 2020 – Sep 11, 2026 evaluation
window produced by a **PSE Fresh Model Training** run.

This repo is deliberately separate from
`https://github.com/AlvinTubtub/pse-stock-price-forecast.git`. It does not
import that repo's code, does not reuse its metric implementations, and does
not trust its reported numbers — it recomputes RMSE / MAE / MASE / R² itself
from the raw predicted-vs-actual Close values and flags any disagreement.

## Why two stages?

The full per-date evaluation data (`evaluation.joblib`) that this recheck
needs is **not committed to git** in the source repo — `backend/artifacts/`
is gitignored there. It only exists:

- locally, after someone runs `scripts/train_all.py --all --fresh` in that
  repo, or
- as the `forecastph-backend-artifacts` GitHub Actions artifact uploaded by
  its `train_models.yml` workflow (90-day retention).

So getting the data out requires one short script (the "bridge") that runs
**inside a clone of the original repo**, because only that repo's Python
environment can unpickle its own `CompanyEvaluation` objects. The bridge's
only job is exporting each company's full evaluation record to plain JSON.
Everything after that — loading, validating, recomputing metrics, comparing,
dashboarding — lives entirely in this repo and never touches the original
code again.

```
┌─────────────────────────────┐        plain JSON        ┌──────────────────────────────┐
│ original repo (unchanged)   │ ────────────────────────▶ │ this repo (independent)      │
│  backend/artifacts/         │   bridge/export_          │  recheck/  (own metrics math) │
│  evaluations/<SYM>/         │   evaluations.py           │  dashboard/ (Streamlit)       │
│  evaluation.joblib          │   (run once, there)        │  scripts/  (CLI)              │
└─────────────────────────────┘                            └──────────────────────────────┘
```

## Step 1 — export the data (run inside the ORIGINAL repo)

1. Clone `https://github.com/AlvinTubtub/pse-stock-price-forecast.git` and
   make sure `backend/artifacts/evaluations/` is populated — either:
   - run `python scripts/train_all.py --all --fresh --verbose` from
     `backend/` yourself, **or**
   - download the `forecastph-backend-artifacts` artifact from a recent
     "PSE Fresh Model Training" GitHub Actions run and unzip it so its
     `evaluations/` folder lands at `backend/artifacts/evaluations/`.
2. Copy `bridge/export_evaluations.py` (from this repo) into that repo's
   `backend/` directory.
3. From `backend/`, with that repo's own virtualenv active (it needs
   `joblib`, `numpy`, `torch`, `scikit-learn`, `statsmodels` — the same deps
   it already installs), run:
   ```bash
   python export_evaluations.py --output ../../evaluation_export
   ```
4. Copy the resulting `evaluation_export/` folder (one JSON file per
   company, plus `manifest.json`) into **this** repo's `data/evaluations/`
   directory.

Full details, including what the script actually does and why it re-uses
the original repo's own checksum-validated loader instead of a raw
`joblib.load`, are in `bridge/README.md`.

## Step 2 (Alternative A) — Tier 3: Train models directly from raw data (Recommended)

You do **not** need the bridge script if you prefer running the exact model training pipeline directly from raw historical market data:

```bash
pip install -r requirements.txt

# 1. Download raw historical data for all 15 PSE tickers (2020-01-02 to 2026-09-11)
python scripts/fetch_raw_data.py

# 2. Train LIR, ARIMA, LSTM, and Naive models manually (no cron jobs)
python scripts/train_models.py --all

# 3. Run independent audit
python scripts/run_recheck_cli.py --data-dir data/evaluations

# 4. Launch interactive Streamlit dashboard
streamlit run dashboard/app.py
```

## Step 2 (Alternative B) — Tier 2: Recheck exported artifacts

If you already exported `<SYMBOL>.json` files using the bridge script:

```bash
pip install -r requirements.txt

# Run automated test suite (25 unit tests)
pytest -v

# Headless recheck (prints formatted audit report)
python scripts/run_recheck_cli.py --data-dir data/evaluations

# Launch interactive Streamlit dashboard
streamlit run dashboard/app.py
```

## Azure for Students Deployment

To deploy this pipeline and host the Streamlit dashboard 24/7 for free using your Azure for Students subscription (with manual training execution), see [AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md).

## What "recheck" actually means here

For each company, for each of the four series (Lag-Informed Regression,
ARIMA, LSTM, Naive), across every date in the reported evaluation window:

1. **Validate** the export — chronological dates, no duplicates, every
   model has a prediction for every date, and the `actual_close` is
   identical across all four series for a given date (if it isn't, the
   export itself is broken and this is flagged before any metric is
   trusted).
2. **Recompute RMSE, MAE, MASE, R²** from scratch, from the raw
   `actual_close` / `predicted_close` arrays, using this repo's own
   `recheck/metrics.py` — an independent implementation of the standard
   mathematical definitions:
   - $\text{RMSE} = \sqrt{\frac{1}{n}\sum (y_t - \hat{y}_t)^2}$
   - $\text{MAE} = \frac{1}{n}\sum |y_t - \hat{y}_t|$
   - $\text{MASE} = \frac{\text{MAE}}{\text{mase\_denominator}}$
   - $R^2 = 1 - \frac{\sum (y_t - \hat{y}_t)^2}{\sum (y_t - \bar{y})^2}$
3. **Compare** the recomputed values against the `reported_metrics` the
   original pipeline shipped in the export, with a configurable numerical
   tolerance (`1e-6`). Anything outside tolerance is surfaced as a mismatch.
4. **Evaluate Naive Benchmark**:
   - Computes whether $\text{MASE} < 1.0$ (indicating the model outperforms the Naive persistence baseline).
   - Tallies cross-company win-rates and identifies the best-performing model per company and sector.

## Repository layout

```
bridge/                  runs INSIDE the original repo, produces JSON only
  export_evaluations.py
  README.md
config/
  companies.py             15-ticker/sector reference metadata
recheck/                 independent recompute/validate/compare logic
  metrics.py              RMSE / MAE / MASE / R² from scratch
  schema.py                export schema + validation
  loader.py                load one/all company export files
  compare.py                recomputed vs. reported diffing
  aggregate.py              cross-company roll-ups (win-rates, sectors)
  denominator.py            optional raw-CSV MASE-denominator cross-check
dashboard/
  app.py                   Streamlit dashboard application
  components/
    charts.py              Plotly visualization routines
    tables.py              Styled summary tables and metrics display
scripts/
  run_recheck_cli.py       headless CLI report
  generate_sample_data.py  synthetic demo data generator
data/
  evaluations/             put exported (or sample) JSON files here
tests/
  test_metrics.py          unit tests for mathematical correctness
  test_schema.py           unit tests for invariant validation
  test_compare.py          unit tests for diffing & tolerances
  test_loader.py           unit tests for loader and error handling
```

## Pushing to your new repo

```bash
git init
git add .
git commit -m "Initial independent Tier-2 accuracy recheck pipeline"
git branch -M main
git remote add origin <your-new-repo-url>
git push -u origin main
```
