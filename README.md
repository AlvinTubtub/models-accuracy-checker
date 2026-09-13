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
   ```
   python export_evaluations.py --output ../../evaluation_export
   ```
4. Copy the resulting `evaluation_export/` folder (one JSON file per
   company, plus `manifest.json`) into **this** repo's `data/evaluations/`
   directory.

Full details, including what the script actually does and why it re-uses
the original repo's own checksum-validated loader instead of a raw
`joblib.load`, are in `bridge/README.md`.

## Step 2 — recheck and view the dashboard (run here, independently)

```
pip install -r requirements.txt

# headless recheck (prints a report, exits non-zero on mismatch/failure)
python scripts/run_recheck_cli.py --data-dir data/evaluations

# interactive dashboard
streamlit run dashboard/app.py
```

If you just want to see the dashboard work before wiring up real data:

```
python scripts/generate_sample_data.py --output data/evaluations
streamlit run dashboard/app.py
```

This generates synthetic (clearly labeled as synthetic) per-company
evaluation exports in the same schema the real bridge script produces, so
the dashboard and CLI are fully exercised without needing the original repo
at all. **A set of these sample files ships in `data/evaluations/` already**
so `streamlit run dashboard/app.py` works immediately after `pip install` —
replace them with real exports (Step 1 above) whenever you're ready, or
re-run the generator to reset them. One sample company (`SCC`) has a
deliberately corrupted metric baked in, so the "Recomputed vs. Reported"
tab has a real mismatch to show on first run.

## What "recheck" actually means here

For each company, for each of the four series (Lag-Informed Regression,
ARIMA, LSTM, Naive), across every date in the reported evaluation window
(not just a 60-session slice):

1. **Validate** the export — chronological dates, no duplicates, every
   model has a prediction for every date, and the `actual_close` is
   identical across all four series for a given date (if it isn't, the
   export itself is broken and this is flagged before any metric is
   trusted).
2. **Recompute RMSE, MAE, MASE, R²** from scratch, from the raw
   `actual_close` / `predicted_close` arrays, using this repo's own
   `recheck/metrics.py` — not a copy of the original's formulas, an
   independent implementation of the same standard definitions.
3. **Compare** the recomputed values against the `reported_metrics` the
   original pipeline shipped in the export, with a configurable numerical
   tolerance. Anything outside tolerance is surfaced as a mismatch, not
   silently averaged away.
4. Optionally, **recompute the MASE denominator itself** from the
   company's raw OHLCV CSV (see `recheck/denominator.py`), rather than
   trusting the denominator the original pipeline used. This step is
   optional because it needs the raw CSVs from `backend/data/raw/`, which
   (unlike the evaluations) **are** committed to the original repo, so you
   can `git clone` them directly with no bridge step.

## Repository layout

```
bridge/                  runs INSIDE the original repo, produces JSON only
  export_evaluations.py
  README.md
recheck/                 independent recompute/validate/compare logic
  metrics.py              RMSE / MAE / MASE / R² from scratch
  schema.py                export schema + validation
  loader.py                load one/all company export files
  compare.py                recomputed vs. reported diffing
  aggregate.py              cross-company roll-ups (win-rates, sectors)
  denominator.py            optional raw-CSV MASE-denominator cross-check
scripts/
  run_recheck_cli.py       headless CLI report
  generate_sample_data.py  synthetic demo data generator
dashboard/
  app.py                   Streamlit dashboard
data/
  evaluations/             put exported (or sample) JSON files here
tests/
  test_metrics.py
  test_loader.py
  test_compare.py
```

## Pushing this to a new repo

```
cd pse-model-accuracy-recheck
git init
git add .
git commit -m "Initial independent Tier-2 accuracy recheck pipeline"
git branch -M main
git remote add origin <your-new-repo-url>
git push -u origin main
```

`data/evaluations/*.json` is **not** gitignored by default — decide for
yourself whether you want the exported evaluation data committed to your
new repo or kept local/private. If you'd rather not commit it, add
`data/evaluations/*.json` to `.gitignore` before your first commit.
