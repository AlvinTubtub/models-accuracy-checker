# Baseline Audit: ForecastPH Models Accuracy Checker

**Date:** September 15, 2026  
**Branch:** `feature/formal-and-prospective-validation`  
**Target Project:** `models-accuracy-checker` (independent validation tool for `pse-stock-price-forecast`)

---

## 1. Executive Summary

This baseline audit assesses the current state of the `models-accuracy-checker` repository prior to implementing formal verification, prospective ledger tracking, and rigorous statistical tests. 

The existing repository is fully operational for its initial scope:
- **Baseline Test Suite:** 25/25 unit tests passing (`pytest`).
- **CLI Tools:** Auditing, training, data fetching, and sample data generation are functional.
- **Streamlit Dashboard:** 4-tab interactive dashboard runs with zero syntax or import errors.
- **Verification Engine:** Pure-Python independent metric recomputation (RMSE, MAE, MASE, $R^2$) with configurable tolerance checking.

---

## 2. Identified Repository Structure & Components

### 2.1 Package Architecture
- **`config/`**:
  - `companies.py`: Defines 15 Philippine Stock Exchange (PSE) listed companies across Financials, Properties, Industrials, Holding Firms, and Services, complete with tickers and metadata.
- **`recheck/`** (Independent Verification Core):
  - `metrics.py`: Mathematical definitions for RMSE, MAE, MASE, and $R^2$.
  - `schema.py`: Pydantic/dataclass-style schema validation for evaluation exports.
  - `loader.py`: Evaluator JSON loaders and manifest parsers.
  - `compare.py`: Core discrepancy check between reported metrics and freshly computed metrics.
  - `aggregate.py`: Multi-symbol cross-sectional summary statistics.
  - `denominator.py`: Naive development-set mean absolute first difference calculator for MASE.
- **`training/`** (Independent Re-implementation):
  - `pipeline.py`: End-to-end training orchestrator for LiR, ARIMA, LSTM, and Naive.
  - `features.py`: Technical indicators (RSI, MACD) and PACF lag selection.
  - `split.py`: 80/20 chronological time-series splitting.
  - `models/`: Implementations for `lir.py`, `arima.py`, `lstm.py` (PyTorch LSTM with gradient clipping and EarlyStopping), and `naive.py` (shift-1 baseline).
- **`bridge/`**:
  - `export_evaluations.py`: Extraction script intended to run against `pse-stock-price-forecast` joblib artifacts to export JSON evaluations.
- **`dashboard/`**:
  - `app.py`: Multi-tab Streamlit dashboard.
  - `components/charts.py` & `components/tables.py`: Plotly charts and formatted metric tables.
- **`scripts/`**:
  - `run_recheck_cli.py`: CLI audit tool with exit codes (0 = pass, 1 = audit failure).
  - `train_models.py`: CLI entry point for training pipelines.
  - `fetch_raw_data.py`: Raw CSV downloader.
  - `generate_sample_data.py`: Synthetic evaluation generator for offline testing.
- **`data/`**:
  - `raw/`: 15 CSV files with historical daily PSE prices (2020-01-02 to 2026-09-11, 1633 rows each).
  - `evaluations/`: 15 evaluation JSONs + `manifest.json`.

---

## 3. Current Behavior Baseline

### 3.1 CLI Entry Points
- `python scripts/run_recheck_cli.py`: Audits all JSON files in `data/evaluations/`. Verified working with 15/15 evaluations passing within tolerance $10^{-4}$.
- `python scripts/train_models.py --symbol <SYM>`: Trains LiR, ARIMA, and LSTM models from local raw CSV data.
- `python scripts/generate_sample_data.py`: Generates synthetic data with reproducible random seeds.

### 3.2 Data & Schema Assumptions
- Current evaluation files (`data/evaluations/<SYM>.json`) conform to `EvaluationSchema`:
  - Requires `symbol`, `split` (development/evaluation pairs and dates), `mase_denominator`, `metrics` per model, `principal_ranking`, and `backtest` (historical predicted vs actual closes).
- **Provenance limitation**: Only `manifest.json` holds an overall `is_synthetic` flag. Individual `<SYM>.json` files do not currently store distinct data provenance, Git commit hashes, input data hashes, or experiment run metadata.

### 3.3 Metric Math Implementations
- **MAE:** $\frac{1}{n} \sum |y_t - \hat{y}_t|$
- **RMSE:** $\sqrt{\frac{1}{n} \sum (y_t - \hat{y}_t)^2}$
- **MASE:** $MAE / \text{mase\_denominator}$, where $\text{mase\_denominator} = \frac{1}{N-1} \sum_{i=2}^N |y_i - y_{i-1}|$ on the development (training) period.
- **$R^2$:** $1 - \frac{\sum (y_t - \hat{y}_t)^2}{\sum (y_t - \bar{y})^2}$.

---

## 4. Current Test Baseline

Executed via `./.venv/bin/pytest --collect-only -q` and verified via `./.venv/bin/pytest -v`:
- `tests/test_compare.py`: 2 tests (exact match, discrepancy detection).
- `tests/test_loader.py`: 3 tests (valid export loading, corrupted export skipping, missing directory handling).
- `tests/test_metrics.py`: 6 tests (perfect predictions, analytical values, length mismatch, invalid denominator, non-finite handling, denominator calculation).
- `tests/test_schema.py`: 6 tests (valid payload parsing, missing root keys, missing model keys, chronological order check, duplicate date check, length mismatch check).
- `tests/test_training.py`: 8 tests (RSI calculation, PACF lag selection, feature dataframe creation, split planning, Naive model check, LiR training smoke test, ARIMA training smoke test, LSTM training smoke test).
- **Total:** **25 passed in 2.61s**. Zero warnings, zero failures.

---

## 5. Known Implementation Gaps (Targeted for Resolution)

1. **Statistical Significance Testing:**
   - Within-company: No Diebold-Mariano tests (with Newey-West/HAC variance and Harvey-Leybourne-Newbold small-sample correction) across all 6 model pairs with Holm multiplicity correction.
   - Cross-company: No Friedman omnibus test across the 15 companies with conditional post-hoc Wilcoxon signed-rank tests.

2. **Error Differential Time Series:**
   - Point-by-point loss differentials ($d_t = L(e_{1,t}) - L(e_{2,t})$) for both squared-error and absolute-error loss are not computed or persisted.

3. **Data Provenance & Formal Schema:**
   - No explicit classification between `DEMO`, `FORMAL_FROZEN`, and `PROSPECTIVE`.
   - No raw CSV SHA-256 digests, PSE trading calendar completeness checks, or duplicate/missing session audits.
   - No canonical research-payload digest or separate manifest digest for formal archives.

4. **Model-Audit Verification (No Retraining):**
   - No automated verification of exported ForecastPH evidence (ARIMA candidate space / no-drift (0,1,0) presence / convergence; LASSO alpha grid boundary check; LSTM seed and lookback consistency).

5. **Prospective Pre-Settlement Ledger:**
   - No immutable storage (SQLite) capturing forward forecasts where `forecast_created_at` strictly precedes target settlement availability.
   - No pre-settlement lock on predictions (`symbol`, `model`, `model_version`, `origin_date`, `target_date`, `predicted_close`, `origin_close`, `source_artifact_hash`).

6. **End-of-Day (EOD) Settlement Reconciliation:**
   - No pipeline to reconcile pending prospective forecasts with official PSE closing prices without altering finalized formal experiments.

7. **Trading-Session Rolling Windows & Drift:**
   - Fixed split only; no rolling trading-session windows (latest 5, 20, 60 sessions, and model lifetime) or concept drift / variance monitoring.

8. **Configurable Promotion Eligibility:**
   - No automated assessment of prospective promotion eligibility (minimum ~20 settled sessions, outperforming Naive on RMSE/MAE, outlier robustness, and convergence audit).
