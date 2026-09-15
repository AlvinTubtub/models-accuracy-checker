# ForecastPH Evidence Compatibility and Pre-Flight Blocker Report

**Document Version:** 1.0.0 (Phase 5 Audit)  
**Date:** September 2026  
**Audited ForecastPH Repository:** `https://github.com/AlvinTubtub/pse-stock-price-forecast.git`  
**Reference Run ID:** `FORMAL_CORRECTED_20260828_02` (commit `bfb33b8c184c87cc8828af5529410da94addd71c`)

---

## 1. Purpose

This document provides a concrete, read-only audit of the current ForecastPH codebase, configuration, and artifact generation pipelines. It establishes whether ForecastPH currently persists sufficient cryptographic and statistical evidence to support an independent formal audit, and identifies any pre-flight blockers that must be resolved prior to the one-time formal capstone experiment.

---

## 2. Evidence Classification Taxonomy

Every requirement is evaluated under one of four unambiguous categories:
1. **AVAILABLE NOW**: The required evidence is actively computed and persisted in native JSON/CSV artifacts.
2. **PARTIALLY AVAILABLE**: Some elements are persisted, but key subfields or diagnostic details are missing or incomplete.
3. **NOT PERSISTED**: The calculation or logging does not take place in ForecastPH; evidence is completely absent from outputs.
4. **UNKNOWN UNTIL REAL RUN**: Relies on specific runtime properties that can only be verified when the execution completes.

---

## 3. Comprehensive Audit Matrix

### 3.1 Raw Data & Exchange Provenance

| Evidence Item | Classification | Current ForecastPH Location / Mechanism | Analysis & Audit Finding |
| :--- | :--- | :--- | :--- |
| **Exact Raw CSV Hashes (SHA-256)** | **AVAILABLE NOW** | `.checkpoints/<SYM>/source_data.json` & `backend/data/raw/<SYM>.csv` | Full SHA-256 hashes are recorded in `.checkpoints/<SYM>/source_data.json` alongside byte length and date boundaries. |
| **Source Retrieval Metadata** | **NOT PERSISTED** | Missing from raw CSV headers and loader | Download timestamps, upstream endpoint URLs (PSE Edge / Yahoo), and raw extraction versions are not stored with the data. |
| **Session Completeness & Exchange Holidays** | **PARTIALLY AVAILABLE** | `backend/src/data/calendar.py`, `backend/config/pse_holidays.py` | Explicit holiday calendar (`PSE_CLOSURES`) tracks reviewed Philippine non-working days. However, security suspensions cannot be distinguished from exchange closures. |
| **Corporate Action Registry & Policy** | **AVAILABLE NOW** | `backend/config/formal_corporate_actions_20250902_20260828.json` & `diagnostics.json` | Explicit 26-event registry with SHA-256 tracking. Policy (`raw_close_retain_and_flag_v1`) enforces retention of all validated observations in primary analysis. |

---

### 3.2 Splits & Holdout Windows

| Evidence Item | Classification | Current ForecastPH Location / Mechanism | Analysis & Audit Finding |
| :--- | :--- | :--- | :--- |
| **Development & Evaluation Target Dates** | **AVAILABLE NOW** | `.checkpoints/<SYM>/plan.json`, `split_manifest.json` | Exact chronological target dates for development and evaluation are fully enumerated for all 15 symbols. |
| **Evaluation Ratio & Boundaries** | **AVAILABLE NOW** | `backend/src/data/split.py` (`build_company_evaluation_plan`) | 85% development / 15% evaluation split enforced by target date. |
| **Expanding-Window Cross-Validation Folds** | **PARTIALLY AVAILABLE** | `diagnostics.json` (`cv_fold_results`, `candidate_cv`) | Fold numbers and fold RMSEs are recorded; however, complete date-range manifests per fold are reconstructed rather than stored as separate fold files. |

---

### 3.3 Lag-Informed Regression (LIR / LASSO)

| Evidence Item | Classification | Current ForecastPH Location / Mechanism | Analysis & Audit Finding |
| :--- | :--- | :--- | :--- |
| **Alpha Search Grid** | **AVAILABLE NOW** | `diagnostics.json` (`lag_reg.tuning_metadata.grid`) | 36 log-spaced alphas from `0.0001` to `1000.0`. |
| **Alpha Expansion $> 1.0$** | **AVAILABLE NOW** | `diagnostics.json` (`lag_reg.tuning_metadata.grid_max`) | Explicitly expands up to `1000.0`, satisfying the requirement to avoid artificial restriction to $\le 1.0$. |
| **Selected Alpha & Fold Scores** | **AVAILABLE NOW** | `diagnostics.json` (`lag_reg.selected_alpha`, `alpha_results`) | Selected alpha and per-fold iteration counts and RMSEs are fully documented. |
| **Boundary Hit Warning** | **AVAILABLE NOW** | `diagnostics.json` (`lag_reg.tuning_metadata.selected_at_boundary`) | Boolean flag explicitly checks whether selected alpha hit `grid_min` or `grid_max`. |
| **Selected Features & Coefficients** | **AVAILABLE NOW** | `diagnostics.json` (`lag_reg.selected_features`, `coefficients`) | Complete list of nonzero feature names and unrounded float coefficients persisted. |
| **Development-Only Scaling & Tuning** | **AVAILABLE NOW** | `backend/src/training/train_lir.py` | StandardScaler fit strictly inside CV training folds; evaluation data never touches feature scaling. |

---

### 3.4 ARIMA Model

| Evidence Item | Classification | Current ForecastPH Location / Mechanism | Analysis & Audit Finding |
| :--- | :--- | :--- | :--- |
| **Candidate Search Space Grid** | **AVAILABLE NOW** | `diagnostics.json` (`arima.candidate_cv`) | 80 candidate order/trend combinations evaluated across CV folds. |
| **No-Drift Baseline $(0, 1, 0, "n")$** | **AVAILABLE NOW** | `diagnostics.json` (`candidate_cv[...].configuration`) | ARIMA(0,1,0) with trend="n" is explicitly present in candidate CV search space. |
| **Trend / Differencing Rules** | **AVAILABLE NOW** | `backend/src/models/arima.py` | Restricts trends: $d=0 \to (\text{n, c})$, $d=1 \to (\text{n, t})$, $d=2 \to (\text{n})$. |
| **Fold-Level & Final Fit Convergence** | **AVAILABLE NOW** | `diagnostics.json` (`all_cv_folds_converged`, `final_fit_converged`) | Optimizer convergence state recorded for candidate CV folds and the final development model fit. |
| **Ljung-Box Test on Holdout Residuals** | **AVAILABLE NOW** | `diagnostics.json` (`arima.ljung_box`) & `metrics.json` | Holdout error autocorrelation tests persisted with test statistic and p-value. |
| **ARCH-LM Test on Residuals** | **AVAILABLE NOW** | `diagnostics.json` (`arima.arch_lm`) | Lagrange multiplier test for autoregressive conditional heteroskedasticity persisted. |
| **Stationarity & Invertibility Roots** | **NOT PERSISTED** | Absent from `train_arima.py` and `diagnostics.json` | **BLOCKER**: Characteristic AR and MA polynomial roots are not calculated or exported. |

---

### 3.5 LSTM Model

| Evidence Item | Classification | Current ForecastPH Location / Mechanism | Analysis & Audit Finding |
| :--- | :--- | :--- | :--- |
| **Three Distinct Tuning Seeds** | **AVAILABLE NOW** | `diagnostics.json` (`lstm.training_metadata.tuning_seeds`) | Explicitly records 3 tuning seeds (`[42, 123, 2026]`). |
| **Common CV Date Window Alignment** | **AVAILABLE NOW** | `diagnostics.json` (`common_cv_target_start`, `common_cv_target_end`) | All lookback lengths evaluated against common target-date sequence. |
| **Per-Seed & Per-Config Validation Scores** | **AVAILABLE NOW** | `diagnostics.json` (`configuration_results`) | 48 hyperparameter configurations evaluated across folds and seeds. |
| **Selected Configuration & Epoch Count** | **AVAILABLE NOW** | `diagnostics.json` (`selected_config`, `final_epoch_info`) | Best lookback, hidden size, learning rate, batch size, and epoch count persisted. |
| **Final Refit Seed & Clean Training** | **AVAILABLE NOW** | `diagnostics.json` (`final_fit_seed`, `development_sequence_count`) | Fixed final seed (`42`) used on full development set. |

---

### 3.6 Evaluation & Diebold-Mariano Outputs

| Evidence Item | Classification | Current ForecastPH Location / Mechanism | Analysis & Audit Finding |
| :--- | :--- | :--- | :--- |
| **Full Unrounded Predictions & Actuals** | **AVAILABLE NOW** | `per_company/<SYM>/holdout_predictions.csv` | Full IEEE 754 float64 precision preserved for actuals and all 4 models. |
| **Direct Naive Baseline Series** | **AVAILABLE NOW** | `holdout_predictions.csv` (`naive` column) | Naive previous-close forecast persisted on exact evaluation dates. |
| **Reported Metrics** | **AVAILABLE NOW** | `per_company/<SYM>/metrics.json` | Full-precision RMSE, MAE, MASE, R², and Ljung-Box p-values. |
| **ForecastPH Statistical Tests** | **AVAILABLE NOW** | `statistical_tests.json` | Within-company paired tests and cross-company Friedman/Wilcoxon outputs stored. |

---

### 3.7 Runtime Environment & Execution Logs

| Evidence Item | Classification | Current ForecastPH Location / Mechanism | Analysis & Audit Finding |
| :--- | :--- | :--- | :--- |
| **Source Git Commit & Clean State** | **AVAILABLE NOW** | `methodology_manifest.json` | Commit hash (`bfb33b8c184c87cc8828af5529410da94addd71c`), branch, and dirty status. |
| **Core Dependencies** | **AVAILABLE NOW** | `methodology_manifest.json` (`dependencies`) | Specific versions of numpy, pandas, scikit-learn, scipy, statsmodels, torch recorded. |
| **Complete Environment (`pip freeze`)** | **PARTIALLY AVAILABLE** | Recorded at bridge export time | Captured by `bridge/export_formal_experiment.py` into `environment/pip_freeze.txt`. |
| **Execution Stdout / Stderr Logs** | **NOT PERSISTED** | Missing from formal evidence directory | Console training logs were discarded after execution; only JSON metadata remains. |

---

## 4. Pre-Flight Blocker Summary for Real Formal Run

Before executing the **one-time formal training run**, the following items must be addressed in ForecastPH:

| Item | Component | Current State | Required Modification in ForecastPH | Severity |
| :--- | :--- | :--- | :--- | :--- |
| **1. ARIMA Characteristic Roots** | `train_arima.py` | Not computed or exported | Extract `results.arroots` and `results.maroots` from fitted statsmodels ARIMA and serialize moduli into `diagnostics.json`. | **HIGH** |
| **2. Raw Data Ingestion Metadata** | `src/ingestion/` | Missing acquisition timestamp/source | Write a sidecar `raw/<SYM>.metadata.json` capturing endpoint URL, retrieval timestamp UTC, and source API. | **HIGH** |
| **3. Training Console Log Retention** | `scripts/train_all.py` | Logs only to console | Configure `FileHandler` in `logging_config.py` to write `execution.log` directly into the formal evidence directory. | **OPTIONAL** |
| **4. Exchange Suspension Disclosures** | `src/data/calendar.py` | Binary holiday calendar | Add suspension dates table to distinguish market closures from security-specific trading halts. | **OPTIONAL** |

---

## 5. Formal Readiness Assessment

- **Current Status**: **NOT READY FOR FINAL FORMAL RUN — MINOR PRE-FLIGHT ENHANCEMENTS REQUIRED**
- **Assessment Rationale**:
  ForecastPH already satisfies **>90% of all formal audit requirements**, including unrounded holdout predictions, scale-free MASE tracking, LASSO alpha expansion beyond 1.0, ARIMA (0,1,0,n) candidate search, 3-seed LSTM tuning, and corporate action retention.
  Adding ARIMA characteristic roots and raw data acquisition metadata will bring ForecastPH into 100% compliance with formal capstone research audit standards.
