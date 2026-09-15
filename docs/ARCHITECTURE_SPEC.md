# Architecture Specification: Dual-System Formal Audit & Prospective Validation

## 1. System Overview & Core Invariant

The ForecastPH accuracy-checking platform is partitioned into two strictly decoupled systems that share independent mathematical validation primitives:

1. **System A: Formal / Frozen Experiment Audit**  
   Audits historical, frozen model evaluations against a fixed chronological holdout with immutable research archives and hypothesis testing across 15 matched company evaluations.
2. **System B: Prospective / Live EOD Monitoring**  
   Monitors forward-looking daily forecasts logged prior to session closing with EOD settlement against validated EOD closes with recorded provenance, rolling trading-session accuracy, and configurable promotion eligibility.

> [!IMPORTANT]
> **Strict Non-Interference Invariant**: Prospective settlements never modify, append to, or alter the evaluation boundary of a finalized formal experiment archive.

---

## 2. Prospective Timing & Settlement Invariant

A forecast is only valid if recorded before the market outcome is knowable. The timing invariant is:

$$\text{forecast\_created\_at} < \text{target\_actual\_available\_at}$$

where `target_actual_available_at` is derived from the target trading-session close / validated EOD availability boundary.

### Timing Evidence Fields
- `origin_date`: The trading day whose closing price provides the base features.
- `target_date`: The trading day being forecasted.
- `target_session_close_at`: The scheduled close / availability boundary of the target session.
- `forecast_created_at`: The cryptographic timestamp when the forecast was registered.
- `actual_received_at`: The timestamp when the settlement close was ingested.

> [!NOTE]
> A forecast created after the origin session close (e.g. at 18:00 PHT or 08:30 PHT the next morning) is fully valid, provided it strictly precedes `target_actual_available_at`. Any forecast recorded after target Close availability is marked **INVALID**, even if the settlement job has not yet ingested the actual Close.

---

## 3. Matched Cross-Company Evaluation Terminology

The 15 Philippine Stock Exchange companies share macro market conditions, monetary policy shifts, and local sentiment. Therefore, they are formally designated as:

> **"15 matched company evaluations"** (NOT independent experimental units).

Cross-company evaluations (Friedman omnibus rank test and post-hoc paired Wilcoxon signed-rank tests) serve as **supporting cross-company evidence** rather than independent trials.

---

## 4. Frozen Raw Data Integrity

The formal verification suite relies strictly on frozen raw CSV inputs.
- `scripts/fetch_raw_data.py` is removed from the core formal experiment workflow.
- Every input dataset is validated against:
  - Filename and relative path
  - SHA-256 cryptographic digest
  - Exact row count
  - Monotonic date range (first date and last date)
  - Provenance tier (`DEMO` vs `FORMAL_FROZEN`)
  - Calendar completeness (duplicate session detection and trading gap checks)
- The platform will **never** silently fetch, substitute, or interpolate newer data into a frozen formal experiment.

---

## 5. Full Numeric Precision & Canonical Hashing

Canonical research-payload digests are computed **without artificial rounding or fixed-precision truncation**:
- Predictions and metric values are preserved at their exact exported floating-point precision.
- Serialization determinism is achieved via:
  - Canonical JSON formatting (sorted keys, compact separators `","` and `":"`)
  - Stable UTF-8 encoding
  - Substantive field filtering (excluding ephemeral run timestamps and local filepaths)
- A separate manifest file records environment metadata (git commit, runtime, timestamp) and its own manifest digest.

---

## 6. Configurable Promotion Policy (`config/promotion.toml`)

Promotion configuration is managed via standard-library `tomllib` (Python 3.11+) in `config/promotion.toml`.

- The approximately 20 settled-session minimum is a configurable default operational policy, not a universal mathematical constant.
- The system evaluates **eligibility only** and never triggers automated deployment or model retraining.

---

## 7. Full Holdout Invariant & 60-Session Display Rule

- **No Hardcoded Sample Sizes**: Observation counts and date ranges are read directly from the frozen run manifest.
- **Formal Metrics & Hypothesis Tests**: Computed exclusively over the **complete frozen formal holdout**.
- **Display Subsets**: Any 60-session historical chart is explicitly titled: **"Latest 60 sessions from the complete formal holdout"**.
- **Prospective Monitoring**: Rolling 60 settled trading sessions is a distinct live monitoring concept that operates strictly within System B.
- **Data Source Wording**: All price series are referenced as **"validated EOD closes with recorded provenance"** unless authoritative upstream metadata confirms an official exchange feed.
