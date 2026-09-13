"""Independent recompute/validate/compare logic for the Tier-2 accuracy recheck.

Nothing in this package imports from, or depends on, the original
ForecastPH repository. It only consumes plain JSON produced by
`bridge/export_evaluations.py` (or raw OHLCV CSVs, for the optional
denominator cross-check in `recheck/denominator.py`).
"""
