#!/usr/bin/env python3
"""Bridge script: run this INSIDE a clone of the original ForecastPH repo.

This script does not belong to, and must never be committed into, the
independent recheck repo's "trusted" logic — it is a one-way export step.
Its only job is to turn the original pipeline's pickled evaluation
artifacts into plain, dependency-free JSON so a completely separate
codebase can recompute and cross-check the reported accuracy numbers
without ever importing the original pipeline's code again.

USAGE (from the original repo's `backend/` directory, with that repo's own
virtualenv active so `src.*` imports resolve):

    cp /path/to/this/export_evaluations.py .
    python export_evaluations.py --output ../../evaluation_export

Requirements this script relies on (already present in the original repo,
NOT installed by the independent recheck repo):
    - backend/src/... importable (run from backend/, or add it to PYTHONPATH)
    - joblib, numpy, torch, scikit-learn, statsmodels (already in
      backend/requirements.txt)
    - backend/data/raw/<SYMBOL>.csv present (used only to validate the
      evaluation artifact's checksum/boundary before trusting it)
    - backend/artifacts/evaluations/<SYMBOL>/evaluation.joblib present for
      each company (produced by `scripts/train_all.py --all --fresh`, or
      restored from the `forecastph-backend-artifacts` CI artifact)

What this script deliberately does NOT do:
    - It does not recompute or adjust any metric. It exports exactly what
      the original pipeline computed, verbatim, so the independent recheck
      repo can compare its own from-scratch numbers against these without
      any risk of the export step having quietly "helped."
    - It does not use a raw `joblib.load()` on the artifact file. It goes
      through the original pipeline's own `load_company_evaluation`, which
      validates the artifact's SHA-256 checksum and its training-boundary
      date against the current raw CSV before returning anything. If that
      validation fails for any company, this script reports the failure
      and continues with the rest rather than silently skipping it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="../evaluation_export",
        help="Directory to write one JSON file per company plus manifest.json "
        "(default: ../evaluation_export, relative to backend/)",
    )
    parser.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        default=None,
        help="Export only this symbol (repeatable). Default: all configured companies.",
    )
    arguments = parser.parse_args(argv)

    try:
        from config.companies import COMPANIES
        from src.data.loader import load_company_history
        from src.training.orchestration import load_company_evaluation, OrchestrationError
    except ImportError as exc:
        print(
            "ERROR: could not import the original repo's backend package.\n"
            "Run this script from inside that repo's backend/ directory, "
            "with its virtualenv active.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        return 2

    symbols = arguments.symbols or [company.symbol for company in COMPANIES]
    output_dir = Path(arguments.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: list[str] = []
    failed: list[str] = []

    for symbol in symbols:
        try:
            records = load_company_history(symbol)
            evaluation = load_company_evaluation(symbol, records)
        except (OrchestrationError, Exception) as exc:  # noqa: BLE001 - report, don't crash the batch
            failed.append(f"{symbol}: {type(exc).__name__}: {exc}")
            print(f"FAILED {symbol}: {exc}", file=sys.stderr)
            continue

        payload = evaluation.as_dict()
        # Attach the company's sector so the independent repo doesn't need
        # to re-derive it from anywhere else.
        company = next(c for c in COMPANIES if c.symbol == symbol)
        payload["sector"] = company.sector
        payload["name"] = company.name

        destination = output_dir / f"{symbol}.json"
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        exported.append(symbol)
        print(f"Exported {symbol} -> {destination}")

    manifest = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": _git_commit(),
        "requested_symbols": symbols,
        "exported_symbols": exported,
        "failed_symbols": failed,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"\nExported {len(exported)}/{len(symbols)} companies to {output_dir}")
    if failed:
        print(f"{len(failed)} company export(s) failed - see manifest.json", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
