#!/usr/bin/env python3
"""CLI Utility: Validate ForecastPH Formal Export Bundle and Audit Readiness.

Usage:
    python scripts/validate_formal_bundle.py /path/to/formal_export

Evaluates cryptographic manifest integrity, raw file provenance, canonical evaluations,
split manifests, and model training evidence for ARIMA, LIR/LASSO, and LSTM.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recheck.formal_bundle import FormalBundleError, load_formal_bundle, verify_formal_bundle_integrity
from recheck.model_audit import audit_formal_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_path", help="Path to the formal export bundle directory")
    args = parser.parse_args(argv)

    bundle_dir = Path(args.bundle_path).resolve()
    print("=" * 70)
    print("  FORECASTPH FORMAL EVIDENCE BUNDLE VERIFICATION")
    print(f"  Target directory: {bundle_dir}")
    print("=" * 70)

    # 1. Manifest and Cryptographic Integrity
    try:
        integrity_report = verify_formal_bundle_integrity(bundle_dir)
    except FormalBundleError as exc:
        print(f"\n[!] MANIFEST INTEGRITY FAILED: {exc}", file=sys.stderr)
        print("\nFINAL: NOT READY — FORECASTPH EVIDENCE BLOCKERS REMAIN")
        return 1

    print(f"\n[*] MANIFEST INTEGRITY:")
    print(f"    Schema Version:        {integrity_report.schema_version}")
    print(f"    Total Manifest Files:  {integrity_report.total_files_checked}")
    print(f"    Checksum Match:        {integrity_report.passed_files}/{integrity_report.total_files_checked}")
    if integrity_report.failed_files > 0:
        print(f"    FAILED Files ({integrity_report.failed_files}):")
        for err in integrity_report.errors:
            print(f"      - {err}")
    else:
        print(f"    Status:                PASSED (100% cryptographic agreement)")

    # 2. Load bundle
    try:
        bundle = load_formal_bundle(bundle_dir, enforce_integrity=False)
    except Exception as exc:
        print(f"\n[!] BUNDLE LOAD FAILED: {exc}", file=sys.stderr)
        print("\nFINAL: NOT READY — FORECASTPH EVIDENCE BLOCKERS REMAIN")
        return 1

    # 3. Model and Provenance Audit
    readiness = audit_formal_bundle(bundle)

    print(f"\n[*] EVIDENCE AUDIT:")
    print(f"    Canonical Evaluations: {'READY' if readiness.canonical_evaluations_ready else 'INCOMPLETE / DEMO'}")
    print(f"    Matched Companies:     {len(bundle.evaluations)}/15")
    print(f"    Raw Input Provenance:  {'READY' if readiness.raw_provenance_ready else 'INCOMPLETE'}")
    print(f"    Split Evidence:        {'READY' if readiness.split_manifests_ready else 'INCOMPLETE'}")
    print(f"    LIR Evidence:          {readiness.lir_evidence_status}")
    print(f"    ARIMA Evidence:        {readiness.arima_evidence_status}")
    print(f"    ARIMA Diagnostics:     {readiness.arima_diagnostics_status}")
    print(f"    LSTM Evidence:         {readiness.lstm_evidence_status}")
    print(f"    Corporate Actions:     {readiness.corporate_action_evidence_status}")
    print(f"    Execution Logs:        {readiness.logs_status}")

    # 4. Missing Evidence Breakdown
    crit_checks = [c for c in readiness.checks if c.requirement_level == "CRITICAL" and c.status != "PASS"]
    high_checks = [c for c in readiness.checks if c.requirement_level == "HIGH" and c.status != "PASS"]

    print(f"\n[*] MISSING / FAILING EVIDENCE SUMMARY:")
    print(f"    Critical Blockers:     {len(crit_checks)}")
    print(f"    High/Optional Items:   {len(high_checks)}")

    if crit_checks:
        print("    Critical Issues:")
        for c in crit_checks[:10]:
            print(f"      - [{c.model.upper()}] {c.symbol}: {c.message}")

    if high_checks:
        print("    Notable High/Optional Observations:")
        for c in high_checks[:5]:
            print(f"      - [{c.model.upper()}] {c.symbol}: {c.message}")

    print("\n" + "=" * 70)
    print(f"  FINAL VERDICT: {readiness.summary_verdict}")
    print("=" * 70)

    return 0 if readiness.ready_for_formal_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
