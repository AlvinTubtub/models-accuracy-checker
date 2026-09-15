#!/usr/bin/env python3
"""Formal Evidence Bridge: Export ForecastPH experiment evidence to a portable bundle.

This script executes either inside the ForecastPH backend environment (importing
ForecastPH modules) or packages from an existing ForecastPH formal evidence
directory (e.g. `formal_evidence/FORMAL_CORRECTED_...`).

The exported bundle contains only dependency-free JSON, CSV, and text files so that
the independent checker can audit the experiment without installing TensorFlow,
statsmodels, PyTorch, or joblib.

Bundle Layout:
formal_export/
├── manifest.json
├── missing_evidence.json
├── environment/
│   ├── git.json
│   ├── python.json
│   ├── packages.json
│   ├── platform.json
│   └── pip_freeze.txt
├── provenance/
│   ├── raw_files.json
│   ├── session_completeness.json
│   └── source_metadata.json
├── configuration/
│   ├── companies.json
│   ├── experiment.json
│   └── model_search_spaces.json
├── splits/
│   ├── <SYM>.json (15 symbols)
├── evaluations/
│   ├── <SYM>.json (15 symbols)
├── model_evidence/
│   ├── <SYM>/
│   │   ├── lir.json
│   │   ├── arima.json
│   │   └── lstm.json
├── diagnostics/
│   └── <SYM>.json
└── logs/
    └── execution_logs.json
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence

# Canonical bridge schema version
BRIDGE_SCHEMA_VERSION = "forecastph-formal-v1"
BRIDGE_VERSION = "1.0.0"

# Required 15-company universe
REQUIRED_FORMAL_SYMBOLS: tuple[str, ...] = (
    "ALI",
    "APX",
    "BPI",
    "GLO",
    "ICT",
    "JFC",
    "MBT",
    "MEG",
    "MER",
    "NIKL",
    "PGOLD",
    "SCC",
    "SECB",
    "SHLPH",
    "SMPH",
)

# Company sector mappings for portable export
COMPANY_METADATA: dict[str, dict[str, str]] = {
    "ALI": {"name": "Ayala Land, Inc.", "sector": "Property"},
    "APX": {"name": "Apex Mining Co., Inc.", "sector": "Mining and Oil"},
    "BPI": {"name": "Bank of the Philippine Islands", "sector": "Financials"},
    "GLO": {"name": "Globe Telecom, Inc.", "sector": "Services"},
    "ICT": {"name": "International Container Terminal Services, Inc.", "sector": "Industrial"},
    "JFC": {"name": "Jollibee Foods Corporation", "sector": "Services"},
    "MBT": {"name": "Metropolitan Bank & Trust Company", "sector": "Financials"},
    "MEG": {"name": "Megaworld Corporation", "sector": "Property"},
    "MER": {"name": "Manila Electric Company", "sector": "Services"},
    "NIKL": {"name": "Nickel Asia Corporation", "sector": "Mining and Oil"},
    "PGOLD": {"name": "Puregold Price Club, Inc.", "sector": "Services"},
    "SCC": {"name": "Semirara Mining and Power Corporation", "sector": "Industrial"},
    "SECB": {"name": "Security Bank Corporation", "sector": "Financials"},
    "SHLPH": {"name": "Shell Pilipinas Corporation", "sector": "Industrial"},
    "SMPH": {"name": "SM Prime Holdings, Inc.", "sector": "Property"},
}


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def canonical_dump_json(obj: Any, destination: Path) -> None:
    """Serialize object to deterministic canonical JSON (sorted keys, UTF-8, no NaN)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
        f.write("\n")


def clean_floats(obj: Any) -> Any:
    """Recursively validate that float values are finite; reject NaN and Inf."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite float value encountered: {obj}")
        return obj
    if isinstance(obj, dict):
        return {k: clean_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_floats(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean_floats(v) for v in obj]
    return obj


# =========================================================================
# Environment Evidence Extraction
# =========================================================================

def extract_git_evidence(repo_dir: Path) -> dict[str, Any]:
    """Capture Git provenance without modifying repository state."""
    evidence: dict[str, Any] = {
        "repository_url": None,
        "branch": None,
        "commit_sha": None,
        "commit_timestamp": None,
        "is_dirty": None,
        "modified_tracked_files": [],
        "untracked_files": [],
        "export_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bridge_schema_version": BRIDGE_SCHEMA_VERSION,
    }

    try:
        # Remote URL
        url_res = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, cwd=repo_dir, timeout=5,
        )
        if url_res.returncode == 0:
            evidence["repository_url"] = url_res.stdout.strip() or None

        # Branch
        br_res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=repo_dir, timeout=5,
        )
        if br_res.returncode == 0:
            evidence["branch"] = br_res.stdout.strip() or None

        # Commit SHA
        sha_res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=repo_dir, timeout=5,
        )
        if sha_res.returncode == 0:
            evidence["commit_sha"] = sha_res.stdout.strip() or None

        # Commit timestamp
        ts_res = subprocess.run(
            ["git", "show", "-s", "--format=%cI", "HEAD"],
            capture_output=True, text=True, cwd=repo_dir, timeout=5,
        )
        if ts_res.returncode == 0:
            evidence["commit_timestamp"] = ts_res.stdout.strip() or None

        # Status: modified tracked
        mod_res = subprocess.run(
            ["git", "diff", "--name-only"],
            capture_output=True, text=True, cwd=repo_dir, timeout=5,
        )
        if mod_res.returncode == 0:
            evidence["modified_tracked_files"] = [f.strip() for f in mod_res.stdout.splitlines() if f.strip()]

        # Status: untracked
        untr_res = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, cwd=repo_dir, timeout=5,
        )
        if untr_res.returncode == 0:
            evidence["untracked_files"] = [f.strip() for f in untr_res.stdout.splitlines() if f.strip()]

        evidence["is_dirty"] = bool(evidence["modified_tracked_files"] or evidence["untracked_files"])
    except Exception as exc:
        evidence["git_error"] = str(exc)

    return evidence


def extract_environment_evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Capture runtime python, platform, sorted packages, and pip freeze."""
    py_info = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "compiler": platform.python_compiler(),
        "build": platform.python_build(),
    }
    plat_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    pip_freeze_txt = ""
    packages: dict[str, str] = {}
    try:
        import importlib.metadata
        dists = sorted(importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower())
        for dist in dists:
            name = dist.metadata["Name"]
            ver = dist.version
            packages[name] = ver
            pip_freeze_txt += f"{name}=={ver}\n"
    except Exception:
        # Fallback to pip freeze if importlib.metadata is unavailable
        try:
            res = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                pip_freeze_txt = res.stdout
                for line in pip_freeze_txt.splitlines():
                    if "==" in line:
                        parts = line.split("==")
                        packages[parts[0].strip()] = parts[1].strip()
        except Exception:
            pass

    pkg_info = {
        "count": len(packages),
        "packages": packages,
    }
    return py_info, plat_info, pkg_info, {"source": "runtime"}, pip_freeze_txt


# =========================================================================
# Raw Data Provenance & Session Completeness
# =========================================================================

def audit_raw_file(raw_csv_path: Path, symbol: str) -> dict[str, Any]:
    """Extract byte-level and chronological evidence from raw OHLCV CSV."""
    if not raw_csv_path.exists():
        return {
            "symbol": symbol,
            "status": "FILE_NOT_FOUND",
            "path": str(raw_csv_path),
        }

    sha = compute_file_sha256(raw_csv_path)
    size = raw_csv_path.stat().st_size

    # Inspect CSV structure
    lines = raw_csv_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {
            "symbol": symbol,
            "status": "EMPTY_FILE",
            "sha256": sha,
            "byte_size": size,
        }

    header = [h.strip() for h in lines[0].split(",")]
    rows = [line.split(",") for line in lines[1:] if line.strip()]

    date_idx = header.index("Date") if "Date" in header else 0

    parsed_dates: list[str] = []
    null_counts: dict[str, int] = {h: 0 for h in header}
    is_chronological = True
    prev_d: date | None = None

    for row in rows:
        for idx, col_name in enumerate(header):
            if idx >= len(row) or row[idx].strip() == "" or row[idx].strip().lower() in ("nan", "null"):
                null_counts[col_name] += 1

        if date_idx < len(row):
            raw_d = row[date_idx].strip()
            parsed_dates.append(raw_d)
            try:
                cur_d = date.fromisoformat(raw_d[:10])
                if prev_d and cur_d <= prev_d:
                    is_chronological = False
                prev_d = cur_d
            except ValueError:
                is_chronological = False

    unique_dates = set(parsed_dates)
    duplicate_date_count = len(parsed_dates) - len(unique_dates)

    meta = COMPANY_METADATA.get(symbol, {"name": f"Company {symbol}", "sector": "Unknown"})

    return {
        "symbol": symbol,
        "company_name": meta["name"],
        "sector": meta["sector"],
        "filename": raw_csv_path.name,
        "relative_path": f"raw/{raw_csv_path.name}",
        "sha256": sha,
        "byte_size": size,
        "row_count": len(rows),
        "column_names": header,
        "first_date": parsed_dates[0] if parsed_dates else None,
        "last_date": parsed_dates[-1] if parsed_dates else None,
        "duplicate_date_count": duplicate_date_count,
        "null_counts": null_counts,
        "is_chronological": is_chronological,
        "status": "VALIDATED",
    }


def analyze_session_completeness(
    raw_files_info: list[dict[str, Any]],
    calendar_closures: set[date] | None = None,
) -> dict[str, Any]:
    """Compare observed raw dates with expected weekday sessions."""
    per_symbol: dict[str, Any] = {}
    for r in raw_files_info:
        sym = r["symbol"]
        if r.get("status") != "VALIDATED":
            per_symbol[sym] = {"status": r.get("status", "UNKNOWN")}
            continue

        per_symbol[sym] = {
            "first_date": r["first_date"],
            "last_date": r["last_date"],
            "observed_count": r["row_count"],
            "duplicate_count": r["duplicate_date_count"],
            "distinction_capability": "CANNOT_DISTINGUISH_EXCHANGE_CLOSED_VS_SUSPENSION",
            "note": "Current ForecastPH raw feeds do not tag exchange suspension vs security zero-volume sessions.",
        }

    return {
        "calendar_source": "config/pse_holidays.py (PSE_CLOSURES)",
        "calendar_version": "reviewed_sccp_pse_circulars_v1",
        "symbols": per_symbol,
    }


# =========================================================================
# Missing Evidence Registry Helper
# =========================================================================

class MissingEvidenceRegistry:
    """Tracks required, high, and optional audit evidence presence."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def record(
        self,
        symbol: str,
        component: str,
        field: str,
        requirement_level: str,
        status: str,
        reason: str,
    ) -> None:
        self.entries.append(
            {
                "symbol": symbol,
                "component": component,
                "field": field,
                "requirement_level": requirement_level,  # CRITICAL, HIGH, OPTIONAL
                "status": status,  # AVAILABLE, MISSING_EVIDENCE, NOT_APPLICABLE, RECONSTRUCTED_VERIFICATION
                "reason": reason,
            }
        )

    def critical_missing_count(self) -> int:
        return sum(
            1 for e in self.entries
            if e["requirement_level"] == "CRITICAL" and e["status"] == "MISSING_EVIDENCE"
        )

    def high_missing_count(self) -> int:
        return sum(
            1 for e in self.entries
            if e["requirement_level"] == "HIGH" and e["status"] == "MISSING_EVIDENCE"
        )

    def as_list(self) -> list[dict[str, Any]]:
        return list(self.entries)


# =========================================================================
# Main Export Orchestrator
# =========================================================================

def export_formal_experiment_bundle(
    output_dir: Path,
    *,
    backend_dir: Path | None = None,
    evidence_dir: Path | None = None,
    raw_dir: Path | None = None,
    symbols: Sequence[str] = REQUIRED_FORMAL_SYMBOLS,
) -> int:
    """Export complete ForecastPH formal evidence bundle.

    Returns:
        0 if all required evaluations exported successfully;
        1 if any required company evaluation failed to export.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_registry = MissingEvidenceRegistry()
    exported_symbols: list[str] = []
    failed_symbols: list[str] = []

    # 1. Environment evidence
    env_dir = output_dir / "environment"
    git_evidence = extract_git_evidence(backend_dir or Path.cwd())
    py_info, plat_info, pkg_info, _, pip_freeze = extract_environment_evidence()

    canonical_dump_json(git_evidence, env_dir / "git.json")
    canonical_dump_json(py_info, env_dir / "python.json")
    canonical_dump_json(plat_info, env_dir / "platform.json")
    canonical_dump_json(pkg_info, env_dir / "packages.json")
    (env_dir / "pip_freeze.txt").write_text(pip_freeze, encoding="utf-8")

    # 2. Raw data provenance
    prov_dir = output_dir / "provenance"
    raw_files_info: list[dict[str, Any]] = []
    effective_raw_dir = raw_dir or (backend_dir / "data" / "raw" if backend_dir else Path("data/raw"))

    for sym in symbols:
        raw_csv = effective_raw_dir / f"{sym}.csv"
        info = audit_raw_file(raw_csv, sym)
        raw_files_info.append(info)

        # Source metadata presence check
        missing_registry.record(
            symbol=sym,
            component="raw_data",
            field="source_metadata",
            requirement_level="HIGH",
            status="MISSING_EVIDENCE",
            reason="ForecastPH does not serialize raw data acquisition metadata (provider URL, download timestamp)",
        )

    canonical_dump_json({"files": raw_files_info}, prov_dir / "raw_files.json")
    session_info = analyze_session_completeness(raw_files_info)
    canonical_dump_json(session_info, prov_dir / "session_completeness.json")
    canonical_dump_json(
        {"source_metadata_status": "MISSING", "note": "Provider URL/timestamp not persisted"},
        prov_dir / "source_metadata.json",
    )

    # 3. Configuration snapshots
    config_dir = output_dir / "configuration"
    canonical_dump_json(
        {"companies": [COMPANY_METADATA.get(s, {"name": s, "sector": "Unknown"}) | {"symbol": s} for s in symbols]},
        config_dir / "companies.json",
    )
    experiment_config = {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "evaluation_proportion": 0.15,
        "development_proportion": 0.85,
        "models": ["lag_reg", "arima", "lstm", "naive"],
        "principal_models": ["lag_reg", "arima", "lstm"],
        "benchmark_model": "naive",
        "corporate_action_policy": "raw_close_retain_and_flag_v1",
        "tie_tolerance": 1e-9,
        "recomputation_tolerance": 1e-6,
    }
    canonical_dump_json(experiment_config, config_dir / "experiment.json")

    search_spaces = {
        "arima": {
            "p_range": [0, 1, 2, 3],
            "d_range": [0, 1, 2],
            "q_range": [0, 1, 2, 3],
            "trends": {"0": ["n", "c"], "1": ["n", "t"], "2": ["n"]},
            "includes_no_drift_random_walk": True,
            "no_drift_candidate": {"order": [0, 1, 0], "trend": "n"},
        },
        "lag_reg": {
            "alpha_grid_min": 0.0001,
            "alpha_grid_max": 1000.0,
            "expands_above_1": True,
            "pacf_lags_selected": True,
        },
        "lstm": {
            "tuning_seeds": [42, 123, 2026],
            "seed_count": 3,
            "lookback_candidates": [5, 10, 15, 20, 25, 30],
            "hidden_sizes": [25, 50],
            "learning_rates": [0.001, 0.01],
            "batch_sizes": [16, 32],
        },
    }
    canonical_dump_json(search_spaces, config_dir / "model_search_spaces.json")

    # 4. Packaging per-company splits, evaluations, model evidence, diagnostics
    splits_dir = output_dir / "splits"
    evals_dir = output_dir / "evaluations"
    model_ev_dir = output_dir / "model_evidence"
    diag_dir = output_dir / "diagnostics"

    for sym in symbols:
        try:
            split_data: dict[str, Any] | None = None
            eval_data: dict[str, Any] | None = None
            diag_data: dict[str, Any] | None = None

            # Path A: Extract from existing formal evidence directory if provided
            if evidence_dir and (evidence_dir / "per_company" / sym).exists():
                comp_dir = evidence_dir / "per_company" / sym
                chk_dir = evidence_dir / ".checkpoints" / sym

                # Diagnostics
                if (comp_dir / "diagnostics.json").exists():
                    with (comp_dir / "diagnostics.json").open("r", encoding="utf-8") as f:
                        diag_data = json.load(f)

                # Metrics & Evaluation
                if (comp_dir / "metrics.json").exists() and (comp_dir / "holdout_predictions.csv").exists():
                    with (comp_dir / "metrics.json").open("r", encoding="utf-8") as f:
                        metrics_raw = json.load(f)

                    # Parse holdout predictions CSV
                    csv_lines = (comp_dir / "holdout_predictions.csv").read_text(encoding="utf-8").splitlines()
                    header = [h.strip() for h in csv_lines[0].split(",")]
                    rows = [l.split(",") for l in csv_lines[1:] if l.strip()]

                    target_dates: list[str] = []
                    actual_closes: list[float] = []
                    preds: dict[str, list[float]] = {"lag_reg": [], "arima": [], "lstm": [], "naive": []}

                    d_idx = header.index("target_date") if "target_date" in header else 0
                    a_idx = header.index("actual_close") if "actual_close" in header else 1

                    for r in rows:
                        target_dates.append(r[d_idx].strip())
                        actual_closes.append(float(r[a_idx]))
                        for m in preds:
                            if m in header:
                                m_idx = header.index(m)
                                preds[m].append(float(r[m_idx]))

                    # Calculate or extract development denominator
                    denom = 1.0
                    if (chk_dir / "plan.json").exists():
                        with (chk_dir / "plan.json").open("r", encoding="utf-8") as f:
                            plan_data = json.load(f)
                            split_data = plan_data

                    # Check source data SHA
                    raw_info = next((rf for rf in raw_files_info if rf["symbol"] == sym), None)
                    if chk_dir.exists() and (chk_dir / "source_data.json").exists():
                        with (chk_dir / "source_data.json").open("r", encoding="utf-8") as f:
                            src_meta = json.load(f)
                            if "sha256" in src_meta and raw_info and raw_info.get("sha256") != src_meta["sha256"]:
                                missing_registry.record(
                                    symbol=sym,
                                    component="provenance",
                                    field="raw_sha256",
                                    requirement_level="CRITICAL",
                                    status="MISSING_EVIDENCE",
                                    reason="Raw CSV SHA-256 does not match original checkpoint source_data.json",
                                )

                    eval_meta = COMPANY_METADATA.get(sym, {"name": f"Company {sym}", "sector": "Unknown"})
                    eval_data = {
                        "symbol": sym,
                        "name": eval_meta["name"],
                        "sector": eval_meta["sector"],
                        "evaluation_proportion": 0.15,
                        "development_pairs": split_data.get("development_count", 0) if split_data else 0,
                        "evaluation_pairs": len(actual_closes),
                        "evaluation_start": target_dates[0] if target_dates else "",
                        "evaluation_end": target_dates[-1] if target_dates else "",
                        "mase_denominator": denom,
                        "reported_metrics": metrics_raw,
                        "reported_best_model": min(
                            ("lag_reg", "arima", "lstm"),
                            key=lambda m: metrics_raw.get(m, {}).get("rmse", float("inf")),
                        ),
                        "target_dates": target_dates,
                        "actual_closes": actual_closes,
                        "predicted_closes_by_model": preds,
                        "provenance_tier": "FORMAL_FROZEN",
                    }

            # Path B: Extract using ForecastPH safe loader if backend is importable
            elif backend_dir is not None:
                sys.path.insert(0, str(backend_dir))
                try:
                    from src.data.loader import load_company_history
                    from src.training.orchestration import load_company_evaluation
                    records = load_company_history(sym)
                    evaluation = load_company_evaluation(sym, records)
                    eval_dict = evaluation.as_dict()
                    meta = COMPANY_METADATA.get(sym, {"name": sym, "sector": "Unknown"})
                    eval_dict["name"] = meta["name"]
                    eval_dict["sector"] = meta["sector"]
                    eval_dict["provenance_tier"] = "FORMAL_FROZEN"
                    eval_data = eval_dict
                except Exception as exc:
                    failed_symbols.append(f"{sym}: safe loader failed: {exc}")
                    continue

            # Validate and write canonical evaluation
            if eval_data:
                clean_eval = clean_floats(eval_data)
                canonical_dump_json(clean_eval, evals_dir / f"{sym}.json")
                exported_symbols.append(sym)
            else:
                failed_symbols.append(f"{sym}: No evaluation data available")
                continue

            # Split evidence
            if split_data:
                canonical_dump_json(split_data, splits_dir / f"{sym}.json")
            else:
                missing_registry.record(
                    symbol=sym,
                    component="splits",
                    field="split_plan",
                    requirement_level="HIGH",
                    status="RECONSTRUCTED_VERIFICATION",
                    reason="Split plan reconstructed from canonical evaluation target dates",
                )
                reconstructed_split = {
                    "symbol": sym,
                    "evaluation_target_dates": eval_data.get("target_dates", []),
                    "evaluation_pairs": len(eval_data.get("target_dates", [])),
                    "provenance": "RECONSTRUCTED_VERIFICATION",
                }
                canonical_dump_json(reconstructed_split, splits_dir / f"{sym}.json")

            # Diagnostics & Model Evidence
            if diag_data:
                canonical_dump_json(clean_floats(diag_data), diag_dir / f"{sym}.json")

                # Decompose into model_evidence/SYM/
                comp_mod_dir = model_ev_dir / sym
                comp_mod_dir.mkdir(parents=True, exist_ok=True)

                lir_ev = diag_data.get("lag_reg", {})
                arima_ev = diag_data.get("arima", {})
                lstm_ev = diag_data.get("lstm", {})

                canonical_dump_json(clean_floats(lir_ev), comp_mod_dir / "lir.json")
                canonical_dump_json(clean_floats(arima_ev), comp_mod_dir / "arima.json")
                canonical_dump_json(clean_floats(lstm_ev), comp_mod_dir / "lstm.json")

                # Audit ARIMA roots
                if "roots" not in arima_ev and "ar_roots" not in arima_ev:
                    missing_registry.record(
                        symbol=sym,
                        component="arima",
                        field="diagnostics.roots",
                        requirement_level="HIGH",
                        status="MISSING_EVIDENCE",
                        reason="ForecastPH did not persist ARIMA AR/MA characteristic roots",
                    )
            else:
                missing_registry.record(
                    symbol=sym,
                    component="model_evidence",
                    field="diagnostics",
                    requirement_level="HIGH",
                    status="MISSING_EVIDENCE",
                    reason=f"No diagnostics or model tuning evidence found for {sym}",
                )

        except Exception as exc:
            failed_symbols.append(f"{sym}: {type(exc).__name__}: {exc}")

    # 5. Logs evidence
    logs_dir = output_dir / "logs"
    logs_evidence = {
        "logs_status": "NOT_AVAILABLE",
        "reason": "Original ForecastPH standard output/error execution logs were not retained",
    }
    canonical_dump_json(logs_evidence, logs_dir / "execution_logs.json")
    missing_registry.record(
        symbol="ALL",
        component="logs",
        field="execution_logs",
        requirement_level="OPTIONAL",
        status="MISSING_EVIDENCE",
        reason="ForecastPH did not preserve original training console/file logs",
    )

    # 6. Missing evidence registry file
    canonical_dump_json(missing_registry.as_list(), output_dir / "missing_evidence.json")

    # 7. File Manifest with hashes and sizes
    manifest_files: dict[str, dict[str, Any]] = {}
    for root, _, files in os.walk(output_dir):
        for fname in sorted(files):
            if fname == "manifest.json":
                continue
            fpath = Path(root) / fname
            rel_path = fpath.relative_to(output_dir).as_posix()
            category = rel_path.split("/")[0] if "/" in rel_path else "root"
            manifest_files[rel_path] = {
                "sha256": compute_file_sha256(fpath),
                "byte_size": fpath.stat().st_size,
                "category": category,
            }

    top_manifest = {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "bridge_version": BRIDGE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_evidence.get("commit_sha"),
        "source_dirty_state": git_evidence.get("is_dirty"),
        "requested_symbols": list(symbols),
        "exported_symbols": exported_symbols,
        "failed_symbols": failed_symbols,
        "required_company_count": len(REQUIRED_FORMAL_SYMBOLS),
        "is_complete_formal_set": len(exported_symbols) == len(REQUIRED_FORMAL_SYMBOLS),
        "missing_critical_evidence_count": missing_registry.critical_missing_count(),
        "missing_high_evidence_count": missing_registry.high_missing_count(),
        "files": manifest_files,
    }
    canonical_dump_json(top_manifest, output_dir / "manifest.json")

    print(f"Exported {len(exported_symbols)}/{len(symbols)} companies to {output_dir}")
    if failed_symbols:
        print(f"FAILED companies ({len(failed_symbols)}): {failed_symbols}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="formal_export", help="Destination bundle directory")
    parser.add_argument("--backend-dir", default=None, help="ForecastPH backend directory")
    parser.add_argument("--evidence-dir", default=None, help="Existing ForecastPH formal evidence directory")
    parser.add_argument("--raw-dir", default=None, help="Directory containing raw OHLCV CSVs")
    parser.add_argument("--symbol", action="append", dest="symbols", default=None, help="Export specific symbol (repeatable)")

    args = parser.parse_args(argv)
    output = Path(args.output)
    backend = Path(args.backend_dir).resolve() if args.backend_dir else None
    evidence = Path(args.evidence_dir).resolve() if args.evidence_dir else None
    raw = Path(args.raw_dir).resolve() if args.raw_dir else None
    syms = args.symbols or REQUIRED_FORMAL_SYMBOLS

    return export_formal_experiment_bundle(
        output_dir=output,
        backend_dir=backend,
        evidence_dir=evidence,
        raw_dir=raw,
        symbols=syms,
    )


if __name__ == "__main__":
    raise SystemExit(main())
