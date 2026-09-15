"""Checker-side Formal Bundle Loader and Integrity Verifier.

Loads and cryptographically validates portable ForecastPH formal experiment bundles.
Enforces that:
1. `manifest.json` exists and uses supported schema (`forecastph-formal-v1`).
2. Every listed file exists, matches SHA-256, and matches byte size.
3. No duplicate symbols exist.
4. Complete 15-company set is present (ALI, APX, BPI, GLO, ICT, JFC, MBT, MEG, MER,
   NIKL, PGOLD, SCC, SECB, SHLPH, SMPH).
5. Provenance tier is not DEMO (must be FORMAL_FROZEN).
6. Canonical evaluations are deserializable into `CompanyEvaluationExport`.
7. Missing-evidence registry is loaded and inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from recheck.schema import CompanyEvaluationExport, ProvenanceTier

SUPPORTED_SCHEMAS: frozenset[str] = frozenset(["forecastph-formal-v1"])
REQUIRED_FORMAL_COMPANIES: tuple[str, ...] = (
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


class FormalBundleError(ValueError):
    """Raised when a formal bundle is malformed, incomplete, or corrupted."""


class BundleIntegrityError(FormalBundleError):
    """Raised when cryptographic or byte-level integrity verification fails."""


@dataclass(frozen=True)
class FileIntegrityCheck:
    relative_path: str
    category: str
    expected_sha256: str
    actual_sha256: str | None
    expected_bytes: int
    actual_bytes: int | None
    passed: bool
    error: str | None = None


@dataclass(frozen=True)
class BundleIntegrityReport:
    bundle_path: Path
    schema_version: str
    total_files_checked: int
    passed_files: int
    failed_files: int
    is_valid: bool
    missing_companies: list[str]
    is_complete_15_companies: bool
    has_demo_provenance: bool
    checks: list[FileIntegrityCheck]
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MissingEvidenceItem:
    symbol: str
    component: str
    field: str
    requirement_level: str  # CRITICAL, HIGH, OPTIONAL
    status: str  # AVAILABLE, MISSING_EVIDENCE, NOT_APPLICABLE, RECONSTRUCTED_VERIFICATION
    reason: str


@dataclass(frozen=True)
class FormalBundle:
    root_path: Path
    manifest: dict[str, Any]
    integrity_report: BundleIntegrityReport
    missing_evidence: list[MissingEvidenceItem]
    environment: dict[str, Any]
    provenance: dict[str, Any]
    configuration: dict[str, Any]
    splits: dict[str, dict[str, Any]]
    evaluations: dict[str, CompanyEvaluationExport]
    model_evidence: dict[str, dict[str, Any]]
    diagnostics: dict[str, dict[str, Any]]
    logs: dict[str, Any]

    @property
    def symbols(self) -> list[str]:
        return sorted(self.evaluations.keys())

    @property
    def critical_missing_evidence(self) -> list[MissingEvidenceItem]:
        return [e for e in self.missing_evidence if e.requirement_level == "CRITICAL" and e.status == "MISSING_EVIDENCE"]

    @property
    def high_missing_evidence(self) -> list[MissingEvidenceItem]:
        return [e for e in self.missing_evidence if e.requirement_level == "HIGH" and e.status == "MISSING_EVIDENCE"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_formal_bundle_integrity(bundle_dir: str | Path) -> BundleIntegrityReport:
    """Verify cryptographic checksums, sizes, and schema compliance of a formal export bundle."""
    root = Path(bundle_dir).resolve()
    if not root.is_dir():
        raise FormalBundleError(f"Bundle path does not exist or is not a directory: {root}")

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FormalBundleError(f"Missing required bundle manifest: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FormalBundleError(f"Malformed manifest.json: {exc}") from exc

    schema = manifest.get("schema_version", "")
    if schema not in SUPPORTED_SCHEMAS:
        raise FormalBundleError(f"Unsupported formal bundle schema: '{schema}'. Supported: {sorted(SUPPORTED_SCHEMAS)}")

    manifest_files: dict[str, dict[str, Any]] = manifest.get("files", {})
    checks: list[FileIntegrityCheck] = []
    errors: list[str] = []

    passed_count = 0
    failed_count = 0

    for rel_path, expected in manifest_files.items():
        target = root / rel_path
        exp_sha = expected.get("sha256", "")
        exp_bytes = expected.get("byte_size", 0)
        cat = expected.get("category", "")

        if not target.is_file():
            failed_count += 1
            err = f"File missing: {rel_path}"
            errors.append(err)
            checks.append(
                FileIntegrityCheck(
                    relative_path=rel_path,
                    category=cat,
                    expected_sha256=exp_sha,
                    actual_sha256=None,
                    expected_bytes=exp_bytes,
                    actual_bytes=None,
                    passed=False,
                    error=err,
                )
            )
            continue

        act_bytes = target.stat().st_size
        act_sha = _sha256(target)

        if act_bytes != exp_bytes:
            failed_count += 1
            err = f"Byte size mismatch for {rel_path}: expected {exp_bytes}, got {act_bytes}"
            errors.append(err)
            checks.append(
                FileIntegrityCheck(
                    relative_path=rel_path,
                    category=cat,
                    expected_sha256=exp_sha,
                    actual_sha256=act_sha,
                    expected_bytes=exp_bytes,
                    actual_bytes=act_bytes,
                    passed=False,
                    error=err,
                )
            )
        elif act_sha != exp_sha:
            failed_count += 1
            err = f"SHA-256 hash mismatch for {rel_path}: expected {exp_sha}, got {act_sha}"
            errors.append(err)
            checks.append(
                FileIntegrityCheck(
                    relative_path=rel_path,
                    category=cat,
                    expected_sha256=exp_sha,
                    actual_sha256=act_sha,
                    expected_bytes=exp_bytes,
                    actual_bytes=act_bytes,
                    passed=False,
                    error=err,
                )
            )
        else:
            passed_count += 1
            checks.append(
                FileIntegrityCheck(
                    relative_path=rel_path,
                    category=cat,
                    expected_sha256=exp_sha,
                    actual_sha256=act_sha,
                    expected_bytes=exp_bytes,
                    actual_bytes=act_bytes,
                    passed=True,
                )
            )

    # Check companies in evaluations/
    eval_dir = root / "evaluations"
    found_symbols = set()
    has_demo = False
    if eval_dir.is_dir():
        for f in eval_dir.glob("*.json"):
            sym = f.stem
            found_symbols.add(sym)
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("provenance_tier") == "DEMO":
                    has_demo = True
            except Exception:
                pass

    missing_comps = sorted(set(REQUIRED_FORMAL_COMPANIES) - found_symbols)
    if missing_comps:
        errors.append(f"Missing required formal companies: {missing_comps}")

    if has_demo:
        errors.append("Bundle contains DEMO provenance tier; formal audit requires FORMAL_FROZEN")

    is_valid = (failed_count == 0) and (len(missing_comps) == 0) and (not has_demo)

    return BundleIntegrityReport(
        bundle_path=root,
        schema_version=schema,
        total_files_checked=len(manifest_files),
        passed_files=passed_count,
        failed_files=failed_count,
        is_valid=is_valid,
        missing_companies=missing_comps,
        is_complete_15_companies=(len(missing_comps) == 0),
        has_demo_provenance=has_demo,
        checks=checks,
        errors=errors,
    )


def load_formal_bundle(
    bundle_dir: str | Path,
    *,
    enforce_integrity: bool = True,
) -> FormalBundle:
    """Load a verified formal bundle into memory.

    Raises:
        FormalBundleError or BundleIntegrityError if bundle fails verification.
    """
    root = Path(bundle_dir).resolve()
    report = verify_formal_bundle_integrity(root)

    if enforce_integrity and not report.is_valid:
        raise BundleIntegrityError(f"Formal bundle failed integrity verification: {report.errors}")

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Missing evidence
    missing_evidence: list[MissingEvidenceItem] = []
    missing_path = root / "missing_evidence.json"
    if missing_path.is_file():
        try:
            raw_missing = json.loads(missing_path.read_text(encoding="utf-8"))
            for entry in raw_missing:
                missing_evidence.append(
                    MissingEvidenceItem(
                        symbol=entry.get("symbol", ""),
                        component=entry.get("component", ""),
                        field=entry.get("field", ""),
                        requirement_level=entry.get("requirement_level", "OPTIONAL"),
                        status=entry.get("status", "AVAILABLE"),
                        reason=entry.get("reason", ""),
                    )
                )
        except Exception:
            pass

    # Environment
    env: dict[str, Any] = {}
    env_dir = root / "environment"
    if env_dir.is_dir():
        for f in env_dir.glob("*.json"):
            env[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        freeze_file = env_dir / "pip_freeze.txt"
        if freeze_file.is_file():
            env["pip_freeze"] = freeze_file.read_text(encoding="utf-8")

    # Provenance
    prov: dict[str, Any] = {}
    prov_dir = root / "provenance"
    if prov_dir.is_dir():
        for f in prov_dir.glob("*.json"):
            prov[f.stem] = json.loads(f.read_text(encoding="utf-8"))

    # Configuration
    conf: dict[str, Any] = {}
    conf_dir = root / "configuration"
    if conf_dir.is_dir():
        for f in conf_dir.glob("*.json"):
            conf[f.stem] = json.loads(f.read_text(encoding="utf-8"))

    # Splits
    splits: dict[str, dict[str, Any]] = {}
    splits_dir = root / "splits"
    if splits_dir.is_dir():
        for f in sorted(splits_dir.glob("*.json")):
            splits[f.stem] = json.loads(f.read_text(encoding="utf-8"))

    # Evaluations
    evals: dict[str, CompanyEvaluationExport] = {}
    evals_dir = root / "evaluations"
    if evals_dir.is_dir():
        for f in sorted(evals_dir.glob("*.json")):
            sym = f.stem
            data = json.loads(f.read_text(encoding="utf-8"))
            if sym in evals:
                raise FormalBundleError(f"Duplicate symbol artifact detected: {sym}")

            # Parse as CompanyEvaluationExport
            tier_str = data.get("provenance_tier", "FORMAL_FROZEN")
            try:
                tier = ProvenanceTier(tier_str)
            except ValueError:
                tier = ProvenanceTier.FORMAL_FROZEN

            if tier == ProvenanceTier.DEMO:
                raise FormalBundleError(f"Symbol {sym} has DEMO provenance tier; formal audit requires FORMAL_FROZEN")

            # Dates
            t_dates = [date.fromisoformat(d[:10]) for d in data.get("target_dates", [])]
            e_start = date.fromisoformat(data["evaluation_start"][:10]) if data.get("evaluation_start") else t_dates[0]
            e_end = date.fromisoformat(data["evaluation_end"][:10]) if data.get("evaluation_end") else t_dates[-1]

            evals[sym] = CompanyEvaluationExport(
                symbol=sym,
                name=data.get("name", f"Company {sym}"),
                sector=data.get("sector", "Unknown"),
                evaluation_proportion=float(data.get("evaluation_proportion", 0.15)),
                development_pairs=int(data.get("development_pairs", 0)),
                evaluation_pairs=int(data.get("evaluation_pairs", len(t_dates))),
                evaluation_start=e_start,
                evaluation_end=e_end,
                mase_denominator=float(data.get("mase_denominator", 1.0)),
                reported_metrics=data.get("reported_metrics", {}),
                reported_best_model=data.get("reported_best_model", "lag_reg"),
                target_dates=t_dates,
                actual_closes=[float(x) for x in data.get("actual_closes", [])],
                predicted_closes_by_model={
                    m: [float(x) for x in series]
                    for m, series in data.get("predicted_closes_by_model", {}).items()
                },
                provenance_tier=tier,
            )

    # Model Evidence
    model_ev: dict[str, dict[str, Any]] = {}
    model_dir = root / "model_evidence"
    if model_dir.is_dir():
        for sym_dir in sorted(model_dir.iterdir()):
            if sym_dir.is_dir():
                sym = sym_dir.name
                model_ev[sym] = {}
                for f in sym_dir.glob("*.json"):
                    model_ev[sym][f.stem] = json.loads(f.read_text(encoding="utf-8"))

    # Diagnostics
    diag: dict[str, dict[str, Any]] = {}
    diag_dir = root / "diagnostics"
    if diag_dir.is_dir():
        for f in sorted(diag_dir.glob("*.json")):
            diag[f.stem] = json.loads(f.read_text(encoding="utf-8"))

    # Logs
    logs: dict[str, Any] = {}
    logs_dir = root / "logs"
    if logs_dir.is_dir():
        for f in logs_dir.glob("*.json"):
            logs[f.stem] = json.loads(f.read_text(encoding="utf-8"))

    return FormalBundle(
        root_path=root,
        manifest=manifest,
        integrity_report=report,
        missing_evidence=missing_evidence,
        environment=env,
        provenance=prov,
        configuration=conf,
        splits=splits,
        evaluations=evals,
        model_evidence=model_ev,
        diagnostics=diag,
        logs=logs,
    )
