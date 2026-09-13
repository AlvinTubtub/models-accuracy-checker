"""Load bridge-exported evaluation JSON files from disk."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from recheck.schema import CompanyEvaluationExport, ExportValidationError, parse_company_export


@dataclass(frozen=True)
class LoadResult:
    exports: dict[str, CompanyEvaluationExport]
    load_errors: dict[str, str]
    manifest: dict | None

    @property
    def symbols(self) -> list[str]:
        return sorted(self.exports)


# Alias for readability and convenience
LoadedEvaluations = LoadResult


def load_company_export(path: Path | str) -> CompanyEvaluationExport:
    """Load and validate a single `<SYMBOL>.json` export file."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return parse_company_export(payload, source_label=path.stem)


def load_all_exports(data_dir: Path | str) -> LoadResult:
    """Load every `<SYMBOL>.json` file in a directory (skips manifest.json)."""

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    exports: dict[str, CompanyEvaluationExport] = {}
    load_errors: dict[str, str] = {}

    for json_path in sorted(data_dir.glob("*.json")):
        if json_path.name == "manifest.json":
            continue
        try:
            export = load_company_export(json_path)
        except (ExportValidationError, json.JSONDecodeError, KeyError, ValueError) as exc:
            load_errors[json_path.stem] = f"{type(exc).__name__}: {exc}"
            continue
        exports[export.symbol] = export

    manifest = None
    manifest_path = data_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception:
            manifest = None

    return LoadResult(exports=exports, load_errors=load_errors, manifest=manifest)


load_all_evaluations = load_all_exports
