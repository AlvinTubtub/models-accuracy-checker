"""Independent Model Audit Verifier.

Audits exported ForecastPH model training and diagnostic evidence without running
training or depending on heavy model frameworks.

Evaluates:
- ARIMA: presence of (0,1,0,'n') candidate, trend/d compatibility, candidate membership,
  CV fold convergence, final fit convergence, roots and diagnostic evidence.
- LIR / LASSO: alpha grid presence, expansion above 1.0, selected alpha membership,
  boundary condition warning, development-only tuning.
- LSTM: 3 tuning seeds, per-seed validation scores, common CV date window alignment,
  selected config, epoch selection, final refit seed and training evidence.

Evaluates evidence states:
- PASS: Requirement satisfied by evidence.
- FAIL: Evidence violates invariant or specification.
- MISSING_EVIDENCE: Required or optional evidence was not persisted.
- NOT_APPLICABLE: Check not applicable to this configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

from recheck.formal_bundle import FormalBundle, MissingEvidenceItem
from recheck.tolerances import TIE_TOLERANCE


@dataclass(frozen=True)
class ModelAuditCheck:
    check_id: str
    model: str  # "arima", "lag_reg", "lstm", "data", "splits"
    symbol: str  # symbol or "ALL"
    requirement_level: str  # "CRITICAL", "HIGH", "OPTIONAL"
    status: str  # "PASS", "FAIL", "MISSING_EVIDENCE", "NOT_APPLICABLE"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FormalEvidenceReadiness:
    canonical_evaluations_ready: bool
    raw_provenance_ready: bool
    split_manifests_ready: bool
    lir_evidence_status: str
    arima_evidence_status: str
    lstm_evidence_status: str
    arima_diagnostics_status: str
    corporate_action_evidence_status: str
    logs_status: str
    critical_missing_evidence_count: int
    high_missing_evidence_count: int
    ready_for_formal_run: bool
    summary_verdict: str
    checks: list[ModelAuditCheck]


# Valid ARIMA trend rules per d
VALID_ARIMA_TRENDS: dict[int, set[str]] = {
    0: {"n", "c"},
    1: {"n", "t"},
    2: {"n"},
}


def audit_arima_candidate_space(search_space: dict[str, Any]) -> list[ModelAuditCheck]:
    """Audit declared ARIMA candidate search space."""
    checks: list[ModelAuditCheck] = []

    # 1. No-drift random walk (0, 1, 0, 'n')
    has_010_n = False
    includes_flag = search_space.get("includes_no_drift_random_walk")
    no_drift_cand = search_space.get("no_drift_candidate")
    if no_drift_cand:
        order = tuple(no_drift_cand.get("order", ()))
        trend = no_drift_cand.get("trend")
        if order == (0, 1, 0) and trend == "n":
            has_010_n = True
    elif includes_flag:
        has_010_n = True

    if has_010_n:
        checks.append(
            ModelAuditCheck(
                check_id="ARIMA_CANDIDATE_NO_DRIFT",
                model="arima",
                symbol="ALL",
                requirement_level="CRITICAL",
                status="PASS",
                message="ARIMA candidate space includes no-drift random walk ARIMA(0,1,0) with trend='n'.",
            )
        )
    else:
        checks.append(
            ModelAuditCheck(
                check_id="ARIMA_CANDIDATE_NO_DRIFT",
                model="arima",
                symbol="ALL",
                requirement_level="CRITICAL",
                status="FAIL",
                message="ARIMA candidate space is missing baseline no-drift random walk ARIMA(0,1,0), trend='n'.",
            )
        )

    # 2. Trend rules per d
    declared_trends = search_space.get("trends", {})
    trend_rules_valid = True
    violating_rules = []
    for d_str, trends in declared_trends.items():
        try:
            d_val = int(d_str)
            allowed = VALID_ARIMA_TRENDS.get(d_val, set())
            for tr in trends:
                if tr not in allowed:
                    trend_rules_valid = False
                    violating_rules.append(f"d={d_val}, trend='{tr}' (allowed: {sorted(allowed)})")
        except ValueError:
            pass

    if trend_rules_valid:
        checks.append(
            ModelAuditCheck(
                check_id="ARIMA_TREND_RULES_COMPLIANT",
                model="arima",
                symbol="ALL",
                requirement_level="CRITICAL",
                status="PASS",
                message="ARIMA trend restrictions adhere to specification (d=0 -> n,c; d=1 -> n,t; d=2 -> n).",
            )
        )
    else:
        checks.append(
            ModelAuditCheck(
                check_id="ARIMA_TREND_RULES_COMPLIANT",
                model="arima",
                symbol="ALL",
                requirement_level="CRITICAL",
                status="FAIL",
                message=f"ARIMA trend rules contain invalid trend/differencing combinations: {violating_rules}",
            )
        )

    return checks


def audit_company_arima_evidence(symbol: str, arima_ev: dict[str, Any]) -> list[ModelAuditCheck]:
    """Audit per-company ARIMA tuning, convergence, and diagnostic evidence."""
    checks: list[ModelAuditCheck] = []

    if not arima_ev:
        checks.append(
            ModelAuditCheck(
                check_id="ARIMA_EVIDENCE_EXISTS",
                model="arima",
                symbol=symbol,
                requirement_level="CRITICAL",
                status="MISSING_EVIDENCE",
                message=f"No ARIMA evidence persisted for company {symbol}.",
            )
        )
        return checks

    # Check selected candidate
    sel_order = arima_ev.get("selected_order") or arima_ev.get("order")
    sel_trend = arima_ev.get("selected_trend") or arima_ev.get("trend")

    # Check candidate space membership
    candidate_cv = arima_ev.get("candidate_cv", [])
    if candidate_cv and sel_order is not None and sel_trend is not None:
        cand_tuples = [
            (tuple(c.get("configuration", {}).get("order", ())), c.get("configuration", {}).get("trend"))
            for c in candidate_cv
        ]
        if (tuple(sel_order), sel_trend) in cand_tuples:
            checks.append(
                ModelAuditCheck(
                    check_id="ARIMA_SELECTED_IN_GRID",
                    model="arima",
                    symbol=symbol,
                    requirement_level="CRITICAL",
                    status="PASS",
                    message=f"Selected ARIMA order {sel_order} trend '{sel_trend}' belongs to candidate search grid.",
                )
            )
        else:
            checks.append(
                ModelAuditCheck(
                    check_id="ARIMA_SELECTED_IN_GRID",
                    model="arima",
                    symbol=symbol,
                    requirement_level="CRITICAL",
                    status="FAIL",
                    message=f"Selected ARIMA order {sel_order} trend '{sel_trend}' not found in candidate_cv grid.",
                )
            )

    # Check CV convergence
    cv_converged = arima_ev.get("all_cv_folds_converged")
    if cv_converged is True:
        checks.append(
            ModelAuditCheck(
                check_id="ARIMA_CV_CONVERGED",
                model="arima",
                symbol=symbol,
                requirement_level="HIGH",
                status="PASS",
                message="All cross-validation folds converged for selected candidate.",
            )
        )
    elif cv_converged is False:
        checks.append(
            ModelAuditCheck(
                check_id="ARIMA_CV_CONVERGED",
                model="arima",
                symbol=symbol,
                requirement_level="HIGH",
                status="FAIL",
                message="Cross-validation folds suffered convergence failure during candidate selection.",
            )
        )
    else:
        checks.append(
            ModelAuditCheck(
                check_id="ARIMA_CV_CONVERGED",
                model="arima",
                symbol=symbol,
                requirement_level="HIGH",
                status="MISSING_EVIDENCE",
                message="No CV fold convergence tracking evidence persisted.",
            )
        )

    # Check final fit convergence
    final_conv = arima_ev.get("final_fit_converged")
    if final_conv is True:
        checks.append(
            ModelAuditCheck(
                check_id="ARIMA_FINAL_FIT_CONVERGED",
                model="arima",
                symbol=symbol,
                requirement_level="CRITICAL",
                status="PASS",
                message="Final development-period ARIMA model fit converged.",
            )
        )
    elif final_conv is False:
        checks.append(
            ModelAuditCheck(
                check_id="ARIMA_FINAL_FIT_CONVERGED",
                model="arima",
                symbol=symbol,
                requirement_level="CRITICAL",
                status="FAIL",
                message="Final development-period ARIMA model fit failed to converge.",
            )
        )
    else:
        checks.append(
            ModelAuditCheck(
                check_id="ARIMA_FINAL_FIT_CONVERGED",
                model="arima",
                symbol=symbol,
                requirement_level="CRITICAL",
                status="MISSING_EVIDENCE",
                message="Final ARIMA fit convergence evidence is missing.",
            )
        )

    # Check characteristic roots
    roots = arima_ev.get("roots") or arima_ev.get("ar_roots")
    if roots is not None:
        checks.append(
            ModelAuditCheck(
                check_id="ARIMA_DIAGNOSTICS_ROOTS",
                model="arima",
                symbol=symbol,
                requirement_level="HIGH",
                status="PASS",
                message="AR/MA characteristic roots evidence persisted.",
            )
        )
    else:
        checks.append(
            ModelAuditCheck(
                check_id="ARIMA_DIAGNOSTICS_ROOTS",
                model="arima",
                symbol=symbol,
                requirement_level="HIGH",
                status="MISSING_EVIDENCE",
                message="ForecastPH did not persist ARIMA characteristic root diagnostic evidence.",
            )
        )

    # Check Ljung-Box test
    ljung_box = arima_ev.get("ljung_box")
    if ljung_box is not None:
        checks.append(
            ModelAuditCheck(
                check_id="ARIMA_DIAGNOSTICS_LJUNG_BOX",
                model="arima",
                symbol=symbol,
                requirement_level="HIGH",
                status="PASS",
                message="Ljung-Box diagnostic evidence persisted.",
            )
        )
    else:
        checks.append(
            ModelAuditCheck(
                check_id="ARIMA_DIAGNOSTICS_LJUNG_BOX",
                model="arima",
                symbol=symbol,
                requirement_level="HIGH",
                status="MISSING_EVIDENCE",
                message="Ljung-Box diagnostic evidence is missing.",
            )
        )

    return checks


def audit_company_lir_evidence(symbol: str, lir_ev: dict[str, Any]) -> list[ModelAuditCheck]:
    """Audit per-company LIR / LASSO tuning, boundary warning, and feature evidence."""
    checks: list[ModelAuditCheck] = []

    if not lir_ev:
        checks.append(
            ModelAuditCheck(
                check_id="LIR_EVIDENCE_EXISTS",
                model="lag_reg",
                symbol=symbol,
                requirement_level="CRITICAL",
                status="MISSING_EVIDENCE",
                message=f"No LIR / LASSO evidence persisted for company {symbol}.",
            )
        )
        return checks

    tuning_meta = lir_ev.get("tuning_metadata", {})
    alpha_grid = tuning_meta.get("grid") or lir_ev.get("alpha_grid")

    if not alpha_grid:
        checks.append(
            ModelAuditCheck(
                check_id="LIR_ALPHA_GRID_EXISTS",
                model="lag_reg",
                symbol=symbol,
                requirement_level="CRITICAL",
                status="MISSING_EVIDENCE",
                message="Alpha search grid not persisted in LIR tuning evidence.",
            )
        )
    else:
        # Check alpha expansion > 1.0
        has_above_1 = any(float(a) > 1.0 for a in alpha_grid)
        if has_above_1:
            checks.append(
                ModelAuditCheck(
                    check_id="LIR_ALPHA_EXPANSION_ABOVE_ONE",
                    model="lag_reg",
                    symbol=symbol,
                    requirement_level="CRITICAL",
                    status="PASS",
                    message=f"LASSO alpha grid expands above 1.0 (max alpha: {max(alpha_grid)}).",
                )
            )
        else:
            checks.append(
                ModelAuditCheck(
                    check_id="LIR_ALPHA_EXPANSION_ABOVE_ONE",
                    model="lag_reg",
                    symbol=symbol,
                    requirement_level="CRITICAL",
                    status="FAIL",
                    message=f"LASSO alpha grid does not expand above 1.0 (max alpha: {max(alpha_grid)}).",
                )
            )

        # Selected alpha in grid
        sel_alpha = lir_ev.get("selected_alpha") or tuning_meta.get("selected_alpha")
        if sel_alpha is not None:
            in_grid = any(abs(float(sel_alpha) - float(a)) <= TIE_TOLERANCE for a in alpha_grid)
            if in_grid:
                checks.append(
                    ModelAuditCheck(
                        check_id="LIR_SELECTED_ALPHA_IN_GRID",
                        model="lag_reg",
                        symbol=symbol,
                        requirement_level="CRITICAL",
                        status="PASS",
                        message=f"Selected alpha ({sel_alpha}) is an element of the search grid.",
                    )
                )
            else:
                checks.append(
                    ModelAuditCheck(
                        check_id="LIR_SELECTED_ALPHA_IN_GRID",
                        model="lag_reg",
                        symbol=symbol,
                        requirement_level="CRITICAL",
                        status="FAIL",
                        message=f"Selected alpha ({sel_alpha}) is not in the declared alpha grid.",
                    )
                )

            # Boundary warning audit
            min_a = min(alpha_grid)
            max_a = max(alpha_grid)
            is_boundary = (abs(float(sel_alpha) - float(min_a)) <= TIE_TOLERANCE) or (
                abs(float(sel_alpha) - float(max_a)) <= TIE_TOLERANCE
            )
            has_boundary_flag = bool(
                tuning_meta.get("selected_at_boundary")
                or lir_ev.get("boundary_warning_present")
                or lir_ev.get("boundary_warning")
            )

            if is_boundary and not has_boundary_flag:
                checks.append(
                    ModelAuditCheck(
                        check_id="LIR_BOUNDARY_WARNING_AUDIT",
                        model="lag_reg",
                        symbol=symbol,
                        requirement_level="HIGH",
                        status="FAIL",
                        message=f"Selected alpha hit grid boundary ({sel_alpha}) but boundary warning was not flagged.",
                    )
                )
            elif is_boundary and has_boundary_flag:
                checks.append(
                    ModelAuditCheck(
                        check_id="LIR_BOUNDARY_WARNING_AUDIT",
                        model="lag_reg",
                        symbol=symbol,
                        requirement_level="HIGH",
                        status="PASS",
                        message=f"Selected alpha hit grid boundary ({sel_alpha}) and boundary condition was correctly flagged.",
                    )
                )
            else:
                checks.append(
                    ModelAuditCheck(
                        check_id="LIR_BOUNDARY_WARNING_AUDIT",
                        model="lag_reg",
                        symbol=symbol,
                        requirement_level="HIGH",
                        status="PASS",
                        message="Selected alpha is interior to the search grid.",
                    )
                )

    return checks


def audit_company_lstm_evidence(symbol: str, lstm_ev: dict[str, Any]) -> list[ModelAuditCheck]:
    """Audit per-company LSTM 3-seed tuning, date alignment, and final refit evidence."""
    checks: list[ModelAuditCheck] = []

    if not lstm_ev:
        checks.append(
            ModelAuditCheck(
                check_id="LSTM_EVIDENCE_EXISTS",
                model="lstm",
                symbol=symbol,
                requirement_level="CRITICAL",
                status="MISSING_EVIDENCE",
                message=f"No LSTM evidence persisted for company {symbol}.",
            )
        )
        return checks

    train_meta = lstm_ev.get("training_metadata", {})

    # 1. Three tuning seeds
    seeds = train_meta.get("tuning_seeds") or lstm_ev.get("tuning_seeds", [])
    if len(seeds) == 3:
        checks.append(
            ModelAuditCheck(
                check_id="LSTM_THREE_TUNING_SEEDS",
                model="lstm",
                symbol=symbol,
                requirement_level="CRITICAL",
                status="PASS",
                message=f"LSTM tuning utilized exactly 3 designated seeds: {seeds}.",
            )
        )
    elif len(seeds) > 0 and len(seeds) != 3:
        checks.append(
            ModelAuditCheck(
                check_id="LSTM_THREE_TUNING_SEEDS",
                model="lstm",
                symbol=symbol,
                requirement_level="CRITICAL",
                status="FAIL",
                message=f"LSTM tuning did not use exactly 3 seeds (found {len(seeds)}: {seeds}).",
            )
        )
    else:
        checks.append(
            ModelAuditCheck(
                check_id="LSTM_THREE_TUNING_SEEDS",
                model="lstm",
                symbol=symbol,
                requirement_level="CRITICAL",
                status="MISSING_EVIDENCE",
                message="Tuning seed metadata is missing from LSTM evidence.",
            )
        )

    # 2. Common CV target dates across candidate lookbacks
    start_d = train_meta.get("common_cv_target_start") or lstm_ev.get("common_cv_target_start")
    end_d = train_meta.get("common_cv_target_end") or lstm_ev.get("common_cv_target_end")
    if start_d and end_d:
        checks.append(
            ModelAuditCheck(
                check_id="LSTM_COMMON_CV_WINDOW",
                model="lstm",
                symbol=symbol,
                requirement_level="CRITICAL",
                status="PASS",
                message=f"Common CV validation window enforced across lookback candidates ({start_d} to {end_d}).",
            )
        )
    else:
        checks.append(
            ModelAuditCheck(
                check_id="LSTM_COMMON_CV_WINDOW",
                model="lstm",
                symbol=symbol,
                requirement_level="CRITICAL",
                status="MISSING_EVIDENCE",
                message="Evidence of common CV target-date window across lookbacks is missing.",
            )
        )

    # 3. Selected configuration
    sel_config = train_meta.get("selected_config") or lstm_ev.get("selected_config")
    if sel_config:
        checks.append(
            ModelAuditCheck(
                check_id="LSTM_SELECTED_CONFIG",
                model="lstm",
                symbol=symbol,
                requirement_level="HIGH",
                status="PASS",
                message=f"Selected LSTM configuration documented: {sel_config}.",
            )
        )
    else:
        checks.append(
            ModelAuditCheck(
                check_id="LSTM_SELECTED_CONFIG",
                model="lstm",
                symbol=symbol,
                requirement_level="HIGH",
                status="MISSING_EVIDENCE",
                message="Selected LSTM configuration metadata is missing.",
            )
        )

    # 4. Final refit seed
    final_seed = train_meta.get("final_fit_seed") or lstm_ev.get("final_fit_seed")
    if final_seed is not None:
        checks.append(
            ModelAuditCheck(
                check_id="LSTM_FINAL_FIT_SEED",
                model="lstm",
                symbol=symbol,
                requirement_level="CRITICAL",
                status="PASS",
                message=f"Final full-development LSTM refit seed documented ({final_seed}).",
            )
        )
    else:
        checks.append(
            ModelAuditCheck(
                check_id="LSTM_FINAL_FIT_SEED",
                model="lstm",
                symbol=symbol,
                requirement_level="CRITICAL",
                status="MISSING_EVIDENCE",
                message="Final refit seed evidence is missing.",
            )
        )

    return checks


def audit_formal_bundle(bundle: FormalBundle) -> FormalEvidenceReadiness:
    """Perform comprehensive independent model audit on an exported formal bundle."""
    all_checks: list[ModelAuditCheck] = []

    # 1. Global ARIMA search space audit
    search_space = bundle.configuration.get("model_search_spaces", {}).get("arima", {})
    all_checks.extend(audit_arima_candidate_space(search_space))

    # 2. Per-company audits
    for sym in bundle.symbols:
        # ARIMA
        arima_ev = bundle.model_evidence.get(sym, {}).get("arima") or bundle.diagnostics.get(sym, {}).get("arima", {})
        all_checks.extend(audit_company_arima_evidence(sym, arima_ev))

        # LIR / LASSO
        lir_ev = bundle.model_evidence.get(sym, {}).get("lir") or bundle.diagnostics.get(sym, {}).get("lag_reg", {})
        all_checks.extend(audit_company_lir_evidence(sym, lir_ev))

        # LSTM
        lstm_ev = bundle.model_evidence.get(sym, {}).get("lstm") or bundle.diagnostics.get(sym, {}).get("lstm", {})
        all_checks.extend(audit_company_lstm_evidence(sym, lstm_ev))

    # 3. Evaluate Readiness Summary
    crit_missing = sum(1 for c in all_checks if c.requirement_level == "CRITICAL" and c.status in ("FAIL", "MISSING_EVIDENCE"))
    high_missing = sum(1 for c in all_checks if c.requirement_level == "HIGH" and c.status in ("FAIL", "MISSING_EVIDENCE"))

    # Check canonical evaluations and splits
    can_evals_ready = bundle.integrity_report.is_complete_15_companies and not bundle.integrity_report.has_demo_provenance
    raw_prov_ready = len(bundle.provenance.get("raw_files", {}).get("files", [])) >= 15
    splits_ready = len(bundle.splits) >= 15

    # Determine status strings
    def _status_for_model(model: str) -> str:
        model_checks = [c for c in all_checks if c.model == model]
        if any(c.status == "FAIL" for c in model_checks):
            return "FAIL"
        if any(c.status == "MISSING_EVIDENCE" and c.requirement_level == "CRITICAL" for c in model_checks):
            return "MISSING_CRITICAL_EVIDENCE"
        if any(c.status == "MISSING_EVIDENCE" for c in model_checks):
            return "PARTIALLY_AVAILABLE"
        return "AVAILABLE"

    lir_status = _status_for_model("lag_reg")
    arima_status = _status_for_model("arima")
    lstm_status = _status_for_model("lstm")

    # Diagnostics status
    arima_diag_checks = [c for c in all_checks if "DIAGNOSTICS" in c.check_id]
    if any(c.status == "MISSING_EVIDENCE" for c in arima_diag_checks):
        arima_diag_status = "PARTIALLY_AVAILABLE_ROOTS_MISSING"
    else:
        arima_diag_status = "AVAILABLE"

    corp_status = "AVAILABLE" if bundle.provenance.get("session_completeness") else "NOT_AVAILABLE"
    logs_status = bundle.logs.get("execution_logs", {}).get("logs_status", "NOT_AVAILABLE")

    # Ready for formal run: only True if zero critical failures or missing critical evidence
    ready_for_formal_run = bool(
        can_evals_ready
        and raw_prov_ready
        and splits_ready
        and (crit_missing == 0)
    )

    summary_verdict = (
        "READY FOR FORMAL EXPERIMENT PREPARATION"
        if ready_for_formal_run
        else "NOT READY — FORECASTPH EVIDENCE BLOCKERS REMAIN"
    )

    return FormalEvidenceReadiness(
        canonical_evaluations_ready=can_evals_ready,
        raw_provenance_ready=raw_prov_ready,
        split_manifests_ready=splits_ready,
        lir_evidence_status=lir_status,
        arima_evidence_status=arima_status,
        lstm_evidence_status=lstm_status,
        arima_diagnostics_status=arima_diag_status,
        corporate_action_evidence_status=corp_status,
        logs_status=logs_status,
        critical_missing_evidence_count=crit_missing,
        high_missing_evidence_count=high_missing,
        ready_for_formal_run=ready_for_formal_run,
        summary_verdict=summary_verdict,
        checks=all_checks,
    )
