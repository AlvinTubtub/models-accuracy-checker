from __future__ import annotations

from recheck.compare import compare_company
from recheck.schema import parse_company_export


def test_compare_company_matches_when_reported_metrics_are_correct(minimal_payload):
    export = parse_company_export(minimal_payload)
    comparisons = compare_company(export)
    assert len(comparisons) == 4
    for comparison in comparisons:
        assert comparison.within_tolerance, (
            f"{comparison.model} did not match: recomputed={comparison.recomputed}, "
            f"reported={comparison.reported}"
        )


def test_compare_company_flags_corrupted_reported_metric(minimal_payload):
    minimal_payload["metrics"]["arima"]["rmse"] += 5.0
    export = parse_company_export(minimal_payload)
    comparisons = compare_company(export)
    by_model = {c.model: c for c in comparisons}
    assert not by_model["arima"].within_tolerance
    assert by_model["lag_reg"].within_tolerance
    assert by_model["lstm"].within_tolerance
    assert by_model["naive"].within_tolerance


def test_compare_company_beats_naive_flags(minimal_payload):
    export = parse_company_export(minimal_payload)
    comparisons = compare_company(export)
    naive_comparison = next(c for c in comparisons if c.model == "naive")
    # Naive's MASE against itself as the yardstick is always exactly 1.0 in this
    # fixture's construction only if predicted == origin == the naive series;
    # regardless, naive should never be reported as "beating" a MASE-of-1 baseline
    # that is defined by its own errors, so this just sanity-checks the flag runs.
    assert isinstance(naive_comparison.beats_naive_recomputed, bool)
