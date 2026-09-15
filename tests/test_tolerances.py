"""Tests for explicit recomputation, demo, tie, and boundary tolerances."""

from recheck.tolerances import (
    DEMO_RECOMPUTATION_TOLERANCE,
    NUMERICAL_BOUNDARY_TOLERANCE,
    RECOMPUTATION_TOLERANCE,
    TIE_TOLERANCE,
)


def test_tolerances_distinct_and_correctly_valued():
    assert RECOMPUTATION_TOLERANCE == 1e-6
    assert DEMO_RECOMPUTATION_TOLERANCE == 1e-4
    assert TIE_TOLERANCE == 1e-9
    assert NUMERICAL_BOUNDARY_TOLERANCE == 1e-9

    # Conceptual independence: recomputation tolerance vs tie tolerance
    assert RECOMPUTATION_TOLERANCE != TIE_TOLERANCE
    assert DEMO_RECOMPUTATION_TOLERANCE != RECOMPUTATION_TOLERANCE
    assert TIE_TOLERANCE < RECOMPUTATION_TOLERANCE < DEMO_RECOMPUTATION_TOLERANCE
