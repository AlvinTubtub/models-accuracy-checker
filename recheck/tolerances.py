"""Explicit numerical tolerances for verification, comparison, and tie evaluations.

Conceptual distinction:
- RECOMPUTATION_TOLERANCE: Tolerance for verifying recomputed metrics from raw holdouts
  against reported metrics in formal unrounded float64 exports (1e-6).
- DEMO_RECOMPUTATION_TOLERANCE: Tolerance for legacy/demo JSON exports rounded to 4 decimals (1e-4).
- TIE_TOLERANCE: Precision threshold below which two models, metrics, or loss differentials
  are treated as exactly tied / equal (1e-9).
- NUMERICAL_BOUNDARY_TOLERANCE: Precision threshold for absorbing tiny floating-point noise
  on theoretical mathematical boundaries such as Kendall's W in [0.0, 1.0] (1e-9).
"""

from __future__ import annotations

# Tolerance for verifying recomputed metrics against unrounded float64 reported values
RECOMPUTATION_TOLERANCE: float = 1e-6

# Tolerance used when auditing legacy/demo evaluation files where metrics were rounded to 4 decimals
DEMO_RECOMPUTATION_TOLERANCE: float = 1e-4

# Single source of truth for determining exact equality / ties between model losses, metrics, and differences
TIE_TOLERANCE: float = 1e-9

# Single source of truth for numerical boundary noise adjustments (e.g. Kendall's W near 0.0 or 1.0)
NUMERICAL_BOUNDARY_TOLERANCE: float = 1e-9
