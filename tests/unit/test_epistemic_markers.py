"""Unit tests for epistemic drift marker detection."""

import pytest

from x4_advisor.curation.epistemic_markers import detect_epistemic_drift


def test_detect_quantifier_drift():
    """Verifies shift from 'often' to 'always' triggers quantifier drift."""
    drift = detect_epistemic_drift(
        c1_qualifier="often recommended for miners",
        c2_qualifier="always recommended for miners",
    )
    assert drift == "quantifier"


def test_detect_modality_drift():
    """Verifies shift from 'can produce' to 'must produce' triggers modality drift."""
    drift = detect_epistemic_drift(
        c1_qualifier="can produce high yields",
        c2_qualifier="must produce high yields",
    )
    assert drift == "modality"


def test_detect_attribution_drift_in_predicate():
    """Verifies attribution shift across predicate/qualifier (e.g. guide recommends -> is best)."""
    drift = detect_epistemic_drift(
        c1_qualifier="the guide recommends",
        c2_qualifier="is the best choice",
    )
    assert drift == "attribution"


def test_no_drift_when_identical():
    """Verifies no drift flagged when C1 and C2 carry identical wording."""
    drift = detect_epistemic_drift(
        c1_qualifier="often recommended",
        c2_qualifier="often recommended",
    )
    assert drift is None


def test_no_drift_when_both_weak():
    """Verifies no drift flagged when both C1 and C2 use weak terms."""
    drift = detect_epistemic_drift(
        c1_qualifier="sometimes useful",
        c2_qualifier="can be useful",
    )
    assert drift is None
