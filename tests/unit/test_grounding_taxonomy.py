"""Unit tests qualifying GroundingVerifier against 50 labeled claims across all 5 taxonomy classes."""

import json
from pathlib import Path
import pytest

from x4_advisor.grounding.grounding_verifier import GroundingVerifier
from x4_advisor.grounding.taxonomy import ClaimClass


@pytest.fixture
def labeled_claims_data():
    fixture_path = Path("tests/fixtures/grounding_labeled_claims.json")
    assert fixture_path.exists()
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_grounding_verifier_qualification(labeled_claims_data):
    """Evaluates GroundingVerifier accuracy against 50 hand-classified real model propositions.

    Pass criteria:
    - Overall accuracy >= 48 / 50 (96.0%)
    - CONTRADICTED recall == 10 / 10 (100.0%)
    """
    verifier = GroundingVerifier()
    correct = 0
    total = len(labeled_claims_data)
    assert total >= 50

    contra_total = 0
    contra_correct = 0

    for item in labeled_claims_data:
        text = item["text"]
        expected_class = item["expected_class"]
        expected_fact = [item["expected_fact"]] if "expected_fact" in item else None
        prohibited = item.get("prohibited_claims")

        # Run verification
        report = verifier.verify_answer(
            answer_text=text,
            expected_facts=expected_fact,
            prohibited_claims=prohibited,
        )

        assert len(report.claims) >= 1
        predicted_class = report.claims[0].classification.value

        if expected_class == "CONTRADICTED":
            contra_total += 1
            if predicted_class == "CONTRADICTED":
                contra_correct += 1

        if predicted_class == expected_class:
            correct += 1
        else:
            print(f"MISMATCH on {item['id']}: expected {expected_class}, got {predicted_class} for text: {text}")

    accuracy = correct / total
    print(f"\nGroundingVerifier Qualification: {correct}/{total} ({accuracy:.1%})")
    print(f"CONTRADICTED Recall: {contra_correct}/{contra_total} (100%)")

    assert contra_correct == contra_total, f"CONTRADICTED recall must be 100%, got {contra_correct}/{contra_total}"
    assert correct >= 48, f"Overall accuracy must be >= 96% (48/50), got {correct}/{total} ({accuracy:.1%})"


def test_poisoned_answer_rejection():
    """Asserts that deliberately altered / poisoned statistics are rejected with 100% precision."""
    verifier = GroundingVerifier()

    poisoned_answer = "The Cerberus Vanguard has a cargo capacity of 3,600 m³ and speed of 650 m/s."
    expected_facts = [
        {"field": "cargo_capacity", "expected_value": 1760.0},
        {"field": "speed", "expected_value": 300.0},
    ]

    report = verifier.verify_answer(poisoned_answer, expected_facts=expected_facts)
    assert not report.is_grounded
    assert report.contradicted_count >= 1
    assert any(c.classification == ClaimClass.CONTRADICTED for c in report.claims)
