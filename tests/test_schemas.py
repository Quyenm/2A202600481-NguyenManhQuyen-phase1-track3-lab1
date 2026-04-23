from pydantic import ValidationError

from src.reflexion_lab.schemas import JudgeResult, ReflectionEntry


def test_judge_result_requires_score_reason_and_lists():
    judge = JudgeResult(
        score=1,
        reason="Exact match.",
        missing_evidence=[],
        spurious_claims=[],
    )

    assert judge.score == 1
    assert judge.missing_evidence == []


def test_judge_result_rejects_score_outside_binary_range():
    try:
        JudgeResult(score=2, reason="bad")
    except ValidationError as exc:
        assert "score" in str(exc)
    else:
        raise AssertionError("Expected score validation to fail")


def test_reflection_entry_exposes_strategy_fields():
    reflection = ReflectionEntry(
        attempt_id=1,
        failure_reason="Stopped after first hop.",
        lesson="Complete both hops.",
        next_strategy="Use the second supporting paragraph.",
    )

    assert reflection.attempt_id == 1
    assert "second" in reflection.next_strategy
