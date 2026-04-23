from src.reflexion_lab.reporting import build_report, failure_breakdown
from src.reflexion_lab.schemas import RunRecord


def make_record(qid: str, agent_type: str, failure_mode: str) -> RunRecord:
    return RunRecord(
        qid=qid,
        question="q",
        gold_answer="gold",
        agent_type=agent_type,
        predicted_answer="pred",
        is_correct=failure_mode == "none",
        attempts=1,
        token_estimate=1,
        latency_ms=1,
        failure_mode=failure_mode,
        reflections=[],
        traces=[],
    )


def test_failure_breakdown_includes_agent_groups_and_overall_group():
    breakdown = failure_breakdown(
        [
            make_record("a", "react", "wrong_final_answer"),
            make_record("b", "reflexion", "none"),
        ]
    )

    assert breakdown["react"] == {"wrong_final_answer": 1}
    assert breakdown["reflexion"] == {"none": 1}
    assert breakdown["overall"] == {"wrong_final_answer": 1, "none": 1}


def test_build_report_discussion_mentions_actual_summary_deltas():
    report = build_report(
        [
            make_record("a", "react", "wrong_final_answer"),
            make_record("b", "reflexion", "none"),
        ],
        dataset_name="demo.json",
        mode="openai",
    )

    assert "Reflexion" in report.discussion
    assert "EM" in report.discussion
    assert "token" in report.discussion.lower()
    assert "latency" in report.discussion.lower()
    assert str(report.summary["delta_reflexion_minus_react"]["em_abs"]) in report.discussion
