from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.schemas import JudgeResult, QAExample, ReflectionEntry


class FakeResult:
    def __init__(self, text=None, judge=None, reflection=None, tokens=5, latency_ms=7):
        self.text = text
        self.judge = judge
        self.reflection = reflection
        self.tokens = tokens
        self.latency_ms = latency_ms


class FakeRuntime:
    failure_modes = {"hp-test": "incomplete_multi_hop"}

    def actor_answer(self, example, attempt_id, agent_type, reflection_memory):
        if agent_type == "reflexion" and reflection_memory:
            return FakeResult(text=example.gold_answer, tokens=11, latency_ms=13)
        return FakeResult(text="London", tokens=3, latency_ms=5)

    def evaluator(self, example, answer):
        if answer == example.gold_answer:
            return FakeResult(
                judge=JudgeResult(score=1, reason="Correct.", missing_evidence=[], spurious_claims=[]),
                tokens=2,
                latency_ms=4,
            )
        return FakeResult(
            judge=JudgeResult(
                score=0,
                reason="Only completed first hop.",
                missing_evidence=["second hop"],
                spurious_claims=["London"],
            ),
            tokens=2,
            latency_ms=4,
        )

    def reflector(self, example, attempt_id, judge):
        return FakeResult(
            reflection=ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=judge.reason,
                lesson="A first-hop answer is insufficient.",
                next_strategy="Use the second paragraph before answering.",
            ),
            tokens=3,
            latency_ms=6,
        )


def make_example() -> QAExample:
    return QAExample(
        qid="hp-test",
        difficulty="medium",
        question="What river flows through the birthplace?",
        gold_answer="River Thames",
        context=[{"title": "Birthplace", "text": "The city is London. The river is River Thames."}],
    )


def test_reflexion_uses_reflection_memory_to_retry_until_correct():
    record = ReflexionAgent(max_attempts=2, runtime=FakeRuntime()).run(make_example())

    assert record.is_correct is True
    assert record.attempts == 2
    assert len(record.reflections) == 1
    assert record.traces[0].reflection is not None
    assert record.token_estimate == 21


def test_react_runs_single_attempt_without_reflection():
    record = ReActAgent(runtime=FakeRuntime()).run(make_example())

    assert record.is_correct is False
    assert record.attempts == 1
    assert record.reflections == []
    assert record.failure_mode == "incomplete_multi_hop"
