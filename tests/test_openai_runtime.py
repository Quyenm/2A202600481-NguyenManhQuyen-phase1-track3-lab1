from types import SimpleNamespace

from src.reflexion_lab.openai_runtime import OpenAIRuntime
from src.reflexion_lab.schemas import QAExample


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def make_example() -> QAExample:
    return QAExample(
        qid="hp-test",
        difficulty="easy",
        question="Which label released the EP?",
        gold_answer="Warner Bros",
        context=[{"title": "EP", "text": "The EP was released through Warner Bros."}],
    )


def test_actor_returns_text_tokens_and_latency_from_openai_response():
    response = SimpleNamespace(
        output_text=" Warner Bros. ",
        usage=SimpleNamespace(input_tokens=10, output_tokens=4, total_tokens=14),
    )
    client = SimpleNamespace(responses=FakeResponses(response))
    runtime = OpenAIRuntime(client=client, model="gpt-test")

    result = runtime.actor_answer(make_example(), 1, "react", [])

    assert result.text == "Warner Bros."
    assert result.tokens == 14
    assert result.latency_ms >= 0
    assert client.responses.calls[0]["model"] == "gpt-test"


def test_evaluator_parses_structured_json_result():
    response = SimpleNamespace(
        output_text='{"score": 0, "reason": "Wrong entity", "missing_evidence": ["second hop"], "spurious_claims": ["London"]}',
        usage={"total_tokens": 21},
    )
    client = SimpleNamespace(responses=FakeResponses(response))
    runtime = OpenAIRuntime(client=client, model="gpt-test")

    result = runtime.evaluator(make_example(), "London")

    assert result.judge.score == 0
    assert result.judge.missing_evidence == ["second hop"]
    assert result.tokens == 21
