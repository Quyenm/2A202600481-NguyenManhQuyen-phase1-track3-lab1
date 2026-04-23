from __future__ import annotations
from dataclasses import dataclass
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}

def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if example.qid not in FIRST_ATTEMPT_WRONG:
        return example.gold_answer
    if agent_type == "react":
        return FIRST_ATTEMPT_WRONG[example.qid]
    if attempt_id == 1 and not reflection_memory:
        return FIRST_ATTEMPT_WRONG[example.qid]
    return example.gold_answer

def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
    if normalize_answer(answer) == "london":
        return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
    return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])

def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
    return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)

@dataclass
class MockResult:
    text: str | None = None
    judge: JudgeResult | None = None
    reflection: ReflectionEntry | None = None
    tokens: int = 0
    latency_ms: int = 0

class MockRuntime:
    failure_modes = FAILURE_MODE_BY_QID

    def actor_answer(self, example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> MockResult:
        answer = actor_answer(example, attempt_id, agent_type, reflection_memory)
        tokens = 320 + (attempt_id * 65) + (120 if agent_type == "reflexion" else 0)
        latency_ms = 160 + (attempt_id * 40) + (90 if agent_type == "reflexion" else 0)
        return MockResult(text=answer, tokens=tokens, latency_ms=latency_ms)

    def evaluator(self, example: QAExample, answer: str) -> MockResult:
        return MockResult(judge=evaluator(example, answer), tokens=25, latency_ms=20)

    def reflector(self, example: QAExample, attempt_id: int, judge: JudgeResult) -> MockResult:
        return MockResult(reflection=reflector(example, attempt_id, judge), tokens=60, latency_ms=35)
