from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .mock_runtime import MockRuntime
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    runtime: object | None = None

    def __post_init__(self) -> None:
        if self.runtime is None:
            self.runtime = MockRuntime()

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            actor_result = self.runtime.actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            answer = actor_result.text or ""
            judge_result = self.runtime.evaluator(example, answer)
            judge = judge_result.judge
            if judge is None:
                raise ValueError("Runtime evaluator returned no judge result")
            token_estimate = actor_result.tokens + judge_result.tokens
            latency_ms = actor_result.latency_ms + judge_result.latency_ms
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=token_estimate, latency_ms=latency_ms)
            final_answer = answer
            final_score = judge.score
            if judge.score == 1:
                traces.append(trace)
                break

            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection_result = self.runtime.reflector(example, attempt_id, judge)
                reflection = reflection_result.reflection
                if reflection is None:
                    raise ValueError("Runtime reflector returned no reflection")
                reflections.append(reflection)
                reflection_memory.append(f"{reflection.lesson} Next strategy: {reflection.next_strategy}")
                trace.reflection = reflection
                trace.token_estimate += reflection_result.tokens
                trace.latency_ms += reflection_result.latency_ms
            traces.append(trace)
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_modes = getattr(self.runtime, "failure_modes", {})
        failure_mode = "none" if final_score == 1 else failure_modes.get(example.qid, "wrong_final_answer")
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self, runtime: object | None = None) -> None:
        super().__init__(agent_type="react", max_attempts=1, runtime=runtime)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, runtime: object | None = None) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts, runtime=runtime)
