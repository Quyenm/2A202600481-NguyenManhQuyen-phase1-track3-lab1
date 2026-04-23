from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry


@dataclass
class OpenAIResult:
    text: str | None = None
    judge: JudgeResult | None = None
    reflection: ReflectionEntry | None = None
    tokens: int = 0
    latency_ms: int = 0


class OpenAIRuntime:
    failure_modes: dict[str, str] = {}

    def __init__(self, client: Any | None = None, model: str | None = None, timeout_seconds: float | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.timeout_seconds = timeout_seconds or float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=self.timeout_seconds)
        self.client = client

    def actor_answer(self, example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> OpenAIResult:
        user_input = "\n\n".join(
            [
                f"Question: {example.question}",
                f"Context:\n{_format_context(example)}",
                f"Attempt: {attempt_id}",
                f"Agent type: {agent_type}",
                "Reflection memory:\n" + ("\n".join(f"- {item}" for item in reflection_memory) if reflection_memory else "None"),
            ]
        )
        response, latency_ms = self._create(ACTOR_SYSTEM, user_input)
        return OpenAIResult(text=_response_text(response).strip(), tokens=_usage_tokens(response), latency_ms=latency_ms)

    def evaluator(self, example: QAExample, answer: str) -> OpenAIResult:
        user_input = "\n\n".join(
            [
                f"Question: {example.question}",
                f"Gold answer: {example.gold_answer}",
                f"Predicted answer: {answer}",
            ]
        )
        response, latency_ms = self._create(EVALUATOR_SYSTEM, user_input)
        payload = _json_payload(_response_text(response))
        judge = JudgeResult.model_validate(payload)
        return OpenAIResult(judge=judge, tokens=_usage_tokens(response), latency_ms=latency_ms)

    def reflector(self, example: QAExample, attempt_id: int, judge: JudgeResult) -> OpenAIResult:
        user_input = "\n\n".join(
            [
                f"Question: {example.question}",
                f"Gold answer: {example.gold_answer}",
                f"Failed attempt: {attempt_id}",
                f"Evaluator reason: {judge.reason}",
                "Missing evidence: " + ", ".join(judge.missing_evidence),
                "Spurious claims: " + ", ".join(judge.spurious_claims),
            ]
        )
        response, latency_ms = self._create(REFLECTOR_SYSTEM, user_input)
        payload = _json_payload(_response_text(response))
        reflection = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=str(payload.get("failure_reason", judge.reason)),
            lesson=str(payload["lesson"]),
            next_strategy=str(payload["next_strategy"]),
        )
        return OpenAIResult(reflection=reflection, tokens=_usage_tokens(response), latency_ms=latency_ms)

    def _create(self, instructions: str, user_input: str) -> tuple[Any, int]:
        started = time.perf_counter()
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user_input,
        )
        latency_ms = round((time.perf_counter() - started) * 1000)
        return response, latency_ms


def _format_context(example: QAExample) -> str:
    return "\n".join(f"[{chunk.title}] {chunk.text}" for chunk in example.context)


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text is not None:
        return str(text)
    if isinstance(response, dict) and response.get("output_text") is not None:
        return str(response["output_text"])
    return ""


def _usage_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get("total_tokens") or usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
    total = getattr(usage, "total_tokens", None)
    if total is not None:
        return int(total)
    return int(getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0))


def _json_payload(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)
