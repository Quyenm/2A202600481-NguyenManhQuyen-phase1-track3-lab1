# Lab 16 - Reflexion Agent with OpenAI

This repository has been completed to run a **ReAct vs Reflexion** benchmark on the **100-sample HotpotQA dataset** using the **OpenAI API**.

## 1. Overview

The main goals of this lab are:

1. Replace the mock runtime with a real LLM runtime.
2. Run the benchmark on at least 100 HotpotQA samples.
3. Export `report.json` and `report.md` in the required format.
4. Compute real token usage from API responses.

Current implementation:

- Supports `mode=openai` for real OpenAI calls.
- Supports `mode=mock` for fast flow testing without API cost.
- Compares `react` and `reflexion` in the same report.
- Shows benchmark progress with `rich progress`.

## 2. Setup and Run

```bash
# install dependencies
pip install -r requirements.txt

# create .env from .env.sample and fill in your API key
# OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_MODEL=gpt-4.1-mini
# OPENAI_TIMEOUT_SECONDS=60

# run the real OpenAI benchmark on 100 HotpotQA samples
python run_benchmark.py --dataset data/hotpot_100.json --out-dir outputs/openai_100 --mode openai --reflexion-attempts 3

# run the mock benchmark for quick flow testing
python run_benchmark.py --dataset data/hotpot_100.json --out-dir outputs/mock_100 --mode mock --reflexion-attempts 3

# run autograde
python autograde.py --report-path outputs/openai_100/report.json
```

Notes:

- `.env` is already included in `.gitignore`
- `.env.sample` is included for configuration sharing

## 3. Output Files

For example, when using `--out-dir outputs/openai_100`, the benchmark generates:

- `outputs/openai_100/react_runs.jsonl`
- `outputs/openai_100/reflexion_runs.jsonl`
- `outputs/openai_100/report.json`
- `outputs/openai_100/report.md`

## 4. Current Result Overview

Current OpenAI benchmark result on `data/hotpot_100.json`:

- ReAct EM: `0.89`
- Reflexion EM: `0.97`
- EM delta: `+0.08`
- Average attempts: `1.00 -> 1.14`
- Average token usage: `1652.21 -> 1928.30`
- Average latency: `3329.50 ms -> 4132.55 ms`

Short summary:

- Reflexion improves accuracy clearly over ReAct.
- The tradeoff is higher token usage and higher latency because Reflexion adds evaluator, reflection, and retry steps.
- `wrong_final_answer` errors drop from `11` to `3`.

## 5. Source Structure

- `src/reflexion_lab/schemas.py`: schemas for judge, reflection, trace, and report data.
- `src/reflexion_lab/prompts.py`: prompts for Actor, Evaluator, and Reflector.
- `src/reflexion_lab/mock_runtime.py`: mock runtime for fast no-cost testing.
- `src/reflexion_lab/openai_runtime.py`: real OpenAI runtime.
- `src/reflexion_lab/agents.py`: main ReAct and Reflexion agent logic.
- `src/reflexion_lab/reporting.py`: metric aggregation and report generation.
- `run_benchmark.py`: benchmark runner with progress bars.
- `autograde.py`: report grading script.

## 6. Extra Work Beyond the Minimum Requirements

In addition to the required lab functionality, this repository also includes:

- `structured_evaluator`: the evaluator returns structured JSON.
- `reflection_memory`: Reflexion stores lessons and next-step strategies for later attempts.
- `benchmark_report_json`: standardized report output for autograding.
- Rich progress bars during benchmark execution.
- Dual runtime support:
  - `openai` for the real benchmark
  - `mock` for fast testing
- Automatically generated `report.md` based on real benchmark metrics instead of placeholder discussion text.
- `.env.sample` for clearer environment setup.
- Verified autograde result:

```text
Auto-grade total: 100/100
```
