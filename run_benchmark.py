from __future__ import annotations
import json
from pathlib import Path
import typer
from dotenv import load_dotenv
from rich import print
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.mock_runtime import MockRuntime
from src.reflexion_lab.openai_runtime import OpenAIRuntime
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)

def run_agent_with_progress(label: str, agent, examples: list, progress) -> list:
    task_id = progress.add_task(f"[cyan]{label}", total=len(examples))
    records = []
    for example in examples:
        records.append(agent.run(example))
        progress.advance(task_id)
    return records

@app.command()
def main(dataset: str = "data/hotpot_100.json", out_dir: str = "outputs/openai_100", reflexion_attempts: int = 3, mode: str = "openai") -> None:
    load_dotenv()
    examples = load_dataset(dataset)
    if mode == "mock":
        runtime = MockRuntime()
    elif mode == "openai":
        runtime = OpenAIRuntime()
    else:
        raise typer.BadParameter("mode must be 'openai' or 'mock'")
    react = ReActAgent(runtime=runtime)
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts, runtime=runtime)
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        react_records = run_agent_with_progress("ReAct", react, examples, progress)
        reflexion_records = run_agent_with_progress("Reflexion", reflexion, examples, progress)
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
