from run_benchmark import run_agent_with_progress


class FakeAgent:
    def __init__(self):
        self.seen = []

    def run(self, example):
        self.seen.append(example)
        return {"qid": example}


class FakeProgress:
    def __init__(self):
        self.tasks = []
        self.advanced = []

    def add_task(self, description, total):
        task_id = len(self.tasks)
        self.tasks.append((description, total))
        return task_id

    def advance(self, task_id):
        self.advanced.append(task_id)


def test_run_agent_with_progress_advances_once_per_example():
    agent = FakeAgent()
    progress = FakeProgress()

    records = run_agent_with_progress("ReAct", agent, ["a", "b", "c"], progress)

    assert records == [{"qid": "a"}, {"qid": "b"}, {"qid": "c"}]
    assert progress.tasks == [("[cyan]ReAct", 3)]
    assert progress.advanced == [0, 0, 0]
