"""The memory-budgeted process pool."""

import time

import _workers

from market_1t import scheduler as sched


def run(tasks, func, **kwargs):
    kwargs.setdefault("max_workers", 3)
    kwargs.setdefault("memory_budget", 10**12)
    return list(sched.run_tasks(tasks, func, **kwargs))


def test_results():
    tasks = [sched.Task(key=i, memory=1, payload=i) for i in range(10)]
    assert {task.key: result for task, result, _ in run(tasks, _workers.square)} == {i: i * i for i in range(10)}


def test_errors_are_reported_per_task():
    tasks = [sched.Task(key=i, memory=1, payload=i) for i in range(6)]
    outcome = {task.key: (result, error) for task, result, error in run(tasks, _workers.fail_on_three)}
    assert isinstance(outcome[3][1], ValueError)
    assert {k: r for k, (r, e) in outcome.items() if e is None} == {0: 0, 1: 1, 2: 2, 4: 4, 5: 5}


def test_worker_death_is_retried_alone_then_reported():
    tasks = [sched.Task(key=i, memory=1, payload=i) for i in range(6)]
    outcome = {task.key: (result, error) for task, result, error in run(tasks, _workers.die_on_two)}
    assert sorted(outcome) == list(range(6))
    assert isinstance(outcome[2][1], sched.WorkerDied)
    assert all(outcome[key] == (key, None) for key in (0, 1, 3, 4, 5))


def test_worker_death_while_results_are_consumed():
    # The pool breaks after wait() returned a result and before the next
    # submit(): the unstarted task must go to the new pool, not crash.
    tasks = [sched.Task("X", 3, ("die", 0.5)), sched.Task("A", 2, ("ok", 0.0)), sched.Task("C", 1, ("ok", 0.0))]
    outcome = {}
    for task, result, error in sched.run_tasks(tasks, _workers.job, max_workers=2, memory_budget=100):
        outcome[task.key] = (result, error)
        if task.key == "A":
            time.sleep(1.5)
    assert outcome["A"] == (("ok", 0.0), None)
    assert outcome["C"] == (("ok", 0.0), None)
    assert isinstance(outcome["X"][1], sched.WorkerDied)


def test_failing_initializer_gives_up_quickly():
    tasks = [sched.Task(i, 1, i) for i in range(8)]
    outcome = run(tasks, _workers.square, max_workers=4, initializer=_workers.failing_initializer)
    assert sorted(task.key for task, _, _ in outcome) == list(range(8))
    assert all(isinstance(error, sched.WorkerDied) for _, _, error in outcome)


def test_memory_budget_limits_concurrency():
    # Budget 10 with tasks of 4: at most two run at the same time.
    tasks = [sched.Task(key=i, memory=4, payload=0.4) for i in range(6)]
    spans = [result for _, result, _ in run(tasks, _workers.sleep_and_report, max_workers=6, memory_budget=10)]
    events = sorted([(start, 1) for start, _ in spans] + [(end, -1) for _, end in spans])
    running = peak = 0
    for _, delta in events:
        running += delta
        peak = max(peak, running)
    assert peak <= 2


def test_task_larger_than_budget_still_runs():
    assert run([sched.Task(key=0, memory=100, payload=3)], _workers.square, memory_budget=10)[0][1] == 9


def test_inline_execution():
    tasks = [sched.Task(key=i, memory=1, payload=i) for i in range(4)]
    assert {task.key: result for task, result, _ in run(tasks, _workers.square, max_workers=1)} == {
        0: 0,
        1: 1,
        2: 4,
        3: 9,
    }
