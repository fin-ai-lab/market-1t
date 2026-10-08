"""Top-level task functions for the scheduler tests (spawned workers import them)."""

import os
import time


def square(x):
    return x * x


def fail_on_three(x):
    if x == 3:
        raise ValueError("three")
    return x


def die_on_two(x):
    if x == 2:
        os._exit(1)  # abrupt worker death, as with the OOM killer
    return x


def sleep_and_report(duration):
    start = time.time()
    time.sleep(duration)
    return start, time.time()


def job(spec):
    kind, delay = spec
    time.sleep(delay)
    if kind == "die":
        os._exit(1)
    return spec


def failing_initializer():
    raise RuntimeError("initializer failed")
