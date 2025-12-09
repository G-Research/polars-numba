from contextlib import contextmanager
from dataclasses import dataclass
from time import process_time


@dataclass
class Elapsed:
    time: float | None = None


@contextmanager
def measure_cpu_time():
    """
    Context manager that measures elapsed process CPU time.
    """
    elapsed = Elapsed()
    start = process_time()
    try:
        yield elapsed
    finally:
        elapsed.time = process_time() - start
