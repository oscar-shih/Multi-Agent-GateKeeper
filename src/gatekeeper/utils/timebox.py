import time
from contextlib import contextmanager
from typing import Generator

class Timer:
    """A simple timer context manager."""
    def __init__(self, name: str = "Task"):
        self.name = name
        self.start = 0.0
        self.end = 0.0
        self.duration = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.duration = self.end - self.start

@contextmanager
def timebox(limit_s: float) -> Generator[None, None, None]:
    """Raises TimeoutError if the block exceeds limit_s."""
    start = time.time()
    yield
    elapsed = time.time() - start
    if elapsed > limit_s:
        raise TimeoutError(f"Timebox exceeded: {elapsed:.2f}s > {limit_s}s")

