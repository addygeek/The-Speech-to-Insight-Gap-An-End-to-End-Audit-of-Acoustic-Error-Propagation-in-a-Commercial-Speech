"""Bounded worker pool for Deepgram HTTP calls."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any

from src.logging_utils import setup_logging

logger = setup_logging()

POOL_WORKERS = 6
REDUCED_WORKERS = 3
SHRINK_AFTER_429 = 3


class WorkerGate:
    """Caps in-flight calls and drops the cap after repeated HTTP 429 responses."""

    def __init__(self, start: int = POOL_WORKERS, reduced: int = REDUCED_WORKERS, shrink_after: int = SHRINK_AFTER_429) -> None:
        self._limit = max(1, start)
        self._reduced = max(1, min(reduced, self._limit))
        self._shrink_after = shrink_after
        self._active = 0
        self._hits = 0
        self._stop = False
        self._cv = threading.Condition()

    @property
    def limit(self) -> int:
        with self._cv:
            return self._limit

    def acquire(self) -> bool:
        with self._cv:
            while not self._stop and self._active >= self._limit:
                self._cv.wait()
            if self._stop:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._cv:
            self._active = max(0, self._active - 1)
            self._cv.notify_all()

    def note_status(self, status: int) -> None:
        if status not in (408, 429):
            return
        with self._cv:
            self._hits += 1
            if self._hits >= self._shrink_after and self._limit > self._reduced:
                self._limit = self._reduced
                logger.warning("Deepgram timed out or rate-limited %s times; workers now %s", self._hits, self._limit)
                self._cv.notify_all()

    def request_stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()


def run_pool(
    jobs: list[Any],
    worker: Callable[[Any, Any, WorkerGate, threading.Lock], None],
    client_factory: Callable[[], Any] | None = None,
    workers: int = POOL_WORKERS,
) -> None:
    if not jobs:
        return
    n_workers = max(1, min(workers, len(jobs)))
    gate = WorkerGate(start=n_workers, reduced=min(REDUCED_WORKERS, n_workers))
    pending: queue.Queue[Any] = queue.Queue()
    for job in jobs:
        pending.put(job)
    book = threading.Lock()

    def run() -> None:
        client = client_factory() if client_factory is not None else None
        try:
            while True:
                if not gate.acquire():
                    return
                try:
                    job = pending.get_nowait()
                except queue.Empty:
                    gate.release()
                    return
                try:
                    worker(job, client, gate, book)
                except Exception:
                    logger.exception("Deepgram worker failed")
                finally:
                    gate.release()
        finally:
            if client is not None:
                client.close()

    threads = [threading.Thread(target=run, name=f"deepgram-{index}") for index in range(n_workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
