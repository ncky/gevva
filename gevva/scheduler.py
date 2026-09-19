"""Bounded FIFO execution with optional replacement of pending stream frames."""
from concurrent.futures import Future
from dataclasses import dataclass
import threading
import time


class QueueFull(RuntimeError):
    pass


class Superseded(RuntimeError):
    pass


class QueueExpired(RuntimeError):
    pass


@dataclass
class Job:
    payload: object
    future: Future
    stream: str | None
    latest: bool
    queued: float
    max_queue_ms: float | None


class Scheduler:
    def __init__(self, execute, capacity=8):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('queue capacity must be a positive integer')
        self.execute = execute
        self.capacity = capacity
        self.pending = []
        self.condition = threading.Condition()
        self.closed = False
        self.thread = threading.Thread(target=self._run, name='gevva-inference', daemon=True)
        self.thread.start()

    def submit(self, payload, *, stream=None, latest=False, max_queue_ms=None):
        future = Future()
        job = Job(payload, future, stream, latest, time.perf_counter(), max_queue_ms)
        replaced = None
        with self.condition:
            if self.closed:
                raise RuntimeError('evaluator is closed')
            self.pending = [old for old in self.pending if not old.future.cancelled()]
            index = next((i for i, old in enumerate(self.pending)
                          if latest and old.latest and old.stream == stream), None)
            if index is not None:
                replaced = self.pending[index]
                self.pending[index] = job  # Preserve this stream's queue position.
            elif len(self.pending) >= self.capacity:
                raise QueueFull('local inference queue is full')
            else:
                self.pending.append(job)
            self.condition.notify()
        if replaced is not None:
            # Mark running first to resolve a simultaneous caller cancellation.
            if replaced.future.set_running_or_notify_cancel():
                replaced.future.set_exception(Superseded('a newer pending request replaced this stream frame'))
        return future

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending)
                if not self.pending:
                    return
                job = self.pending.pop(0)
            if not job.future.set_running_or_notify_cancel():
                continue
            queue_ms = (time.perf_counter() - job.queued) * 1000
            try:
                if job.max_queue_ms is not None and queue_ms > job.max_queue_ms:
                    raise QueueExpired('request exceeded max_queue_ms before inference')
                result = self.execute(job.payload)
                result['gevva']['queue_ms'] = queue_ms
                result['gevva']['total_ms'] = (time.perf_counter() - job.queued) * 1000
                job.future.set_result(result)
            except Exception as error:
                job.future.set_exception(error)

    def close(self, *, wait=True):
        with self.condition:
            self.closed = True
            pending, self.pending = self.pending, []
            self.condition.notify_all()
        for job in pending:
            job.future.cancel()
        if wait and threading.current_thread() is not self.thread:
            self.thread.join()
