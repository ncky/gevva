"""Persistent engine transport. One deadline covers each complete JSONL exchange."""
import json
import os
import selectors
import subprocess
import threading
import time
from .config import load_config

MAX_RESPONSE_BYTES = 64 * 1024 * 1024


class Worker:
    def __init__(self, engine=None, timeout=None, *, config=None, gpu=None):
        self.config = load_config(config, overrides={
            'engine.binary': engine, 'engine.worker_timeout': timeout, 'engine.gpu': gpu})
        self.config.validate_files()
        self.timeout = self.config.engine.worker_timeout
        self.process = subprocess.Popen([self.config.engine.binary, 'decision-serve'],
            env=self.config.worker_env(), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, bufsize=0)
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        self.closed = False
        self.starting = True
        self.deadline = None
        self._transport_lock = threading.Lock()
        self._stop_lock = threading.Lock()

    @property
    def healthy(self):
        return (not self.closed and self.process.poll() is None and
                (self.deadline is None or time.monotonic() < self.deadline))

    def _stop(self):
        # May be called by close() while the scheduler is inside run(). Killing
        # the child wakes the nonblocking transport; always collect its status.
        with self._stop_lock:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()

    def run(self, request):
        with self._transport_lock:
            if self.closed or self.process.poll() is not None:
                raise RuntimeError('decision worker is not running')
            timeout = self.config.engine.startup_timeout if self.starting else self.timeout
            self.deadline = time.monotonic() + timeout
            try:
                payload = (json.dumps(request, allow_nan=False) + '\n').encode()
                line = self._exchange(payload)
                try:
                    response = json.loads(line)
                    if not isinstance(response, dict):
                        raise ValueError('response must be an object')
                except (ValueError, UnicodeError) as error:
                    self._stop()
                    raise RuntimeError('decision worker returned invalid JSON and was stopped') from error
                self.starting = False
                if 'error' in response:
                    if response.get('error_kind') == 'validation':
                        raise ValueError(response['error'])
                    raise RuntimeError(response['error'])
                return response
            finally:
                self.deadline = None

    def _exchange(self, payload):
        sent = 0
        response = bytearray()
        complete = False
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdin, selectors.EVENT_WRITE)
                selector.register(self.process.stdout, selectors.EVENT_READ)
                while sent < len(payload) or not complete:
                    remaining = self.deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError
                    if self.closed:
                        raise OSError('worker closed')
                    for key, _ in selector.select(remaining):
                        if time.monotonic() >= self.deadline:
                            raise TimeoutError
                        try:
                            if key.fileobj is self.process.stdin:
                                sent += os.write(key.fd, memoryview(payload)[sent:sent + 65536])
                                if sent == len(payload):
                                    selector.unregister(self.process.stdin)
                            else:
                                chunk = os.read(key.fd, min(65536, MAX_RESPONSE_BYTES + 1 - len(response)))
                                if not chunk:
                                    raise OSError('worker closed response pipe')
                                response.extend(chunk)
                                if len(response) > MAX_RESPONSE_BYTES:
                                    raise OSError('worker response exceeds 64 MiB limit')
                                if b'\n' in chunk:
                                    if response.find(b'\n') != len(response) - 1:
                                        raise OSError('worker returned multiple response lines')
                                    complete = True
                                    selector.unregister(self.process.stdout)
                        except BlockingIOError:
                            continue
            return response
        except TimeoutError as error:
            self._stop()
            raise RuntimeError('decision worker timed out and was stopped') from error
        except OSError as error:
            self._stop()
            raise RuntimeError(f'decision worker transport failed: {error}') from error

    def close(self):
        self.closed = True
        self._stop()
        with self._transport_lock:
            self.process.stdin.close()
            self.process.stdout.close()
