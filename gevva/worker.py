"""Persistent engine transport. One scheduler owns all reads and writes."""
import json
import selectors
import subprocess
from .config import load_config


class Worker:
    def __init__(self, engine=None, timeout=None, *, config=None, gpu=None):
        self.config = load_config(config, overrides={
            'engine.binary': engine, 'engine.worker_timeout': timeout, 'engine.gpu': gpu})
        self.config.validate_files()
        self.timeout = self.config.engine.worker_timeout
        self.process = subprocess.Popen([self.config.engine.binary, 'decision-serve'],
            env=self.config.worker_env(), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, bufsize=1)
        self.closed = False
        self.starting = True

    def run(self, request):
        if self.closed or self.process.poll() is not None:
            raise RuntimeError('decision worker is not running')
        try:
            self.process.stdin.write(json.dumps(request, allow_nan=False) + '\n')
            self.process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                timeout = self.config.engine.startup_timeout if self.starting else self.timeout
                if not selector.select(timeout):
                    self.process.kill()  # Never read an old response as a new one.
                    raise RuntimeError('decision worker timed out and was stopped')
            line = self.process.stdout.readline()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError('decision worker transport failed') from error
        if not line:
            raise RuntimeError(f'decision worker exited: {self.process.poll()}')
        try:
            response = json.loads(line)
        except ValueError as error:
            self.process.kill()
            raise RuntimeError('decision worker returned invalid JSON and was stopped') from error
        self.starting = False
        if 'error' in response:
            if response.get('error_kind') == 'validation':
                raise ValueError(response['error'])
            raise RuntimeError(response['error'])
        return response

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()
