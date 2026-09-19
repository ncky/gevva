"""Startup may load cold weights; subsequent requests retain their own deadline."""
from pathlib import Path
import tempfile
import threading
import time
from concurrent.futures import CancelledError
import unittest
from unittest.mock import patch

from gevva.config import Config, ConfigError, load_config
from gevva.worker import Worker
from gevva.api import LocalEvaluator


class WorkerStartupTests(unittest.TestCase):
    def test_cold_start_deadline_does_not_extend_later_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Path(directory) / 'slow-worker'
            engine.write_text('#!/usr/bin/env python3\nimport sys,time\n'
                              'for line in sys.stdin:\n'
                              ' time.sleep(0.2)\n'
                              ' print("{}", flush=True)\n')
            engine.chmod(0o755)
            config = load_config(environ={}, overrides={
                'engine.binary': engine, 'engine.startup_timeout': 5,
                'engine.worker_timeout': 0.05})
            with patch.object(Config, 'validate_files'):
                worker = Worker(config=config)
            try:
                self.assertEqual(worker.run({}), {})
                with self.assertRaisesRegex(RuntimeError, 'timed out'):
                    worker.run({})
            finally:
                worker.close()
            self.assertIsNotNone(worker.process.returncode)

    def test_invalid_startup_timeout(self):
        for value in (0, -1, True, float('nan'), float('inf'), '300'):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                load_config(environ={}, overrides={'engine.startup_timeout': value})


class WorkerTransportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.engine = Path(self.directory.name) / 'worker'

    def make_config(self, body, timeout=.15):
        self.engine.write_text('#!/usr/bin/env python3\nimport os,sys,time\n' + body)
        self.engine.chmod(0o755)
        return load_config(environ={}, overrides={
            'engine.binary': self.engine, 'engine.startup_timeout': timeout,
            'engine.worker_timeout': timeout})

    def check_failure(self, body, payload, message, limit=None):
        config = self.make_config(body)
        with patch.object(Config, 'validate_files'):
            worker = Worker(config=config)
        result = []
        def run():
            try:
                worker.run(payload)
            except Exception as error:
                result.append(error)
        limit_patch = patch('gevva.worker.MAX_RESPONSE_BYTES', limit or 64 * 1024 * 1024)
        with limit_patch:
            thread = threading.Thread(target=run, daemon=True)
            start = time.monotonic()
            thread.start()
            thread.join(1.5)
            try:
                self.assertFalse(thread.is_alive(), 'transport exceeded its deadline')
                self.assertLess(time.monotonic() - start, 1.5)
                self.assertEqual(len(result), 1)
                self.assertIsInstance(result[0], RuntimeError)
                self.assertIn(message, str(result[0]))
                self.assertIsNotNone(worker.process.returncode, 'child must already be reaped')
                self.assertFalse(worker.healthy)
            finally:
                # A regression must fail the test rather than hang the suite.
                if worker.process.poll() is None:
                    worker.process.kill()
                thread.join(2)
                worker.close()

    def test_partial_response_has_deadline(self):
        self.check_failure('sys.stdin.readline()\nsys.stdout.write("{")\nsys.stdout.flush()\ntime.sleep(60)\n',
                           {}, 'timed out')

    def test_full_stdin_pipe_has_deadline(self):
        self.check_failure('time.sleep(60)\n', {'state': 'x' * (1024 * 1024)}, 'timed out')

    def test_slow_response_cannot_reset_deadline(self):
        self.check_failure('sys.stdin.readline()\nwhile True:\n os.write(1,b" ")\n time.sleep(.02)\n',
                           {}, 'timed out')

    def test_oversized_response_is_rejected_and_reaped(self):
        self.check_failure('sys.stdin.readline()\nos.write(1,b"x"*4096)\ntime.sleep(60)\n',
                           {}, 'exceeds', limit=1024)

    def test_invalid_json_is_rejected_and_reaped(self):
        self.check_failure('sys.stdin.readline()\nprint("not json",flush=True)\ntime.sleep(60)\n',
                           {}, 'invalid JSON')

    def test_chunked_utf8_responses_and_large_writes(self):
        config = self.make_config('import json\nfor line in sys.stdin:\n'
            ' data=json.loads(line)\n response=json.dumps({"length":len(data["state"]),"text":"é"},ensure_ascii=False).encode()+b"\\n"\n'
            ' for byte in response:\n  os.write(1,bytes([byte]))\n  time.sleep(.001)\n', timeout=3)
        with patch.object(Config, 'validate_files'):
            worker = Worker(config=config)
        try:
            for size in [1024 * 1024, 3]:
                self.assertEqual(worker.run({'state': 'x' * size}), {'length': size, 'text': 'é'})
            self.assertTrue(worker.healthy)
        finally:
            worker.close()

    def test_shutdown_interrupts_active_transport_and_cancels_queue(self):
        for mode in ('no-read', 'partial'):
            with self.subTest(mode=mode):
                body = ('sys.stdin.readline()\nos.write(1,b"{")\n' if mode == 'partial' else '') + 'time.sleep(60)\n'
                config = self.make_config(body, timeout=30)
                with patch.object(Config, 'validate_files'):
                    evaluator = LocalEvaluator(config=config)
                payload = {'state': 'x' * (1024 * 1024), 'questions': {
                    'flag': {'type': 'noul', 'instructions': 'Check.'}}}
                active = evaluator.submit(payload)
                end = time.monotonic() + 2
                while not active.running() and time.monotonic() < end:
                    time.sleep(.005)
                self.assertTrue(active.running())
                pending = evaluator.submit(payload)
                closer = threading.Thread(target=evaluator.close, daemon=True)
                closer.start()
                closer.join(1.5)
                try:
                    self.assertFalse(closer.is_alive(), 'shutdown blocked on transport')
                    self.assertFalse(evaluator.scheduler.thread.is_alive())
                    self.assertFalse(evaluator.healthy)
                    self.assertIsNotNone(evaluator.worker.process.returncode)
                    with self.assertRaises(RuntimeError):
                        active.result(timeout=1)
                    with self.assertRaises(CancelledError):
                        pending.result(timeout=1)
                finally:
                    if evaluator.worker.process.poll() is None:
                        evaluator.worker.process.kill()
                    closer.join(2)
                    evaluator.close()


if __name__ == '__main__':
    unittest.main()
