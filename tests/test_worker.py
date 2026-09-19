"""Startup may load cold weights; subsequent requests retain their own deadline."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gevva.config import Config, ConfigError, load_config
from gevva.worker import Worker


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


if __name__ == '__main__':
    unittest.main()
