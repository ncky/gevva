"""Deterministic queue tests: no GPU, timed sleeps or inference assumptions."""
from concurrent.futures import CancelledError
import asyncio
import threading
import unittest
from unittest.mock import patch
from gevva.scheduler import Scheduler, QueueFull, Superseded, QueueExpired
from gevva.api import LocalEvaluator


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []
        def execute(payload):
            self.calls.append(payload)
            if payload == 'active':
                self.started.set()
                assert self.release.wait(5)
            return {'gevva': {}, 'value': payload}
        self.scheduler = Scheduler(execute, capacity=2)
        self.active = self.scheduler.submit('active')
        self.assertTrue(self.started.wait(5))

    def tearDown(self):
        self.release.set()
        self.scheduler.close()

    def test_latest_preserves_fifo_slot_and_does_not_drop_email(self):
        old = self.scheduler.submit('old-frame', stream='game', latest=True)
        email = self.scheduler.submit('email')
        new = self.scheduler.submit('new-frame', stream='game', latest=True)
        with self.assertRaises(Superseded):
            old.result()
        with self.assertRaises(QueueFull):
            self.scheduler.submit('overflow')
        self.release.set()
        self.assertEqual(new.result(5)['value'], 'new-frame')
        self.assertEqual(email.result(5)['value'], 'email')
        self.assertEqual(self.calls, ['active', 'new-frame', 'email'])
        self.assertGreaterEqual(new.result()['gevva']['total_ms'], new.result()['gevva']['queue_ms'])

    def test_cancelled_request_frees_capacity(self):
        cancelled = self.scheduler.submit('cancelled')
        self.assertTrue(cancelled.cancel())
        first = self.scheduler.submit('first')
        second = self.scheduler.submit('second')
        self.release.set()
        first.result(5)
        second.result(5)
        self.assertNotIn('cancelled', self.calls)

    def test_expiry_skips_inference_and_next_request_runs(self):
        with patch('gevva.scheduler.time.perf_counter', return_value=0):
            expired = self.scheduler.submit('expired', max_queue_ms=1)
        following = self.scheduler.submit('following')
        self.release.set()
        with self.assertRaises(QueueExpired):
            expired.result(5)
        following.result(5)
        self.assertNotIn('expired', self.calls)

    def test_streams_and_fifo_are_independent(self):
        one = self.scheduler.submit('one', stream='game', latest=False)
        two = self.scheduler.submit('two', stream='game', latest=True)
        self.release.set()
        one.result(5)
        two.result(5)
        self.assertEqual(self.calls, ['active', 'one', 'two'])

    def test_close_cancels_pending_and_is_idempotent(self):
        waiting = self.scheduler.submit('waiting')
        closer = threading.Thread(target=self.scheduler.close)
        closer.start()
        with self.assertRaises(CancelledError):
            waiting.result(5)
        self.release.set()
        closer.join(5)
        self.assertFalse(closer.is_alive())
        with self.assertRaises(RuntimeError):
            self.scheduler.submit('after-close')


class EvaluatorTests(unittest.TestCase):
    def test_snapshot_validation_and_request_identity(self):
        class FakeWorker:
            def __init__(self, *args, **kwargs):
                self.requests = []
            def run(self, request):
                self.requests.append(request)
                return {'answers': [{'id': 'flag', 'probabilities': [.2, .8], 'choice_index': 1}],
                    'metrics': {'prefix_tokens': 1, 'suffix_tokens': 2}, 'readout_mode': 'options'}
            def close(self):
                pass
        with patch('gevva.api.Worker', FakeWorker), LocalEvaluator() as evaluator:
            payload = {'state': {'subject': 'before'}, 'request_id': 'email-1', 'questions': {
                'flag': {'type': 'noul', 'instructions': 'Is this important?'}}}
            future = evaluator.submit(payload)
            payload['state']['subject'] = 'after'
            response = future.result(5)
            self.assertEqual(evaluator.worker.requests[0]['state']['subject'], 'before')
            self.assertEqual(response['gevva']['request_id'], 'email-1')
            async_response = asyncio.run(evaluator.evaluate_async(payload))
            self.assertEqual(async_response['gevva']['request_id'], 'email-1')
            payload['scheduling'] = {'policy': 'latest'}
            with self.assertRaises(ValueError):
                evaluator.submit(payload)

if __name__ == '__main__':
    unittest.main()
