"""Connection admission and authentication checks against real TCP sockets."""
import http.client
import socket
import threading
import time
import unittest

from gevva.config import ConfigError
from gevva.server import create_server


class ServerLimitsTests(unittest.TestCase):
    def start_server(self, **kwargs):
        class Evaluator:
            healthy = True
            def evaluate(self, payload):
                return {'answers': {}}
        server = create_server(Evaluator(), port=0, **kwargs)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def stop():
            server.shutdown()
            server.server_close()
            thread.join(2)
        self.addCleanup(stop)
        return server

    def test_idle_connections_are_admitted_before_thread_creation(self):
        server = self.start_server(capacity=1, api_key='secret', connection_timeout=.3)
        started = threading.Event()
        setups = []
        setup = server.RequestHandlerClass.setup
        def observed_setup(handler):
            setup(handler)
            setups.append(handler.connection.gettimeout())
            started.set()
        server.RequestHandlerClass.setup = observed_setup
        first = socket.create_connection(server.server_address, timeout=1)
        self.addCleanup(first.close)
        self.assertTrue(started.wait(1))
        self.assertEqual(setups, [.3])
        for _ in range(4):
            with socket.create_connection(server.server_address, timeout=1) as excess:
                try:
                    self.assertEqual(excess.recv(1), b'')
                except ConnectionResetError:
                    pass
        self.assertEqual(len(setups), 1, 'rejected sockets must not get handler threads')
        # A header fragment cannot keep the admitted connection alive forever.
        first.sendall(b'GET /health HTTP/1.1\r\nHost:')
        self.assertEqual(first.recv(1), b'')
        # The timed-out connection releases its admission slot.
        end = time.monotonic() + 2
        while True:
            connection = http.client.HTTPConnection(*server.server_address, timeout=1)
            try:
                connection.request('GET', '/health', headers={'Authorization': 'Bearer secret'})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                response.read()
                break
            except (ConnectionError, http.client.RemoteDisconnected):
                if time.monotonic() >= end:
                    raise
                time.sleep(.01)
            finally:
                connection.close()

    def test_post_admission_is_separate_from_connection_admission(self):
        entered = threading.Event()
        release = threading.Event()
        class Evaluator:
            healthy = True
            def evaluate(self, payload):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('test request never released')
                return {'answers': {}}
        server = create_server(Evaluator(), port=0, capacity=2, request_capacity=1)
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        errors = []
        def first_request():
            connection = http.client.HTTPConnection(*server.server_address, timeout=3)
            try:
                connection.request('POST', '/v1/systemone', body='{}')
                response = connection.getresponse()
                if response.status != 200:
                    raise AssertionError(response.status)
                response.read()
            except Exception as error:
                errors.append(error)
            finally:
                connection.close()
        first = threading.Thread(target=first_request, daemon=True)
        first.start()
        try:
            self.assertTrue(entered.wait(1))
            connection = http.client.HTTPConnection(*server.server_address, timeout=1)
            try:
                connection.request('POST', '/v1/systemone', body='{}')
                response = connection.getresponse()
                self.assertEqual(response.status, 529)
                response.read()
            finally:
                connection.close()
        finally:
            release.set()
            first.join(3)
            server.shutdown()
            server.server_close()
            serving.join(2)
        self.assertFalse(first.is_alive())
        self.assertEqual(errors, [])

    def test_non_ascii_authorization_get_and_post_return_401(self):
        server = self.start_server(api_key='secret')
        for method, path in [('GET', '/health'), ('POST', '/v1/systemone')]:
            for value in ['Bearer é', 'Bearer ÿ', 'Bearer wrong', '']:
                with self.subTest(method=method, value=value):
                    connection = http.client.HTTPConnection(*server.server_address, timeout=1)
                    try:
                        connection.request(method, path, body='{}' if method == 'POST' else None,
                                           headers={'Authorization': value})
                        response = connection.getresponse()
                        self.assertEqual(response.status, 401)
                        self.assertIn(b'Invalid API key', response.read())
                    finally:
                        connection.close()

    def test_invalid_configured_keys_fail_before_binding(self):
        for key in ['', 'é', 'white space', 'line\nbreak', '\x00', 123]:
            with self.subTest(key=key), self.assertRaises(ConfigError):
                create_server(object(), port=0, api_key=key)


if __name__ == '__main__':
    unittest.main()
