"""Small persistent HTTP client. Use one instance per concurrent producer."""
import http.client
import json
from urllib.parse import urlsplit
from .scheduler import QueueFull, Superseded, QueueExpired


class Client:
    def __init__(self, base_url='http://127.0.0.1:8081', *, api_key=None, timeout=120):
        url = urlsplit(base_url)
        if url.scheme not in ('http', 'https') or not url.hostname or url.path.rstrip('/'):
            raise ValueError('base_url must be an http(s) origin without a path')
        connection = http.client.HTTPSConnection if url.scheme == 'https' else http.client.HTTPConnection
        self.connection = connection(url.hostname, url.port, timeout=timeout)
        self.api_key = api_key

    def evaluate(self, payload):
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        body = json.dumps(payload, allow_nan=False).encode()
        try:
            self.connection.request('POST', '/v1/systemone', body=body, headers=headers)
            response = self.connection.getresponse()
            result = json.loads(response.read())
        except Exception:
            self.connection.close()
            raise
        if response.status != 200:
            message = result.get('error', {}).get('message', f'HTTP {response.status}')
            error_type = {409: Superseded, 408: QueueExpired, 529: QueueFull, 422: ValueError}.get(response.status, RuntimeError)
            raise error_type(message)
        return result

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
