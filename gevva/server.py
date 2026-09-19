"""Run the local Jev-schema HTTP endpoint: python -m gevva.server --port 8081."""
import argparse
from concurrent.futures import CancelledError
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sys
import threading
from .api import LocalEvaluator, MODEL, MODEL_ALIASES
from .scheduler import QueueFull, Superseded, QueueExpired
from .config import ConfigError, load_config

MAX_BODY = 64 * 1024 * 1024


def create_server(evaluator, host='127.0.0.1', port=8081, api_key=None, capacity=32):
    admission = threading.BoundedSemaphore(capacity)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        disable_nagle_algorithm = True  # Avoid delayed-ACK stalls between headers and small JSON bodies.

        def reply(self, status, payload, close=False):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            if close:
                self.send_header('Connection', 'close')
                self.close_connection = True
            if status == 529:
                self.send_header('Retry-After', '1')
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            if api_key and not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + api_key):
                self.reply(401, {'error': {'message': 'Invalid API key'}}, close=True)
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path == '/health':
                healthy = getattr(evaluator, 'healthy', True)
                self.reply(200 if healthy else 503, {'status': 'ready' if healthy else 'unavailable', 'model': MODEL})
            elif self.path == '/v1/models':
                self.reply(200, {'models': [{'name': name, 'release_date': '2026-09-18',
                    'description': 'Local Gemma decision backend; Jev names are compatibility aliases.'}
                    for name in sorted(MODEL_ALIASES)]})
            else:
                self.reply(404, {'error': {'message': 'Unknown endpoint'}})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != '/v1/systemone':
                self.reply(404, {'error': {'message': 'Unknown endpoint'}}, close=True)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                length = 0
            if not 0 < length <= MAX_BODY:
                self.reply(422, {'error': {'message': 'Content-Length must be 1 byte..64 MiB'}}, close=True)
                return
            if not admission.acquire(blocking=False):
                self.reply(529, {'error': {'message': 'Local inference queue is full'}}, close=True)
                return
            try:
                self.connection.settimeout(30)
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError('Incomplete request body')
                def invalid_constant(value):
                    raise ValueError('Non-finite JSON number: ' + value)
                payload = json.loads(body, parse_constant=invalid_constant)
                result = evaluator.evaluate(payload)
                self.reply(200, result)
            except QueueFull as error:
                self.reply(529, {'error': {'code': 'queue_full', 'message': str(error)}})
            except Superseded as error:
                self.reply(409, {'error': {'code': 'superseded', 'message': str(error)}})
            except QueueExpired as error:
                self.reply(408, {'error': {'code': 'queue_expired', 'message': str(error)}})
            except CancelledError:
                self.reply(503, {'error': {'code': 'shutdown', 'message': 'Evaluator is shutting down'}})
            except (ValueError, KeyError, TypeError, UnicodeError) as error:
                self.reply(422, {'error': {'message': str(error)}})
            except RuntimeError as error:
                self.reply(500, {'error': {'message': str(error)}})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True
            finally:
                admission.release()

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run the local SystemOne API.')
    parser.add_argument('--config', help='TOML file; otherwise GEVVA_CONFIG or ./gevva.toml')
    parser.add_argument('--host', '--bind', dest='host', help='Bind address')
    parser.add_argument('--port', type=int)
    parser.add_argument('--engine', help='Native gevva-engine executable')
    parser.add_argument('--gpu', choices=['pro6000', '5090'])
    parser.add_argument('--model-root', help='Directory containing the four model components')
    parser.add_argument('--worker-timeout', type=float)
    parser.add_argument('--queue-capacity', type=int)
    parser.add_argument('--max-connections', type=int)
    parser.add_argument('--warmup-images', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--common-shapes', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--minimal-warmup', dest='common_shapes', action='store_false', default=None,
                        help='Alias for --no-common-shapes')
    parser.add_argument('--print-config', action='store_true', help='Print resolved settings and exit')
    parser.add_argument('--check-config', action='store_true', help='Check settings and files without loading the GPU')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, overrides={
            'engine.binary': args.engine, 'engine.gpu': args.gpu,
            'engine.worker_timeout': args.worker_timeout, 'model.root': args.model_root,
            'server.host': args.host, 'server.port': args.port,
            'server.queue_capacity': args.queue_capacity, 'server.max_connections': args.max_connections,
            'warmup.images': args.warmup_images, 'warmup.common_shapes': args.common_shapes})
        if args.print_config:
            print(json.dumps(config.as_dict(), indent=2))
            if not args.check_config:
                return 0
        config.validate_files()
        if args.check_config:
            print(json.dumps({'status': 'config_valid', 'config': config.as_dict()}))
            return 0
        print(f"Loading model on {config.engine.gpu} and warming up; waiting for the ready URL...",
              file=sys.stderr, flush=True)
        with LocalEvaluator(config=config) as evaluator:
            from .warmup import warmup
            preparation = warmup(evaluator, images=config.warmup.images, common_shapes=config.warmup.common_shapes)
            server = create_server(evaluator, config.server.host, config.server.port,
                                   os.environ.get(config.server.api_key_env), capacity=config.server.max_connections)
            print(json.dumps({'status': 'ready', 'gpu': config.engine.gpu, 'warmup': preparation,
                              'url': f'http://{config.server.host}:{server.server_port}/v1/systemone'}), flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
    except (ConfigError, OSError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
