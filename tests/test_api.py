import json
import math
from pathlib import Path
import sys
import threading
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gevva.api import compile_request, format_response, concentration, MODEL
from gevva.server import create_server
from gevva import Client, QueueFull, QueueExpired, Superseded


class ApiTests(unittest.TestCase):
    def test_structured_entries_and_keys(self):
        payload = {'state': {'x': 1}, 'model': 'jev-latest', 'questions': {
            'private_question_id': {'type': 'choice', 'instructions': {'task': 'route'},
                'criteria': {'A key': {'description': ['one', 'two']}, 'other': None}},
            'score': {'type': 'score', 'instructions': None, 'criteria': [None, {'level': 'high'}]},
            'noul': {'type': 'noul', 'instructions': ['Check x'],
                     'criteria': {'true': {'rule': 'yes'}, 'false': ['not yes']}}}}
        engine, questions = compile_request(payload)
        self.assertEqual(engine['state'], payload['state'])
        self.assertEqual(engine['readout_mode'], 'options')
        self.assertNotIn('private_question_id', engine['questions'][0]['question'])
        self.assertEqual(engine['questions'][0]['options'], ['A key: {"description":["one","two"]}', 'other'])
        result = {'answers': [
            {'id': 'private_question_id', 'probabilities': [.8, .2], 'choice_index': 0},
            {'id': 'score', 'probabilities': [.3, .7], 'choice_index': 1},
            {'id': 'noul', 'probabilities': [.1, .9], 'choice_index': 1}],
            'metrics': {'prefix_tokens': 10, 'suffix_tokens': 20}, 'readout_mode': 'options'}
        response = format_response(result, questions)
        self.assertEqual(response['model'], MODEL)
        self.assertEqual(response['answers']['private_question_id']['choice'], 'A key')
        self.assertEqual(response['answers']['score']['legend'], {'0': None, '1': {'level': 'high'}})
        self.assertEqual(response['answers']['score']['score'], .7)
        self.assertEqual(response['answers']['noul'], {'type': 'noul', 'noul': .9})
        self.assertFalse(response['gevva']['probabilities_calibrated'])

    def test_255_options_and_limits(self):
        request = {'state': '', 'questions': {'q': {'type': 'choice', 'instructions': 'Select.',
                   'criteria': {str(i): None for i in range(255)}}}}
        self.assertEqual(len(compile_request(request)[0]['questions'][0]['options']), 255)
        request['questions']['q']['criteria']['256'] = None
        with self.assertRaises(ValueError):
            compile_request(request)
        request['questions']['q'] = {'type': 'score', 'instructions': 'Rate.', 'criteria': ['x']}
        with self.assertRaises(ValueError):
            compile_request(request)

    def test_sdk_optional_instructions_and_legacy_model(self):
        request = {'state': 'A refund was requested.', 'model': 'jev-1.12',
                   'questions': {'route': {'type': 'choice', 'criteria': {'billing': None, 'sales': None}}}}
        native, _ = compile_request(request)
        self.assertTrue(native['questions'][0]['question'])

    def test_confidence_definition(self):
        self.assertEqual(concentration([1]), 1)
        self.assertEqual(concentration([1, 0]), 1)
        self.assertAlmostEqual(concentration([.5, .5]), 0)
        self.assertAlmostEqual(concentration([.75, .25]), 1 + (.75 * math.log(.75) + .25 * math.log(.25)) / math.log(2))

    def test_http_schema_and_errors(self):
        class FakeEvaluator:
            def evaluate(self, payload):
                request, questions = compile_request(payload)
                answers = [{'id': q['id'], 'probabilities': [.25, .75], 'choice_index': 1} for q in request['questions']]
                return format_response({'answers': answers, 'readout_mode': 'options',
                    'metrics': {'prefix_tokens': 4, 'suffix_tokens': 6}}, questions)
        server = create_server(FakeEvaluator(), port=0, api_key='test-local')
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(base + '/v1/models')
            self.assertEqual(caught.exception.code, 401)
            caught.exception.close()
            payload = {'state': 'hello', 'questions': {'q': {'type': 'noul', 'instructions': 'Is this a greeting?'}}}
            request = urllib.request.Request(base + '/v1/systemone', data=json.dumps(payload).encode(),
                headers={'Authorization': 'Bearer test-local', 'Content-Type': 'application/json'})
            response = json.load(urllib.request.urlopen(request))
            self.assertEqual(response['answers']['q'], {'type': 'noul', 'noul': .75})
            request.data = b'{"questions":{}}'
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request)
            self.assertEqual(caught.exception.code, 422)
            caught.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

class PersistentClientTests(unittest.TestCase):
    def test_reuse_errors_and_recovery(self):
        class FakeEvaluator:
            healthy = True
            def evaluate(self, payload):
                errors = {'full': QueueFull, 'expired': QueueExpired, 'old': Superseded}
                if payload.get('case') in errors:
                    raise errors[payload['case']]('test error')
                return {'answers': {'flag': {'type': 'noul', 'noul': .8}}}
        evaluator = FakeEvaluator()
        server = create_server(evaluator, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with Client(f'http://127.0.0.1:{server.server_port}') as client:
                self.assertEqual(client.evaluate({})['answers']['flag']['noul'], .8)
                connection = client.connection.sock
                for kind, error in [('full', QueueFull), ('expired', QueueExpired), ('old', Superseded)]:
                    with self.assertRaises(error):
                        client.evaluate({'case': kind})
                self.assertEqual(client.evaluate({})['answers']['flag']['noul'], .8)
                self.assertIs(client.connection.sock, connection)
            evaluator.healthy = False
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/health')
            self.assertEqual(caught.exception.code, 503)
            caught.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

if __name__ == '__main__':
    unittest.main()
