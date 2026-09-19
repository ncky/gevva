"""Real-GPU tests for Jev-schema values, image reuse, and image replacement."""
import base64
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import threading
import urllib.request
import urllib.error
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gevva import LocalEvaluator
from gevva.server import create_server

ROOT = Path(__file__).resolve().parents[1]
base = json.loads((ROOT / 'examples/systemone-image.json').read_text())
base['images'] = [str(ROOT / 'examples/shapes.png')]
base['parameters'] = {'verify_prompt': True}
records = []
with LocalEvaluator() as evaluator:
    text = evaluator.evaluate(json.loads((ROOT / 'examples/systemone.json').read_text()))
    assert text['answers']['department']['choice'] == 'billing', text
    assert text['answers']['refund']['noul'] > .9, text
    with tempfile.TemporaryDirectory() as directory:
        changing = Path(directory) / 'same-path.png'
        changing.write_bytes((ROOT / 'examples/shapes.png').read_bytes())
        requests = [base, base]
        data = copy.deepcopy(base)
        data['images'] = ['data:image/png;base64,' + base64.b64encode(changing.read_bytes()).decode()]
        requests.append(data)
        changed = copy.deepcopy(base)
        changed['images'] = [str(changing)]
        for index, request in enumerate(requests):
            start = time.perf_counter()
            response = evaluator.evaluate(request)
            records.append({'case': index, 'round_trip_ms': (time.perf_counter() - start) * 1000, 'response': response})
            answers = response['answers']
            assert answers['square_colour']['choice'] == 'red', response
            assert answers['circle_colour']['choice'] == 'blue', response
            assert answers['square_left']['noul'] > .9, response
            assert abs(answers['shape_count']['score'] - 2) < .1, response
            assert response['gevva']['image_metrics']['images_encoded'] == (1 if index == 0 else 0), response
        changing.write_bytes((ROOT / 'examples/shapes-swapped.png').read_bytes())
        response = evaluator.evaluate(changed)
        records.append({'case': 'replace-bytes', 'response': response})
        assert response['answers']['square_colour']['choice'] == 'blue', response
        assert response['answers']['circle_colour']['choice'] == 'red', response
        assert not response['gevva']['metrics']['prefix_cache_hit'], response
        assert response['gevva']['image_metrics']['images_encoded'] == 1, response
        multi = {'state': 'Compare the numbered images.', 'images': [base['images'][0], str(changing)],
            'questions': {'first': {'type': 'choice', 'instructions': 'What colour is the square in Image 1?', 'criteria': {'red': None, 'blue': None}},
                          'second': {'type': 'choice', 'instructions': 'What colour is the square in Image 2?', 'criteria': {'red': None, 'blue': None}}}}
        response = evaluator.evaluate(multi)
        records.append({'case': 'two-images', 'response': response})
        assert response['answers']['first']['choice'] == 'red', response
        assert response['answers']['second']['choice'] == 'blue', response
        assert response['gevva']['image_metrics']['images_encoded'] == 2, response
        repeated = evaluator.evaluate(multi)
        assert repeated['gevva']['image_metrics']['images_encoded'] == 0
        assert repeated['gevva']['metrics']['prefix_cache_hit']
    # Exercise singleton and high-cardinality Choice through the public schema.
    request = {'state': 'The selected code is OPT137.', 'questions': {
        'code': {'type': 'choice', 'instructions': 'Which exact code is selected?',
                 'criteria': {f'OPT{i:03}': None for i in range(255)}},
        'only': {'type': 'choice', 'instructions': 'Select the only option.', 'criteria': {'one': None}}}}
    response = evaluator.evaluate(request)
    assert response['answers']['code']['choice'] == 'OPT137', response
    assert response['answers']['only']['probabilities'] == {'one': 1.0}, response
    records.append({'case': '255-options', 'response': response})
    server = create_server(evaluator, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f'http://127.0.0.1:{server.server_port}/v1/systemone'
        payload = json.loads((ROOT / 'examples/systemone.json').read_text())
        call = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(call) as http:
            response = json.load(http)
        assert response['answers']['department']['choice'] == 'billing', response
        payload['parameters'] = {'prefix_mode': 'invalid'}
        call.data = json.dumps(payload).encode()
        try:
            urllib.request.urlopen(call)
            raise AssertionError('invalid native parameter accepted')
        except urllib.error.HTTPError as error:
            assert error.code == 422, error
            error.close()
        records.append({'case': 'real-http', 'response': response})
        payload['parameters'] = {'reset_prefix': True, 'reset_images': True}
        payload['images'] = ['data:image/jpeg;base64,bm90LWEtanBlZw==']
        call.data = json.dumps(payload).encode()
        try:
            urllib.request.urlopen(call)
            raise AssertionError('malformed image accepted')
        except urllib.error.HTTPError as error:
            assert error.code == 422, error
            error.close()
        del payload['images']
        call.data = json.dumps(payload).encode()
        with urllib.request.urlopen(call) as http:
            recovered = json.load(http)
        assert recovered['answers']['department']['choice'] == 'billing', recovered
        records.append({'case': 'image-error-recovery', 'response': recovered})
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
path = ROOT / 'runs/multimodal-api-validation.json'
path.write_text(json.dumps({'passed': True, 'cases': records}, indent=2) + '\n')
print(json.dumps({'passed': True, 'cases': len(records), 'results': str(path)}))
