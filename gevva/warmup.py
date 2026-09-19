"""Prepare common continuous-request shapes before accepting traffic."""
import base64
from importlib.resources import files
import time


def warmup(evaluator, images=False, common_shapes=True):
    started = time.perf_counter()
    fixture = 'data:image/png;base64,' + base64.b64encode(
        files('gevva').joinpath('assets/shapes.png').read_bytes()).decode('ascii')
    questions = (
        'Is a red square present?', 'Is a blue circle present?', 'Are two shapes described?',
        'Is the square red?', 'Is the circle blue?', 'Is a triangle present?',
        'Is a green shape present?', 'Are any shapes visible?', 'Is the square blue?',
        'Is the circle red?')
    calls = 0
    last = None

    def run(state, count, budget=None):
        nonlocal calls, last
        request = {'state': state, 'questions': {
            str(i): {'type': 'noul', 'instructions': question}
            for i, question in enumerate(questions[:count])},
            'parameters': {'reset_prefix': True, 'reset_images': True}}
        if budget is not None:
            request['images'] = [fixture]
            request['parameters']['image_soft_tokens'] = budget
        last = evaluator.evaluate(request)
        calls += 1

    # Prepare padded text-prefix plans and projection sizes, independent of real
    # observations. Longer contexts and unusual batch sizes remain lazy.
    if common_shapes:
        for words in range(0, 449, 16):
            run('Warmup scene.' + ' object' * words, (3, 5, 10)[(words // 16) % 3])
    for budget in ([None, 140, 280] if images else [None]):
        for count in (3, 5, 10):
            run('A red square is beside a blue circle.', count, budget)
    return {'seconds': time.perf_counter() - started, 'requests': calls,
            'common_shapes': common_shapes, 'images': images,
            'device_used_bytes': last['gevva']['metrics']['device_used_bytes']}
