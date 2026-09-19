"""TypeSafe's public Choice/Score/Noul schema, backed by the local decision worker.

Schema compatibility does not imply Jev model, calibration, confidence-formula,
latency or token-accounting equivalence. Sources: docs.typesafe.ai/api and
/primitives/advanced, inspected 2026-09-18.
"""
import json
import math
import asyncio
import copy
from .worker import Worker
from .config import load_config
from .scheduler import Scheduler

MODEL = 'gevva-gemma4-26b-a4b'
MODEL_ALIASES = {MODEL, 'gevva-latest', 'jev-latest', 'jev-preview', 'jev-1.13.0', 'jev-1.12'}
PARAMETERS = {'batch_size', 'prefix_mode', 'temperature', 'readout_mode', 'reset_prefix',
              'reset_images', 'image_soft_tokens', 'compact_prompt', 'verify_readout', 'verify_prompt'}


def entry(value, field):
    if value is not None and not isinstance(value, (str, dict, list)):
        raise ValueError(f'{field} must be string, object, array or null')
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def compile_request(payload):
    if not isinstance(payload, dict):
        raise ValueError('request must be a JSON object')
    if 'state' not in payload or not isinstance(payload['state'], (str, dict, list)):
        raise ValueError('state must be a string, object or array')
    model = payload.get('model', 'gevva-latest')
    if not isinstance(model, str) or model not in MODEL_ALIASES:
        raise ValueError(f'unsupported local model: {model!r}')
    questions = payload.get('questions')
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 256:
        raise ValueError('questions must be a map containing 1..256 entries')
    parameters = payload.get('parameters', {})
    if not isinstance(parameters, dict) or parameters.keys() - PARAMETERS:
        raise ValueError('unsupported local parameters')
    for name in ('reset_prefix', 'reset_images', 'compact_prompt', 'verify_readout', 'verify_prompt'):
        if name in parameters and not isinstance(parameters[name], bool):
            raise ValueError(f'{name} must be boolean')
    if 'temperature' in parameters and (isinstance(parameters['temperature'], bool) or
            not isinstance(parameters['temperature'], (int, float)) or
            not math.isfinite(parameters['temperature']) or parameters['temperature'] <= 0):
        raise ValueError('temperature must be finite and positive')
    if 'batch_size' in parameters and (type(parameters['batch_size']) is not int or not 1 <= parameters['batch_size'] <= 256):
        raise ValueError('batch_size must be an integer in 1..256')
    engine = dict(parameters, state=payload['state'], questions=[])
    engine.setdefault('readout_mode', 'options')
    engine.setdefault('prefix_mode', 'readonly')
    if 'images' in payload:
        images = payload['images']
        if not isinstance(images, list) or len(images) > 8 or not all(isinstance(x, str) and x for x in images):
            raise ValueError('images must contain at most 8 nonempty local paths or inline image data URLs')
        engine['images'] = images
    for identifier, question in questions.items():
        if not isinstance(identifier, str) or not identifier or not isinstance(question, dict):
            raise ValueError('each question needs a nonempty string id and an object value')
        kind = question.get('type')
        if kind not in ('choice', 'score', 'noul'):
            raise ValueError(f'{identifier}: type must be choice, score, or noul')
        raw_instructions = question.get('instructions')
        instructions = entry(raw_instructions, f'{identifier}.instructions')
        if raw_instructions is None or not instructions:
            instructions = 'Evaluate the provided state using the listed criteria.'
        criteria = question.get('criteria')
        if kind == 'choice':
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                raise ValueError(f'{identifier}: Choice criteria must contain 1..255 named options')
            options = []
            for name, description in criteria.items():
                if not isinstance(name, str) or not name:
                    raise ValueError(f'{identifier}: option names must be nonempty strings')
                rendered = entry(description, f'{identifier}.criteria.{name}')
                options.append(name if description is None else name + ': ' + rendered)
        elif kind == 'score':
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError(f'{identifier}: Score criteria must contain 2..10 ordered levels')
            options = [entry(value, f'{identifier}.criteria') or 'Unspecified level' for value in criteria]
        else:
            if criteria is not None and (not isinstance(criteria, dict) or criteria.keys() - {'true', 'false'}):
                raise ValueError(f'{identifier}: Noul criteria only accepts true and false descriptions')
            criteria = criteria or {}
            options = []
            for key in ('false', 'true'):
                description = criteria.get(key)
                rendered = entry(description, f'{identifier}.criteria.{key}')
                options.append(key.capitalize() if description is None else key.capitalize() + ': ' + rendered)
        # Named ids are retained for response routing only, never in the prompt.
        engine['questions'].append({'id': identifier, 'question': instructions,
                                    'type': 'score' if kind == 'score' else 'choice', 'options': options})
    return engine, questions


def concentration(probabilities):
    """Explicit local confidence statistic; TypeSafe's exact formula is unpublished."""
    if len(probabilities) == 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return min(1.0, max(0.0, 1.0 - entropy / math.log(len(probabilities))))


def format_response(result, questions):
    answers = {}
    for answer in result['answers']:
        identifier = answer['id']
        question = questions[identifier]
        kind = question['type']
        p = answer['probabilities']
        if kind == 'noul':
            answers[identifier] = {'type': kind, 'noul': p[1]}
            continue
        keys = list(question['criteria']) if kind == 'choice' else [str(i) for i in range(len(p))]
        formatted = {'type': kind, 'probabilities': dict(zip(keys, p)), 'confidence': concentration(p)}
        if kind == 'choice':
            formatted['choice'] = keys[answer['choice_index']]
        else:
            formatted.update(score=sum(i * probability for i, probability in enumerate(p)),
                             legend=dict(zip(keys, question['criteria'])))
        answers[identifier] = formatted
    metrics = result['metrics']
    return {'model': MODEL, 'answers': answers,
            'usage': {'input_tokens': metrics['prefix_tokens'] + metrics['suffix_tokens'],
                      'output_tokens': len(answers)},
            'gevva': {'probabilities_calibrated': False, 'confidence_method': '1 - entropy / log(option_count)',
                      'usage_method': 'shared rendered prefix plus suffix tokens; one scored output token per question',
                      'readout_mode': result['readout_mode'], 'metrics': metrics,
                      'image_metrics': result.get('image_metrics', {})}}


class LocalEvaluator:
    """One resident GPU worker; submit from multiple producers without blocking."""
    def __init__(self, engine=None, *, config=None, gpu=None, queue_capacity=None, worker_timeout=None):
        self.config = load_config(config, overrides={'engine.binary': engine, 'engine.gpu': gpu,
            'server.queue_capacity': queue_capacity, 'engine.worker_timeout': worker_timeout})
        self.worker = Worker(config=self.config)
        self.scheduler = Scheduler(self._execute, self.config.server.queue_capacity)

    @property
    def healthy(self):
        return not self.scheduler.closed and self.worker.healthy

    def _execute(self, item):
        request, questions, request_id = item
        response = format_response(self.worker.run(request), questions)
        if request_id is not None:
            response['gevva']['request_id'] = request_id
        return response

    def submit(self, payload):
        # Own the snapshot: callers may immediately reuse their request template.
        payload = copy.deepcopy(payload)
        request, questions = compile_request(payload)
        scheduling = payload.get('scheduling', {})
        if not isinstance(scheduling, dict) or scheduling.keys() - {'stream', 'policy', 'max_queue_ms'}:
            raise ValueError('unsupported scheduling fields')
        stream = scheduling.get('stream')
        policy = scheduling.get('policy', 'fifo')
        if policy not in ('fifo', 'latest'):
            raise ValueError('scheduling.policy must be fifo or latest')
        if stream is not None and (not isinstance(stream, str) or not 1 <= len(stream) <= 128):
            raise ValueError('scheduling.stream must be a nonempty string of at most 128 characters')
        if policy == 'latest' and stream is None:
            raise ValueError('latest scheduling requires a stream')
        max_queue_ms = scheduling.get('max_queue_ms')
        if max_queue_ms is not None and (isinstance(max_queue_ms, bool) or
                not isinstance(max_queue_ms, (int, float)) or not math.isfinite(max_queue_ms) or max_queue_ms <= 0):
            raise ValueError('max_queue_ms must be finite and positive')
        request_id = payload.get('request_id')
        if request_id is not None and (not isinstance(request_id, str) or not 1 <= len(request_id) <= 256):
            raise ValueError('request_id must be a nonempty string of at most 256 characters')
        return self.scheduler.submit((request, questions, request_id), stream=stream,
                                     latest=policy == 'latest', max_queue_ms=max_queue_ms)

    def evaluate(self, payload):
        return self.submit(payload).result()

    async def evaluate_async(self, payload):
        return await asyncio.wrap_future(self.submit(payload))

    def close(self):
        self.scheduler.close(wait=False)
        self.worker.close()  # Interrupt active transport before joining its owner.
        self.scheduler.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
