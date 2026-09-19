"""Integration parity, scoring, state replacement and branch isolation on the selected GPU."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gevva.config import load_config
CONFIG = load_config()
base = json.loads((ROOT / 'examples/parcel.json').read_text())
base['state'] += '\n' + 'Archive metadata: this line is unrelated to parcel contents and payment.\n' * 180
base['verify_readout'] = True
requests = []
for mode in ('none', 'copy', 'copy', 'shared', 'shared'):
    requests.append(dict(base, prefix_mode=mode))
requests.append(dict(base, prefix_mode='shared', questions=list(reversed(base['questions']))))
requests.append(dict(base, prefix_mode='shared', batch_size=1))
changed = copy.deepcopy(base)
changed['state'] = changed['state'].replace('blue parcel', 'red parcel')
requests.extend([dict(changed, prefix_mode='shared'), dict(changed, prefix_mode='shared')])
requests.append(dict(base, prefix_mode='shared'))
requests.append(dict(base, prefix_mode='copy'))
requests.append(dict(base, prefix_mode='none', batch_size=1))
requests.append(dict(base, prefix_mode='shared', temperature=0))
requests.append(dict(base, prefix_mode='shared'))
# Crossing the chunk boundary exercises sliding-ring restoration and global growth.
long = copy.deepcopy(base)
long['state'] += '\nUnrelated archive entry.' * 900
requests.extend([dict(long, prefix_mode='none'), dict(long, prefix_mode='shared')])
requests.append(dict(base, prefix_mode='shared', questions=[base['questions'][0]]))
ambiguous = copy.deepcopy(base)
ambiguous['questions'] = [
    {'id': 'delivery', 'question': 'Which delivery speed is best for this customer?', 'options': ['Standard', 'Express', 'Pickup']},
    {'id': 'fragility', 'type': 'boolean', 'question': 'Is this parcel fragile?'},
    {'id': 'customer', 'question': 'What is the customer most likely buying these books for?', 'options': ['Personal reading', 'A gift', 'Schoolwork']},
    {'id': 'value', 'type': 'score', 'question': 'How expensive is this order?', 'options': ['Inexpensive', 'Moderate', 'Expensive']},
]
requests.extend([dict(ambiguous, prefix_mode='copy'), dict(ambiguous, prefix_mode='shared')])
requests.extend([dict(base, prefix_mode='readonly', verify_prompt=True),
                 dict(base, prefix_mode='readonly'),
                 dict(ambiguous, prefix_mode='readonly'),
                 dict(long, prefix_mode='readonly')])
requests.extend([dict(base, prefix_mode='readonly', verify_readout=False),
                 dict(ambiguous, prefix_mode='readonly', verify_readout=False)])
requests.extend([dict(base, prefix_mode='readonly', verify_readout=False, readout_mode='options'),
                 dict(ambiguous, prefix_mode='readonly', verify_readout=False, readout_mode='options')])
requests.extend([dict(base, prefix_mode='readonly', reset_prefix=True),
                 dict(changed, prefix_mode='readonly'),
                 dict(base, prefix_mode='readonly')])
proc = subprocess.run([CONFIG.engine.binary, 'decision-serve'], env=CONFIG.worker_env(),
    input=''.join(json.dumps(r) + '\n' for r in requests), text=True,
    capture_output=True, timeout=600)
(ROOT / 'runs').mkdir(exist_ok=True)
(ROOT / 'runs/validation.stderr').write_text(proc.stderr)
(ROOT / 'runs/validation.jsonl').write_text(proc.stdout)
if proc.returncode:
    raise RuntimeError(proc.stderr[-4000:])
outputs = [json.loads(line) for line in proc.stdout.splitlines()]
assert len(outputs) == len(requests), (len(outputs), len(requests), proc.stdout[:1000])
assert 'error' in outputs[12], outputs[12]
for i, out in enumerate(outputs):
    if i == 12:
        continue
    assert 'error' not in out, (i, out)
    if requests[i].get('verify_readout'):
        assert out['readout_reference_max_error'] < 0.05, (i, out['readout_reference_max_error'])
    assert out['probabilities_calibrated'] is False
    for a in out['answers']:
        assert abs(sum(a['probabilities']) - 1) < 1e-10
        if requests[i].get('readout_mode') == 'options':
            assert a['allowed_mass'] is None and a['full_vocab_logprobs'] is None
        else:
            assert 0 <= a['allowed_mass'] <= 1 + 1e-10
            assert abs(a['allowed_mass'] - sum(map(math.exp, a['full_vocab_logprobs']))) < 1e-5
        assert len(a['probabilities']) == len(a['options'])
        if a['type'] == 'boolean':
            assert a['p_true'] == a['probabilities'][1]
        if a['type'] == 'score':
            assert abs(a['score'] - sum(p * x for p, x in zip(a['probabilities'], a['levels']))) < 1e-10

def compare(first, second):
    a = {r['id']: r for r in outputs[first]['answers']}
    b = {r['id']: r for r in outputs[second]['answers']}
    difference = max(abs(x - y) for key in a for x, y in zip(a[key]['probabilities'], b[key]['probabilities']))
    assert difference < 0.02, (first, second, difference)
    assert all(a[key]['choice_index'] == b[key]['choice_index'] for key in a)
    return difference

comparisons = [(0, i) for i in (1, 2, 3, 4, 5, 6, 9, 10, 11, 13)] + [(7, 8), (14, 15), (0, 19), (14, 22)]
differences = {f'{a}:{b}': compare(a, b) for a, b in comparisons}
assert outputs[2]['metrics']['prefix_cache_hit']
assert outputs[4]['metrics']['prefix_cache_hit']
assert not outputs[7]['metrics']['prefix_cache_hit']
assert outputs[8]['metrics']['prefix_cache_hit']
assert outputs[3]['metrics']['shared_global_bytes'] > 0
assert outputs[7]['answers'][0]['choice_index'] == 0
assert outputs[9]['answers'][0]['choice_index'] == 1
assert outputs[16]['answers'][0]['choice_index'] == 1
# Physical sharing must preserve logits, not merely saturated argmax answers.
for a, b in [(1, 3), (17, 18), (19, 23), (21, 24)]:
    error = max(abs(x - y) for qa, qb in zip(outputs[a]['answers'], outputs[b]['answers'])
                for x, y in zip(qa['option_logits'], qb['option_logits']))
    assert error < 1e-4, (a, b, error)

# Restricted projection must retain the full head's active-label logits.
head_errors = []
for a, b in [(23, 25), (24, 26)]:
    error = max(abs(x - y) for qa, qb in zip(outputs[a]['answers'], outputs[b]['answers'])
                for x, y in zip(qa['option_logits'], qb['option_logits']))
    assert error < 0.005, (a, b, error)
    head_errors.append(error)

# Fresh-state allocation reuse must not retain answers or prior-state KV content.
for index in (27, 28, 29):
    assert not outputs[index]['metrics']['prefix_cache_hit']
    assert outputs[index]['metrics']['prefix_storage_reused']
assert outputs[28]['answers'][0]['choice_index'] == 0
for index in (27, 29):
    error = max(abs(x-y) for a,b in zip(outputs[23]['answers'],outputs[index]['answers'])
                for x,y in zip(a['option_logits'],b['option_logits']))
    assert error < 1e-4, (index,error)

summary = {'max_restricted_head_logit_error': max(head_errors), 'passed': True, 'cases': len(requests), 'max_probability_difference': max(differences.values()),
           'max_cpu_readout_error': max(x.get('readout_reference_max_error') or 0 for x in outputs),
           'metrics': [x.get('metrics', x) for x in outputs]}
(ROOT / 'runs/validation-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
