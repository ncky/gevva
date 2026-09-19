"""GPU selection integration checks on the two-card workstation (no model load)."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'build/gevva-engine'

def run(target=None, visible=None):
    env = dict(os.environ)
    env.pop('GEVVA_GPU', None)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    if target is not None:
        env['GEVVA_GPU'] = target
    if visible is not None:
        env['CUDA_VISIBLE_DEVICES'] = visible
    return subprocess.run([ENGINE, 'gpu-info'], env=env, capture_output=True, text=True)

for target, name in [(None, 'RTX PRO 6000'), ('pro6000', 'RTX PRO 6000'), ('5090', 'GeForce RTX 5090')]:
    result = run(target)
    assert result.returncode == 0 and name in result.stdout, result
result = run('auto')
assert result.returncode != 0 and 'GEVVA_GPU must be' in result.stderr, result
# Restricting visibility must not silently select the other card, even though
# both support the architecture. Use the selected UUID instead of an ordinal.
pro = run('pro6000')
uuid = next(line.split('=', 1)[1] for line in pro.stdout.splitlines() if line.startswith('gpu.uuid='))
if not uuid.startswith('GPU-'):
    uuid = 'GPU-' + uuid
result = run('5090', uuid)
assert result.returncode != 0 and 'not visible' in result.stderr, result
result = run('pro6000', uuid)
assert result.returncode == 0 and 'RTX PRO 6000' in result.stdout, result
print(json.dumps({'passed': True, 'cases': 6}))
