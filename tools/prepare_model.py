#!/usr/bin/env python3
"""Download (optionally) and prepare the three model sidecars used by Gevva."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gevva.config import ConfigError, load_config

REPOSITORY = 'nvidia/Gemma-4-26B-A4B-NVFP4'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Gevva TOML configuration')
    parser.add_argument('--gpu', choices=['5090', 'pro6000'])
    parser.add_argument('--download', action='store_true', help='Download the supported NVIDIA checkpoint first')
    parser.add_argument('--force', action='store_true', help='Rebuild existing sidecars')
    parser.add_argument('--dry-run', action='store_true', help='Print paths and commands without downloading or packing')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, overrides={'engine.gpu': args.gpu})
        paths = config.model
        if len({p.resolve() for p in (paths.target, paths.experts, paths.dense, paths.vocab)}) != 4:
            raise ConfigError('model component directories must be distinct')
        common = ['--gpu', config.engine.gpu]
        if config.source:
            common += ['--config', str(config.source)]
        commands = [
            [sys.executable, str(ROOT / 'tools/pack_trtllm_experts.py'),
             '--source', str(paths.target), '--output', str(paths.experts), *common],
            [sys.executable, str(ROOT / 'tools/pack_fp8_dense.py'),
             '--source', str(paths.target), '--output', str(paths.dense), *common],
            [sys.executable, str(ROOT / 'tools/pack_vocab_int8.py'),
             '--model', str(paths.target), '--output', str(paths.vocab), *common],
        ]
        if args.force:
            for command in commands: command.append('--force')
        plan = {'repository': REPOSITORY, 'download': args.download, 'gpu': config.engine.gpu,
                'model': config.as_dict()['model'], 'commands': commands}
        if args.dry_run:
            print(json.dumps(plan, indent=2), flush=True)
            return 0
        print(f"Checking preparation dependencies and GPU {config.engine.gpu}...", flush=True)
        # Check the selected GPU and converter dependencies before a large download.
        from _preparation import select_gpu
        select_gpu(config)
        import safetensors
        from flashinfer import nvfp4_block_scale_interleave
        if args.download:
            from huggingface_hub import snapshot_download
            print(f"Downloading {REPOSITORY} to {paths.target}...", flush=True)
            snapshot_download(repo_id=REPOSITORY, local_dir=str(paths.target))
        source_config = paths.target / 'config.json'
        if not source_config.is_file():
            raise ConfigError(f'Checkpoint missing: {source_config}; use --download or set model.target')
        text = json.loads(source_config.read_text()).get('text_config', {})
        if (text.get('hidden_size'), text.get('vocab_size'), text.get('num_hidden_layers')) != (2816, 262144, 30):
            raise ConfigError('Only the supported Gemma 4 26B-A4B checkpoint can be prepared')
        for label, command in zip(("NVFP4 experts (first run may compile CUDA helpers)", "FP8 dense projections", "INT8 vocabulary"), commands):
            print(f"Preparing {label}...", flush=True)
            subprocess.run(command, env=config.worker_env(), check=True)
        print("Checking prepared model files...", flush=True)
        config.validate_files()
        print(json.dumps({'status': 'prepared', 'model': config.as_dict()['model']}), flush=True)
    except (ConfigError, OSError, ValueError, ImportError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'error: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
