"""Common launch settings for the offline model converters."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gevva.config import load_config


def add_launch_arguments(parser):
    parser.add_argument('--config', help='Gevva TOML configuration')
    parser.add_argument('--gpu', choices=['5090', 'pro6000'], help='Override configured GPU')


def settings(args):
    return load_config(args.config, overrides={'engine.gpu': args.gpu})


def select_gpu(config):
    # Set visibility before importing Torch / initializing CUDA in this converter.
    if config.engine.cuda_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.engine.cuda_visible_devices
    import torch
    expected = {'5090': 'NVIDIA GeForce RTX 5090',
                'pro6000': 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition'}[config.engine.gpu]
    for ordinal in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(ordinal)
        if props.name == expected:
            if (props.major, props.minor) != (12, 0):
                raise RuntimeError(f'Packing requires SM120, got {props.name}')
            torch.cuda.set_device(ordinal)
            return props
    raise RuntimeError(f'{expected} is not visible; refusing to use another GPU')
