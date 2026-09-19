"""Validated launch settings. Paths in TOML are relative to that TOML file."""
from dataclasses import asdict, dataclass, replace
import math
import json
import os
from pathlib import Path
import shutil
import tomllib


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class EngineConfig:
    binary: str
    gpu: str = 'pro6000'
    worker_timeout: float = 120.0
    startup_timeout: float = 300.0
    cuda_visible_devices: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    root: Path
    target: Path
    experts: Path
    dense: Path
    vocab: Path
    _explicit: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ServerConfig:
    host: str = '127.0.0.1'
    port: int = 8081
    queue_capacity: int = 8
    max_connections: int = 32
    api_key_env: str = 'GEVVA_API_KEY'


@dataclass(frozen=True)
class WarmupConfig:
    images: bool = False
    common_shapes: bool = True


@dataclass(frozen=True)
class Config:
    engine: EngineConfig
    model: ModelConfig
    server: ServerConfig
    warmup: WarmupConfig
    source: Path | None = None

    def worker_env(self, environ=None):
        env = dict(os.environ if environ is None else environ)
        env['GEVVA_GPU'] = self.engine.gpu
        env['GEVVA_MODEL_ROOT'] = str(self.model.root)
        for key in MODEL_NAMES:
            env['GEVVA_MODEL_' + key.upper()] = str(getattr(self.model, key))
        if self.engine.cuda_visible_devices is not None:
            env['CUDA_VISIBLE_DEVICES'] = self.engine.cuda_visible_devices
        return env

    def as_dict(self):
        result = asdict(self)
        result['source'] = str(self.source) if self.source else None
        result['model'] = {key: str(value) for key, value in result['model'].items() if not key.startswith('_')}
        result['schema_version'] = 1
        return result

    def validate_files(self):
        binary = Path(self.engine.binary)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ConfigError(f'engine.binary is not executable: {binary}; build gevva-engine or configure its path')
        required = [self.model.target / 'config.json', self.model.target / 'tokenizer.json',
                    self.model.vocab / 'model.safetensors']
        required += [self.model.experts / f'experts-{i:02d}.safetensors' for i in range(30)]
        required += [self.model.dense / f'dense-{i:02d}.safetensors' for i in range(30)]
        index = self.model.target / 'model.safetensors.index.json'
        if index.is_file():
            try:
                mapping = json.loads(index.read_text())['weight_map']
                if not isinstance(mapping, dict) or not mapping or not all(isinstance(x, str) for x in mapping.values()):
                    raise ValueError('weight_map must contain shard filenames')
                required += [self.model.target / name for name in set(mapping.values())]
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ConfigError(f'Invalid model index {index}: {error}') from error
        missing = [str(path) for path in required if not path.is_file()]
        if not any(self.model.target.glob('*.safetensors')):
            missing.append(str(self.model.target / '*.safetensors'))
        if missing:
            preview = ', '.join(missing[:4])
            if len(missing) > 4:
                preview += f' (+{len(missing) - 4} more)'
            raise ConfigError('Missing model files: ' + preview + '; set [model] paths in your TOML config')


MODEL_NAMES = {
    'target': 'gemma4-26b-a4b-nvfp4',
    'experts': 'gemma4-26b-a4b-trtllm',
    'dense': 'gemma4-26b-a4b-dense-fp8',
    'vocab': 'gemma4-26b-a4b-target-vocab-int8',
}
FIELDS = {
    'engine': {'binary', 'gpu', 'worker_timeout', 'startup_timeout', 'cuda_visible_devices'},
    'model': {'root', *MODEL_NAMES},
    'server': {'host', 'port', 'queue_capacity', 'max_connections', 'api_key_env'},
    'warmup': {'images', 'common_shapes'},
}
ENV_FIELDS = {
    'GEVVA_ENGINE': ('engine', 'binary'), 'GEVVA_GPU': ('engine', 'gpu'),
    'GEVVA_MODEL_ROOT': ('model', 'root'),
    **{'GEVVA_MODEL_' + key.upper(): ('model', key) for key in MODEL_NAMES},
}


def _string(value, name):
    if not isinstance(value, str) or not value.strip() or '\0' in value:
        raise ConfigError(f'{name} must be a nonempty string without NUL characters')
    return value


def _path(value, base, name):
    path = Path(_string(str(value) if isinstance(value, Path) else value, name)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _binary(value, base):
    text = _string(str(value) if isinstance(value, Path) else value, 'engine.binary')
    if '/' not in text and not text.startswith('~'):
        located = shutil.which(text)
        return located or str((base / text).resolve())
    return str(_path(text, base, 'engine.binary'))


def _default_binary():
    local = Path(__file__).resolve().parents[1] / 'build/gevva-engine'
    return str(local) if local.is_file() else 'gevva-engine'


def _validate(settings):
    if settings.engine.gpu not in ('pro6000', '5090'):
        raise ConfigError('engine.gpu must be "pro6000" or "5090"')
    for name in ('worker_timeout', 'startup_timeout'):
        timeout = getattr(settings.engine, name)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ConfigError(f'engine.{name} must be finite and positive')
    _string(settings.server.host, 'server.host')
    key_env = _string(settings.server.api_key_env, 'server.api_key_env')
    if '=' in key_env:
        raise ConfigError('server.api_key_env must be an environment variable name')
    for name, minimum, maximum in [('port', 0, 65535), ('queue_capacity', 1, None), ('max_connections', 1, None)]:
        value = getattr(settings.server, name)
        if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
            raise ConfigError(f'server.{name} must be an integer in {minimum}..{maximum or "unbounded"}')
    for name in ('images', 'common_shapes'):
        if type(getattr(settings.warmup, name)) is not bool:
            raise ConfigError(f'warmup.{name} must be boolean')
    visible = settings.engine.cuda_visible_devices
    if visible is not None and (not isinstance(visible, str) or '\0' in visible):
        raise ConfigError('engine.cuda_visible_devices must be a string without NUL characters')
    return settings


def load_config(path=None, *, environ=None, overrides=None):
    """Precedence: explicit overrides > environment > TOML > defaults.

    Config objects are already resolved snapshots: no environment re-reading.
    Override keys use dotted names, e.g. engine.gpu and server.port.
    """
    env = os.environ if environ is None else environ
    cwd = Path.cwd()
    if isinstance(path, Config):
        settings = path
    else:
        source = path if path is not None else env.get('GEVVA_CONFIG')
        if source is None and (cwd / 'gevva.toml').is_file():
            source = cwd / 'gevva.toml'
        data = {}
        if source is not None:
            source = _path(source, cwd, 'config path')
            try:
                with source.open('rb') as file:
                    data = tomllib.load(file)
            except (OSError, tomllib.TOMLDecodeError) as error:
                raise ConfigError(f'Cannot read config {source}: {error}') from error
        base = source.parent if source else cwd
        unknown = data.keys() - {'schema_version', *FIELDS}
        if unknown:
            raise ConfigError('Unknown configuration keys: ' + ', '.join(sorted(unknown)))
        version = data.get('schema_version', 1)
        if type(version) is not int or version != 1:
            raise ConfigError('schema_version must be 1')
        sections = {}
        for section, allowed in FIELDS.items():
            values = data.get(section, {})
            if not isinstance(values, dict):
                raise ConfigError(f'{section} must be a TOML table')
            unknown = values.keys() - allowed
            if unknown:
                raise ConfigError('Unknown configuration keys: ' + ', '.join(section + '.' + k for k in sorted(unknown)))
            sections[section] = dict(values)
        # Resolve TOML paths before environment/CLI overrides, whose paths are
        # relative to the caller's working directory instead of the TOML file.
        for name in ('root', *MODEL_NAMES):
            if name in sections['model']:
                sections['model'][name] = _path(sections['model'][name], base, 'model.' + name)
        if 'binary' in sections['engine']:
            sections['engine']['binary'] = _binary(sections['engine']['binary'], base)
        for variable, (section, name) in ENV_FIELDS.items():
            if variable in env:
                value = env[variable]
                if section == 'model':
                    value = _path(value, cwd, 'environment ' + variable)
                elif name == 'binary':
                    value = _binary(value, cwd)
                sections[section][name] = value
        if 'CUDA_VISIBLE_DEVICES' in env:
            # An empty CUDA_VISIBLE_DEVICES intentionally hides all GPUs.
            sections['engine']['cuda_visible_devices'] = env['CUDA_VISIBLE_DEVICES']
        model = sections['model']
        root = model.get('root', base / 'models')
        settings = Config(
            EngineConfig(**{'binary': _binary(_default_binary(), cwd), **sections['engine']}),
            ModelConfig(root=root, _explicit=frozenset(model.keys() - {'root'}),
                        **{key: model.get(key, root / leaf) for key, leaf in MODEL_NAMES.items()}),
            ServerConfig(**sections['server']), WarmupConfig(**sections['warmup']), source)
    # Explicit root overrides also move derived component paths. Explicit
    # component paths stay fixed, including when they were set in the TOML.
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        if '.' not in dotted:
            raise ConfigError(f'Unknown override: {dotted}')
        section, name = dotted.split('.', 1)
        if section not in FIELDS or name not in FIELDS[section]:
            raise ConfigError(f'Unknown override: {dotted}')
        if section == 'model':
            value = _path(value, cwd, dotted)
            if name == 'root':
                old = settings.model
                components = {key: value / leaf for key, leaf in MODEL_NAMES.items()
                              if key not in old._explicit}
                settings = replace(settings, model=replace(old, root=value, **components))
                continue
            settings = replace(settings, model=replace(settings.model,
                _explicit=settings.model._explicit | {name}))
        elif dotted == 'engine.binary':
            value = _binary(value, cwd)
        settings = replace(settings, **{section: replace(getattr(settings, section), **{name: value})})
    return _validate(settings)
