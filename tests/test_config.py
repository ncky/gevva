"""Portable launch configuration and isolated worker settings; no GPU required."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from contextlib import contextmanager

from gevva.config import ConfigError, load_config
from gevva.worker import Worker
from gevva.server import main
from gevva.warmup import warmup


@contextmanager
def working_directory(path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / 'install' / 'gevva.toml'
        self.path.parent.mkdir()
        self.path.write_text('[engine]\nbinary="./bin/gevva-engine"\ngpu="5090"\n[model]\nroot="./weights"\n'
                             '[server]\nport=9000\n[warmup]\nimages=true\n')

    def tearDown(self):
        self.temporary.cleanup()

    def load(self, **kwargs):
        return load_config(self.path, environ={}, **kwargs)

    def test_paths_follow_config_not_cwd(self):
        with working_directory(self.root):
            config = self.load()
        self.assertEqual(config.engine.binary, str(self.path.parent / 'bin/gevva-engine'))
        self.assertEqual(config.model.root, self.path.parent / 'weights')
        self.assertEqual(config.model.target, self.path.parent / 'weights/gemma4-26b-a4b-nvfp4')
        self.assertEqual(config.server.port, 9000)
        self.assertTrue(config.warmup.images)

    def test_precedence_and_relative_override_paths(self):
        with working_directory(self.root):
            config = load_config(self.path, environ={'GEVVA_GPU': 'pro6000', 'GEVVA_MODEL_ROOT': './env-models'},
                                 overrides={'engine.gpu': '5090', 'server.port': 8123})
        self.assertEqual(config.engine.gpu, '5090')
        self.assertEqual(config.model.root, self.root / 'env-models')
        self.assertEqual(config.model.dense.parent, config.model.root)
        self.assertEqual(config.server.port, 8123)

    def test_root_override_preserves_explicit_components_even_at_default_path(self):
        self.path.write_text('[model]\nroot="./weights"\ntarget="./weights/gemma4-26b-a4b-nvfp4"\n')
        config = self.load(overrides={'model.root': self.root / 'new'})
        self.assertEqual(config.model.target, self.path.parent / 'weights/gemma4-26b-a4b-nvfp4')
        self.assertEqual(config.model.dense.parent, self.root / 'new')

    def test_config_discovery_and_explicit_missing_file(self):
        with working_directory(self.path.parent):
            self.assertEqual(load_config(environ={}).source, self.path)
        other = self.root / 'other.toml'
        other.write_text('[server]\nport=8124\n')
        self.assertEqual(load_config(environ={'GEVVA_CONFIG': str(other)}).server.port, 8124)
        self.assertEqual(load_config(self.path, environ={'GEVVA_CONFIG': str(other)}).server.port, 9000)
        with self.assertRaisesRegex(ConfigError, 'Cannot read config'):
            load_config(self.root / 'missing.toml', environ={})

    def test_unknown_keys_and_invalid_values(self):
        bad = ['unknown=1', '[engine]\ngup="5090"', '[model]\nroot=7', 'server=3',
               'schema_version=true', 'schema_version=2', '[server]\nport=true',
               '[server]\nport=65536', '[server]\nqueue_capacity=0',
               '[server]\nmax_connections=-1', '[server]\nhost=""',
               '[engine]\ngpu="auto"', '[engine]\nworker_timeout=nan',
               '[engine]\nworker_timeout=true', '[engine]\nworker_timeout=0',
               '[warmup]\nimages="yes"', '[warmup]\ncommon_shapes=1',
               '[engine]\ncuda_visible_devices=3', '[server]\napi_key_env="A=B"']
        for text in bad:
            with self.subTest(text=text):
                self.path.write_text(text)
                with self.assertRaises(ConfigError):
                    self.load()

    def test_bad_toml_has_filename(self):
        self.path.write_text('[engine\n')
        with self.assertRaisesRegex(ConfigError, str(self.path)):
            self.load()

    def test_snapshot_does_not_reread_environment(self):
        config = self.load()
        with patch.dict(os.environ, {'GEVVA_GPU': 'pro6000'}):
            self.assertEqual(load_config(config).engine.gpu, '5090')
        self.assertEqual(load_config(config, overrides={'engine.gpu': 'pro6000'}).engine.gpu, 'pro6000')

    def test_worker_environment_is_private_and_honors_empty_visibility(self):
        parent = {'GEVVA_GPU': 'pro6000', 'UNCHANGED': 'value', 'CUDA_VISIBLE_DEVICES': ''}
        config = load_config(self.path, environ=parent, overrides={'engine.gpu': '5090'})
        env = config.worker_env(parent)
        self.assertEqual(parent['GEVVA_GPU'], 'pro6000')
        self.assertEqual(env['GEVVA_GPU'], '5090')
        self.assertEqual(env['UNCHANGED'], 'value')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(env['GEVVA_MODEL_TARGET'], str(config.model.target))

    def test_missing_files_fail_before_spawn(self):
        with patch('gevva.worker.subprocess.Popen') as spawn:
            with self.assertRaisesRegex(ConfigError, 'engine.binary'):
                Worker(config=self.load())
            spawn.assert_not_called()
        binary = self.path.parent / 'bin/gevva-engine'
        binary.parent.mkdir();binary.write_text('#!/bin/sh\nexit 0\n');binary.chmod(0o755)
        with patch('gevva.worker.subprocess.Popen') as spawn:
            with self.assertRaisesRegex(ConfigError, 'Missing model files'):
                Worker(config=self.load())
            spawn.assert_not_called()

    def test_two_workers_do_not_change_parent_gpu(self):
        config = self.load()
        with patch('gevva.config.Config.validate_files'), patch('gevva.worker.subprocess.Popen') as spawn:
            before = os.environ.get('GEVVA_GPU')
            first = Worker(config=config)
            second = Worker(config=config, gpu='pro6000')
            self.assertEqual(spawn.call_args_list[0].kwargs['env']['GEVVA_GPU'], '5090')
            self.assertEqual(spawn.call_args_list[1].kwargs['env']['GEVVA_GPU'], 'pro6000')
            self.assertEqual(os.environ.get('GEVVA_GPU'), before)
            first.close();second.close()

    def test_cli_prints_overrides_without_reading_secrets_or_starting_worker(self):
        from contextlib import redirect_stdout
        import io
        output = io.StringIO()
        with patch.dict(os.environ, {'GEVVA_API_KEY': 'do-not-print-me'}, clear=True), \
             patch('gevva.server.LocalEvaluator') as evaluator, redirect_stdout(output):
            result = main(['--config', str(self.path), '--gpu', 'pro6000', '--bind', '0.0.0.0',
                           '--port', '8125', '--no-warmup-images', '--minimal-warmup', '--print-config'])
        self.assertEqual(result, 0)
        evaluator.assert_not_called()
        data = json.loads(output.getvalue())
        self.assertEqual(data['server']['host'], '0.0.0.0')
        self.assertEqual(data['server']['port'], 8125)
        self.assertEqual(data['engine']['gpu'], 'pro6000')
        self.assertEqual(data['warmup'], {'images': False, 'common_shapes': False})
        self.assertNotIn('do-not-print-me', output.getvalue())

    def test_warmup_image_is_packaged_inline_not_repo_relative(self):
        class Evaluator:
            def __init__(self): self.requests=[]
            def evaluate(self, request):
                self.requests.append(request)
                return {'gevva': {'metrics': {'device_used_bytes': 123}}}
        evaluator=Evaluator()
        with working_directory(self.root):
            report=warmup(evaluator, images=True, common_shapes=False)
        self.assertEqual(report['requests'], 9)
        self.assertEqual(sum('images' in r for r in evaluator.requests), 6)
        for request in evaluator.requests:
            if 'images' in request:
                self.assertTrue(request['images'][0].startswith('data:image/png;base64,'))

class PreparationTests(unittest.TestCase):
    def test_dry_run_resolves_all_outputs_without_loading_gpu(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('prepare_model', Path(__file__).resolve().parents[1] / 'tools/prepare_model.py')
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from contextlib import redirect_stdout
        import io
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'launch.toml';path.write_text('[engine]\ngpu="5090"\n[model]\nroot="./model files"\n')
            out=io.StringIO()
            with patch.dict(os.environ, {}, clear=True), patch.object(module.subprocess, 'run') as run, redirect_stdout(out):
                self.assertEqual(module.main(['--config',str(path),'--dry-run','--download']),0)
            run.assert_not_called()
            plan=json.loads(out.getvalue())
            self.assertEqual(len(plan['commands']),3)
            self.assertEqual(plan['gpu'],'5090')
            for command in plan['commands']:
                self.assertIn('--config',command)
                self.assertIn(str(path),command)
                self.assertTrue(any('model files' in x for x in command))

    def test_converter_gpu_selection_uses_model_not_ordinal(self):
        import importlib.util
        from types import SimpleNamespace
        spec=importlib.util.spec_from_file_location('_preparation',Path(__file__).resolve().parents[1]/'tools/_preparation.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        props=[SimpleNamespace(name='NVIDIA GeForce RTX 5090',major=12,minor=0),
               SimpleNamespace(name='NVIDIA RTX PRO 6000 Blackwell Workstation Edition',major=12,minor=0)]
        cuda=MagicMock();cuda.device_count.return_value=2;cuda.get_device_properties.side_effect=lambda i:props[i]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.toml';path.write_text('[engine]\ngpu="pro6000"\n')
            config=load_config(path,environ={})
            with patch.dict(sys.modules, {'torch':SimpleNamespace(cuda=cuda)}):
                self.assertIs(module.select_gpu(config),props[1])
                cuda.set_device.assert_called_once_with(1)
                cuda.device_count.return_value=1
                with self.assertRaisesRegex(RuntimeError,'not visible'):
                    module.select_gpu(config)


if __name__ == '__main__':
    unittest.main()
