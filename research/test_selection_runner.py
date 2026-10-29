#!/usr/bin/env python3
"""Test orchestration, resume, aggregation, and failure recovery without GPU jobs."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('selection_runner', Path(__file__).with_name('run_selection_experiments.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTest(unittest.TestCase):
    def test_resume_failure_recovery_and_config_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'research').mkdir()
            for name in ('train_ace.py', 'ace_trainer.py', 'replay_buffer.py', 'test_ace.py',
                         'research/run_selection_experiments.py', 'fps_sampling.py', 'precision.py', 'ace_encoder_pretrained.pt'):
                (root / name).write_text('fixture')
            for scene in ('7scenes_chess', '7scenes_fire'):
                for split in ('train', 'test'):
                    (root / 'datasets' / scene / split).mkdir(parents=True)
            calls = []
            fail = [True]

            def execute(command, **kwargs):
                if command == ['nvidia-smi']:
                    return types.SimpleNamespace(stdout='fixture GPU', stderr='')
                calls.append(command)
                is_train = Path(command[1]).name == 'train_ace.py'
                def value(flag):
                    return command[command.index(flag) + 1]
                if is_train:
                    if '--replay_buffer' in command and fail[0]:
                        raise subprocess.CalledProcessError(1, command)
                    metrics = dict(total_seconds=2, peak_gpu_memory_allocated_bytes=100)
                    if '--export_only' in command:
                        Path(value('--export_replay_buffer')).write_bytes(b'buffer')
                        metrics['selection_metadata'] = dict(candidate_geometry_sha256='same',
                            selection_seconds=0.5, selected_occupied_voxels=7)
                    else:
                        Path(command[3]).write_bytes(b'head')
                    Path(value('--training_metrics_json')).write_text(json.dumps(metrics))
                else:
                    score = 95 if value('--session') == 'chess_before' else 90
                    Path(value('--metrics_json')).write_text(json.dumps(
                        dict(accuracy_percent={'5cm_5deg': score})))
                return types.SimpleNamespace(returncode=0)

            args = ['runner', '--gpu', '2', '--seeds', '2089', '2090', '--sizes', '10',
                    '--methods', 'filtered_random', 'voxel_quality', '--buffer-size', '100',
                    '--batch-size', '20', '--epochs', '1']
            torch_stub = types.SimpleNamespace(__version__='fixture', cuda=types.SimpleNamespace(
                is_available=lambda: True, get_device_name=lambda _: 'fixture'))
            with patch.object(runner, 'ROOT', root), patch.dict(sys.modules, {'torch': torch_stub}), \
                    patch.object(runner.subprocess, 'run', side_effect=execute), \
                    contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys, 'argv', args), self.assertRaises(SystemExit):
                    runner.main()
                out = root / 'research/results/selection-study-v1'
                self.assertFalse(runner.read(out / 'summary.json')['complete'])
                fail[0] = False
                calls.clear()
                with patch.object(sys, 'argv', args):
                    runner.main()
                summary = runner.read(out / 'summary.json')
                self.assertTrue(summary['complete'])
                self.assertEqual(summary['completed_runs'], 4)
                self.assertEqual(summary['aggregate'][0]['chess_percent_std'], 0)
                self.assertEqual(len(calls), 12)  # Fire + 2 evaluations for 4 conditions
                calls.clear()
                with patch.object(sys, 'argv', args):
                    runner.main()
                self.assertEqual(calls, [])
                with patch.object(sys, 'argv', [*args, '--voxel-size', '0.5']), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    runner.main()


if __name__ == '__main__':
    unittest.main()
