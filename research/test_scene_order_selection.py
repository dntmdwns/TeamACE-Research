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

spec = importlib.util.spec_from_file_location('scene_order_runner', Path(__file__).with_name('run_scene_order_selection.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTest(unittest.TestCase):
    def test_resume_failure_recovery_and_config_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'research').mkdir()
            for name in ('train_ace.py', 'ace_trainer.py', 'replay_buffer.py', 'test_ace.py',
                         'fps_sampling.py', 'precision.py', 'ace_encoder_pretrained.pt'):
                (root / name).write_text('fixture')
            for scene in ('7scenes_chess', '7scenes_fire'):
                for split in ('train', 'test'):
                    (root / 'datasets' / scene / split).mkdir(parents=True)
            calls = []
            fail = [True]
            pool_mismatch = [False]

            def execute(command, **kwargs):
                if command[0] == 'git':
                    return types.SimpleNamespace(stdout='task-branch', stderr='')
                if command == ['nvidia-smi']:
                    return types.SimpleNamespace(stdout='fixture GPU', stderr='')
                calls.append(command)
                # Source scene must be Fire; replay training must target Chess.
                if Path(command[1]).name == 'train_ace.py':
                    scene = Path(command[2]).name
                    self.assertEqual(scene, '7scenes_chess' if '--replay_buffer' in command else '7scenes_fire')
                else:
                    label = command[command.index('--session') + 1]
                    self.assertEqual(Path(command[2]).name, '7scenes_chess' if label == 'new_after' else '7scenes_fire')
                is_train = Path(command[1]).name == 'train_ace.py'
                def value(flag):
                    return command[command.index(flag) + 1]
                if is_train:
                    if '--replay_buffer' in command and fail[0]:
                        raise subprocess.CalledProcessError(1, command)
                    metrics = dict(total_seconds=2, peak_gpu_memory_allocated_bytes=100)
                    if '--export_only' in command:
                        Path(value('--export_replay_buffer')).write_bytes(b'buffer')
                        mismatched = pool_mismatch[0] and value('--replay_selection') == 'fps'
                        metrics['selection_metadata'] = dict(candidate_geometry_sha256='different' if mismatched else 'same',
                            selection_seconds=0.5, selected_occupied_voxels=7)
                    else:
                        Path(command[3]).write_bytes(b'head')
                    Path(value('--training_metrics_json')).write_text(json.dumps(metrics))
                else:
                    score = 95 if value('--session') == 'old_before' else 90
                    Path(value('--metrics_json')).write_text(json.dumps(
                        dict(accuracy_percent={'5cm_5deg': score})))
                return types.SimpleNamespace(returncode=0)

            args = ['runner', '--checkout', str(root), '--gpu', '2', '--seeds', '2089', '2090', '--sizes', '10',
                    '--methods', 'filtered_random', 'fps', '--buffer-size', '100',
                    '--batch-size', '20', '--epochs', '1']
            torch_stub = types.SimpleNamespace(__version__='fixture', cuda=types.SimpleNamespace(
                is_available=lambda: True, get_device_name=lambda _: 'fixture'))
            with patch.object(runner, 'ROOT', root), patch.dict(sys.modules, {'torch': torch_stub}), patch.object(Path, 'home', return_value=root), \
                    patch.object(runner.subprocess, 'run', side_effect=execute), \
                    contextlib.redirect_stdout(io.StringIO()):
                preview = io.StringIO()
                with patch.object(sys, 'argv', ['runner', '--checkout', str(root), '--gpu', '2', '--dry-run']), \
                        contextlib.redirect_stdout(preview):
                    runner.main()
                self.assertEqual(preview.getvalue().count('--export_only'), 3)
                self.assertEqual(preview.getvalue().count('--replay_ratio_percent 10'), 3)
                self.assertFalse((root / 'research/results').exists())
                with patch.object(sys, 'argv', [*args, '--dry-run']):
                    runner.main()
                self.assertFalse((root / 'research/results').exists())
                self.assertEqual(calls, [])
                with patch.object(sys, 'argv', args), self.assertRaises(SystemExit):
                    runner.main()
                out = root / 'research/results/fire-to-chess-selection-pilot-v1'
                self.assertFalse(runner.read(out / 'summary.json')['complete'])
                fail[0] = False
                calls.clear()
                with patch.object(sys, 'argv', args):
                    runner.main()
                summary = runner.read(out / 'summary.json')
                self.assertTrue(summary['complete'])
                self.assertEqual(summary['runs'][0]['old_scene'], '7scenes_fire')
                self.assertEqual(summary['runs'][0]['new_scene'], '7scenes_chess')
                self.assertEqual(summary['completed_runs'], 4)
                self.assertEqual(summary['aggregate'][0]['old_percent_std'], 0)
                self.assertEqual(summary['runs'][0]['forgetting_points'], 5)
                self.assertEqual(len(calls), 12)  # Fire + 2 evaluations for 4 conditions
                calls.clear()
                with patch.object(sys, 'argv', args):
                    runner.main()
                self.assertEqual(calls, [])
                with patch.object(sys, 'argv', [*args, '--voxel-size', '0.5']), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    runner.main()
                # A changed completed model cannot be silently reused.
                model = out / 'seed_2089/stage_a_head.pt'
                model.write_bytes(b'changed')
                with patch.object(sys, 'argv', args), self.assertRaises(RuntimeError):
                    runner.main()
                model.write_bytes(b'head')
                # Mismatched export pools must stop that condition before Chess training.
                pool_mismatch[0] = True
                calls.clear()
                bad_out = root / 'research/results/mismatch'
                with patch.object(sys, 'argv', [*args, '--output-dir', str(bad_out)]), self.assertRaises(SystemExit):
                    runner.main()
                bad = runner.read(bad_out / 'summary.json')
                self.assertEqual(bad['completed_runs'], 2)
                self.assertEqual(len(runner.read(bad_out / 'failures.json')), 2)
                self.assertFalse(any('--replay_buffer' in cmd and 'fps_' in ' '.join(cmd) for cmd in calls))


if __name__ == '__main__':
    unittest.main()
