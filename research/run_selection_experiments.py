#!/usr/bin/env python3
# HISTORICAL runner, kept to document past results; new experiments use research/run_standard.sh.
# Defaults to: calibrated (D-SLAM) poses (datasets/7scenes_*), Chess -> Fire, --use_half False with an fp32 store,
# filtered_random/voxel/quality/voxel_quality at K 10,000/50,000, 5 px filter, repeated candidates NOT removed,
# 10% stratified, seeds 2089-2091.
"""Resumable Chess -> Fire selection study; invoke from the lab's ace environment."""
import argparse
from contextlib import ExitStack
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def table(path, rows):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def summarize(out, expected):
    rows = [read(p) for p in sorted(out.glob('seed_*/**/result.json'))]
    table(out / 'runs.csv', rows)
    groups = {}
    for row in rows:
        groups.setdefault((row['method'], row['k']), []).append(row)
    aggregate = []
    for (method, k), group in sorted(groups.items()):
        item = dict(method=method, k=k, completed_seeds=len(group))
        for key in ('chess_percent', 'fire_percent', 'average_percent', 'forgetting_points',
                    'replay_bytes', 'selection_seconds', 'export_total_seconds',
                    'fire_total_seconds', 'peak_gpu_bytes', 'selected_voxels'):
            values = [r[key] for r in group]
            item[key + '_mean'] = statistics.mean(values)
            item[key + '_std'] = statistics.stdev(values) if len(values) > 1 else None
        aggregate.append(item)
    table(out / 'aggregate.csv', aggregate)
    write(out / 'summary.json', dict(expected_runs=expected, completed_runs=len(rows),
                                    complete=len(rows) == expected, runs=rows, aggregate=aggregate))


def run_study(stack):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', required=True, help='Physical GPU index selected after checking nvidia-smi')
    p.add_argument('--datasets-root', type=Path, default=ROOT / 'datasets')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'research/results/selection-study-v1')
    p.add_argument('--seeds', type=int, nargs='+', default=[2089, 2090, 2091])
    p.add_argument('--sizes', type=int, nargs='+', default=[10000, 50000])
    p.add_argument('--methods', nargs='+', default=['filtered_random', 'voxel', 'quality', 'voxel_quality'],
                   choices=['filtered_random', 'voxel', 'quality', 'voxel_quality', 'voxel_feature_dedup', 'fps'])
    p.add_argument('--dedup-cosine', type=float, default=0.98)
    p.add_argument('--dedup-pool-factor', type=int, default=4)
    p.add_argument('--buffer-size', type=int, default=4000000)
    p.add_argument('--batch-size', type=int, default=5120)
    p.add_argument('--epochs', type=int, default=16)
    p.add_argument('--ratio', type=float, default=10)
    p.add_argument('--voxel-size', type=float, default=0.25)
    p.add_argument('--max-error', type=float, default=5)
    p.add_argument('--encoding', choices=['fp32', 'fp16', 'int8'], default='fp32')
    p.add_argument('--use-half', choices=['True', 'False'], default='False',
                   help='False disables FP16 autocast and uses FP32 training features')
    p.add_argument('--batch-mode', choices=['stratified', 'shuffle'], default='stratified')
    p.add_argument('--dry-run', action='store_true', help='Print every command without writing or running jobs')
    a = p.parse_args()
    for values in (a.seeds, a.sizes, a.methods):
        if len(values) != len(set(values)):
            p.error('Duplicate seeds, sizes, or methods are not allowed')
    if (not 0 < a.ratio < 100 or a.batch_size <= 0 or a.buffer_size < a.batch_size
            or a.epochs <= 0 or min(a.sizes) <= 0 or max(a.sizes) > a.buffer_size
            or a.voxel_size <= 0 or a.max_error <= 0
            or not 0 < a.dedup_cosine <= 1 or a.dedup_pool_factor < 1):
        p.error('Invalid experiment dimensions or ratio')
    if a.batch_mode == 'stratified':
        old = round(a.batch_size * a.ratio / 100)
        count = round(a.buffer_size * a.ratio / 100)
        steps = a.buffer_size // a.batch_size
        if not 0 < old < a.batch_size or steps * old > count or steps * (a.batch_size-old) > a.buffer_size-count:
            p.error('Exact batch quotas cannot be supplied by this buffer; choose compatible dimensions')
    out = a.output_dir.resolve()
    data = a.datasets_root.resolve()
    expected = len(a.methods) * len(a.sizes) * len(a.seeds)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
              if k not in ('dry_run', 'output_dir')}
    config['datasets_root'] = str(data)
    config['python'] = sys.executable
    config['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ('train_ace.py', 'ace_trainer.py', 'replay_buffer.py', 'test_ace.py',
                     'research/run_selection_experiments.py', 'fps_sampling.py', 'precision.py')}
    config['expected_runs'] = expected
    config['evaluation'] = 'full test splits; no max_test_images'
    env = os.environ.copy()
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=a.gpu)
    if not a.dry_run:
        for scene in ('7scenes_chess', '7scenes_fire'):
            for split in ('train', 'test'):
                if not (data / scene / split).is_dir():
                    p.error(f'Missing {data / scene / split}')
        if not (ROOT / 'ace_encoder_pretrained.pt').is_file():
            p.error('Missing ace_encoder_pretrained.pt')
        out.mkdir(parents=True, exist_ok=True)
        lock = stack.enter_context((out / '.runner.lock').open('w'))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error('Another runner already owns this output directory')
        manifest = out / 'experiment.json'
        if manifest.exists() and read(manifest) != config:
            p.error('Configuration/source changed: use a NEW --output-dir; existing results are preserved')
        write(manifest, config)
        import torch
        if not torch.cuda.is_available():
            p.error('CUDA unavailable: execute inside the lab ace environment')
        info = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        (out / 'gpu_run.log').write_text(info.stdout + info.stderr)
        write(out / 'runtime.json', {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                                   'started_unix': time.time()})

    def step(directory, label, command, outputs):
        command = [str(x) for x in command]
        print('+ ' + ' '.join(command), flush=True)
        if a.dry_run:
            return
        directory.mkdir(parents=True, exist_ok=True)
        stamp = directory / (label + '.done.json')
        if stamp.exists():
            saved = read(stamp)
            if saved['command'] != command or any(not Path(x).is_file() for x in outputs):
                raise RuntimeError(f'Invalid completion record: {stamp}')
            for x in outputs:
                if str(x).endswith('.json'):
                    read(Path(x))
            print('  completed; skipping', flush=True)
            return
        with (directory / (label + '_run.log')).open('a') as log:
            log.write('\nCOMMAND: ' + json.dumps(command) + '\n')
            log.flush()
            subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if any(not Path(x).is_file() for x in outputs):
            raise RuntimeError(f'Missing output after {label}')
        for x in outputs:
            if str(x).endswith('.json'):
                read(Path(x))
        write(stamp, dict(command=command, completed_unix=time.time()))

    def train(directory, label, scene, head, extra):
        metrics = directory / (label + '.json')
        cmd = [sys.executable, ROOT / 'train_ace.py', data / scene, head,
               '--training_buffer_size', a.buffer_size, '--batch_size', a.batch_size,
               '--epochs', a.epochs, '--seed', seed, '--use_half', a.use_half,
               '--training_metrics_json', metrics, *extra]
        outputs = [metrics] if '--export_only' in extra else [metrics, head]
        if '--export_replay_buffer' in extra:
            outputs.append(extra[extra.index('--export_replay_buffer') + 1])
        step(directory, label, cmd, outputs)
        return metrics

    def evaluate(directory, label, scene, head):
        metrics = directory / (label + '.json')
        step(directory, label, [sys.executable, ROOT / 'test_ace.py', data / scene, head,
             '--session', label, '--metrics_json', metrics,
             '--use_half', a.use_half, '--seed', seed], [metrics])
        return metrics

    failures = []
    for seed in a.seeds:
        base = out / f'seed_{seed}'
        head_a = base / 'stage_a_head.pt'
        train(base, 'train_chess', '7scenes_chess', head_a, [])
        before = evaluate(base, 'chess_before', '7scenes_chess', head_a)
        # The shared checkpoint and deterministic seed reconstruct the same candidate pool.
        for k in a.sizes:
            for method in a.methods:
                directory = base / f'{method}_k{k}'
                try:
                    replay = directory / 'selected_replay_buffer.pt'
                    export = train(directory, 'export', '7scenes_chess', directory / 'unused.pt', [
                        '--resume_head', head_a, '--export_only', '--export_replay_buffer', replay,
                        '--replay_buffer_features', k, '--replay_selection', method,
                        '--replay_feature_encoding', a.encoding, '--replay_voxel_size_m', a.voxel_size,
                        '--replay_reprojection_error_px', a.max_error,
                        '--replay_dedup_cosine', a.dedup_cosine,
                        '--replay_dedup_pool_factor', a.dedup_pool_factor])
                    if not a.dry_run:
                        metadata = read(export)['selection_metadata']
                        pool_record = base / 'candidate_pool.json'
                        pool_hash = metadata['candidate_geometry_sha256']
                        if pool_record.exists() and read(pool_record)['sha256'] != pool_hash:
                            raise RuntimeError('Candidate pool differs across conditions; comparison stopped')
                        write(pool_record, {'sha256': pool_hash})
                    head_b = directory / 'stage_b_head.pt'
                    training = train(directory, 'train_fire', '7scenes_fire', head_b, [
                        '--resume_head', head_a, '--replay_buffer', replay,
                        '--replay_ratio_percent', a.ratio, '--replay_batch_mode', a.batch_mode])
                    chess = evaluate(directory, 'chess_after', '7scenes_chess', head_b)
                    fire = evaluate(directory, 'fire_after', '7scenes_fire', head_b)
                    if not a.dry_run:
                        c = read(chess)['accuracy_percent']['5cm_5deg']
                        f = read(fire)['accuracy_percent']['5cm_5deg']
                        t = read(training)
                        e = read(export)
                        write(directory / 'result.json', dict(seed=seed, method=method, k=k,
                            chess_percent=c, fire_percent=f, average_percent=(c+f)/2,
                            forgetting_points=read(before)['accuracy_percent']['5cm_5deg']-c,
                            replay_bytes=replay.stat().st_size,
                            selection_seconds=metadata['selection_seconds'],
                            export_total_seconds=e['total_seconds'], fire_total_seconds=t['total_seconds'],
                            peak_gpu_bytes=t['peak_gpu_memory_allocated_bytes'],
                            selected_voxels=metadata['selected_occupied_voxels'],
                            dedup_shortlist_candidates=metadata.get('dedup_shortlist_candidates'),
                            dedup_survivors=metadata.get('dedup_survivors'),
                            dedup_rejected=metadata.get('dedup_rejected'),
                            dedup_backfill_count=metadata.get('dedup_backfill_count'),
                            dedup_backfill_fraction=metadata.get('dedup_backfill_fraction')))
                        (directory / "failure.json").unlink(missing_ok=True)
                except (subprocess.CalledProcessError, RuntimeError, ValueError, KeyError) as error:
                    failures.append(dict(seed=seed, method=method, k=k, error=str(error)))
                    write(directory / 'failure.json', failures[-1])
                    print(f'FAILED {directory}: {error}; see *_run.log', flush=True)
                finally:
                    if not a.dry_run:
                        summarize(out, expected)
                        write(out / 'failures.json', failures)
    if not a.dry_run:
        print(f'Results: {out}/aggregate.csv ({read(out / "summary.json")["completed_runs"]}/{expected} complete)')
    if failures:
        raise SystemExit(1)


def main():
    with ExitStack() as stack:
        run_study(stack)


if __name__ == '__main__':
    main()
