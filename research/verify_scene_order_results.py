"""Validate and import metrics-only ZIP from run_scene_order_selection.py."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
from zipfile import ZipFile


def verify(archive, output):
    checks = 0

    def check(condition, label):
        nonlocal checks
        if not condition:
            raise ValueError(label)
        checks += 1

    with ZipFile(archive) as z:
        def read(name):
            return json.loads(z.read(name))

        manifest = read('experiment.json')
        summary = read('summary.json')
        rows = list(csv.DictReader(io.StringIO(z.read('runs.csv').decode())))
        expected = {(s, m, k) for s in manifest['seeds'] for m in manifest['methods'] for k in manifest['sizes']}
        check(summary['complete'] and summary['completed_runs'] == len(expected), 'Incomplete study')
        check(read('failures.json') == [], 'Non-empty failures')
        check(len(rows) == len(expected), 'Unexpected run count')
        check({(int(r['seed']), r['method'], int(r['k'])) for r in rows} == expected, 'Missing/duplicate condition')
        for r in rows:
            base = f"seed_{r['seed']}/"
            folder = base + f"{r['method']}_k{r['k']}/"
            before = read(base + 'old_before.json')['accuracy_percent']['5cm_5deg']
            check(r['old_scene'] == manifest['old_scene'] and r['new_scene'] == manifest['new_scene'], 'Scene labels')
            for role, scene in [('old', manifest['old_scene']), ('new', manifest['new_scene'])]:
                evaluation = read(folder + role + '_after.json')
                check(Path(evaluation['scene']).name == scene, 'Evaluation scene')
                check(math.isclose(evaluation['accuracy_percent']['5cm_5deg'], float(r[role + '_percent'])), 'Accuracy mismatch')
            check(math.isclose(float(r['forgetting_points']), before - float(r['old_percent'])), 'Forgetting mismatch')
            check(math.isclose(float(r['average_percent']), (float(r['old_percent']) + float(r['new_percent'])) / 2), 'Average mismatch')
            export = read(folder + 'export.json')
            check(export['selection_metadata']['candidate_geometry_sha256'] == read(base + 'candidate_pool.json')['sha256'], 'Pool mismatch')
            check(export['replay_features_stored'] == int(r['k']), 'Replay size')
            check(export['replay_buffer_bytes'] == int(r['replay_bytes']), 'Replay bytes')
            training = read(folder + 'train_new.json')
            check(Path(training['scene']).name == manifest['new_scene'], 'Training scene')
            check(training['replay_selection'] == r['method'], 'Training replay selector')
            check(training['replay_features_stored'] == int(r['k']), 'Training replay K')
            check(training['replay_ratio_percent'] == manifest['ratio'], 'Training ratio')
            check(training['replay_per_batch'] == round(manifest['batch_size'] * manifest['ratio'] / 100), 'Training quota')
            check(training['replay_features_injected'] == round(manifest['buffer_size'] * manifest['ratio'] / 100), 'Training replay slots')
            check(training['replay_batch_mode'] == manifest['batch_mode'], 'Training batch mode')
            check(training['replay_feature_encoding'] == manifest['encoding'], 'Training encoding')
        aggregate = list(csv.DictReader(io.StringIO(z.read('aggregate.csv').decode())))
        check(len(aggregate) == len(manifest['methods']) * len(manifest['sizes']), 'Aggregate count')
        for a in aggregate:
            group = [r for r in rows if r['method'] == a['method'] and r['k'] == a['k']]
            check(int(a['completed_seeds']) == len(group), 'Aggregate seeds')
            for key in rows[0]:
                if key in ('seed', 'method', 'k', 'old_scene', 'new_scene'):
                    continue
                values = [float(r[key]) for r in group]
                check(math.isclose(statistics.mean(values), float(a[key + '_mean']), abs_tol=1e-9), 'Aggregate mean')
                check(a[key + '_std'] == '' if len(values) == 1 else math.isclose(statistics.stdev(values), float(a[key + '_std']), abs_tol=1e-9), 'Aggregate SD')
        for name in z.namelist():
            if name.endswith('.done.json'):
                for path, digest in read(name)['output_sha256'].items():
                    if path.endswith('.json'):
                        marker = '/seed_'
                        member = 'seed_' + path.split(marker, 1)[1]
                        check(hashlib.sha256(z.read(member)).hexdigest() == digest, 'Output JSON digest')
            path = Path(name)
            check(not path.is_absolute() and '..' not in path.parts and path.suffix in ('.csv', '.json'), 'Unsafe member')
        # Do not write partial imports when any validation fails.
        for name in z.namelist():
            path = output / 'server-records' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read(name))
    result = dict(checks=checks, conditions=len(rows), archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  scope='JSON/CSV consistency and recorded digests; model/replay contents not included')
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return manifest, rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    verify(a.archive, a.output)
