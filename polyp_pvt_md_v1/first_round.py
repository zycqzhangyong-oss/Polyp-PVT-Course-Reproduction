"""Run the three prespecified variants, then test them once with UACANet metrics.

Run from Anaconda Prompt: python -u polyp_pvt_md_v1/first_round.py
Stop with Ctrl+C. Re-run the same command to resume from completed epochs.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
VARIANTS = ('baseline', 'spatial', 'multidomain')
METRICS = ('meanDic', 'meanIoU', 'wFm', 'Sm', 'meanEm', 'mae')


def execute(arguments, root):
    print('\nRUN: ' + subprocess.list2cmdline([str(x) for x in arguments]), flush=True)
    subprocess.run([str(x) for x in arguments], cwd=str(root), check=True)


def completed_json(path):
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding='utf-8')).get('status') == 'complete'


def summarize(suite, seed):
    tables, configs = {}, {}
    for variant in VARIANTS:
        folder = suite / (variant + '_seed' + str(seed))
        summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        if summary['status'] != 'complete' or not completed_json(folder / 'evaluation/protocol.json'):
            raise ValueError('Incomplete training/evaluation for ' + variant)
        configs[variant] = summary
        with (folder / 'evaluation/summary.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 5 or len({r['dataset'] for r in rows}) != 5:
            raise ValueError('Expected five distinct test datasets.')
        tables[variant] = {row['dataset']: row for row in rows}
    reference = configs['baseline']
    for variant in VARIANTS[1:]:
        current = configs[variant]
        a, b = dict(reference['config']), dict(current['config'])
        a.pop('variant'); b.pop('variant')
        if (a != b or reference['fingerprints'] != current['fingerprints'] or
                reference['metadata']['common_initial_weights_sha256'] != current['metadata']['common_initial_weights_sha256']):
            raise ValueError('Cannot compare runs with different protocols, data or common initialization.')
        if set(tables[variant]) != set(tables['baseline']):
            raise ValueError('Test dataset sets differ.')
    rows = []
    for dataset in tables['baseline']:
        for variant in VARIANTS:
            source = tables[variant][dataset]
            if source['count'] != tables['baseline'][dataset]['count']:
                raise ValueError('Test counts differ.')
            row = {'dataset': dataset, 'variant': variant, 'count': source['count'],
                   'best_epoch': configs[variant]['best_epoch'],
                   'parameters': configs[variant]['metadata']['parameters']}
            for metric in METRICS:
                value = float(source[metric])
                delta = value - float(tables['baseline'][dataset][metric])
                row[metric] = value
                row[metric + '_delta_vs_baseline'] = delta
                row[metric + '_display'] = '{:.4f} ({:+.4f})'.format(value, delta)
            rows.append(row)
    path = suite / ('comparison_seed' + str(seed) + '.csv')
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print('COMPARISON COMPLETE: ' + str(path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=HERE.parent)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--metrics-workers', type=int, default=2)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    root = args.project_root.resolve()
    suite = root / 'runs/polyp_pvt_md_v1'
    if args.summarize_only:
        summarize(suite, args.seed)
        return
    common = ['--project-root', root, '--batch-size', args.batch_size, '--seed', args.seed]
    python = [sys.executable, '-u']
    if not (suite / 'data').exists():
        execute(python + [HERE / 'prepare_data.py', '--train-root', root / 'dataset/TrainDataset',
                         '--test-root', root / 'dataset/TestDataset', '--output', suite / 'data'], root)
    execute(python + [HERE / 'run.py', 'check'] + common, root)
    # No benchmark scores are computed until ALL three variants finish training.
    for variant in VARIANTS:
        folder = suite / (variant + '_seed' + str(args.seed))
        summary = folder / 'summary.json'
        if completed_json(summary):
            # Revalidate finished checkpoint/configuration; does not train more epochs.
            execute(python + [HERE / 'run.py', 'train', '--variant', variant, '--resume'] + common, root)
            continue
        resume = (folder / 'last_checkpoint.pt').is_file()
        if not folder.exists():
            execute(python + [HERE / 'run.py', 'smoke', '--variant', variant] + common, root)
        if folder.exists() and not resume:
            raise RuntimeError('Run folder exists but has no complete epoch checkpoint: ' + str(folder)
                               + '. Preserve it under a different name before starting a new run.')
        execute(python + [HERE / 'run.py', 'train', '--variant', variant]
                + (['--resume'] if resume else []) + common, root)
    for variant in VARIANTS:
        folder = suite / (variant + '_seed' + str(args.seed))
        if not completed_json(folder / 'prediction_summary.json'):
            execute(python + [HERE / 'run.py', 'predict', '--variant', variant] + common, root)
        if not completed_json(folder / 'evaluation/protocol.json'):
            if (folder / 'evaluation').exists():
                archive = folder / ('evaluation_incomplete_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                (folder / 'evaluation').rename(archive)
                print('Preserved incomplete evaluation: ' + str(archive), flush=True)
            execute(python + [HERE / 'evaluation/evaluate.py', '--project-root', root,
                              '--pred-root', folder / 'predictions', '--gt-root', root / 'dataset/TestDataset',
                              '--output-dir', folder / 'evaluation', '--workers', args.metrics_workers], root)
    summarize(suite, args.seed)


if __name__ == '__main__':
    main()
