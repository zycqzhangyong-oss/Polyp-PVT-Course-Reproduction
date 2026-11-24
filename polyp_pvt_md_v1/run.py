"""Check, smoke-test, train and predict Polyp-PVT CIM replacement experiments."""
import argparse
import ast
import csv
from datetime import datetime
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import numpy as np
from PIL import Image
import torch

PACKAGE = Path(__file__).resolve().parent
from protocol import (SCHEMA, DEFAULTS, DATASETS, load_json, write_json, write_csv,
                      seed_all, load_development_data, read_manifest, verify_rows,
                      make_loader, input_tensor, structure_loss, scale_batch,
                      probability_map, validation_scores)
from prepare_data import sha256_file, read_image, read_gray_mask


def ast_hash(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    # Permit comments and explanatory docstrings without accepting code changes.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body = node.body[1:]
    # ast.dump varies between Python releases; use a version-neutral field walk.
    def canonical(value):
        if isinstance(value, ast.AST):
            if type(value).__name__ == 'Index':  # Python 3.8 subscript wrapper
                return canonical(value.value)
            if type(value).__name__ == 'ExtSlice':
                return ['Tuple', {'elts': canonical(value.dims), 'ctx': ['Load', {}]}]
            return [type(value).__name__, {name: canonical(child) for name, child in ast.iter_fields(value)
                                         if name not in ('type_params', 'type_comment', 'kind')}]
        if isinstance(value, list):
            return [canonical(child) for child in value]
        return value
    return hashlib.sha256(json.dumps(canonical(tree), sort_keys=True).encode()).hexdigest()


def activate_project(root):
    root = root.resolve()
    reference = load_json(PACKAGE / 'reference.json')
    for name in ('lib/pvt.py', 'lib/pvtv2.py'):
        if not (root / name).is_file():
            raise FileNotFoundError('Missing official project file: ' + str(root / name))
        if ast_hash(root / name) != reference['ast_sha256'][name]:
            raise ValueError(name + ' differs from the verified official version. Keep your file; share it for adaptation. No file was overwritten.')
    if not (root / 'pretrained_pth/pvt_v2_b2.pth').is_file():
        raise FileNotFoundError('Missing pretrained_pth/pvt_v2_b2.pth (Backbone pretraining, not PolypPVT.pth).')
    sys.path.insert(0, str(root))
    os.chdir(str(root))
    return root


def fingerprints(root, data_fingerprint):
    import timm
    import PIL
    import scipy
    import torchvision
    files = {p.relative_to(PACKAGE).as_posix(): sha256_file(p) for p in sorted(PACKAGE.rglob('*.py'))}
    files.update({name: sha256_file(root / name) for name in ('lib/pvt.py', 'lib/pvtv2.py')})
    return {'data': data_fingerprint, 'code': files,
            'pretrained': sha256_file(root / 'pretrained_pth/pvt_v2_b2.pth'),
            'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                            'numpy': np.__version__, 'timm': timm.__version__,
                            'pillow': PIL.__version__, 'scipy': scipy.__version__,
                            'torchvision': torchvision.__version__,
                            'cuda_runtime': torch.version.cuda}}


def atomic_save(payload, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def cpu_weights(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def configure_logging(path):
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(str(path), encoding='utf-8')]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=handlers, force=True)
    logging.captureWarnings(True)


def train_epoch(model, optimizer, loader, config, device, epoch, max_batches=None):
    model.train()
    totals, samples = [0., 0., 0.], 0
    started = time.monotonic()
    for step, (images, masks, _) in enumerate(loader, 1):
        images, masks = images.to(device), masks.to(device)
        if not torch.isfinite(images).all() or not torch.isfinite(masks).all():
            raise FloatingPointError('Nonfinite input.')
        for rate in config['scales']:
            size = int(round(config['size'] * rate / 32) * 32)
            x, y = scale_batch(images, masks, size, config['size'])
            optimizer.zero_grad()
            p1, p2 = model(x)
            if p1.shape != y.shape or p2.shape != y.shape:
                raise ValueError('Prediction/target shape mismatch.')
            loss1, loss2 = structure_loss(p1, y), structure_loss(p2, y)
            loss = loss1 + loss2
            if not torch.isfinite(loss).item():
                raise FloatingPointError('Nonfinite loss.')
            loss.backward()
            gradients = []
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    raise RuntimeError('Unused trainable parameter: ' + name)
                gradients.append(torch.isfinite(parameter.grad).all())
            if not torch.stack(gradients).all().item():
                raise FloatingPointError('Nonfinite gradients.')
            for parameter in model.parameters():
                parameter.grad.data.clamp_(-config['clip_value'], config['clip_value'])
            optimizer.step()
            if rate == 1.:
                count = len(images)
                samples += count
                for i, value in enumerate((loss, loss1, loss2)):
                    totals[i] += value.item() * count
            if max_batches is not None:
                logging.info('SMOKE scale=%d batch=%d loss=%.6f', size, len(images), loss.item())
        if step == 1 or step % 10 == 0 or step == len(loader):
            logging.info('TRAIN epoch=%d step=%d/%d loss=%.6f elapsed_s=%.1f',
                         epoch, step, len(loader), totals[0] / samples, time.monotonic() - started)
        if max_batches is not None and step >= max_batches:
            break
    optimizer.zero_grad()
    if max_batches is None and samples != len(loader.dataset):
        raise RuntimeError('Training did not cover every manifest row.')
    return dict(zip(('train_loss', 'loss_p1', 'loss_p2'), [v / samples for v in totals]))


def validate(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for images, _, indices in loader:
            row = loader.dataset.rows[int(indices[0])]
            gray = read_gray_mask(row['mask'])
            probability = probability_map(model(images.to(device)), (gray.height, gray.width))
            soft, hard = validation_scores(probability, gray)
            rows.append({'id': row['id'], 'soft_dice': soft, 'hard_dice_05': hard})
    if len(rows) != len(loader.dataset):
        raise RuntimeError('Validation did not cover every manifest row.')
    return {'val_soft_dice': float(np.mean([r['soft_dice'] for r in rows])),
            'val_hard_dice_05': float(np.mean([r['hard_dice_05'] for r in rows]))}, rows


def export_reports(directory, state):
    write_csv(directory / 'epochs.csv', state['history'])
    write_csv(directory / 'val_latest.csv', state['val_rows'])
    write_json(directory / 'summary.json', {
        'schema': SCHEMA, 'variant': state['config']['variant'], 'completed_epoch': state['epoch'],
        'planned_epochs': state['config']['epochs'], 'best_epoch': state['best_epoch'],
        'best_val_soft_dice': state['best_score'], 'metadata': state['metadata'],
        'status': 'complete' if state['epoch'] == state['config']['epochs'] else 'incomplete_resumable',
        'selection': state['config']['selection'], 'config': state['config'],
        'fingerprints': state['fingerprints']})
    atomic_save({'schema': SCHEMA, 'kind': 'best_weights', 'config': state['config'],
                 'fingerprints': state['fingerprints'], 'epoch': state['best_epoch'],
                 'score': state['best_score'], 'model': state['best_model'],
                 'metadata': state['metadata']}, directory / 'best_weights.pt')


def run_training(args, root, data, config, directory, fp, device):
    from model import build_model
    if args.resume:
        state = torch.load(str(directory / 'last_checkpoint.pt'), map_location='cpu')
        if (state.get('schema') != SCHEMA or state.get('kind') != 'training_checkpoint'
                or state['config'] != config or state['fingerprints'] != fp):
            raise ValueError('Resume config, source code, environment or data differs from checkpoint.')
    else:
        directory.mkdir(parents=True, exist_ok=False)
        state = None
    configure_logging(directory / 'train.log')
    seed_all(config['seed'])
    model, metadata = build_model(config['variant'], config['seed'])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    seed_all(config['seed'] + 500000)
    write_json(directory / 'config.json', {'config': config, 'fingerprints': fp, 'metadata': metadata})
    if state:
        if state['metadata'] != metadata:
            raise ValueError('Initialization differs from the original run.')
        if len(state['history']) != state['epoch'] or not 1 <= state['epoch'] <= config['epochs']:
            raise ValueError('Invalid checkpoint epoch/history.')
        if state['epoch'] == config['epochs']:
            if not (directory / 'best_weights.pt').is_file() or not (directory / 'summary.json').is_file():
                export_reports(directory, state)
            logging.info('ALREADY COMPLETE; configuration and initialization verified: %s', directory)
            return
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'])
        export_reports(directory, state)
    history = state['history'] if state else []
    best_score = state['best_score'] if state else -1.
    best_epoch = state['best_epoch'] if state else 0
    best_model = state['best_model'] if state else None
    start = state['epoch'] + 1 if state else 1
    last = config['epochs'] if args.stop_after is None else min(config['epochs'], start + args.stop_after - 1)
    val_loader = make_loader(data['val'], config, training=False)
    logging.info('START variant=%s train=%d val=%d epochs=%d..%d device=%s parameters=%d',
                 config['variant'], len(data['train']), len(data['val']), start, last, device, metadata['parameters'])
    for epoch in range(start, last + 1):
        started = time.monotonic()
        scores = train_epoch(model, optimizer, make_loader(data['train'], config, epoch), config, device, epoch)
        validation, val_rows = validate(model, val_loader, device)
        if validation['val_soft_dice'] > best_score:
            best_score, best_epoch = validation['val_soft_dice'], epoch
            best_model = cpu_weights(model)
        history.append({'epoch': epoch, **scores, **validation, 'seconds': time.monotonic() - started,
                        'lr': optimizer.param_groups[0]['lr']})
        state = {'schema': SCHEMA, 'kind': 'training_checkpoint', 'epoch': epoch, 'config': config,
                 'fingerprints': fp, 'metadata': metadata, 'model': cpu_weights(model),
                 'optimizer': optimizer.state_dict(), 'rng': capture_rng(), 'history': history,
                 'best_model': best_model, 'best_score': best_score, 'best_epoch': best_epoch,
                 'val_rows': val_rows}
        # Authoritative checkpoint includes best weights, allowing atomic recovery.
        atomic_save(state, directory / 'last_checkpoint.pt')
        export_reports(directory, state)
        logging.info('EPOCH %d/%d val_soft_dice=%.6f best=%.6f at_epoch=%d seconds=%.1f',
                     epoch, config['epochs'], validation['val_soft_dice'], best_score, best_epoch,
                     history[-1]['seconds'])
    logging.info('TRAINING %s: %s', 'COMPLETE' if last == config['epochs'] else 'STOPPED AT REQUESTED EPOCH', directory)


def run_smoke(args, data, config, suite, device):
    from model import build_model
    directory = suite / ('smoke_' + args.variant + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    configure_logging(directory / 'smoke.log')
    seed_all(config['seed'])
    model, metadata = build_model(args.variant, config['seed'])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    before = cpu_weights(model)
    loss = train_epoch(model, optimizer, make_loader(data['train'], config, 1), config, device, 1, max_batches=1)
    if not any(not torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items()):
        raise RuntimeError('Smoke optimizer did not change the model.')
    validation, _ = validate(model, make_loader(data['val'][:2], config, training=False), device)
    report = {'status': 'passed', 'scope': 'one real-data batch, three scales, two validation images; no trained result',
              'config': config, 'metadata': metadata, **loss, **validation,
              'max_cuda_allocated_MiB': torch.cuda.max_memory_allocated() / 1024**2 if device.type == 'cuda' else None}
    write_json(directory / 'smoke_report.json', report)
    print('SMOKE PASSED: ' + str(directory), flush=True)


def run_check(data, suite, seed, fp):
    from model import build_model
    results = []
    for variant in ('baseline', 'spatial', 'multidomain'):
        model, metadata = build_model(variant, seed)
        results.append({'variant': variant, **metadata})
        print('MODEL CHECK: ' + json.dumps(results[-1]), flush=True)
        del model
        gc.collect()
    if len({r['common_initial_weights_sha256'] for r in results}) != 1:
        raise RuntimeError('Common modules do not share identical initial weights.')
    suite.mkdir(parents=True, exist_ok=True)
    write_json(suite / 'preflight.json', {'status': 'passed', 'models': results, 'fingerprints': fp,
                                         'train': len(data['train']), 'val': len(data['val'])})
    print('CHECK PASSED. Common initial tensors match. No training started.', flush=True)


def run_prediction(args, config, directory, fp, data_dir, device):
    from model import build_model
    summary = load_json(directory / 'summary.json')
    if summary['status'] != 'complete':
        raise ValueError('Finish the planned training before evaluating formal test sets.')
    checkpoint = torch.load(str(directory / 'best_weights.pt'), map_location='cpu')
    if (checkpoint.get('schema') != SCHEMA or checkpoint.get('kind') != 'best_weights'
            or checkpoint['config'] != config or checkpoint['fingerprints'] != fp):
        raise ValueError('Checkpoint belongs to a different experiment/data/code/environment.')
    destination = directory / 'predictions'
    if destination.exists() and not (directory / 'prediction_summary.json').exists():
        archive = directory / ('predictions_incomplete_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        destination.rename(archive)
        print('Preserved incomplete predictions: ' + str(archive), flush=True)
    destination.mkdir(parents=True, exist_ok=False)
    configure_logging(directory / 'predict.log')
    report = load_json(data_dir / 'report.json')
    model, _ = build_model(args.variant, config['seed'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device).eval()
    records = []
    for dataset in DATASETS:
        rows = read_manifest(data_dir, 'test_' + dataset, report)
        verify_rows(rows, report['test_root'])
        folder = destination / dataset
        folder.mkdir()
        with torch.no_grad():
            for i, row in enumerate(rows, 1):
                original = read_image(row['image']).convert('RGB')
                x = input_tensor(original, config['size']).unsqueeze(0).to(device)
                probability = probability_map(model(x), (original.height, original.width))
                output = folder / (row['id'] + '.png')
                Image.fromarray(np.rint(probability * 255).clip(0, 255).astype(np.uint8)).save(str(output))
                records.append({'dataset': dataset, 'id': row['id'], 'path': str(output),
                                'sha256': sha256_file(output)})
                if i % 50 == 0 or i == len(rows):
                    logging.info('PREDICT %s %d/%d', dataset, i, len(rows))
    write_csv(directory / 'prediction_manifest.csv', records)
    write_json(directory / 'prediction_summary.json', {'status': 'complete', 'count': len(records),
        'best_epoch': checkpoint['epoch'], 'checkpoint_sha256': sha256_file(directory / 'best_weights.pt'),
        'output': str(destination), 'encoding': 'sum logits -> resize -> sigmoid -> per-image minmax -> round uint8'})
    print('PREDICTION COMPLETE: ' + str(destination), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'smoke', 'train', 'predict'))
    parser.add_argument('--project-root', type=Path, default=PACKAGE.parent)
    parser.add_argument('--suite-dir', type=Path)
    parser.add_argument('--variant', choices=('baseline', 'spatial', 'multidomain', 'spatial_frequency', 'spatial_wavelet'), default='multidomain')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Stop after this many additional epochs; checkpoint remains resumable.')
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 <= args.seed < 2**32 - 500000 or (args.stop_after is not None and args.stop_after < 1):
        parser.error('Invalid batch size, seed or stop-after.')
    if (args.resume or args.stop_after) and args.command != 'train':
        parser.error('--resume and --stop-after apply only to train.')
    root = activate_project(args.project_root)
    suite = args.suite_dir.resolve() if args.suite_dir else root / 'runs/polyp_pvt_md_v1'
    data_dir = suite / 'data'
    data, report, data_fp = load_development_data(data_dir)
    config = dict(DEFAULTS, variant=args.variant, seed=args.seed, batch_size=args.batch_size)
    fp = fingerprints(root, data_fp)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this environment. Run in the server polyp_pvt environment.')
    print('DEVICE: {} | Torch {} | train={} val={}'.format(
        torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU', torch.__version__,
        len(data['train']), len(data['val'])), flush=True)
    directory = suite / (args.variant + '_seed' + str(args.seed))
    if args.command == 'check':
        run_check(data, suite, args.seed, fp)
    elif args.command == 'smoke':
        run_smoke(args, data, config, suite, device)
    elif args.command == 'train':
        run_training(args, root, data, config, directory, fp, device)
    else:
        run_prediction(args, config, directory, fp, data_dir, device)


if __name__ == '__main__':
    main()
