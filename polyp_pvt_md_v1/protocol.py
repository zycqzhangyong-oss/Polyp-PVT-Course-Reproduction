"""Frozen data manifests and explicit Polyp-PVT training/evaluation conventions."""
import csv
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from prepare_data import MASK_POLICY, sha256_file, read_image, read_gray_mask

SCHEMA = 'polyp_pvt_cim_multidomain_v1_gray_masks'
DATASETS = ('CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB')
DEFAULTS = dict(epochs=100, batch_size=16, seed=42, workers=0, lr=1e-4,
                weight_decay=1e-4, clip_value=0.5, size=352,
                loss='legacy_mean_bce_plus_weighted_iou', scales=[0.75, 1., 1.25],
                optimizer='AdamW', lr_schedule='constant_for_100_epochs',
                augmentation=False, mask_policy=MASK_POLICY,
                selection='mean_original_resolution_soft_dice_on_internal_validation')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refusing to write an empty result table.')
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_manifest(data_dir, name, report):
    path = Path(data_dir) / (name + '.csv')
    if sha256_file(path) != report['manifest_sha256'].get(path.name):
        raise ValueError('Frozen manifest changed: ' + path.name)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r['id'].casefold() for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate IDs in ' + name)
    return rows


def verify_rows(rows, root):
    root = Path(root).resolve()
    for row in rows:
        for key in ('image', 'mask'):
            path = Path(row[key]).resolve()
            if root not in path.parents or not path.is_file():
                raise ValueError('Missing or out-of-root dataset file: ' + str(path))
            if sha256_file(path) != row[key + '_sha256']:
                raise ValueError('Data changed after audit: ' + str(path))


def load_development_data(data_dir, expected_count=1450):
    report = load_json(Path(data_dir) / 'report.json')
    if report['status'] != 'passed_automated_checks':
        raise ValueError('Data preparation has not passed.')
    if report.get('mask_policy_id') != MASK_POLICY:
        raise ValueError('Data manifests use an older mask policy. Preserve the old report and prepare a fresh split.')
    tables = {name: read_manifest(data_dir, name, report) for name in ('train', 'val', 'train_pool')}
    if len(tables['train_pool']) != expected_count:
        raise ValueError('Training pool count changed.')
    pool = {r['id']: r for r in tables['train_pool']}
    train = {r['id']: r for r in tables['train']}
    val = {r['id']: r for r in tables['val']}
    if set(train) & set(val) or set(train) | set(val) != set(pool):
        raise ValueError('Train/validation IDs must be disjoint and cover the full pool.')
    for row in tables['train'] + tables['val']:
        if row != pool[row['id']]:
            raise ValueError('Split row differs from original pool: ' + row['id'])
    for key in ('rgb_sha256', 'image_sha256'):
        if {r[key] for r in tables['train']} & {r[key] for r in tables['val']}:
            raise ValueError('Training/validation pixel overlap.')
    tg = {r['group'] for r in tables['train'] if r['group']}
    vg = {r['group'] for r in tables['val'] if r['group']}
    if tg & vg:
        raise ValueError('Training/validation group overlap.')
    verify_rows(tables['train_pool'], report['train_root'])
    fingerprint = hashlib.sha256(json.dumps(tables, sort_keys=True).encode()).hexdigest()
    return tables, report, fingerprint


def input_tensor(image, size=352):
    resampling = getattr(Image, 'Resampling', Image)
    x = np.asarray(image.convert('RGB').resize((size, size), resampling.BILINEAR), dtype=np.float32) / 255.
    x = (x - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self, rows, size=352):
        self.rows, self.size = rows, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = read_image(row['image']).convert('RGB')
        gray = read_gray_mask(row['mask'])
        if image.size != gray.size or image.size != (int(row['width']), int(row['height'])):
            raise ValueError('Changed image/mask size: ' + row['id'])
        resampling = getattr(Image, 'Resampling', Image)
        # Match official transforms.Resize default BILINEAR for training masks.
        target = np.asarray(gray.resize((self.size, self.size), resampling.BILINEAR), dtype=np.float32) / 255.
        return input_tensor(image, self.size), torch.from_numpy(target[None].copy()), index


def make_loader(rows, config, epoch=0, training=True):
    generator = torch.Generator().manual_seed(config['seed'] + epoch)
    return torch.utils.data.DataLoader(ManifestDataset(rows, config['size']),
                                      batch_size=config['batch_size'] if training else 1,
                                      shuffle=training, num_workers=config['workers'],
                                      pin_memory=torch.cuda.is_available(), drop_last=False,
                                      generator=generator)


def structure_loss(logits, target):
    """Preserve public Train.py's ACTUAL reduce='none' behavior explicitly.

    The old boolean 'reduce' argument receives a truthy string, so BCE is a
    scalar mean; subsequent boundary weighting cancels. This is NOT paper wBCE.
    All variants use the same legacy mean-BCE + weighted-IoU objective.
    """
    weight = 1 + 5 * (F.avg_pool2d(target, 31, stride=1, padding=15) - target).abs()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='mean')
    probability = logits.sigmoid()
    intersection = (probability * target * weight).sum((2, 3))
    total = ((probability + target) * weight).sum((2, 3))
    wiou = 1 - (intersection + 1) / (total - intersection + 1)
    return bce + wiou.mean()


def scale_batch(images, masks, size, base_size):
    if size == base_size:
        return images, masks
    return (F.interpolate(images, (size, size), mode='bilinear', align_corners=True),
            F.interpolate(masks, (size, size), mode='bilinear', align_corners=True))


def probability_map(outputs, shape):
    logits = F.interpolate(outputs[0] + outputs[1], size=shape, mode='bilinear', align_corners=False)
    probability = logits.sigmoid()[0, 0].cpu().numpy()
    if not np.isfinite(probability).all():
        raise FloatingPointError('Nonfinite prediction.')
    return (probability - probability.min()) / (probability.max() - probability.min() + 1e-8)


def validation_scores(probability, gray):
    # Official Train.py normalizes native-resolution GT by its own maximum.
    target = np.asarray(gray, dtype=np.float32)
    target = target / (target.max() + 1e-8)
    soft_dice = (2 * (probability * target).sum() + 1) / (probability.sum() + target.sum() + 1)
    hard = probability >= .5
    hard_target = target >= .5
    inter = (hard & hard_target).sum()
    hard_dice = (2 * inter + 1) / (hard.sum() + hard_target.sum() + 1)
    return float(soft_dice), float(hard_dice)
