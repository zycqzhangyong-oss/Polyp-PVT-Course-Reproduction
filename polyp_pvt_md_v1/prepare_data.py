#!/usr/bin/env python3
"""Audit the existing polyp data and save a reproducible development split.

Requires Python >=3.8 and Pillow. Reads source images; never moves or rewrites them.
Place polyp_pvt_md_v1 inside the Polyp-PVT project root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

TEST_COUNTS = {"Kvasir": 100, "CVC-ClinicDB": 62, "CVC-ColonDB": 380,
               "ETIS-LaribPolypDB": 196, "CVC-300": 60}
ALIASES = {"Kvasir": ("Kvasir", "Kvasir-SEG"),
           "ETIS-LaribPolypDB": ("ETIS-LaribPolypDB", "ETIS")}
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_POLICY = "original_grayscale_train255_valmax_testthreshold_v1"
FIELDS = ["id", "dataset", "image", "mask", "width", "height", "mask_scale",
          "foreground_fraction", "soft_foreground_fraction", "intermediate_fraction",
          "mask_min", "mask_max", "gray_levels", "image_sha256", "mask_sha256", "rgb_sha256",
          "decoded_mask_sha256", "group"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_hash(image):
    digest = hashlib.sha256(f"{image.mode}:{image.width}:{image.height}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def unique_dir(base, names):
    matches = [base / name for name in names if (base / name).is_dir()]
    if len(matches) != 1:
        raise ValueError(f"{base}: expected exactly one directory from {names}; found {matches}")
    return matches[0]


def index_files(folder):
    result = {}
    for path in sorted(folder.iterdir()):
        if path.name.startswith("."):
            continue
        if path.is_dir():
            raise ValueError(f"Nested directory is unsupported: {path}")
        if path.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"Unexpected non-image file (move it outside the data folder): {path}")
        if path.stem.casefold() in {key.casefold() for key in result}:
            raise ValueError(f"Duplicate basename: {result[path.stem]} and {path}")
        result[path.stem] = path
    return result


def read_image(path):
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Multi-frame file is unsupported: {path}")
        if image.getexif().get(274, 1) != 1:
            raise ValueError(f"Non-default EXIF orientation requires checking image-mask alignment: {path}")
        image.load()
        return image.copy()


def read_gray_mask(path):
    """Read validated grayscale labels without changing their decoded values.

    Official Polyp-PVT uses convert('L'), not convert('1') or a threshold.
    Intermediate values in byte grayscale masks therefore remain intact.
    """
    original = read_image(path)
    if original.mode not in ("1", "L", "P", "RGB", "RGBA"):
        raise ValueError(f"Unsupported mask mode {original.mode}: {path}")
    if "A" in original.getbands() or "transparency" in original.info:
        if original.convert("RGBA").getchannel("A").getextrema() != (255, 255):
            raise ValueError(f"Mask has transparent pixels: {path}")
    red, green, blue = original.convert("RGB").split()
    if red.tobytes() != green.tobytes() or red.tobytes() != blue.tobytes():
        raise ValueError(f"Mask has colored labels; inspect its label mapping first: {path}")
    return original.convert("L")


def mask_statistics(gray):
    histogram = gray.histogram()
    pixels = gray.width * gray.height
    values = [value for value, count in enumerate(histogram) if count]
    return dict(mask_scale=255, foreground_fraction=sum(histogram[128:]) / pixels,
                soft_foreground_fraction=sum(i * n for i, n in enumerate(histogram)) / (255 * pixels),
                intermediate_fraction=sum(histogram[1:255]) / pixels,
                mask_min=min(values), mask_max=max(values), gray_levels=len(values))


def scan_dataset(base, name, expected):
    image_dir = unique_dir(base, ("images", "image"))
    mask_dir = unique_dir(base, ("masks", "mask"))
    images, masks = index_files(image_dir), index_files(mask_dir)
    if images.keys() != masks.keys():
        raise ValueError(f"{name}: pairing mismatch; missing masks={sorted(images.keys() - masks.keys())[:10]}, "
                         f"missing images={sorted(masks.keys() - images.keys())[:10]}")
    if len(images) != expected:
        raise ValueError(f"{name}: expected {expected} pairs, found {len(images)}")
    rows = []
    for index, stem in enumerate(sorted(images), 1):
        image_path, mask_path = images[stem], masks[stem]
        rgb = read_image(image_path).convert("RGB")
        gray = read_gray_mask(mask_path)
        stats = mask_statistics(gray)
        if 0 < stats['mask_max'] <= 1:
            raise ValueError(f"Ambiguous 0/1 byte mask encoding: {mask_path}. "
                             "Official training divides byte masks by 255; inspect the label scale. No auto-rescaling.")
        if rgb.size != gray.size:
            raise ValueError(f"{name}/{stem}: image {rgb.size} != mask {gray.size}")
        rows.append(dict(id=stem, dataset=name, image=str(image_path.resolve()),
                         mask=str(mask_path.resolve()), width=rgb.width, height=rgb.height,
                         **stats,
                         image_sha256=sha256_file(image_path), mask_sha256=sha256_file(mask_path),
                         rgb_sha256=pixel_hash(rgb), decoded_mask_sha256=pixel_hash(gray), group=""))
        if index % 100 == 0 or index == expected:
            print(f"{name}: {index}/{expected}", flush=True)
    return rows


def load_groups(path, rows):
    if path is None:
        return {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "group"} <= set(reader.fieldnames or []):
            raise ValueError("Group CSV requires columns image,group (image is filename including extension)")
        groups = {}
        for row in reader:
            name, group = row["image"], row["group"].strip()
            if not group or name in groups:
                raise ValueError(f"Empty or duplicate group entry: {name}")
            groups[name] = group
    names = {Path(row["image"]).name for row in rows}
    if set(groups) != names:
        raise ValueError("Group CSV must contain exactly one entry for every training-pool image")
    return groups


def split_rows(rows, fraction, seed, groups):
    """Keep supplied groups AND identical decoded images in the same partition."""
    rows = sorted(rows, key=lambda row: row["id"])
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        keys = [("pixel", row["rgb_sha256"])]
        if groups:
            keys.append(("group", groups[Path(row["image"]).name]))
        for key in keys:
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    components = {}
    for index, row in enumerate(rows):
        components.setdefault(find(index), []).append(row)
    units = sorted(components.values(), key=lambda unit: unit[0]["id"])
    if len(units) < 2:
        raise ValueError("All training images belong to one group; a disjoint validation split is impossible")
    random.Random(seed).shuffle(units)
    target = max(1, min(len(rows) - 1, round(len(rows) * fraction)))
    reachable = {0: None}
    for index, unit in enumerate(units):
        for total in list(reachable):
            new_total = total + len(unit)
            if new_total not in reachable:
                reachable[new_total] = (total, index)
        if target in reachable:
            break
    candidates = [total for total in reachable if 0 < total < len(rows)]
    if not candidates:
        raise ValueError("Cannot construct a nonempty disjoint validation set")
    total = min(candidates, key=lambda value: (abs(value - target), value))
    selected = set()
    while total:
        total, index = reachable[total]
        selected.add(index)
    train, val = [], []
    for index, unit in enumerate(units):
        for row in unit:
            row["group"] = groups.get(Path(row["image"]).name, "")
        (val if index in selected else train).extend(unit)
    return sorted(train, key=lambda row: row["id"]), sorted(val, key=lambda row: row["id"])


def duplicate_report(all_rows):
    by_hash = {}
    for row in all_rows:
        by_hash.setdefault(row["rgb_sha256"], []).append(row)
    return [dict(rgb_sha256=key, members=[dict(dataset=row["dataset"], id=row["id"],
                 image=row["image"], decoded_mask_sha256=row["decoded_mask_sha256"]) for row in rows])
            for key, rows in by_hash.items() if len(rows) > 1]


def write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def preview(rows, output, seed, stem="preview_10"):
    sample = random.Random(seed).sample(rows, min(10, len(rows)))
    cell_w, cell_h = 256, 192
    canvas = Image.new("RGB", (cell_w * 3, 34 + len(sample) * (cell_h + 26)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(("Image", "Original grayscale mask", "Raw-mask overlay (green)")):
        draw.text((column * cell_w + 8, 8), label, fill="black")
    resampling = getattr(Image, "Resampling", Image)
    for index, row in enumerate(sample):
        rgb = read_image(Path(row["image"])).convert("RGB")
        mask = read_gray_mask(Path(row["mask"]))
        tinted = Image.blend(rgb, Image.new("RGB", rgb.size, (0, 255, 80)), 0.4)
        overlay = Image.composite(tinted, rgb, mask)
        top = 34 + index * (cell_h + 26)
        for column, image in enumerate((rgb, mask.convert("RGB"), overlay)):
            image.thumbnail((cell_w, cell_h), resampling.NEAREST if column == 1 else resampling.LANCZOS)
            canvas.paste(image, (column * cell_w + (cell_w - image.width) // 2, top + (cell_h - image.height) // 2))
        label = (f"{index + 1:02d}  {row['id']}  FG>0.5={row['foreground_fraction']:.3f}"
                 f"  gray={row['intermediate_fraction']:.3%}")
        draw.text((8, top + cell_h + 4), label.encode("ascii", "replace").decode()[:110], fill="black")
    canvas.save(output / (stem + ".png"))
    write_csv(output / (stem + ".csv"), sample)


def prepare(train_root, test_root, output, seed=42, val_fraction=0.1, group_csv=None,
            expected_train=1450, test_counts=None):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    train_root, test_root, output = (Path(p).expanduser().resolve() for p in (train_root, test_root, output))
    for root in (train_root, test_root):
        if not root.is_dir():
            raise FileNotFoundError(f"Data directory not found: {root}")
        if output == root or root in output.parents:
            raise ValueError("Output must be outside the source dataset directories")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", created_utc=datetime.now(timezone.utc).isoformat(),
                  mask_policy_id=MASK_POLICY,
                  train_root=str(train_root), test_root=str(test_root), seed=seed,
                  requested_val_fraction=val_fraction, script_sha256=sha256_file(Path(__file__)),
                  group_csv=str(Path(group_csv).resolve()) if group_csv else None,
                  warnings=[], limits=["Only exact decoded RGB duplicates are checked; near-duplicates are not detected.",
                                       "Geometric annotation alignment requires manual review of the previews.",
                                       "No patient/video independence is claimed for the public train/test split."])
    write_json(output / "report.json", report)
    try:
        report["group_csv_sha256"] = sha256_file(group_csv) if group_csv else None
        pool = scan_dataset(train_root, "train_pool", expected_train)
        tests = {}
        for name, expected in (TEST_COUNTS if test_counts is None else test_counts).items():
            folder = unique_dir(test_root, (name,))
            if not (folder / 'images').is_dir() or not (folder / 'masks').is_dir():
                raise ValueError('The bundled evaluator expects canonical test images/ and masks/: ' + str(folder))
            tests[name] = scan_dataset(folder, name, expected)
            for row in tests[name]:
                # Fail before training if the bundled evaluator cannot read GT.
                if row['foreground_fraction'] == 0:
                    raise ValueError('The unchanged UACANet evaluator needs foreground after GT/255 > 0.5: ' + row['mask'])
                with Image.open(row['mask']) as original:
                    if original.mode not in ('L', 'RGB', 'RGBA'):
                        raise ValueError('Test mask storage must be uint8 grayscale/RGB for the unchanged evaluator: ' + row['mask'])
        all_rows = pool + [row for rows in tests.values() for row in rows]
        report['mask_summary'] = {}
        for name, rows in [('train_pool', pool)] + list(tests.items()):
            gray_rows = [r for r in rows if r['intermediate_fraction'] > 0]
            report['mask_summary'][name] = dict(
                total=len(rows), masks_with_intermediate_values=len(gray_rows),
                source_values_preserved=True,
                top10=[dict(id=r['id'], fraction=r['intermediate_fraction'],
                            gray_levels=r['gray_levels'], mask_min=r['mask_min'], mask_max=r['mask_max'])
                       for r in sorted(gray_rows, key=lambda r: r['intermediate_fraction'], reverse=True)[:10]])
            print(f"MASK AUDIT {name}: intermediate-gray masks={len(gray_rows)}/{len(rows)}; source values preserved", flush=True)
        if report['mask_summary']['train_pool']['masks_with_intermediate_values']:
            report['warnings'].append('Training source masks include intermediate grayscale values, preserved as in official Polyp-PVT. Inspect preview_gray_masks.png; statistics alone do not verify annotation quality or establish the cause of gray pixels.')
        duplicates = duplicate_report(all_rows)
        write_json(output / "duplicates.json", duplicates)
        report["counts"] = {"train_pool": len(pool), **{key: len(rows) for key, rows in tests.items()}}
        for duplicate in duplicates:
            members = duplicate["members"]
            datasets = {row["dataset"] for row in members}
            if "train_pool" in datasets and len(datasets) > 1:
                raise ValueError("Exact train/test image overlap found; inspect duplicates.json before training")
            if "train_pool" in datasets and len({row["decoded_mask_sha256"] for row in members}) > 1:
                raise ValueError("Identical training images have conflicting masks; inspect duplicates.json")
        if duplicates:
            report["warnings"].append("Exact duplicates are listed in duplicates.json; training duplicates stay together, no files are removed.")
        empty_or_full = [dict(dataset=row["dataset"], id=row["id"], fraction=row["soft_foreground_fraction"])
                         for row in all_rows if row["soft_foreground_fraction"] in (0, 1)]
        report["empty_or_full_masks"] = empty_or_full
        if empty_or_full:
            report["warnings"].append("Empty/full masks require inspection; see empty_or_full_masks.")
        groups = load_groups(group_csv, pool)
        train, val = split_rows(pool, val_fraction, seed, groups)
        report["split_method"] = "supplied group IDs plus exact-image groups" if groups else "image-level random split with exact-image groups"
        if not groups:
            report["warnings"].append("No patient/video IDs provided; correlated frames can cross training/validation. No source-stratification is inferred from filenames.")
        report["counts"].update(train=len(train), val=len(val))
        report["actual_val_fraction"] = len(val) / len(pool)
        report["mask_policy"] = ("Source grayscale values preserved with convert('L'). Training: PIL bilinear resize then /255, no threshold or per-image maximum normalization. Internal validation selection: native-resolution mask/(mask.max()+1e-8), no threshold; auxiliary hard Dice thresholds both normalized GT and prediction at >=0.5. Final UACANet benchmark evaluation unchanged: native byte GT/255 >0.5. Foreground fraction in audit uses >0.5 for description only. Source files unchanged.")
        preview(pool, output, seed)
        gray_examples = sorted((r for r in pool if r['intermediate_fraction'] > 0),
                               key=lambda r: r['intermediate_fraction'], reverse=True)[:10]
        if gray_examples:
            preview(gray_examples, output, seed, stem='preview_gray_masks')
        write_csv(output / "train.csv", train)
        write_csv(output / "val.csv", val)
        write_csv(output / "train_pool.csv", pool)
        for name, rows in tests.items():
            write_csv(output / f"test_{name}.csv", rows)
        report["manifest_sha256"] = {path.name: sha256_file(path) for path in sorted(output.glob("*.csv"))}
        report["status"] = "passed_automated_checks"
        write_json(output / "report.json", report)
        print(f"DATA CHECK PASSED | train={len(train)} val={len(val)} test={sum(len(rows) for rows in tests.values())}", flush=True)
        print(f"Output: {output}\nNext: inspect preview_10.png and report.json warnings; training has NOT started.", flush=True)
        for warning in report["warnings"]:
            print(f"WARNING: {warning}", flush=True)
        return report
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "report.json", report)
        raise


def main():
    base = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=base / "dataset" / "TrainDataset")
    parser.add_argument("--test-root", type=Path, default=base / "dataset" / "TestDataset")
    parser.add_argument("--output", type=Path, default=base / "runs" / "polyp_pvt_md_v1" / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--groups-csv", type=Path, help="Optional complete training-pool CSV with columns image,group")
    args = parser.parse_args()
    try:
        prepare(args.train_root, args.test_root, args.output, args.seed, args.val_fraction, args.groups_csv)
    except Exception as error:
        print(f"DATA CHECK FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
