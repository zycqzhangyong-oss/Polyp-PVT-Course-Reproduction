"""Evaluate Polyp-PVT maps with the UACANet Python metric implementation.

Python >= 3.8; dependencies: NumPy, SciPy, Pillow (already in polyp_pvt).
Run this file from the Windows computer that contains the actual predictions.
See README.txt and LICENSE-UACANet.txt for protocol and attribution.
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
import PIL
from PIL import Image

import uaca_eval_functions as metrics


DATASETS = ["CVC-300", "CVC-ClinicDB", "Kvasir", "CVC-ColonDB", "ETIS-LaribPolypDB"]
THRESHOLDS = np.linspace(1, 0, 256)
EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
COLUMNS = ["meanDic", "meanIoU", "wFm", "Sm", "meanEm", "mae", "maxEm",
           "maxDic", "maxIoU", "meanSen", "maxSen", "meanSpe", "maxSpe"]


def index_images(folder):
    if not folder.is_dir():
        raise ValueError("Folder not found: {}".format(folder))
    indexed = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in indexed:
                raise ValueError("Duplicate image stem in {}: {}".format(folder, path.stem))
            indexed[path.stem] = path
    if not indexed:
        raise ValueError("No image files in {}".format(folder))
    return indexed


def find_pairs(pred_root, gt_root, datasets):
    result = {}
    for name in datasets:
        preds = index_images(pred_root / name)
        gts = index_images(gt_root / name / "masks")
        missing = sorted(set(gts) - set(preds))
        extra = sorted(set(preds) - set(gts))
        print("{}: GT={}, Pred={}, Missing={}, Extra={}".format(
            name, len(gts), len(preds), len(missing), len(extra)), flush=True)
        if missing or extra:
            raise ValueError("{} name mismatch. Missing: {}; Extra: {}".format(
                name, missing[:10], extra[:10]))
        result[name] = [(str(preds[key]), str(gts[key])) for key in sorted(gts)]
    return result


def read_pair(pair):
    pred_path, gt_path = pair
    with Image.open(pred_path) as im:
        pred = np.array(im)
    with Image.open(gt_path) as im:
        gt = np.array(im)
    # Same first-channel convention as UACANet run/Eval.py.
    if pred.ndim == 3:
        pred = pred[:, :, 0]
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape != gt.shape:
        raise ValueError("Shape mismatch: {} {} vs {} {}".format(
            pred_path, pred.shape, gt_path, gt.shape))
    if pred.dtype != np.uint8 or gt.dtype != np.uint8:
        raise ValueError("Expected uint8 maps as in UACANet; check encoding: {} / {}".format(
            pred_path, gt_path))
    pred = pred.astype(np.float64) / 255.0
    gt = (gt.astype(np.float64) / 255.0 > 0.5).astype(np.float64)
    if not np.any(gt):
        raise ValueError("GT has no foreground after >127.5 threshold: {}. "
                         "Check whether this mask is encoded as 0/1 instead of 0/255. "
                         "The upstream weighted F metric does not define empty-GT handling.".format(gt_path))
    return pred, gt


def evaluate_one(pair):
    pred, gt = read_pair(pair)
    scalar = {"Sm": float(metrics.StructureMeasure(pred, gt)),
              "wFm": float(metrics.original_WFb(pred, gt)),
              "mae": float(np.mean(np.abs(gt - pred)))}
    curves = {key: np.empty(256, dtype=np.float64) for key in ("Dic", "IoU", "Em", "Sen", "Spe")}
    for j, threshold in enumerate(THRESHOLDS):
        _, recall, specificity, dice, _, iou = metrics.Fmeasure_calu(pred, gt, threshold)
        binary = np.zeros_like(pred)
        binary[pred >= threshold] = 1
        curves["Dic"][j] = dice
        curves["IoU"][j] = iou
        curves["Sen"][j] = recall
        curves["Spe"][j] = specificity
        curves["Em"][j] = metrics.EnhancedMeasure(binary, gt)
    values = list(scalar.values()) + [v for curve in curves.values() for v in curve]
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite metric for {}. Inspect this image and GT.".format(pair[0]))
    return {"stem": Path(pair[0]).stem, "scalar": scalar, "curves": curves}


def aggregate(samples):
    row = {key: float(np.mean([s["scalar"][key] for s in samples]))
           for key in ("Sm", "wFm", "mae")}
    for key in ("Dic", "IoU", "Em", "Sen", "Spe"):
        dataset_curve = np.mean(np.stack([s["curves"][key] for s in samples]), axis=0)
        row["mean" + key] = float(np.mean(dataset_curve))
        row["max" + key] = float(np.max(dataset_curve))
    return row


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_manifest(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--pred-root", type=Path, default=Path("result_map/PolypPVT"))
    parser.add_argument("--gt-root", type=Path, default=Path("dataset/TestDataset"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--check-only", action="store_true", help="Check matching names, encoding and image sizes only")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    root = args.project_root.resolve()
    pred_root = (root / args.pred_root).resolve()
    gt_root = (root / args.gt_root).resolve()
    print("Project: {}\nPredictions: {}\nGround truth: {}".format(root, pred_root, gt_root), flush=True)
    pairs = find_pairs(pred_root, gt_root, args.datasets)
    if args.check_only:
        for name, jobs in pairs.items():
            for job in jobs:
                read_pair(job)
            print("{}: all names, shapes and encodings OK".format(name), flush=True)
        print("CHECK PASSED. No metrics have been computed.", flush=True)
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (root / args.output_dir if args.output_dir else root / "evaluation_results" / ("uacanet_python_" + stamp)).resolve()
    # Never replace a previous evaluation run.
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "created_at": datetime.now().astimezone().isoformat(),
                "project_root": str(root), "pred_root": str(pred_root), "gt_root": str(gt_root),
                "datasets": {name: len(jobs) for name, jobs in pairs.items()},
                "implementation": "UACANet Python metrics; standalone Polyp-PVT wrapper",
                "upstream_metrics": "https://github.com/plemeri/UACANet/blob/main/utils/eval_functions.py",
                "upstream_protocol": "https://github.com/plemeri/UACANet/blob/main/run/Eval.py",
                "retrieved_on": "2026-10-11",
                "upstream_metrics_sha256": sha256(Path(metrics.__file__)),
                "wrapper_sha256": sha256(Path(__file__)),
                "normalization": "uint8 / 255; GT > 0.5; RGB uses first channel; no resizing or extra min-max normalization",
                "thresholds": "numpy.linspace(1, 0, 256); pred >= threshold",
                "aggregation": "Equal image weights, then mean/max over dataset threshold curve",
                "matlab_equivalence": "Not verified against the paper's MATLAB evaluator",
                "python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__, "scipy": scipy.__version__, "pillow": PIL.__version__,
                "workers": args.workers}
    manifest_file = output / "protocol.json"
    save_manifest(manifest_file, manifest)
    summary = []
    try:
        for name, jobs in pairs.items():
            print("\nEvaluating {} ({} images, 256 thresholds)...".format(name, len(jobs)), flush=True)
            samples = []
            executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
            try:
                iterator = executor.map(evaluate_one, jobs) if executor else map(evaluate_one, jobs)
                for i, sample in enumerate(iterator, 1):
                    samples.append(sample)
                    if i == 1 or i % 10 == 0 or i == len(jobs):
                        print("  {}: {}/{}".format(name, i, len(jobs)), flush=True)
            finally:
                if executor:
                    executor.shutdown(wait=True)
            row = {"dataset": name, "count": len(samples), **aggregate(samples)}
            summary.append(row)
            write_csv(output / "summary.csv", ["dataset", "count"] + COLUMNS, summary)
            per_image = []
            for sample, (pred_file, gt_file) in zip(samples, jobs):
                item = {"stem": sample["stem"], **sample["scalar"]}
                item.update({"mean" + key: float(np.mean(sample["curves"][key])) for key in ("Dic", "IoU", "Em")})
                item.update({"pred_file": pred_file, "gt_file": gt_file,
                             "pred_sha256": sha256(Path(pred_file)), "gt_sha256": sha256(Path(gt_file))})
                per_image.append(item)
            write_csv(output / (name + "_per_image.csv"),
                      ["stem", "meanDic", "meanIoU", "wFm", "Sm", "meanEm", "mae", "pred_file", "gt_file", "pred_sha256", "gt_sha256"], per_image)
            print("{}: meanDic={:.4f}, meanIoU={:.4f}, wFm={:.4f}, Sm={:.4f}, meanEm={:.4f}, MAE={:.4f}".format(
                name, row["meanDic"], row["meanIoU"], row["wFm"], row["Sm"], row["meanEm"], row["mae"]), flush=True)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        save_manifest(manifest_file, manifest)
    print("\n" + "dataset".ljust(21) + "count".rjust(7) + "meanDic".rjust(11) + "meanIoU".rjust(11), flush=True)
    for row in summary:
        print("{:<21}{:>7d}{:>11.4f}{:>11.4f}".format(row["dataset"], row["count"], row["meanDic"], row["meanIoU"]), flush=True)
    print("\nSaved: {}\nEVALUATION COMPLETE".format(output), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print("ERROR: {}".format(error), file=sys.stderr, flush=True)
        sys.exit(1)
