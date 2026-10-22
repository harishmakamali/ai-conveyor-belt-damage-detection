# -*- coding: utf-8 -*-
"""
inspect_datasets.py
===================
SIH26008 -- Conveyor Belt Health Monitoring AI
Audit Step: Inspect all datasets, models, and notebooks WITHOUT modifying originals.

Run: python inspect_datasets.py
Output: reports/audit_report.txt + console
"""

import os
import sys
# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import json
import zipfile
import shutil
import tempfile
import textwrap
from pathlib import Path
from collections import defaultdict, Counter

# ── Optional imports ────────────────────────────────────────────────────────
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent  # Conveyor Belt Project/
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

DATASETS = {
    "DS1_OBB":      ROOT / "Conveyor-belt-damage 3.v1i.yolov8-obb.zip",
    "DS2_STANDARD": ROOT / "CONVEYOR BELT 2.v2i.yolov8.zip",
    "DS3_SEG":      ROOT / "Conveyor-Belt-Defects 1.v2i.yolov8.zip",
}

MODELS = {
    "old_five_class": ROOT / "old_five_class_model.pt",
    "best_tear":      ROOT / "best.pt",
}

REPORT_LINES = []

# ── Helpers ──────────────────────────────────────────────────────────────────
def log(msg=""):
    print(msg)
    REPORT_LINES.append(msg)

def section(title):
    bar = "=" * 70
    log()
    log(bar)
    log(f"  {title}")
    log(bar)

def subsection(title):
    log()
    log(f"  ── {title} ──")

def count_files(directory, extension):
    return sum(1 for _ in Path(directory).rglob(f"*{extension}"))

def read_label_file(path):
    """Return list of rows; each row is a list of floats."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                rows.append(parts)  # keep as-is if unparseable
    return rows

def classify_label_format(rows):
    """
    Classify label rows as:
      - 'standard'     → 5 values  (cls cx cy w h)
      - 'obb'          → 9 values  (cls x1 y1 x2 y2 x3 y3 x4 y4)
      - 'segmentation' → 5+ values (cls x1 y1 x2 y2 ...)  where count is odd
      - 'empty'        → no rows
      - 'mixed'        → inconsistent
    """
    if not rows:
        return "empty"
    counts = set(len(r) for r in rows)
    if counts == {5}:
        return "standard"
    if counts == {9}:
        return "obb"
    if all(len(r) >= 5 and len(r) % 2 == 1 for r in rows):
        return "segmentation"
    return f"mixed({sorted(counts)})"

def validate_standard_labels(rows, filename):
    """Check for out-of-range or negative values in standard YOLO labels."""
    issues = []
    for i, row in enumerate(rows):
        if len(row) != 5:
            issues.append(f"  Row {i}: expected 5 values, got {len(row)}")
            continue
        cls, cx, cy, w, h = row
        if not (0.0 <= cx <= 1.0):
            issues.append(f"  Row {i}: cx={cx:.4f} out of [0,1]")
        if not (0.0 <= cy <= 1.0):
            issues.append(f"  Row {i}: cy={cy:.4f} out of [0,1]")
        if not (0.0 < w <= 1.0):
            issues.append(f"  Row {i}: w={w:.4f} out of (0,1]")
        if not (0.0 < h <= 1.0):
            issues.append(f"  Row {i}: h={h:.4f} out of (0,1]")
        if int(cls) < 0:
            issues.append(f"  Row {i}: negative class id {cls}")
    return issues

def validate_obb_labels(rows, filename):
    """Check OBB rows for out-of-range coords."""
    issues = []
    for i, row in enumerate(rows):
        if len(row) != 9:
            issues.append(f"  Row {i}: expected 9 values, got {len(row)}")
            continue
        cls = row[0]
        coords = row[1:]
        for j, v in enumerate(coords):
            # OBB coords are normalised, should be 0-1
            if not (0.0 <= v <= 1.0):
                issues.append(f"  Row {i}, coord {j}: {v:.4f} out of [0,1]")
    return issues

def sample_label_files(label_dir, n=3):
    """Return up to n label file paths from the directory."""
    files = list(Path(label_dir).rglob("*.txt"))
    # prefer non-empty files
    non_empty = [f for f in files if f.stat().st_size > 0]
    sample = non_empty[:n] if non_empty else files[:n]
    return sample

# ── Model inspection ─────────────────────────────────────────────────────────
def inspect_model(name, path):
    subsection(f"Model: {name}")
    log(f"  Path : {path}")
    log(f"  Exists: {path.exists()}")
    if not path.exists():
        log("  ❌ File not found — skipping")
        return

    size_mb = path.stat().st_size / 1024 / 1024
    log(f"  Size : {size_mb:.2f} MB")

    if not HAS_TORCH:
        log("  ⚠️  torch not installed — cannot inspect model internals")
        return

    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            log(f"  Checkpoint keys: {list(ckpt.keys())}")
            if "model" in ckpt:
                model_obj = ckpt["model"]
                # Try to get number of classes
                nc = None
                if hasattr(model_obj, "nc"):
                    nc = model_obj.nc
                elif hasattr(model_obj, "model"):
                    for m in model_obj.model.modules():
                        if hasattr(m, "nc"):
                            nc = m.nc
                            break
                log(f"  Number of classes (nc): {nc}")
                # Try to get class names
                names = ckpt.get("train_args", {}).get("names") or \
                        getattr(model_obj, "names", None)
                if names:
                    if isinstance(names, dict):
                        log(f"  Class names ({len(names)}): {dict(names)}")
                    else:
                        log(f"  Class names ({len(names)}): {list(names)}")
                # epochs trained
                epoch = ckpt.get("epoch", "unknown")
                log(f"  Epoch saved at  : {epoch}")
                # training args
                args = ckpt.get("train_args", {})
                if args:
                    for k in ["imgsz", "batch", "data", "optimizer", "lr0", "epochs"]:
                        if k in args:
                            log(f"  train_args.{k}  : {args[k]}")
            # metrics
            metrics = ckpt.get("metrics", {})
            if metrics:
                log(f"  Saved metrics   : {metrics}")
        else:
            log(f"  Checkpoint type : {type(ckpt)}")
    except Exception as e:
        log(f"  ⚠️  Could not fully load checkpoint: {e}")


# ── Dataset inspection ────────────────────────────────────────────────────────
def inspect_dataset_zip(ds_key, zip_path):
    section(f"DATASET: {ds_key}")
    log(f"  ZIP : {zip_path.name}")
    log(f"  Size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")

    if not zip_path.exists():
        log("  ❌ ZIP not found — skipping")
        return

    # Extract to temp dir
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"sih_{ds_key}_"))
    log(f"  Extracting to temp: {str(tmp_dir)[:80]}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    # ── Identify structure ────────────────────────────────────────────────
    all_files = list(tmp_dir.rglob("*"))
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    label_exts = {".txt"}
    yaml_files = [f for f in all_files if f.suffix in {".yaml", ".yml"}]
    label_files = [f for f in all_files if f.suffix in label_exts]
    image_files = [f for f in all_files if f.suffix.lower() in image_exts]

    log(f"\n  Total extracted files : {len(all_files)}")
    log(f"  Image files           : {len(image_files)}")
    log(f"  Label files           : {len(label_files)}")
    log(f"  YAML files            : {[f.name for f in yaml_files]}")

    # ── Read data.yaml ─────────────────────────────────────────────────────
    subsection("data.yaml content")
    yaml_data = {}
    for yf in yaml_files:
        log(f"\n  File: {yf.relative_to(tmp_dir)}")
        try:
            with open(yf) as f:
                content = f.read()
            log(content)
            if HAS_YAML:
                yaml_data = yaml.safe_load(content) or {}
        except Exception as e:
            log(f"  ⚠️  Could not read YAML: {e}")

    nc = yaml_data.get("nc", "?")
    class_names = yaml_data.get("names", [])
    log(f"\n  Declared classes (nc={nc}): {class_names}")

    # ── Split counts ───────────────────────────────────────────────────────
    subsection("Per-split image and label counts")
    splits = ["train", "valid", "val", "test"]
    for split in splits:
        # find split folders
        split_dirs = [d for d in tmp_dir.rglob("*") if d.is_dir() and d.name == split]
        if not split_dirs:
            continue
        for sd in split_dirs:
            imgs = list(sd.rglob("*.jpg")) + list(sd.rglob("*.jpeg")) + list(sd.rglob("*.png"))
            # Try to find labels sibling folder
            labels_dir = sd.parent.parent / "labels" / sd.name \
                         if "images" in str(sd) \
                         else sd.parent / "labels" / sd.name
            if not labels_dir.exists():
                labels_dir = sd.parent / split  # fallback
            lbl_count = len(list(sd.rglob("*.txt"))) if sd.name not in {"images"} else 0
            # proper label dir
            lbl_path = tmp_dir
            for candidate in [
                tmp_dir / "labels" / split,
                list(tmp_dir.rglob(f"labels/{split}"))[0] if list(tmp_dir.rglob(f"labels/{split}")) else None,
            ]:
                if candidate and Path(candidate).exists():
                    lbl_path = candidate
                    break
            lbl_files_here = list(Path(lbl_path).glob("*.txt")) if lbl_path != tmp_dir else []
            log(f"  [{split}] images: {len(imgs)}   labels: {len(lbl_files_here)}")

    # ── Label format detection ────────────────────────────────────────────
    subsection("Label format detection (sample 10 files)")
    format_counter = Counter()
    all_label_paths = list(tmp_dir.rglob("*.txt"))
    # exclude data.yaml-style txt
    all_label_paths = [f for f in all_label_paths if "label" in str(f).lower() or f.parent.name in ["labels","train","valid","val","test"]]
    for lp in all_label_paths[:200]:  # scan first 200 for format
        rows = read_label_file(lp)
        fmt = classify_label_format(rows)
        format_counter[fmt] += 1
    log(f"  Label format distribution (from up to 200 files): {dict(format_counter)}")

    # ── Class distribution ────────────────────────────────────────────────
    subsection("Class distribution across ALL label files")
    class_counter = Counter()
    empty_label_count = 0
    invalid_label_count = 0
    total_bboxes = 0
    all_issues = []

    for lp in all_label_paths:
        rows = read_label_file(lp)
        if not rows:
            empty_label_count += 1
            continue
        fmt = classify_label_format(rows)
        for row in rows:
            if not row:
                continue
            try:
                cls_id = int(row[0])
                class_counter[cls_id] += 1
                total_bboxes += 1
            except (ValueError, IndexError):
                invalid_label_count += 1

        # Validate
        if fmt == "standard":
            issues = validate_standard_labels(rows, lp.name)
        elif fmt == "obb":
            issues = validate_obb_labels(rows, lp.name)
        else:
            issues = []
        if issues:
            all_issues.append((lp.relative_to(tmp_dir), issues[:3]))  # cap at 3 per file

    log(f"  Total bounding boxes / polygons : {total_bboxes}")
    log(f"  Empty label files               : {empty_label_count}")
    log(f"  Invalid/unparseable rows        : {invalid_label_count}")
    log(f"\n  Class ID → count:")
    for cid in sorted(class_counter):
        name = class_names[cid] if isinstance(class_names, list) and cid < len(class_names) else "?"
        log(f"    [{cid}] {name:30s} : {class_counter[cid]}")

    # ── Sample label files ─────────────────────────────────────────────────
    subsection("Sample label file contents (first 3 non-empty)")
    samples = sample_label_files(tmp_dir, n=3)
    for sf in samples:
        log(f"\n  File: {sf.relative_to(tmp_dir)}")
        rows = read_label_file(sf)
        fmt = classify_label_format(rows)
        log(f"  Format: {fmt}  | Rows: {len(rows)}")
        for r in rows[:5]:
            log(f"    {r}")

    # ── Issues summary ─────────────────────────────────────────────────────
    if all_issues:
        subsection(f"Coordinate/format issues (first {min(5, len(all_issues))} files)")
        for fpath, issues in all_issues[:5]:
            log(f"  {fpath}")
            for iss in issues:
                log(f"  {iss}")

    # ── Cleanup ────────────────────────────────────────────────────────────
    shutil.rmtree(tmp_dir, ignore_errors=True)
    log(f"\n  Temp dir cleaned up.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    section("SIH26008 — CONVEYOR BELT AI — FULL DATASET & MODEL AUDIT")
    log(f"  Python    : {sys.version}")
    log(f"  torch     : {'installed [OK]' if HAS_TORCH else 'NOT installed [WARN]'}")
    log(f"  pyyaml    : {'installed [OK]' if HAS_YAML else 'NOT installed [WARN]'}")
    log(f"  Root dir  : {ROOT}")

    # ── Models ────────────────────────────────────────────────────────────
    section("MODEL INSPECTION")
    for name, path in MODELS.items():
        inspect_model(name, path)

    # ── Datasets ──────────────────────────────────────────────────────────
    for key, zpath in DATASETS.items():
        inspect_dataset_zip(key, zpath)

    # ── Summary table ─────────────────────────────────────────────────────
    section("AUDIT COMPLETE — Summary")
    log("""
  Dataset                       | Format       | Notes
  ------------------------------|--------------|-------------------------------
  DS1 Conveyor-belt-damage v1i  | OBB          | Needs conversion; filter 3 classes
  DS2 CONVEYOR BELT v2i         | Standard     | Ready to use; 1-class (tear)
  DS3 Conveyor-Belt-Defects v2i | Segmentation | Needs polygon→bbox conversion
  old_five_class_model.pt       | YOLOv8n      | 5 classes; inspect head before reuse
  best.pt                       | YOLOv8n      | 1 class (tear)
  """)

    # ── Write report ──────────────────────────────────────────────────────
    report_path = REPORTS_DIR / "audit_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT_LINES))
    print(f"\n  📄 Full audit report saved: {report_path}")
