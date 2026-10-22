"""
convert_datasets.py
===================
SIH26008 — Conveyor Belt Health Monitoring AI

Converts all datasets into a single standard YOLO format:
  class_id  x_center  y_center  width  height   (all normalised 0-1)

Operations:
  DS1 (OBB)          → axis-aligned bbox from 4 corner points
  DS2 (Standard)     → copy as-is after validation
  DS3 (Segmentation) → axis-aligned bbox from polygon vertices

Filtering (DS1):
  Remove: Human, Other Objects, Roller  (but keep images as negatives if needed)

Output: datasets/converted/DS1/  DS2/  DS3/
        Each with images/ and labels/ sub-folders for train/valid/test.

Run ONLY after inspect_datasets.py confirms formats.
"""

import os, sys, shutil, zipfile, tempfile, json
from pathlib import Path
from collections import defaultdict, Counter

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

ROOT         = Path(__file__).parent.parent
OUT_ROOT     = Path(__file__).parent / "datasets" / "converted"
REPORTS_DIR  = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

DATASETS = {
    "DS1_OBB":      ROOT / "Conveyor-belt-damage 3.v1i.yolov8-obb.zip",
    "DS2_STANDARD": ROOT / "CONVEYOR BELT 2.v2i.yolov8.zip",
    "DS3_SEG":      ROOT / "Conveyor-Belt-Defects 1.v2i.yolov8.zip",
}

# ── DS1 class filtering config ──────────────────────────────────────────────
# DS1 declared classes (from data.yaml — verify with audit):
# 0:Hole  1:Human  2:Other Objects  3:Puncture  4:Roller  5:Tear
# 6:impact damage  7:patch work
DS1_REMOVE_CLASS_NAMES = {"human", "other objects", "roller"}
# DS1 unified mapping (after removal):
# Hole→hole, Puncture→puncture, Tear→tear, impact damage→impact_damage, patch work→patch_work
# These will be remapped to final class IDs during merge step.

# ── DS3 class names ──────────────────────────────────────────────────────────
# DS3: crack, hole-tear  (segmentation polygons)


def obb_to_aabb(x1,y1,x2,y2,x3,y3,x4,y4):
    """OBB corner points → axis-aligned bbox (xmin,ymin,xmax,ymax) clamped 0-1."""
    xs = [x1,x2,x3,x4]; ys = [y1,y2,y3,y4]
    xmin,xmax = max(0.0,min(xs)), min(1.0,max(xs))
    ymin,ymax = max(0.0,min(ys)), min(1.0,max(ys))
    return xmin,ymin,xmax,ymax

def polygon_to_aabb(coords):
    """Polygon (x1,y1,x2,y2,...) → axis-aligned bbox, clamped 0-1."""
    xs = coords[0::2]; ys = coords[1::2]
    xmin,xmax = max(0.0,min(xs)), min(1.0,max(xs))
    ymin,ymax = max(0.0,min(ys)), min(1.0,max(ys))
    return xmin,ymin,xmax,ymax

def aabb_to_yolo(xmin,ymin,xmax,ymax):
    """Axis-aligned bbox → YOLO cx cy w h (normalised)."""
    cx = (xmin+xmax)/2; cy = (ymin+ymax)/2
    w  = xmax-xmin;     h  = ymax-ymin
    return cx,cy,w,h

def read_yaml(path):
    if not HAS_YAML:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}

def convert_ds1_obb(zip_path, out_dir):
    """Convert DS1 OBB → standard YOLO, filtering unwanted classes."""
    print(f"\n[DS1] Converting OBB dataset: {zip_path.name}")
    stats = {
        "removed_labels": Counter(),
        "kept_labels": Counter(),
        "background_images": 0,
        "total_images": 0,
        "total_label_files": 0,
    }

    tmp = Path(tempfile.mkdtemp(prefix="sih_ds1_"))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)

    # Read class names
    yaml_files = list(tmp.rglob("data.yaml")) + list(tmp.rglob("*.yaml"))
    class_names = []
    if yaml_files:
        d = read_yaml(yaml_files[0])
        names_data = d.get("names", [])
        if isinstance(names_data, dict):
            max_id = max(names_data.keys()) if names_data else -1
            class_names = [names_data.get(i, "") for i in range(max_id + 1)]
        elif isinstance(names_data, list):
            class_names = names_data
    print(f"  DS1 classes: {class_names}")

    # Build remove set by index
    remove_ids = {i for i, n in enumerate(class_names)
                  if str(n).lower() in DS1_REMOVE_CLASS_NAMES}
    print(f"  Remove class IDs: {remove_ids} ({[class_names[i] for i in remove_ids if i < len(class_names)]})")

    # Find all label files in all splits
    for label_file in tmp.rglob("*.txt"):
        if "label" not in str(label_file).lower() or label_file.name.lower().startswith("readme"):
            continue
        stats["total_label_files"] += 1
        lines = label_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        kept_rows = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            try:
                cls_id = int(parts[0])
            except ValueError:
                continue
            if cls_id in remove_ids:
                name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                stats["removed_labels"][name] += 1
                continue
            # OBB → standard
            if len(parts) == 9:
                coords = list(map(float, parts[1:]))
                xmin,ymin,xmax,ymax = obb_to_aabb(*coords)
                cx,cy,w,h = aabb_to_yolo(xmin,ymin,xmax,ymax)
                if w <= 0 or h <= 0:
                    continue
                kept_rows.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                stats["kept_labels"][name] += 1
            else:
                print(f"  ⚠️  Unexpected OBB row length {len(parts)} in {label_file.name}")

        # Determine output path
        # Preserve split structure: find relative parts
        rel_parts = label_file.relative_to(tmp).parts
        out_label_path = out_dir / Path(*rel_parts)
        out_label_path.parent.mkdir(parents=True, exist_ok=True)

        if not kept_rows:
            stats["background_images"] += 1
            out_label_path.write_text("")  # empty = background
        else:
            out_label_path.write_text("\n".join(kept_rows) + "\n", encoding="utf-8")

    # Copy images (no modification)
    for img_file in tmp.rglob("*"):
        if img_file.suffix.lower() not in {".jpg",".jpeg",".png",".bmp"}:
            continue
        stats["total_images"] += 1
        rel_parts = img_file.relative_to(tmp).parts
        out_img_path = out_dir / Path(*rel_parts)
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_file, out_img_path)

    # Copy yaml
    for yf in yaml_files:
        shutil.copy2(yf, out_dir / yf.name)

    shutil.rmtree(tmp, ignore_errors=True)

    print(f"  ✅ DS1 done.")
    print(f"     Removed labels: {dict(stats['removed_labels'])}")
    print(f"     Kept labels   : {dict(stats['kept_labels'])}")
    print(f"     Background images (became empty): {stats['background_images']}")
    return stats


def convert_ds2_standard(zip_path, out_dir):
    """Copy DS2 as-is (standard YOLO). Validate labels."""
    print(f"\n[DS2] Copying standard YOLO dataset: {zip_path.name}")
    stats = {"total_images": 0, "total_labels": 0, "issues": []}

    tmp = Path(tempfile.mkdtemp(prefix="sih_ds2_"))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)

    for f in tmp.rglob("*"):
        if not f.is_file():
            continue
        rel_parts = f.relative_to(tmp).parts
        out_path = out_dir / Path(*rel_parts)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out_path)
        if f.suffix.lower() in {".jpg",".jpeg",".png"}:
            stats["total_images"] += 1
        elif f.suffix == ".txt" and "label" in str(f).lower() and not f.name.lower().startswith("readme"):
            stats["total_labels"] += 1

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  ✅ DS2 done. Images: {stats['total_images']}  Labels: {stats['total_labels']}")
    return stats


def convert_ds3_segmentation(zip_path, out_dir):
    """Convert DS3 segmentation polygons and standard bboxes → axis-aligned YOLO bbox."""
    print(f"\n[DS3] Converting segmentation/bbox dataset: {zip_path.name}")
    stats = {"total_images":0, "total_labels":0, "converted_polygons":0, "standard_boxes":0}

    tmp = Path(tempfile.mkdtemp(prefix="sih_ds3_"))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)

    yaml_files = list(tmp.rglob("data.yaml")) + list(tmp.rglob("*.yaml"))
    class_names = []
    if yaml_files:
        d = read_yaml(yaml_files[0])
        names_data = d.get("names", [])
        if isinstance(names_data, dict):
            max_id = max(names_data.keys()) if names_data else -1
            class_names = [names_data.get(i, "") for i in range(max_id + 1)]
        elif isinstance(names_data, list):
            class_names = names_data
    print(f"  DS3 classes: {class_names}")

    for label_file in tmp.rglob("*.txt"):
        if "label" not in str(label_file).lower() or label_file.name.lower().startswith("readme"):
            continue
        lines = label_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        converted = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            if len(parts) == 5:
                # Already standard YOLO bbox: cls cx cy w h
                try:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    if 0 < w <= 1.0 and 0 < h <= 1.0:
                        converted.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                        stats["standard_boxes"] += 1
                except ValueError:
                    continue
            elif len(parts) > 5 and len(parts) % 2 == 1:
                # Segmentation polygon: cls x1 y1 x2 y2 ...
                try:
                    cls_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                except ValueError:
                    continue
                xmin,ymin,xmax,ymax = polygon_to_aabb(coords)
                cx,cy,w,h = aabb_to_yolo(xmin,ymin,xmax,ymax)
                if w <= 0 or h <= 0:
                    continue
                converted.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                stats["converted_polygons"] += 1

        rel_parts = label_file.relative_to(tmp).parts
        out_path = out_dir / Path(*rel_parts)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
        stats["total_labels"] += 1

    for img_file in tmp.rglob("*"):
        if img_file.suffix.lower() not in {".jpg",".jpeg",".png",".bmp"}:
            continue
        stats["total_images"] += 1
        rel_parts = img_file.relative_to(tmp).parts
        out_path = out_dir / Path(*rel_parts)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_file, out_path)

    for yf in yaml_files:
        shutil.copy2(yf, out_dir / yf.name)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  ✅ DS3 done. Images: {stats['total_images']}  Polygons: {stats['converted_polygons']}  Standard boxes: {stats['standard_boxes']}")
    return stats


if __name__ == "__main__":
    print("=" * 70)
    print("  SIH26008 — DATASET FORMAT CONVERSION")
    print("  NOTE: Run inspect_datasets.py first and confirm class mappings.")
    print("=" * 70)

    if not HAS_YAML:
        print("⚠️  pyyaml not installed. Some YAML reading will be skipped.")

    ds1_out = OUT_ROOT / "DS1_OBB"
    ds2_out = OUT_ROOT / "DS2_STANDARD"
    ds3_out = OUT_ROOT / "DS3_SEG"

    ds1_out.mkdir(parents=True, exist_ok=True)
    ds2_out.mkdir(parents=True, exist_ok=True)
    ds3_out.mkdir(parents=True, exist_ok=True)

    stats1 = convert_ds1_obb(DATASETS["DS1_OBB"], ds1_out)
    stats2 = convert_ds2_standard(DATASETS["DS2_STANDARD"], ds2_out)
    stats3 = convert_ds3_segmentation(DATASETS["DS3_SEG"], ds3_out)

    report = {
        "DS1": stats1,
        "DS2": stats2,
        "DS3": stats3,
    }
    report_path = REPORTS_DIR / "conversion_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n📄 Conversion report: {report_path}")
    print("\n✅ All datasets converted. Run merge_datasets.py next.")
