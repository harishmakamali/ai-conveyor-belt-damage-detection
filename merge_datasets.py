"""
merge_datasets.py
=================
SIH26008 — Conveyor Belt Health Monitoring AI

Merges all three converted datasets into one unified YOLO dataset
with a consistent class mapping and produces a single data.yaml.

IMPORTANT: Run ONLY after:
  1. inspect_datasets.py  ← confirms formats
  2. convert_datasets.py  ← converts to standard YOLO bbox

Unified class taxonomy (6 classes — belt_splice, longitudinal_tear, scratch dropped
because they have zero training data in all three datasets):
  0: tear
  1: hole
  2: crack
  3: puncture
  4: patch_work
  5: impact_damage
"""

import os, sys, shutil, json, hashlib
from pathlib import Path
from collections import Counter, defaultdict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

ROOT         = Path(__file__).parent.parent
CONVERTED    = Path(__file__).parent / "datasets" / "converted"
MERGED       = Path(__file__).parent / "datasets" / "merged"
REPORTS_DIR  = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ── Unified class taxonomy (6 classes) ──────────────────────────────────────
# belt_splice, longitudinal_tear, scratch dropped — no training data.
UNIFIED_CLASSES = [
    "tear",          # 0  — DS1 Tear, DS2 tear
    "hole",          # 1  — DS1 Hole, DS3 hole-tear
    "crack",         # 2  — DS3 crack
    "puncture",      # 3  — DS1 Puncture
    "patch_work",    # 4  — DS1 patch work
    "impact_damage", # 5  — DS1 impact damage
]
UNIFIED_INDEX = {name: i for i, name in enumerate(UNIFIED_CLASSES)}

# ── Per-dataset class name → unified class name mapping ──────────────────────
# Keys must be lowercase for case-insensitive matching.
# Value of None means SKIP this class.
DS1_MAPPING = {
    "hole":          "hole",
    "human":         None,           # removed in convert step
    "other objects": None,           # removed in convert step
    "puncture":      "puncture",
    "roller":        None,           # removed in convert step
    "tear":          "tear",
    "impact damage": "impact_damage",
    "patch work":    "patch_work",
}

DS2_MAPPING = {
    "tear": "tear",
}

DS3_MAPPING = {
    "crack":     "crack",
    "hole-tear": "hole",  # hole-tear → hole (review visually if needed)
}

# old_five_class_model class names (reference only — not re-training here)
# belt_splice, longitudinal_tear, scratch have no data in current datasets.
OLD_MODEL_CLASSES = {
    "belt splice":       None,  # no data — dropped from taxonomy
    "deep scratch":      None,  # no data — dropped from taxonomy
    "longitudinal tear": None,  # no data — dropped from taxonomy
    "normal belt":       None,  # background — no annotation
    "slight scratch":    None,  # no data — dropped from taxonomy
}


def get_ds_class_names(converted_ds_dir):
    """Read class names from converted dataset's data.yaml."""
    yamls = list(Path(converted_ds_dir).rglob("data.yaml"))
    if yamls and HAS_YAML:
        with open(yamls[0], encoding="utf-8", errors="replace") as f:
            d = yaml.safe_load(f) or {}
        names_data = d.get("names", [])
        if isinstance(names_data, dict):
            max_id = max(names_data.keys()) if names_data else -1
            return [names_data.get(i, "") for i in range(max_id + 1)]
        elif isinstance(names_data, list):
            return names_data
    return []


def image_hash(img_path):
    """MD5 hash of image file for deduplication."""
    h = hashlib.md5()
    with open(img_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def remap_label_file(label_path, class_names, class_mapping):
    """
    Read a converted (standard YOLO) label file and remap class IDs
    using class_mapping (source name → unified name).
    Returns list of remapped lines, and count of skipped annotations.
    """
    remapped = []
    skipped = 0
    if not label_path.exists():
        return remapped, skipped
    for line in label_path.read_text(encoding="utf-8", errors="replace").strip().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            src_id = int(parts[0])
        except ValueError:
            continue
        if src_id >= len(class_names):
            skipped += 1
            continue
        src_name = str(class_names[src_id]).lower()
        unified_name = class_mapping.get(src_name)
        if unified_name is None:
            skipped += 1
            continue
        if unified_name not in UNIFIED_INDEX:
            print(f"  ⚠️  Unified class '{unified_name}' not in taxonomy — skipping")
            skipped += 1
            continue
        new_id = UNIFIED_INDEX[unified_name]
        remapped.append(f"{new_id} " + " ".join(parts[1:]))
    return remapped, skipped


def copy_split(ds_dir, split, class_names, class_mapping,
               out_split_img, out_split_lbl, seen_hashes, stats):
    """Process one split (train/valid/test) of a converted dataset."""
    # Locate images and labels for this split
    img_dirs = list(ds_dir.rglob(f"{split}/images")) + list(ds_dir.rglob(f"images/{split}"))
    lbl_dirs = list(ds_dir.rglob(f"{split}/labels")) + list(ds_dir.rglob(f"labels/{split}"))

    if not img_dirs:
        return

    img_dir = img_dirs[0]
    lbl_dir = lbl_dirs[0] if lbl_dirs else None

    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in {".jpg",".jpeg",".png",".bmp"}:
            continue
        stats["total_images"] += 1

        # Deduplication
        h = image_hash(img_path)
        if h in seen_hashes:
            stats["duplicates_removed"] += 1
            continue
        seen_hashes.add(h)

        # Output filename: ds-prefix + original name to avoid collisions
        stem = img_path.stem
        suffix = img_path.suffix

        # Find matching label
        lbl_path = (lbl_dir / f"{stem}.txt") if lbl_dir else None

        # Remap labels
        if lbl_path and lbl_path.exists():
            remapped, skipped = remap_label_file(lbl_path, class_names, class_mapping)
            stats["skipped_annotations"] += skipped
        else:
            remapped = []

        # Copy image
        out_img = out_split_img / (stem + suffix)
        if out_img.exists():
            # resolve name collision
            out_img = out_split_img / (h[:8] + suffix)
        shutil.copy2(img_path, out_img)

        # Write label (even if empty → background image)
        out_lbl = out_split_lbl / (out_img.stem + ".txt")
        out_lbl.write_text("\n".join(remapped) + ("\n" if remapped else ""))

        if not remapped:
            stats["background_images"] += 1
        else:
            for line in remapped:
                cls_id = int(line.split()[0])
                stats["class_counter"][cls_id] += 1


def merge_all():
    MERGED.mkdir(parents=True, exist_ok=True)

    splits = ["train", "valid", "test"]
    for split in splits:
        (MERGED / "images" / split).mkdir(parents=True, exist_ok=True)
        (MERGED / "labels" / split).mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    total_stats = defaultdict(lambda: defaultdict(int))

    datasets = [
        ("DS1", CONVERTED / "DS1_OBB",      DS1_MAPPING),
        ("DS2", CONVERTED / "DS2_STANDARD",  DS2_MAPPING),
        ("DS3", CONVERTED / "DS3_SEG",       DS3_MAPPING),
    ]

    for ds_name, ds_dir, ds_mapping in datasets:
        print(f"\n[MERGE] Processing {ds_name} from {ds_dir}")
        if not ds_dir.exists():
            print(f"  ⚠️  Not found — run convert_datasets.py first")
            continue

        class_names = get_ds_class_names(ds_dir)
        print(f"  Source classes: {class_names}")

        for split in splits:
            stats = {
                "total_images": 0,
                "duplicates_removed": 0,
                "background_images": 0,
                "skipped_annotations": 0,
                "class_counter": Counter(),
            }
            out_img = MERGED / "images" / split
            out_lbl = MERGED / "labels" / split
            copy_split(ds_dir, split, class_names, ds_mapping,
                       out_img, out_lbl, seen_hashes, stats)
            for k,v in stats.items():
                if k != "class_counter":
                    total_stats[ds_name][k] += v
                else:
                    for cid, cnt in v.items():
                        total_stats[ds_name]["class_counter_" + str(cid)] += cnt

            if stats["total_images"] > 0:
                print(f"  [{split}] images: {stats['total_images']}  "
                      f"dups: {stats['duplicates_removed']}  "
                      f"bg: {stats['background_images']}  "
                      f"skipped_ann: {stats['skipped_annotations']}")

    # ── Write data.yaml ───────────────────────────────────────────────────
    resolved_merged_path = str(MERGED.resolve()).replace("\\", "/")
    yaml_content = {
        "path":  resolved_merged_path,
        "train": "images/train",
        "val":   "images/valid",
        "test":  "images/test",
        "nc":    len(UNIFIED_CLASSES),
        "names": UNIFIED_CLASSES,
    }
    yaml_path = MERGED / "data.yaml"
    if HAS_YAML:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    else:
        # Fallback manual write
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(f"path: {resolved_merged_path}\n")
            f.write("train: images/train\nval: images/valid\ntest: images/test\n")
            f.write(f"nc: {len(UNIFIED_CLASSES)}\n")
            f.write("names:\n")
            for n in UNIFIED_CLASSES:
                f.write(f"  - {n}\n")
    print(f"\n✅ data.yaml written: {yaml_path}")

    # ── Count final merged dataset ─────────────────────────────────────────
    print("\n[MERGED DATASET SUMMARY]")
    final_class_dist = Counter()
    for split in splits:
        lbl_dir = MERGED / "labels" / split
        imgs_dir = MERGED / "images" / split
        n_imgs = len(list(imgs_dir.glob("*"))) if imgs_dir.exists() else 0
        n_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        for lf in lbl_dir.glob("*.txt") if lbl_dir.exists() else []:
            for line in lf.read_text().strip().splitlines():
                parts = line.split()
                if parts:
                    try:
                        final_class_dist[int(parts[0])] += 1
                    except ValueError:
                        pass
        print(f"  [{split}] images: {n_imgs}  label files: {n_lbls}")

    print("\n  Final unified class distribution:")
    for cid, name in enumerate(UNIFIED_CLASSES):
        cnt = final_class_dist.get(cid, 0)
        bar = "█" * min(30, cnt // max(1, max(final_class_dist.values()) // 30))
        print(f"  [{cid}] {name:20s}: {cnt:5d}  {bar}")

    report = {
        "per_dataset_stats": {k: dict(v) for k,v in total_stats.items()},
        "unified_classes": UNIFIED_CLASSES,
        "class_distribution": {
            UNIFIED_CLASSES[k]: v for k,v in final_class_dist.items()
        },
        "class_mapping": {
            "DS1": DS1_MAPPING,
            "DS2": DS2_MAPPING,
            "DS3": DS3_MAPPING,
            "OLD_MODEL (reference only)": OLD_MODEL_CLASSES,
        },
        "note_hole_tear": (
            "hole-tear from DS3 was mapped to 'hole'. "
            "Inspect DS3 images manually to verify this is appropriate."
        ),
    }
    with open(REPORTS_DIR / "merge_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Merge report: {REPORTS_DIR / 'merge_report.json'}")
    print("\n✅ Merge complete. Review class distribution above, then run train.py.")


if __name__ == "__main__":
    print("=" * 70)
    print("  SIH26008 — DATASET MERGE & CLASS REMAPPING")
    print("=" * 70)
    merge_all()
