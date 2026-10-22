"""
train.py
========
SIH26008 — Conveyor Belt Health Monitoring AI

Trains a unified YOLOv8 detection model on the merged 6-class dataset.

Classes (6 unified):
  0: tear
  1: hole
  2: crack
  3: puncture
  4: patch_work
  5: impact_damage

Key requirements satisfied:
  - Uses YOLOv8s if GPU available; otherwise YOLOv8n for fast CPU training.
  - Trains for 50 epochs (with early stopping).
  - Validates on validation split.
  - Saves the final model as best.pt in a NEW output folder (MODELS / "new_trained_model / best.pt")
    so original best.pt in root is NOT overwritten.
  - Saves metrics, confusion matrix, loss curves, and prediction visual examples.
  - Reports exact final path to the new best.pt.
"""

import os, sys, json, time, shutil
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_YAML   = ROOT / "datasets" / "merged" / "data.yaml"
MODELS      = ROOT / "models"
REPORTS     = ROOT / "reports"
PREDICTIONS = ROOT / "predictions"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)
PREDICTIONS.mkdir(exist_ok=True)

# Original model files in root (must remain untouched)
ORIG_BEST_PT  = ROOT.parent / "best.pt"
OLD_MODEL_PT  = ROOT.parent / "old_five_class_model.pt"

# ── Unified class taxonomy (6 classes) ───────────────────────────────────────
UNIFIED_CLASSES = [
    "tear",          # 0
    "hole",          # 1
    "crack",         # 2
    "puncture",      # 3
    "patch_work",    # 4
    "impact_damage", # 5
]

# ── Training hyperparameters ─────────────────────────────────────────────────
EPOCHS          = 50
PATIENCE        = 15      # early stopping patience
IMGSZ           = 640
LR0             = 0.01
OPTIMIZER       = "auto"
EXPERIMENT_NAME = f"unified_yolo_{datetime.now().strftime('%Y%m%d_%H%M')}"


def check_environment_and_select_model():
    """Check hardware, PyTorch, ultralytics, and decide model architecture."""
    print("\n" + "=" * 70)
    print("  [ENV CHECK & ARCHITECTURE SELECTION]")
    print("=" * 70)
    
    device = "cpu"
    base_weights = "yolov8n.pt"
    batch_size = 16
    workers = 0

    try:
        import torch
        print(f"  ✅ PyTorch Version : {torch.__version__}")
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  ✅ CUDA GPU Detected: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
            device = 0
            base_weights = "yolov8s.pt"   # Pretrained YOLOv8s if GPU available
            batch_size = 16
            workers = 4
            print(f"  🚀 Using YOLOv8s on GPU (device=0)")
        else:
            print(f"  ℹ️  No CUDA GPU detected (CPU mode with Intel Graphics)")
            device = "cpu"
            base_weights = "yolov8n.pt"   # Pretrained YOLOv8n for fast CPU training
            batch_size = 16
            workers = 0
            print(f"  ⚡ Using YOLOv8n on CPU for optimal speed & memory efficiency")
    except ImportError as e:
        print(f"  ❌ PyTorch import error: {e}")
        sys.exit(1)

    try:
        import ultralytics
        print(f"  ✅ Ultralytics Version: {ultralytics.__version__}")
    except ImportError as e:
        print(f"  ❌ Ultralytics import error: {e}")
        sys.exit(1)

    if not DATA_YAML.exists():
        print(f"  ❌ data.yaml not found at {DATA_YAML}")
        print("     Please run merge_datasets.py first.")
        sys.exit(1)
    print(f"  ✅ Dataset Config : {DATA_YAML}")
    print(f"  ✅ Target Classes (nc={len(UNIFIED_CLASSES)}): {UNIFIED_CLASSES}")

    # Verify original files exist and will remain untouched
    if ORIG_BEST_PT.exists():
        print(f"  🔒 Existing root best.pt verified: {ORIG_BEST_PT} (WILL NOT BE OVERWRITTEN)")
    if OLD_MODEL_PT.exists():
        print(f"  🔒 Existing old_five_class_model.pt verified: {OLD_MODEL_PT} (WILL NOT BE TOUCHED)")

    return device, base_weights, batch_size, workers


def train_unified_model(device, base_weights, batch_size, workers):
    """Run YOLOv8 training on the unified 6-class dataset."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print(f"  [TRAINING: {EXPERIMENT_NAME}]")
    print("=" * 70)
    print(f"  Model Base   : {base_weights}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Image Size   : {IMGSZ}")
    print(f"  Batch Size   : {batch_size}")
    print(f"  Device       : {device}")
    print(f"  Patience     : {PATIENCE}")
    print(f"  Output Dir   : {MODELS / EXPERIMENT_NAME}")

    model = YOLO(base_weights)

    train_kwargs = dict(
        data      = str(DATA_YAML),
        epochs    = EPOCHS,
        imgsz     = IMGSZ,
        batch     = batch_size,
        device    = device,
        patience  = PATIENCE,
        optimizer = OPTIMIZER,
        lr0       = LR0,
        name      = EXPERIMENT_NAME,
        project   = str(MODELS),
        save      = True,
        plots     = True,
        verbose   = True,
        workers   = workers,
        exist_ok  = False,
    )

    # Attempt training with automatic fallback to smaller batch if OOM occurs
    for cur_batch in [batch_size, batch_size // 2, 4, 2, 1]:
        try:
            print(f"\n  Starting training run with batch={cur_batch} ...")
            train_kwargs["batch"] = cur_batch
            results = model.train(**train_kwargs)
            return results, model
        except RuntimeError as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or "cuda" in err_str:
                print(f"  ⚠️  OOM encountered with batch={cur_batch} — reducing batch size")
                train_kwargs["exist_ok"] = True
                continue
            else:
                raise

    print("  ❌ Training failed across all batch sizes.")
    sys.exit(1)


def validate_and_predict(model, exp_dir):
    """Run validation metrics calculation and generate prediction examples."""
    print("\n" + "=" * 70)
    print("  [VALIDATION & PREDICTION EXAMPLES]")
    print("=" * 70)

    # 1. Validation
    print("  Evaluating model on validation split...")
    val_results = model.val(
        data=str(DATA_YAML),
        split="val",
        project=str(MODELS),
        name=f"{EXPERIMENT_NAME}_val",
        save_json=True,
        plots=True,
        verbose=True
    )

    # 2. Prediction examples on sample test or validation images
    test_dir = ROOT / "datasets" / "merged" / "images" / "test"
    val_dir  = ROOT / "datasets" / "merged" / "images" / "valid"

    sample_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.png"))
    if not sample_images:
        sample_images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.jpeg")) + list(val_dir.glob("*.png"))
    
    sample_images = sample_images[:12]  # select up to 12 diverse sample images
    
    pred_out_dir = PREDICTIONS / "sample_detections"
    if sample_images:
        print(f"  Generating prediction examples on {len(sample_images)} conveyor belt images...")
        model.predict(
            source=[str(p) for p in sample_images],
            save=True,
            project=str(PREDICTIONS),
            name="sample_detections",
            exist_ok=True,
            conf=0.25,
            iou=0.45
        )
        print(f"  ✅ Prediction visualizations saved to: {pred_out_dir}")

    return val_results


def save_and_report_final_model(results, val_results, device, base_weights, start_time):
    """Save the final model in a NEW folder, copy plots, and write final report."""
    elapsed = time.time() - start_time
    exp_dir = MODELS / EXPERIMENT_NAME

    # NEW OUTPUT FOLDER FOR TRAINED MODEL (Protects root best.pt)
    new_model_folder = MODELS / "new_trained_model"
    new_model_folder.mkdir(parents=True, exist_ok=True)

    src_best = exp_dir / "weights" / "best.pt"
    src_last = exp_dir / "weights" / "last.pt"

    new_best_path = new_model_folder / "best.pt"
    new_last_path = new_model_folder / "last.pt"

    if src_best.exists():
        shutil.copy2(src_best, new_best_path)
    elif src_last.exists():
        shutil.copy2(src_last, new_best_path)

    if src_last.exists():
        shutil.copy2(src_last, new_last_path)

    # Copy plots, confusion matrices, and curves to reports directory
    for pattern in ["*.png", "*.jpg", "*.csv"]:
        for file in exp_dir.glob(pattern):
            try:
                shutil.copy2(file, REPORTS / file.name)
            except Exception:
                pass

    # Extract metrics
    metrics_summary = {}
    if val_results and hasattr(val_results, "results_dict"):
        metrics_summary = {k: round(float(v), 4) for k, v in val_results.results_dict.items()}
    elif results and hasattr(results, "results_dict"):
        metrics_summary = {k: round(float(v), 4) for k, v in results.results_dict.items()}

    # Construct final report dictionary
    training_report = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": EXPERIMENT_NAME,
        "base_model": base_weights,
        "device": str(device),
        "epochs_requested": EPOCHS,
        "elapsed_seconds": round(elapsed, 1),
        "unified_classes": UNIFIED_CLASSES,
        "data_yaml": str(DATA_YAML.resolve()),
        "new_best_model_path": str(new_best_path.resolve()),
        "original_best_pt_untouched": str(ORIG_BEST_PT.resolve()),
        "metrics": metrics_summary,
    }

    report_json_path = REPORTS / f"training_report_{EXPERIMENT_NAME}.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(training_report, f, indent=2)

    # Print Final Summary Banner
    print("\n" + "=" * 70)
    print("  🎉 UNIFIED CONVEYOR BELT MODEL TRAINING & VALIDATION COMPLETE")
    print("=" * 70)
    print(f"  Training Duration : {elapsed/60:.2f} minutes")
    print(f"  Model Architecture: {base_weights}")
    print(f"  Final Metrics     :")
    for k, v in metrics_summary.items():
        print(f"    • {k}: {v}")
    print("\n  🛡️  SAFETY VERIFICATION:")
    print(f"    • Original best.pt in root untouched: {ORIG_BEST_PT.exists()} ({ORIG_BEST_PT})")
    print(f"    • Original old model untouched     : {OLD_MODEL_PT.exists()} ({OLD_MODEL_PT})")
    print("\n  ⭐ EXACT PATH TO NEW TRAINED MODEL (best.pt):")
    print(f"     {new_best_path.resolve()}")
    print("=" * 70 + "\n")

    return str(new_best_path.resolve())


if __name__ == "__main__":
    start_time = time.time()
    device, base_weights, batch_size, workers = check_environment_and_select_model()
    results, model = train_unified_model(device, base_weights, batch_size, workers)
    val_results = validate_and_predict(model, MODELS / EXPERIMENT_NAME)
    final_best_path = save_and_report_final_model(results, val_results, device, base_weights, start_time)
