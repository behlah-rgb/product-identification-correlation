"""
YOLOv8 helpers: dataset download/merging, training (any device), evaluation
with results saved as JSON for the paper.
"""
import json
import shutil
import time
from pathlib import Path

import yaml

from .env import DATASETS_DIR, RESULTS_DIR, RUNS_DIR, default_batch, device_name, get_device, get_secret

SPLITS = {"train": "train", "val": "valid", "test": "test"}  # Roboflow folder names


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------
def download_roboflow(workspace, project, version, dest_name, api_key=None):
    """Download a Roboflow project in YOLOv8 format into DATASETS_DIR/dest_name."""
    from roboflow import Roboflow

    api_key = api_key or get_secret("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Roboflow API key. Add ROBOFLOW_API_KEY to Colab secrets (key icon, left sidebar) "
            "or to local_config.json. Get it at https://app.roboflow.com/settings/api"
        )
    dest = DATASETS_DIR / dest_name
    rf = Roboflow(api_key=api_key)
    rf.workspace(workspace).project(project).version(int(version)).download("yolov8", location=str(dest))
    return dest


def read_names(dataset_dir):
    with open(Path(dataset_dir) / "data.yaml") as f:
        names = yaml.safe_load(f)["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def write_local_yaml(dataset_dir, names=None):
    """
    Roboflow's data.yaml uses relative paths that break across machines.
    Write data_local.yaml with absolute paths for THIS machine.
    """
    dataset_dir = Path(dataset_dir).resolve()
    cfg = {"path": str(dataset_dir), "names": names or read_names(dataset_dir)}
    for key, folder in SPLITS.items():
        if (dataset_dir / folder / "images").exists():
            cfg[key] = f"{folder}/images"
    if "val" not in cfg:
        raise FileNotFoundError(f"No valid/images folder in {dataset_dir}")
    out = dataset_dir / "data_local.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def merge_datasets(primary_dir, extra_dir, out_name):
    """
    Merge two YOLO datasets (e.g. public AgentSK47 + your own shelf photos)
    into DATASETS_DIR/out_name, remapping class indices BY NAME so the two
    Roboflow exports don't need identical class orders.
    Classes only in `extra` are appended to the end of the class list.
    """
    primary_dir, extra_dir = Path(primary_dir), Path(extra_dir)
    names = read_names(primary_dir)
    extra_names = read_names(extra_dir)
    names += [n for n in extra_names if n not in names]
    out = DATASETS_DIR / out_name
    if out.exists():
        shutil.rmtree(out)

    for src_dir, src_names, tag in [(primary_dir, read_names(primary_dir), "pub"), (extra_dir, extra_names, "shelf")]:
        remap = {i: names.index(n) for i, n in enumerate(src_names)}
        for folder in SPLITS.values():
            img_dir = src_dir / folder / "images"
            if not img_dir.exists():
                continue
            (out / folder / "images").mkdir(parents=True, exist_ok=True)
            (out / folder / "labels").mkdir(parents=True, exist_ok=True)
            for img in img_dir.iterdir():
                shutil.copy2(img, out / folder / "images" / f"{tag}_{img.name}")
                lbl = src_dir / folder / "labels" / (img.stem + ".txt")
                lines = []
                if lbl.exists():
                    for line in lbl.read_text().strip().splitlines():
                        parts = line.split()
                        if parts:
                            lines.append(" ".join([str(remap[int(parts[0])])] + parts[1:]))
                (out / folder / "labels" / f"{tag}_{img.stem}.txt").write_text("\n".join(lines))

    with open(out / "data.yaml", "w") as f:
        yaml.safe_dump({"names": names}, f)
    return out


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def latest_run_weights(prefix="", which="last.pt"):
    """Most recently modified RUNS_DIR/<prefix>*/weights/<which>, or None."""
    cands = sorted(RUNS_DIR.glob(f"{prefix}*/weights/{which}"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def train(data_yaml, run_name, weights="yolov8n.pt", epochs=50, imgsz=640, batch=None,
          device=None, patience=20, **extra):
    """
    Train YOLOv8 and save the run to RUNS_DIR/<run_name>_<device-tag>.

    weights: 'yolov8n.pt' to start from COCO-pretrained weights, or a path to
             a previous run's best.pt/last.pt to CONTINUE training on another
             machine (fine-tunes from those weights; optimizer restarts).
    """
    from ultralytics import YOLO

    device = device or get_device()
    batch = batch or default_batch(device)
    name = f"{run_name}_{device_name()}"
    if device == "mps":
        extra.setdefault("deterministic", False)  # some Apple-GPU ops have no deterministic version
    model = YOLO(str(weights))
    t0 = time.time()
    model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
        project=str(RUNS_DIR), name=name, exist_ok=True, patience=patience, seed=42,
        workers=4 if device == "0" else 2, plots=True, **extra,
    )
    save_dir = Path(model.trainer.save_dir)
    print(f"\nTraining took {(time.time() - t0) / 60:.1f} min on {device_name()}")
    print(f"Run saved to: {save_dir}")
    return save_dir


def resume(run_dir):
    """Exact resume of an interrupted run ON THE SAME MACHINE / same Colab Drive paths."""
    from ultralytics import YOLO

    last = Path(run_dir) / "weights" / "last.pt"
    YOLO(str(last)).train(resume=True)
    return Path(run_dir)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(weights, data_yaml, tag, split=None, imgsz=640, device=None, conf=0.001):
    """
    Validate a trained model and save paper-ready numbers to
    RESULTS_DIR/yolo_<tag>.json (overall + per-class P, R, mAP50, mAP50-95).
    """
    from ultralytics import YOLO

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    split = split or ("test" if "test" in cfg else "val")
    device = device or get_device()
    model = YOLO(str(weights))
    m = model.val(data=str(data_yaml), split=split, imgsz=imgsz, device=device, conf=conf,
                  project=str(RUNS_DIR), name=f"eval_{tag}", exist_ok=True, plots=True, verbose=False)

    names = model.names
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[int(c)]] = {"precision": float(p), "recall": float(r),
                                   "mAP50": float(ap50), "mAP50_95": float(ap)}
    result = {
        "tag": tag, "weights": str(weights), "data": str(data_yaml), "split": split,
        "device": device_name(), "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
        "per_class": per_class, "plots_dir": str(m.save_dir),
    }
    out = RESULTS_DIR / f"yolo_{tag}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved metrics -> {out}")
    return result
