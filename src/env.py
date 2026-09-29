"""
Environment helpers: pick the best compute device and resolve shared paths,
so the SAME notebook runs unchanged on Colab, a Mac, or a Windows/NVIDIA laptop.

Path resolution for DATA_DIR (datasets, photos, trained weights, results):
  1. IVP_DATA_DIR environment variable, if set
  2. "data_dir" in <repo>/local_config.json, if present (gitignored, per laptop)
  3. On Colab: /content/drive/MyDrive/IVP_Project_data  (Google Drive)
  4. Otherwise: <repo>/data

Point every laptop's DATA_DIR at the same Google Drive folder (via Google
Drive for Desktop) and all results - no matter which machine trained them -
end up in one shared place.
"""
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_COLAB = "google.colab" in sys.modules
COLAB_DATA_DIR = Path("/content/drive/MyDrive/IVP_Project_data")


def _local_config():
    cfg_path = ROOT / "local_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def get_data_dir():
    if os.environ.get("IVP_DATA_DIR"):
        d = Path(os.environ["IVP_DATA_DIR"]).expanduser()
    elif _local_config().get("data_dir"):
        d = Path(_local_config()["data_dir"]).expanduser()
    elif IN_COLAB:
        d = COLAB_DATA_DIR
    else:
        d = ROOT / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


DATA_DIR = get_data_dir()
DATASETS_DIR = DATA_DIR / "datasets"   # Roboflow / Kaggle downloads
SESSIONS_DIR = DATA_DIR / "sessions"   # before/after shelf photos
RUNS_DIR = DATA_DIR / "runs"           # YOLO training runs (weights, curves)
RESULTS_DIR = DATA_DIR / "results"     # JSON/CSV numbers for the paper
for _d in (DATASETS_DIR, SESSIONS_DIR, RUNS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def get_secret(name):
    """Read an API key from Colab secrets, env vars, or local_config.json."""
    if IN_COLAB:
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception:
            pass
    return os.environ.get(name) or _local_config().get(name)


def get_device():
    """'0' (first NVIDIA GPU) -> 'mps' (Apple Silicon) -> 'cpu'."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_name():
    """Human-readable machine tag used to label training runs."""
    dev = get_device()
    if dev == "0":
        import torch
        return "cuda-" + torch.cuda.get_device_name(0).replace(" ", "_")
    if IN_COLAB:
        return "colab-" + dev
    return f"{dev}-{platform.node().split('.')[0]}"


def default_batch(device=None):
    """Conservative batch sizes that fit a 4 GB laptop GPU / 8 GB Mac."""
    device = device or get_device()
    return {"0": 16, "mps": 8}.get(device, 4)


def summary():
    print(f"Running on : {'Google Colab' if IN_COLAB else platform.platform()}")
    print(f"Device     : {get_device()}  ({device_name()})")
    print(f"Repo root  : {ROOT}")
    print(f"DATA_DIR   : {DATA_DIR}")
