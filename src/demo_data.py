"""
Synthetic DEMO data so every notebook runs end-to-end before the real
dataset / shelf photos exist. Nothing here is used for paper results.

  make_demo_detection_dataset() - YOLO-format dataset of drawn "products"
  make_demo_sessions()          - before/after photos of a 3x3 shelf with
                                  ArUco corner markers, camera tilt and
                                  lighting changes, plus ground truth
"""
import json

import cv2
import numpy as np
import yaml

from .preprocess import ARUCO_DICT, CORNER_IDS

DEMO_CLASSES = [
    "bournvita", "mysore_sandal", "nescafe", "nivea_lotion", "nivea_creme",
    "parachute_oil", "tresemme_shampoo", "tresemme_conditioner", "colgate", "maggi",
]
_COLORS = [(40, 90, 200), (60, 170, 230), (30, 30, 140), (160, 60, 20), (200, 120, 40),
           (60, 160, 60), (30, 30, 30), (180, 180, 180), (40, 40, 220), (20, 200, 240)]


def render_product(cls_id, w, h, rng):
    """Draw a class-specific 'package': colour + shape + stripe pattern + label text."""
    img = np.full((h, w, 3), 245, np.uint8)
    color = tuple(int(c) for c in np.clip(np.array(_COLORS[cls_id]) + rng.integers(-15, 16, 3), 0, 255))
    shape = cls_id % 3
    if shape == 0:    # tall jar / bottle
        cv2.rectangle(img, (int(w * .2), int(h * .1)), (int(w * .8), int(h * .95)), color, -1)
        cv2.rectangle(img, (int(w * .35), 0), (int(w * .65), int(h * .12)), (90, 90, 90), -1)
    elif shape == 1:  # box
        cv2.rectangle(img, (int(w * .08), int(h * .25)), (int(w * .92), int(h * .95)), color, -1)
    else:             # round tub / tube
        cv2.ellipse(img, (w // 2, int(h * .6)), (int(w * .4), int(h * .35)), 0, 0, 360, color, -1)
    for k in range(cls_id % 4 + 1):  # stripes differ per class
        y = int(h * (0.4 + 0.1 * k))
        cv2.line(img, (int(w * .25), y), (int(w * .75), y), (255, 255, 255), 2)
    cv2.putText(img, DEMO_CLASSES[cls_id][:6], (int(w * .1), int(h * .55)),
                cv2.FONT_HERSHEY_SIMPLEX, max(w / 260, 0.3), (255, 255, 255), 1)
    return img


def _paste(canvas, patch, x, y):
    h, w = patch.shape[:2]
    mask = (patch < 240).any(axis=2)
    canvas[y:y + h, x:x + w][mask] = patch[mask]


# ----------------------------------------------------------------------------
# Detection dataset
# ----------------------------------------------------------------------------
def make_demo_detection_dataset(out_dir, n=(60, 15, 15), imgsz=640, seed=0):
    rng = np.random.default_rng(seed)
    for split, count in zip(["train", "valid", "test"], n):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(count):
            bg = rng.integers(170, 235)
            img = np.full((imgsz, imgsz, 3), bg, np.uint8)
            labels, cells = [], rng.permutation(9)[: rng.integers(3, 8)]
            for cell in cells:  # one product per 3x3 cell, random size/offset
                r, c = divmod(int(cell), 3)
                cw = imgsz // 3
                w, h = int(cw * rng.uniform(.55, .9)), int(cw * rng.uniform(.6, .95))
                x = c * cw + rng.integers(0, cw - w + 1)
                y = r * cw + rng.integers(0, cw - h + 1)
                cls = int(rng.integers(0, len(DEMO_CLASSES)))
                _paste(img, render_product(cls, w, h, rng), x, y)
                labels.append(f"{cls} {(x + w / 2) / imgsz:.6f} {(y + h / 2) / imgsz:.6f} {w / imgsz:.6f} {h / imgsz:.6f}")
            img = cv2.convertScaleAbs(img, alpha=rng.uniform(.7, 1.2), beta=rng.uniform(-30, 20))
            cv2.imwrite(str(out_dir / split / "images" / f"{i:04d}.jpg"), img)
            (out_dir / split / "labels" / f"{i:04d}.txt").write_text("\n".join(labels))
    with open(out_dir / "data.yaml", "w") as f:
        yaml.safe_dump({"names": DEMO_CLASSES, "nc": len(DEMO_CLASSES)}, f)
    return out_dir


# ----------------------------------------------------------------------------
# Shelf before/after sessions
# ----------------------------------------------------------------------------
def render_shelf(planogram, present, grid=900, pad=70, rng=None):
    """Front-on shelf; ArUco marker CENTRES sit exactly on the grid corners."""
    rng = rng or np.random.default_rng(0)
    size = grid + 2 * pad
    img = np.full((size, size, 3), 225, np.uint8)
    cv2.rectangle(img, (pad, pad), (pad + grid, pad + grid), (150, 175, 200), -1)  # shelf back
    cw = grid // 3
    for k in range(1, 3):  # shelf boards
        cv2.line(img, (pad, pad + k * cw), (pad + grid, pad + k * cw), (60, 90, 120), 8)
    for key, cls_name in planogram.items():
        if not present.get(key, True):
            continue
        r, c = map(int, key.split(","))
        w, h = int(cw * .7), int(cw * .8)
        x, y = pad + c * cw + (cw - w) // 2, pad + r * cw + (cw - h) // 2
        _paste(img, render_product(DEMO_CLASSES.index(cls_name), w, h, rng), x, y)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    m = 80
    for mid, (cx, cy) in zip(CORNER_IDS, [(pad, pad), (pad + grid, pad), (pad + grid, pad + grid), (pad, pad + grid)]):
        marker = cv2.cvtColor(cv2.aruco.generateImageMarker(dictionary, mid, m), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img, (cx - m // 2 - 10, cy - m // 2 - 10), (cx + m // 2 + 10, cy + m // 2 + 10), (255, 255, 255), -1)
        img[cy - m // 2: cy + m // 2, cx - m // 2: cx + m // 2] = marker
    return img


def camera_view(img, rng, tilt=0.06, out=(1200, 1000), light=(0.75, 1.1), noise=6):
    """Simulate a phone/webcam: perspective tilt, lighting change, sensor noise."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    ow, oh = out
    base = np.float32([[ow * .1, oh * .06], [ow * .9, oh * .06], [ow * .9, oh * .94], [ow * .1, oh * .94]])
    dst = base + rng.uniform(-tilt, tilt, (4, 2)).astype(np.float32) * np.float32([ow, oh])
    view = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), out, borderValue=(200, 200, 200))
    view = cv2.convertScaleAbs(view, alpha=rng.uniform(*light), beta=rng.uniform(-15, 10))
    view = np.clip(view + rng.normal(0, noise, view.shape), 0, 255).astype(np.uint8)
    return view


def make_demo_sessions(out_dir, n_sessions=12, seed=1):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = [f"{r},{c}" for r in range(3) for c in range(3)]
    planogram = dict(zip(cells, rng.permutation(DEMO_CLASSES)[:9].tolist()))
    with open(out_dir / "planogram.json", "w") as f:
        json.dump(planogram, f, indent=2)

    rows = ["session_id,removed"]
    for s in range(n_sessions):
        sid = f"demo_s{s + 1:03d}"
        taken = rng.choice(cells, size=int(rng.integers(1, 4)), replace=False).tolist()
        before = render_shelf(planogram, {}, rng=rng)
        after = render_shelf(planogram, {k: k not in taken for k in cells}, rng=rng)
        (out_dir / sid).mkdir(exist_ok=True)
        cam_seed = int(rng.integers(1e6))  # same camera pose before/after, small jitter
        cv2.imwrite(str(out_dir / sid / "before.jpg"), camera_view(before, np.random.default_rng(cam_seed)))
        cv2.imwrite(str(out_dir / sid / "after.jpg"), camera_view(after, np.random.default_rng(cam_seed + 1), tilt=0.065))
        rows.append(f"{sid},{';'.join(sorted(planogram[k] for k in taken))}")
    (out_dir / "sessions.csv").write_text("\n".join(rows) + "\n")
    return out_dir
