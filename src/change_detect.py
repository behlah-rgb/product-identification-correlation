"""
Before/after change detection: which products did the customer remove?

Two methods, compared in the paper:
  1. YOLO-based  - detect + count products in both images; removed = before - after.
                   Needs no prior knowledge of the shelf layout.
  2. Classical   - colour differencing (CIELAB) + thresholding + morphology on
                   the rectified shelf; a cell with enough changed pixels is
                   "emptied", and a planogram (cell -> product map) says which
                   product that was. Pure image processing, no CNN.

Session folder layout (DATA_DIR/sessions/):
    planogram.json          {"0,0": "bournvita", "0,1": "nescafe", ...}
    sessions.csv            session_id,removed      e.g.  s001,nescafe;nivea
    s001/before.jpg
    s001/after.jpg
"""
from collections import Counter

import cv2
import numpy as np

from .preprocess import cell_of_point, grid_cells, preprocess, rectify_shelf


# ----------------------------------------------------------------------------
# Shared: bring both images into the same rectified, cleaned frame
# ----------------------------------------------------------------------------
def prepare(img, use_aruco=True, pre_cfg=None, out_size=(900, 900)):
    """Rectify (if markers found) + photometric preprocessing. Returns (img, rectified?)."""
    rectified = False
    if use_aruco:
        warped, _ = rectify_shelf(img, out_size)
        if warped is not None:
            img, rectified = warped, True
    if pre_cfg is not False:
        img = preprocess(img, pre_cfg)
    return img, rectified


# ----------------------------------------------------------------------------
# Method 1: YOLO-based
# ----------------------------------------------------------------------------
def yolo_detect(model, img, conf=0.35, device=None):
    """Run YOLO on a BGR image -> list of dicts {cls, conf, box}."""
    res = model.predict(img, conf=conf, device=device, verbose=False)[0]
    dets = []
    for b in res.boxes:
        dets.append({
            "cls": res.names[int(b.cls)],
            "conf": float(b.conf),
            "box": [float(v) for v in b.xyxy[0].tolist()],
        })
    return dets


def dets_by_cell(dets, width, height, rows=3, cols=3):
    """Assign each detection to the grid cell containing its box centre."""
    cells = {(r, c): [] for r, c, _ in grid_cells(width, height, rows, cols)}
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cells[cell_of_point((x1 + x2) / 2, (y1 + y2) / 2, width, height, rows, cols)].append(d["cls"])
    return cells


def removed_yolo(model, before, after, conf=0.35, device=None, **prep_kw):
    """Removed items = multiset difference of class counts (before - after)."""
    b_img, _ = prepare(before, **prep_kw)
    a_img, _ = prepare(after, **prep_kw)
    b_dets = yolo_detect(model, b_img, conf, device)
    a_dets = yolo_detect(model, a_img, conf, device)
    removed = Counter(d["cls"] for d in b_dets) - Counter(d["cls"] for d in a_dets)
    return {
        "removed": sorted(removed.elements()),
        "before_dets": b_dets,
        "after_dets": a_dets,
        "before_img": b_img,
        "after_img": a_img,
    }


# ----------------------------------------------------------------------------
# Method 2: classical per-cell SSIM + planogram
# ----------------------------------------------------------------------------
def ssim(a, b):
    """Mean structural similarity of two grayscale images (Wang et al., 2004)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(s.mean())


def change_mask(before, after, delta_e=30.0, l_weight=0.5, kernel=7):
    """
    Binary 'something changed here' mask:
      1. Gaussian blur both images (suppress noise / sub-pixel misalignment)
      2. Colour difference in CIELAB; lightness is down-weighted so a global
         lighting change between the two shots isn't mistaken for a removal
      3. Threshold + morphological opening (remove speckle) and closing (fill holes)
    """
    h, w = before.shape[:2]
    lab_b = cv2.cvtColor(cv2.GaussianBlur(before, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_a = cv2.cvtColor(cv2.GaussianBlur(cv2.resize(after, (w, h)), (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    d = lab_b - lab_a
    de = np.sqrt((l_weight * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    mask = (de > delta_e).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def cell_changes(before, after, rows=3, cols=3, margin=0.08):
    """
    Per cell (inner region only, to ignore marker/edge jitter):
      changed_frac - fraction of pixels flagged by change_mask()  (decision score)
      ssim         - grayscale structural similarity             (reported for comparison)
    """
    h, w = before.shape[:2]
    after = cv2.resize(after, (w, h))
    mask = change_mask(before, after)
    gb, ga = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    out = {}
    for r, c, (x1, y1, x2, y2) in grid_cells(w, h, rows, cols):
        mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
        sl = (slice(y1 + my, y2 - my), slice(x1 + mx, x2 - mx))
        out[(r, c)] = {"changed_frac": float(mask[sl].mean()), "ssim": ssim(gb[sl], ga[sl])}
    return out, mask


def removed_classical(before, after, planogram, change_threshold=0.15, **prep_kw):
    """Cells where more than `change_threshold` of the pixels changed are treated as 'product removed'."""
    b_img, _ = prepare(before, **prep_kw)
    a_img, _ = prepare(after, **prep_kw)
    scores, mask = cell_changes(b_img, a_img)
    changed = [cell for cell, s in scores.items() if s["changed_frac"] > change_threshold]
    removed = [planogram[f"{r},{c}"] for r, c in changed if f"{r},{c}" in planogram]
    return {"removed": sorted(removed), "cell_scores": scores, "changed_cells": changed,
            "change_mask": mask, "before_img": b_img, "after_img": a_img}


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def score_removals(predicted, truth):
    """Multiset precision / recall / F1 of removed items for one session."""
    p, t = Counter(predicted), Counter(truth)
    tp = sum((p & t).values())
    return {"tp": tp, "fp": sum(p.values()) - tp, "fn": sum(t.values()) - tp}


def aggregate(scores):
    tp = sum(s["tp"] for s in scores)
    fp = sum(s["fp"] for s in scores)
    fn = sum(s["fn"] for s in scores)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
