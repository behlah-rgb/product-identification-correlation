"""
IVP preprocessing for the 3x3 smart shelf.

Pipeline (each step can be switched on/off for the ablation study):
    rectify_shelf()  -> perspective-correct the shelf using 4 ArUco markers
    white_balance()  -> gray-world colour constancy
    denoise()        -> bilateral / non-local-means noise removal
    apply_clahe()    -> CLAHE on the L channel (fixes dim / uneven lighting)
    split_grid()     -> cut the rectified shelf into 3x3 cells

All functions take and return BGR uint8 images (OpenCV convention).
"""
import cv2
import numpy as np

# ----------------------------------------------------------------------------
# ArUco markers: IDs 0,1,2,3 stuck at the shelf's Top-Left, Top-Right,
# Bottom-Right, Bottom-Left corners (see tools/make_aruco_markers.py)
# ----------------------------------------------------------------------------
ARUCO_DICT = cv2.aruco.DICT_4X4_50
CORNER_IDS = (0, 1, 2, 3)  # TL, TR, BR, BL


def _aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


def detect_markers(img):
    """Return {marker_id: (4,2) corner array} for every marker found."""
    corners, ids, _ = _aruco_detector().detectMarkers(img)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)}


def rectify_shelf(img, out_size=(900, 900)):
    """
    Warp the shelf to a flat, top-down (front-on) view.

    Uses the CENTRE of each corner marker as the reference point, so the
    rectified image spans marker-centre to marker-centre. Returns
    (rectified_img, homography) or (None, None) if any marker is missing.
    """
    markers = detect_markers(img)
    if not all(i in markers for i in CORNER_IDS):
        return None, None
    src = np.float32([markers[i].mean(axis=0) for i in CORNER_IDS])
    w, h = out_size
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, H, out_size), H


# ----------------------------------------------------------------------------
# Photometric corrections
# ----------------------------------------------------------------------------
def white_balance(img):
    """Gray-world assumption: scale each channel so its mean equals the global mean."""
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    f *= means.mean() / np.maximum(means, 1e-6)
    return np.clip(f, 0, 255).astype(np.uint8)


def denoise(img, method="bilateral"):
    if method == "bilateral":   # edge-preserving, fast
        return cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)
    if method == "nlm":         # stronger, slower
        return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    if method == "median":
        return cv2.medianBlur(img, 3)
    raise ValueError(f"unknown denoise method: {method}")


def apply_clahe(img, clip_limit=2.0, tile_grid=8):
    """Contrast Limited Adaptive Histogram Equalisation on luminance only (LAB 'L')."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


DEFAULT_CFG = {"white_balance": True, "denoise": "bilateral", "clahe": True}


def preprocess(img, cfg=None):
    """Photometric pipeline used before detection. cfg keys: white_balance, denoise, clahe."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    out = img
    if cfg["white_balance"]:
        out = white_balance(out)
    if cfg["denoise"]:
        out = denoise(out, cfg["denoise"])
    if cfg["clahe"]:
        out = apply_clahe(out)
    return out


# ----------------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------------
def grid_cells(width, height, rows=3, cols=3):
    """List of (row, col, (x1, y1, x2, y2)) boxes covering the image."""
    cells = []
    for r in range(rows):
        for c in range(cols):
            x1, x2 = int(c * width / cols), int((c + 1) * width / cols)
            y1, y2 = int(r * height / rows), int((r + 1) * height / rows)
            cells.append((r, c, (x1, y1, x2, y2)))
    return cells


def split_grid(img, rows=3, cols=3):
    """Return [(row, col, cell_img), ...] for a rectified shelf image."""
    h, w = img.shape[:2]
    return [(r, c, img[y1:y2, x1:x2]) for r, c, (x1, y1, x2, y2) in grid_cells(w, h, rows, cols)]


def cell_of_point(x, y, width, height, rows=3, cols=3):
    """Which (row, col) cell a pixel falls in."""
    c = min(int(x / width * cols), cols - 1)
    r = min(int(y / height * rows), rows - 1)
    return r, c


def draw_grid(img, rows=3, cols=3, color=(0, 255, 255)):
    out = img.copy()
    h, w = out.shape[:2]
    for r, c, (x1, y1, x2, y2) in grid_cells(w, h, rows, cols):
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{r},{c}", (x1 + 6, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


# ----------------------------------------------------------------------------
# Controlled degradations for the lighting / noise ablation study
# ----------------------------------------------------------------------------
def degrade(img, kind, seed=0):
    """
    kind:
      'dim'     - gamma 2.2 darkening (poorly lit store)
      'uneven'  - strong left-to-right illumination gradient (side lighting)
      'noise'   - Gaussian sensor noise, sigma=20 (cheap webcam in low light)
      'blur'    - slight defocus / motion blur
      'combined'- dim + uneven + noise
    """
    rng = np.random.default_rng(seed)
    f = img.astype(np.float32) / 255.0
    if kind in ("dim", "combined"):
        f = f ** 2.2
    if kind in ("uneven", "combined"):
        h, w = f.shape[:2]
        grad = np.linspace(0.25, 1.0, w, dtype=np.float32)[None, :, None]
        f = f * grad
    if kind in ("noise", "combined"):
        f = f + rng.normal(0, 20 / 255.0, f.shape).astype(np.float32)
    if kind == "blur":
        f = cv2.GaussianBlur(f, (7, 7), 2.0)
    if kind not in ("dim", "uneven", "noise", "blur", "combined"):
        raise ValueError(f"unknown degradation: {kind}")
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def image_stats(img):
    """Mean brightness, RMS contrast and a Laplacian sharpness score (for reporting)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }
