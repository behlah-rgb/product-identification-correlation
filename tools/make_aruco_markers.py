"""
Generate the 4 ArUco corner markers for the shelf as a printable A4 PDF.

    python tools/make_aruco_markers.py            -> tools/aruco_markers.pdf

Print at 100% scale (no "fit to page"). Stick marker 0 on the shelf's
top-left corner, 1 top-right, 2 bottom-right, 3 bottom-left - with the
CENTRE of each marker on the corner of the 3x3 grid area.
"""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARUCO_DICT = cv2.aruco.DICT_4X4_50   # must match src/preprocess.py
MARKER_CM = 6.0
LABELS = {0: "TOP-LEFT", 1: "TOP-RIGHT", 2: "BOTTOM-RIGHT", 3: "BOTTOM-LEFT"}


def main():
    out = Path(__file__).resolve().parent / "aruco_markers.pdf"
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    a4_w, a4_h = 8.27, 11.69
    size_in = MARKER_CM / 2.54
    fig = plt.figure(figsize=(a4_w, a4_h))
    fig.text(0.5, 0.965, "Smart Shelf ArUco markers (DICT_4X4_50) - print at 100% scale",
             ha="center", fontsize=11)
    positions = [(1.0, 6.6), (4.6, 6.6), (4.6, 1.6), (1.0, 1.6)]  # inches from bottom-left
    for mid, (x, y) in zip(range(4), positions):
        marker = cv2.aruco.generateImageMarker(dictionary, mid, 600)
        ax = fig.add_axes([x / a4_w, y / a4_h, size_in / a4_w, size_in / a4_h])
        ax.imshow(marker, cmap="gray", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        fig.text((x + size_in / 2) / a4_w, (y - 0.35) / a4_h,
                 f"ID {mid} - {LABELS[mid]}  ({MARKER_CM:.0f} cm)", ha="center", fontsize=10)
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
