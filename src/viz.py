"""Small plotting helpers shared by the notebooks."""
import cv2
import matplotlib.pyplot as plt


def show(images, titles=None, cols=None, size=4.5, suptitle=None, save=None):
    """Show BGR (or grayscale) images in a row/grid."""
    n = len(images)
    cols = cols or n
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, img in enumerate(images):
        ax = axes.flat[i]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if titles:
            ax.set_title(titles[i], fontsize=10)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()


def draw_dets(img, dets, color=(0, 200, 0)):
    """Draw detection dicts {cls, conf, box} on a copy of a BGR image."""
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{d['cls']} {d['conf']:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out
