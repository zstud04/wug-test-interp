#!/usr/bin/env python3
"""
Training and held-out dev images for the blob condition.

One fixed, lumpy blob shape (a smooth closed curve whose radius is modulated by
a few random harmonics, drawn once from a fixed seed) is used in every image,
so [wug] names one kind of thing and only the number of blobs varies. Style
follows the circle images (data/train/im/wugs_subset/): 394 x 394 white
canvas, pink fill (RGB 255,178,178) with a thin darker anti-aliased outline.
Each blob gets a random position and rotation; blobs never overlap.

  singular: 1 blob        plural: 2-4 blobs (counts balanced across images)

Writes
  data/train/im/blobs/singular{1..5}.png, plural{1..5}.png        (training)
  data/dev/held_out_images_blobs/{singular,plural}/*_{1..10}.png  (dev)
  data/dev/dev_image_number_blobs.csv                             (dev CSV, same format as dev_image_number.csv)
Training and dev images use different random streams, so no dev image
repeats a training layout.
"""
import pathlib

import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train" / "im" / "blobs"
DEV_DIR = ROOT / "data" / "dev" / "held_out_images_blobs"
DEV_CSV = ROOT / "data" / "dev" / "dev_image_number_blobs.csv"

SIZE, SS = 394, 4                      # canvas size, supersampling factor
FILL, EDGE = (255, 178, 178), (232, 148, 148)
EDGE_W = 1.5                           # outline width in final pixels
RADIUS = 50                            # mean blob radius (circles were ~47)
MARGIN, GAP = 5, 10


def blob_shape(seed, k=6, n=240):
    """The one blob outline, centred at the origin."""
    rng = np.random.default_rng(seed)
    amps = rng.uniform(0.08, 0.22, k) / np.arange(1, k + 1) ** 0.6
    phases = rng.uniform(0, 2 * np.pi, k)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = RADIUS * (1 + sum(a * np.cos((i + 2) * t + p) for i, (a, p) in enumerate(zip(amps, phases))))
    return np.stack([r * np.cos(t), r * np.sin(t)], 1)


SHAPE = blob_shape(seed=7)
R_MAX = np.linalg.norm(SHAPE, axis=1).max()


def place(rng, n):
    lo, hi = R_MAX + MARGIN, SIZE - R_MAX - MARGIN
    for _ in range(100000):
        c = rng.uniform(lo, hi, size=(n, 2))
        d = np.linalg.norm(c[:, None] - c[None], axis=-1)[np.triu_indices(n, 1)]
        if n == 1 or d.min() > 2 * R_MAX + GAP:
            return c
    raise RuntimeError(f"could not place {n} blobs")


def render(rng, n):
    big = Image.new("RGB", (SIZE * SS, SIZE * SS), (255, 255, 255))
    d = ImageDraw.Draw(big)
    for cx, cy in place(rng, n):
        a = rng.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        pts = SHAPE @ rot + [cx, cy]
        d.polygon([tuple(p * SS) for p in pts], fill=FILL, outline=EDGE, width=int(EDGE_W * SS))
    return big.resize((SIZE, SIZE), Image.LANCZOS)


def plural_counts(m):
    return [2 + (i % 3) for i in range(m)]   # 2, 3, 4, 2, 3, 4, ...


def main():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(101)
    for i in range(1, 6):
        render(rng, 1).save(TRAIN_DIR / f"singular{i}.png")
    for i, n in enumerate(plural_counts(5), 1):
        render(rng, n).save(TRAIN_DIR / f"plural{i}.png")

    rng = np.random.default_rng(202)
    rows = ["image_path,image_class"]
    for cls in ["singular", "plural"]:
        (DEV_DIR / cls).mkdir(parents=True, exist_ok=True)
        counts = [1] * 10 if cls == "singular" else plural_counts(10)
        for i, n in enumerate(counts, 1):
            path = DEV_DIR / cls / f"{cls}_{i}.png"
            render(rng, n).save(path)
            rows.append(f"{path},{cls}")
    DEV_CSV.write_text("\n".join(rows) + "\n")
    print(f"training images -> {TRAIN_DIR}\ndev images -> {DEV_DIR}\ndev csv -> {DEV_CSV}")


if __name__ == "__main__":
    main()
