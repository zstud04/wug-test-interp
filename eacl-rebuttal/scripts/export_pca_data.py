#!/usr/bin/env python3
"""
Copy the files the PCA figures need out of the git-ignored results/ folder
into eacl-rebuttal/pca_data/ (tracked in git), so the figures can be
reproduced from a clone without the server-only run directories.

Run after scripts/movement_pca.py; scripts/plot_movement_pca.py and
analysis/movement-rebuttal.R read from pca_data/. Files are copied (never
moved) and checked byte-for-byte.

  pca_data/<model>/sg_pl_reduced.csv          PCA positions of the real nouns
  pca_data/<model>/wug_wugs_reduced_image.csv [wug]/[wugs] init + final positions per seed
  pca_data/<model>/wug_wugs_movement_image.csv movement along the sg-pl axis per seed
"""
import filecmp
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "movement-analysis"
DST = ROOT / "pca_data"
MODELS = ["Qwen_Qwen3-VL-2B-Instruct", "Qwen_Qwen3-VL-4B-Instruct"]
FILES = ["sg_pl_reduced.csv", "wug_wugs_reduced_image.csv", "wug_wugs_movement_image.csv"]


def main():
    for model in MODELS:
        (DST / model).mkdir(parents=True, exist_ok=True)
        for name in FILES:
            src, dst = SRC / model / name, DST / model / name
            if not src.is_file():
                raise FileNotFoundError(f"{src} missing — run scripts/movement_pca.py first")
            shutil.copy2(src, dst)
            assert filecmp.cmp(src, dst, shallow=False), dst
            print(f"copied {dst.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
