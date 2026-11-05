#!/usr/bin/env python3
"""
LR x init-variant x seed sweep for the image-condition, matched-caption
wug/wugs experiment. Dispatches one job per GPU across a fixed GPU pool via
gpu_pool.run_job_pool; each job calls a dev-eval fork (never the original
core/train/embed_train.py) — --train_script/--eval_csv select which dev
metric drives early stopping (default: the noun-vs-verb margin fork).

Usage (full sweep, default noun-vs-verb margin fork):
  python3 eacl-rebuttal/scripts/run_sweep.py --gpus 0,1,2

Usage (held-out-image logit-gap fork instead):
  python3 eacl-rebuttal/scripts/run_sweep.py --gpus 0,1,2 \
      --train_script eacl-rebuttal/scripts/embed_train_dev_image.py \
      --eval_csv eacl-rebuttal/data/dev/dev_image_number.csv \
      --out_dir eacl-rebuttal/results/sweep_image_dev

Usage (tiny smoke test):
  python3 eacl-rebuttal/scripts/run_sweep.py --gpus 0 --lrs 0.001 --seeds 17 \
      --epochs 3 --out_dir eacl-rebuttal/results/smoke_test
"""
import argparse
import csv
import glob
import os
import pathlib
import sys

from gpu_pool import run_job_pool

_REPO_ROOT = pathlib.Path(__file__).resolve().parent
while not (_REPO_ROOT / "core").is_dir():
    _parent = _REPO_ROOT.parent
    if _parent == _REPO_ROOT:
        raise RuntimeError(f"Could not find repo root (no core/ dir above {__file__})")
    _REPO_ROOT = _parent

DEFAULT_TRAIN_SCRIPT = _REPO_ROOT / "eacl-rebuttal" / "scripts" / "embed_train_dev_margin.py"
IMAGE_DIR = _REPO_ROOT / "eacl-rebuttal" / "data" / "train" / "im" / "creature_1"
TRAIN_CSV = _REPO_ROOT / "eacl-rebuttal" / "data" / "train" / "text" / "image_train_matched_v2.csv"
DEFAULT_EVAL_CSV = _REPO_ROOT / "eacl-rebuttal" / "data" / "dev" / "dev_noun_vs_verb.csv"

DEFAULT_LRS = [0.0001, 0.0003, 0.0005, 0.0007, 0.0009,
                0.001, 0.003, 0.005, 0.0075, 0.01, 0.03, 0.05, 0.075, 0.1]
DEFAULT_SEEDS = [17, 42, 123, 2024, 98765]
DEFAULT_INITS = [
    ("small", str(_REPO_ROOT / "data" / "embeddings" / "init" / "noun_init.txt")),
    ("large", str(_REPO_ROOT / "eacl-rebuttal" / "data" / "init" / "noun_init_large.txt")),
]


def lr_tag(lr):
    return f"{lr:.10f}".rstrip("0").rstrip(".").replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2", help="Comma-separated GPU ids")
    parser.add_argument("--lrs", default=",".join(str(x) for x in DEFAULT_LRS))
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--inits", default=None,
                        help="Comma-separated tag:path pairs. Default: small+large noun init.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--terminate_cond_epochs", type=int, default=5)
    parser.add_argument("--batch_mode", default="alternating", choices=["alternating", "joint"],
                        help="'alternating' (default): separate singular/plural steps, never mixed. "
                             "'joint': one combined step per batch, singular+plural together.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Examples per class per optimizer step (default: 1, i.e. single-example "
                             "steps). E.g. --batch_mode joint --batch_size 8 gives 8 singular + 8 plural "
                             "combined per step.")
    parser.add_argument("--train_script", default=str(DEFAULT_TRAIN_SCRIPT),
                        help="Which dev-eval fork to run (default: noun-vs-verb margin fork).")
    parser.add_argument("--eval_csv", default=str(DEFAULT_EVAL_CSV),
                        help="Dev-set CSV matching --train_script's expected format.")
    parser.add_argument("--training_condition", default="image", choices=["image", "syntax"],
                        help="'image': train with images. 'syntax': train with text-only "
                             "(dev/test can still use images independently of this).")
    parser.add_argument("--train_csv", default=None,
                        help="Override the default TRAIN_CSV (e.g. for syntax-condition stimuli).")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--cache_dir", default="/home/shared/hf_cache")
    parser.add_argument("--out_dir", default=str(_REPO_ROOT / "eacl-rebuttal" / "results" / "sweep"))
    parser.add_argument("--force", action="store_true",
                        help="Re-run jobs even if a learned_embeddings.pt already exists for them "
                             "(default: skip already-completed jobs, e.g. after a stopped/resumed sweep).")
    parser.add_argument("--write_all", action="store_true",
                        help="Pass --write_all to the training script so every epoch's embeddings "
                             "are saved (useful for retrospective checkpoint-selection ablations).")
    parser.add_argument("--init_mode",
                        choices=["independent", "symmetric", "spherical_symmetric"],
                        default="independent")
    parser.add_argument("--symmetric_separation_scales", default=None,
                        help="Comma-separated exact separation scales for spherical_symmetric "
                             "initialization. Each is a multiple of the mean noun-pair distance.")
    parser.add_argument("--selection_metric",
                        choices=["raw_margin", "balanced_logistic"],
                        default="raw_margin")
    args = parser.parse_args()

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    separation_scales = ([float(x) for x in args.symmetric_separation_scales.split(",")
                          if x.strip()]
                         if args.symmetric_separation_scales is not None else [None])
    if any(x is not None and x < 0 for x in separation_scales):
        parser.error("--symmetric_separation_scales must be non-negative")
    if args.symmetric_separation_scales is not None and args.init_mode != "spherical_symmetric":
        parser.error("--symmetric_separation_scales requires --init_mode spherical_symmetric")
    if args.inits:
        inits = [tuple(p.split(":", 1)) for p in args.inits.split(",")]
    else:
        inits = DEFAULT_INITS

    for tag, path in inits:
        assert os.path.isfile(path), f"init file for '{tag}' not found: {path}"

    # Resolve to absolute so manifest.csv's run_dir is valid regardless of the
    # cwd that later reads it (run_test_eval.py / aggregate_results.py).
    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"

    # Non-default batch settings get their own path/name suffix so they can't collide
    # with (or be mistaken by --force-less resume logic for) the default alternating/bs1 runs.
    is_default_batch = (args.batch_mode == "alternating" and args.batch_size == 1)
    batch_suffix = "" if is_default_batch else f"_{args.batch_mode}_bs{args.batch_size}"

    jobs = []
    manifest_rows = []
    n_skipped = 0
    for init_tag, init_path in inits:
        for separation_scale in separation_scales:
            scale_suffix = ("" if separation_scale is None
                            else f"_sep{lr_tag(separation_scale)}")
            for lr in lrs:
                for seed in seeds:
                    run_out = (out_dir / f"init_{init_tag}" /
                               f"lr_{lr_tag(lr)}{batch_suffix}{scale_suffix}" / f"seed_{seed}")
                    name = (f"init_{init_tag}_lr{lr_tag(lr)}{batch_suffix}"
                            f"{scale_suffix}_seed{seed}")

                    already_done = not args.force and glob.glob(
                        str(run_out / "**" / "learned_embeddings.pt"), recursive=True)
                    if already_done:
                        n_skipped += 1
                        manifest_rows.append({
                            "name": name, "init_tag": init_tag, "init_path": init_path,
                            "lr": lr, "seed": seed, "separation_scale": separation_scale,
                            "run_dir": str(run_out), "returncode": 0,
                        })
                        continue

                    train_csv = args.train_csv if args.train_csv else str(TRAIN_CSV)
                    argv = [
                        sys.executable, args.train_script,
                        "--training_condition", args.training_condition,
                        "--train_csv", train_csv,
                        "--eval_csv", args.eval_csv,
                        "--image_dir", str(IMAGE_DIR),
                        "--embed_init", init_path,
                        "--lr", str(lr),
                        "--seed", str(seed),
                        "--epochs", str(args.epochs),
                        "--terminate_cond_epochs", str(args.terminate_cond_epochs),
                        "--batch_mode", args.batch_mode,
                        "--batch_size", str(args.batch_size),
                        "--model", args.model,
                        "--cache_dir", args.cache_dir,
                        "--out_dir", str(run_out),
                        "--init_mode", args.init_mode,
                        "--selection_metric", args.selection_metric,
                        "--write_embeddings",
                    ]
                    if separation_scale is not None:
                        argv.extend(["--symmetric_separation_scale", str(separation_scale)])
                    if args.write_all:
                        argv.append("--write_all")
                    jobs.append({"name": name, "argv": argv})
                    manifest_rows.append({
                        "name": name, "init_tag": init_tag, "init_path": init_path,
                        "lr": lr, "seed": seed, "separation_scale": separation_scale,
                        "run_dir": str(run_out),
                    })

    total = len(jobs) + n_skipped
    print(f"Sweep: {len(inits)} inits x {len(separation_scales)} separation scales x "
          f"{len(lrs)} lrs x {len(seeds)} seeds = {total} cells "
          f"({n_skipped} already done, {len(jobs)} to run)")
    print(f"GPUs: {gpu_ids}  |  out_dir: {out_dir}")

    results = run_job_pool(jobs, gpu_ids, log_dir=str(log_dir)) if jobs else []

    ret_by_name = {r["name"]: r["returncode"] for r in results}
    for row in manifest_rows:
        if row["name"] in ret_by_name:
            row["returncode"] = ret_by_name[row["name"]]
        # else: already-done row, returncode=0 was set when it was skipped above

    # Merge with any existing manifest.csv rather than overwrite it — this script is
    # routinely invoked again with a narrower --lrs/--seeds/--inits scope (e.g. one
    # follow-up config), and a blind overwrite would silently drop every other run
    # already recorded for this out_dir.
    manifest_path = out_dir / "manifest.csv"
    by_name = {}
    if manifest_path.is_file():
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                by_name[row["name"]] = row
    for row in manifest_rows:
        by_name[row["name"]] = row
    all_rows = list(by_name.values())
    fieldnames = list(dict.fromkeys(key for row in all_rows for key in row.keys()))
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote manifest: {manifest_path} ({len(all_rows)} total rows)")

    n_fail = sum(1 for r in manifest_rows if r["returncode"] != 0)
    if n_fail:
        print(f"WARNING: {n_fail}/{len(manifest_rows)} runs failed — see logs under {log_dir}")


if __name__ == "__main__":
    main()
