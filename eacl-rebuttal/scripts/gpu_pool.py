#!/usr/bin/env python3
"""Run a list of subprocess jobs over a fixed pool of GPU ids, one job per GPU
at a time, launching the next queued job on a GPU as soon as it frees up.

Shared by run_sweep.py (training) and run_test_eval.py (scoring).
"""
import os
import subprocess
import time


def run_job_pool(jobs, gpu_ids, log_dir, poll_interval=5.0, cwd=None):
    """
    jobs: list of dicts with keys "name" (str, unique) and "argv" (list[str]).
    gpu_ids: list of ints/strings; CUDA_VISIBLE_DEVICES is pinned per job.
    log_dir: directory to write "<name>.log" (stdout+stderr) for each job.
    Returns: list of dicts (each input job dict plus "returncode" and "log_path"),
             in completion order.
    """
    os.makedirs(log_dir, exist_ok=True)
    free_gpus = list(gpu_ids)
    pending = list(jobs)
    running = []  # list of (proc, gpu, job, log_file_handle)
    results = []

    total = len(jobs)
    print(f"[gpu_pool] {total} jobs queued over {len(gpu_ids)} GPUs: {gpu_ids}")

    while pending or running:
        while pending and free_gpus:
            job = pending.pop(0)
            gpu = free_gpus.pop(0)
            log_path = os.path.join(log_dir, f"{job['name']}.log")
            log_f = open(log_path, "w")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            print(f"[gpu_pool] START {job['name']} on gpu {gpu} "
                  f"({total - len(pending) - len(running)}/{total})")
            proc = subprocess.Popen(job["argv"], stdout=log_f, stderr=subprocess.STDOUT,
                                     env=env, cwd=cwd)
            running.append((proc, gpu, job, log_f, log_path))

        time.sleep(poll_interval)

        still_running = []
        for proc, gpu, job, log_f, log_path in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, gpu, job, log_f, log_path))
                continue
            log_f.close()
            free_gpus.append(gpu)
            status = "OK" if ret == 0 else f"FAIL(rc={ret})"
            print(f"[gpu_pool] {status:10s} {job['name']} -> {log_path}")
            results.append({**job, "returncode": ret, "log_path": log_path})
        running = still_running

    n_fail = sum(1 for r in results if r["returncode"] != 0)
    print(f"[gpu_pool] done. {total - n_fail}/{total} succeeded.")
    return results
