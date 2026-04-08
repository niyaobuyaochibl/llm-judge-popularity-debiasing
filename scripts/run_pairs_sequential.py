#!/usr/bin/env python3
"""Run each model pair as a separate subprocess to avoid cumulative memory leaks.

Each subprocess loads the model, evaluates one pair, writes the JSON, and exits.
After all pairs finish, rebuild_result_dir.py creates all_results.json + summary.json.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PAIRS = {
    "yelp": [
        ("dcd", "dice"),
        ("dcd", "apdcl_bmax10"),
        ("dice", "apdcl_bmax10"),
        ("pdcl_b05", "apdcl_bmax10"),
        ("invcf", "paac"),
    ],
    "amazon": [
        ("dcd", "dice"),
        ("dcd", "apdcl_bmax10"),
        ("dice", "apdcl_bmax10"),
        ("pdcl_b05", "apdcl_bmax10"),
        ("invcf", "paac"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/root/autodl-tmp/models/phi-2")
    parser.add_argument("--datasets", nargs="+", default=["amazon"])
    parser.add_argument("--dimensions", nargs="+", default=["diversity"])
    parser.add_argument("--n-users", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result-subdir", type=str, required=True)
    parser.add_argument("--title-only", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    result_root = Path("/root/new_paper/results/llm_eval") / args.result_subdir
    result_root.mkdir(parents=True, exist_ok=True)
    full_env = {**os.environ, "PYTHONPATH": "/root/new_paper/llm_eval"}
    failed = []

    for dataset in args.datasets:
        for model_a, model_b in PAIRS[dataset]:
            for dimension in args.dimensions:
                out_file = result_root / f"{dataset}_{model_a}_vs_{model_b}_{dimension}.json"
                if out_file.exists():
                    print(f"[skip] {out_file.name} already exists", flush=True)
                    continue

                pair_tag = f"{model_a}_vs_{model_b}"
                cmd = [
                    sys.executable,
                    "/root/new_paper/llm_eval/run_debiasing_eval.py",
                    "--model-name", args.model_name,
                    "--datasets", dataset,
                    "--dimensions", dimension,
                    "--pair", pair_tag,
                    "--n-users", str(args.n_users),
                    "--seed", str(args.seed),
                    "--result-subdir", args.result_subdir,
                    "--skip-summary",
                ]
                if args.title_only:
                    cmd.append("--title-only")
                if args.load_in_4bit:
                    cmd.append("--load-in-4bit")

                print(f"[launch] {dataset} {pair_tag} {dimension}", flush=True)
                result = subprocess.run(cmd, env=full_env)
                if result.returncode != 0:
                    print(f"[FAIL] exit={result.returncode} for {out_file.name}", flush=True)
                    failed.append(out_file.name)
                else:
                    print(f"[done] {out_file.name}", flush=True)

    print(f"\n[run_pairs_sequential] finished. failed={len(failed)}", flush=True)
    if failed:
        for f in failed:
            print(f"  - {f}", flush=True)
    else:
        print("[run_pairs_sequential] rebuilding summary...", flush=True)
        subprocess.run(
            [sys.executable, "/root/new_paper/scripts/rebuild_result_dir.py",
             "--result-dir", str(result_root)],
            check=True,
        )
        print("[run_pairs_sequential] done!", flush=True)


if __name__ == "__main__":
    main()
