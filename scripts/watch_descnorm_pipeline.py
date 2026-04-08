#!/usr/bin/env python3
"""Wait for DescNorm balanced shard, then run diversity and finalize results."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("/root/new_paper/results/llm_eval"))
    parser.add_argument("--model-name", default="/root/autodl-tmp/models/phi-2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-users", type=int, default=50)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--balanced-shard", default="descnorm3_balanced")
    parser.add_argument("--diversity-shard", default="descnorm3_diversity")
    parser.add_argument("--merged-name", default="descnorm3_merged")
    args = parser.parse_args()

    balanced_done = args.result_root / args.balanced_shard / "all_results.json"
    diversity_done = args.result_root / args.diversity_shard / "all_results.json"

    while not balanced_done.exists():
        time.sleep(args.poll_seconds)

    if not diversity_done.exists():
        run(
            [
                "python3",
                "/root/new_paper/llm_eval/run_debiasing_eval.py",
                "--model-name",
                args.model_name,
                "--dimensions",
                "diversity",
                "--n-users",
                str(args.n_users),
                "--seed",
                str(args.seed),
                "--result-subdir",
                args.diversity_shard,
                "--title-only",
            ]
        )

    run(
        [
            "python3",
            "/root/new_paper/scripts/finalize_descnorm_results.py",
            "--result-root",
            str(args.result_root),
            "--balanced-shard",
            args.balanced_shard,
            "--diversity-shard",
            args.diversity_shard,
            "--merged-name",
            args.merged_name,
        ]
    )


if __name__ == "__main__":
    main()
