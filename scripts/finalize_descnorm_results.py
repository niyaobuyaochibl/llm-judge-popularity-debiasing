#!/usr/bin/env python3
"""Merge, repair, and analyze DescNorm shard outputs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def require_all_results(result_root: Path, shard: str) -> None:
    path = result_root / shard / "all_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing shard summary: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("/root/new_paper/results/llm_eval"))
    parser.add_argument("--balanced-shard", default="descnorm3_balanced")
    parser.add_argument("--diversity-shard", default="descnorm3_diversity")
    parser.add_argument("--merged-name", default="descnorm3_merged")
    args = parser.parse_args()

    require_all_results(args.result_root, args.balanced_shard)
    require_all_results(args.result_root, args.diversity_shard)

    run(
        [
            "python3",
            "/root/new_paper/scripts/merge_llm_results.py",
            "--result-root",
            str(args.result_root),
            "--target",
            args.merged_name,
            "--sources",
            args.balanced_shard,
            args.diversity_shard,
        ]
    )
    run(
        [
            "python3",
            "/root/new_paper/scripts/repair_judge_results.py",
            "--result-dir",
            str(args.result_root / args.merged_name),
        ]
    )
    run(
        [
            "python3",
            "/root/new_paper/llm_eval/analyze_results.py",
            "--result-subdir",
            args.merged_name,
        ]
    )


if __name__ == "__main__":
    main()
