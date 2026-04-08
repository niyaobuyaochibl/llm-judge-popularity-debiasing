#!/usr/bin/env python3
"""Run popularity-aware variants of the debiasing evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/new_paper/llm_eval")
DEFAULT_LOCAL_MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--dimensions", nargs="+", default=["balanced", "diversity"])
    parser.add_argument("--datasets", nargs="+", default=["yelp", "amazon"])
    parser.add_argument("--pair", type=str, default=None)
    parser.add_argument("--n-users", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--result-subdir", type=str, default="popaware")
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--api-model", type=str, default="qwen-plus")
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--skip-summary", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "run_debiasing_eval.py"),
        "--model-name",
        args.model_name,
        "--dimensions",
        *args.dimensions,
        "--datasets",
        *args.datasets,
        "--n-users",
        str(args.n_users),
        "--seed",
        str(args.seed),
        "--result-subdir",
        args.result_subdir,
        "--prompt-style",
        "popularity_aware",
        "--max-tokens",
        str(args.max_tokens),
    ]
    if args.pair:
        cmd.extend(["--pair", args.pair])
    if args.load_in_4bit:
        cmd.append("--load-in-4bit")
    if args.use_api:
        cmd.append("--use-api")
        cmd.extend(["--api-model", args.api_model])
        if args.api_base:
            cmd.extend(["--api-base", args.api_base])
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.skip_summary:
        cmd.append("--skip-summary")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
