#!/usr/bin/env python3
"""Run DescNorm variants of the debiasing evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path("/root/new_paper/llm_eval")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-users", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "run_debiasing_eval.py"),
        "--model-name",
        args.model_name,
        "--dimensions",
        "balanced",
        "diversity",
        "--n-users",
        str(args.n_users),
        "--seed",
        str(args.seed),
        "--result-subdir",
        "descnorm",
        "--title-only",
    ]
    if args.load_in_4bit:
        cmd.append("--load-in-4bit")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
