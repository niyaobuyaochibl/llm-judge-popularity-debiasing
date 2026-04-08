#!/usr/bin/env python3
"""Merge multiple llm_eval shard directories into one result directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("/root/new_paper/results/llm_eval"))
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    args = parser.parse_args()

    target_dir = args.result_root / args.target
    target_dir.mkdir(parents=True, exist_ok=True)

    merged = []
    for source in args.sources:
        source_dir = args.result_root / source
        with (source_dir / "all_results.json").open("r", encoding="utf-8") as f:
            merged.extend(json.load(f))
        for path in source_dir.glob("*.json"):
            if path.name in {"all_results.json", "summary.json"}:
                continue
            shutil.copy2(path, target_dir / path.name)

    with (target_dir / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    summary = {
        "num_records": len(merged),
        "sources": args.sources,
    }
    with (target_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
