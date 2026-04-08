#!/usr/bin/env python3
"""Build traditional metrics summary files from exported recommendation payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rec-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_dir in sorted(args.rec_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        metrics = {}
        for path in sorted(dataset_dir.glob(f"*_s{args.seed}.json")):
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            metrics[payload["model"]] = payload.get("metrics", {})
        out_path = args.output_dir / f"{dataset_dir.name}_metrics.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
