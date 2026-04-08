#!/usr/bin/env python3
"""Rebuild all_results.json and summary.json from per-record JSON files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def ordered_unique(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def load_existing_summary(result_dir: Path) -> dict:
    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--elapsed-seconds", type=float, default=None)
    args = parser.parse_args()

    records = []
    for path in sorted(args.result_dir.glob("*.json")):
        if path.name in {"all_results.json", "summary.json"}:
            continue
        with path.open("r", encoding="utf-8") as f:
            records.append(json.load(f))

    with (args.result_dir / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    existing_summary = load_existing_summary(args.result_dir)
    summary = {
        "model_name": args.model_name if args.model_name is not None else existing_summary.get("model_name"),
        "datasets": ordered_unique([r.get("dataset") for r in records]) or existing_summary.get("datasets", []),
        "dimensions": ordered_unique([r.get("dimension") for r in records]) or existing_summary.get("dimensions", []),
        "n_users": max((int(r.get("n_trials", 0)) for r in records), default=existing_summary.get("n_users", 0)),
        "seed": args.seed if args.seed is not None else existing_summary.get("seed"),
        "elapsed_seconds": args.elapsed_seconds if args.elapsed_seconds is not None else existing_summary.get("elapsed_seconds"),
        "num_records": len(records),
        "llm_winner_counts": Counter(r.get("llm_winner", "unknown") for r in records),
        "avg_consistency_rate": sum(r.get("consistency_rate", 0.0) for r in records) / max(1, len(records)),
        "avg_alignment_to_ndcg": sum(r.get("alignment_to_ndcg", 0.0) for r in records) / max(1, len(records)),
        "avg_alignment_to_coverage": sum(r.get("alignment_to_coverage", 0.0) for r in records) / max(1, len(records)),
    }
    with (args.result_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
