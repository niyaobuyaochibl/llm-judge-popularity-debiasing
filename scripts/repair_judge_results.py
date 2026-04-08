#!/usr/bin/env python3
"""Re-parse stored judge raw outputs and recompute experiment summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/root/new_paper/llm_eval")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judge_pipeline import parse_verdict


def flip_verdict(verdict: str) -> str:
    return {"A": "B", "B": "A", "TIE": "TIE"}.get(verdict, "UNPARSEABLE")


def recompute_record(record: dict) -> dict:
    model_a = record["model_a"]
    model_b = record["model_b"]
    trials = record["per_trial"]

    for trial in trials:
        fwd = parse_verdict(trial["raw_forward"])
        rev = parse_verdict(trial["raw_reverse"])
        rev_mapped = flip_verdict(rev)
        consistent = fwd == rev_mapped
        if consistent:
            if fwd == "A":
                effective = model_a
            elif fwd == "B":
                effective = model_b
            else:
                effective = "TIE"
        else:
            effective = "INCONSISTENT"

        trial["fwd_verdict"] = fwd
        trial["rev_verdict"] = rev
        trial["rev_mapped"] = rev_mapped
        trial["consistent"] = consistent
        trial["effective_preference"] = effective

    prefer_a = sum(t["effective_preference"] == model_a for t in trials)
    prefer_b = sum(t["effective_preference"] == model_b for t in trials)
    tie = sum(t["effective_preference"] == "TIE" for t in trials)
    inconsistent = sum(t["effective_preference"] == "INCONSISTENT" for t in trials)
    consistency_rate = (len(trials) - inconsistent) / len(trials) if trials else 0.0
    llm_winner = model_a if prefer_a > prefer_b else (model_b if prefer_b > prefer_a else "tie")

    record["prefer_a"] = prefer_a
    record["prefer_b"] = prefer_b
    record["tie"] = tie
    record["inconsistent"] = inconsistent
    record["consistency_rate"] = consistency_rate
    record["llm_winner"] = llm_winner
    record["alignment_to_ndcg"] = 1.0 if llm_winner == record["ndcg_winner"] else 0.5 if llm_winner == "tie" else 0.0
    record["alignment_to_coverage"] = 1.0 if llm_winner == record["coverage_winner"] else 0.5 if llm_winner == "tie" else 0.0
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for path in sorted(args.result_dir.glob("*.json")):
        if path.name in {"all_results.json", "summary.json"}:
            continue
        with path.open("r", encoding="utf-8") as f:
            record = json.load(f)
        record = recompute_record(record)
        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        records.append(record)

    with (args.result_dir / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    summary = {
        "num_records": len(records),
        "avg_consistency_rate": sum(r["consistency_rate"] for r in records) / max(1, len(records)),
        "avg_alignment_to_ndcg": sum(r["alignment_to_ndcg"] for r in records) / max(1, len(records)),
        "avg_alignment_to_coverage": sum(r["alignment_to_coverage"] for r in records) / max(1, len(records)),
    }
    with (args.result_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
