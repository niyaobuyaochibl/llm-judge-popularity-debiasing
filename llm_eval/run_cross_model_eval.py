#!/usr/bin/env python3
"""Run cross-LLM validation on a reduced set of key pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/root/new_paper/llm_eval")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_config import DATASETS, LLM_EVAL_RESULTS_DIR
from judge_pipeline import JudgeConfig, LLMJudge
from run_debiasing_eval import run_pairwise_dimension_eval


KEY_PAIRS = {
    "yelp": [("dcd", "apdcl_bmax10"), ("pdcl_b05", "apdcl_bmax10")],
    "amazon": [("dcd", "apdcl_bmax10"), ("pdcl_b05", "apdcl_bmax10")],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--dimensions", nargs="+", default=["balanced", "diversity"])
    parser.add_argument("--n-users", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--result-subdir", type=str, default="cross_model")
    args = parser.parse_args()

    out_dir = LLM_EVAL_RESULTS_DIR / args.result_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    judge = LLMJudge(
        JudgeConfig(
            model_name=args.model_name,
            load_in_4bit=args.load_in_4bit,
            max_new_tokens=32,
        )
    )

    all_results = []
    for dataset in DATASETS:
        for model_a, model_b in KEY_PAIRS[dataset]:
            for dimension in args.dimensions:
                result = run_pairwise_dimension_eval(
                    judge=judge,
                    dataset=dataset,
                    model_a=model_a,
                    model_b=model_b,
                    dimension=dimension,
                    n_users=args.n_users,
                    seed=args.seed,
                )
                payload = result.__dict__
                all_results.append(payload)
                with (out_dir / f"{dataset}_{model_a}_vs_{model_b}_{dimension}.json").open("w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)

    with (out_dir / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    judge.cleanup()


if __name__ == "__main__":
    main()
