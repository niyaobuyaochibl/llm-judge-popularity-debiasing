#!/usr/bin/env python3
"""Run balanced and dimension-specific LLM judging experiments."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path("/root/new_paper/llm_eval")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_utils import (
    format_recommendation_list,
    load_item_text_subset,
    load_metrics,
    load_popularity_bucket_map,
    load_recommendations,
    load_recommendation_user_ids,
    load_user_histories,
    render_user_profile_summary,
    summarize_list_popularity,
)
from experiment_config import DATASETS, LLM_EVAL_RESULTS_DIR, MODEL_PAIRS, N_USERS, SEED
from judge_pipeline import JudgeConfig, create_judge
from prompts import get_dimension_prompt


@dataclass
class PairExperimentResult:
    dataset: str
    model_a: str
    model_b: str
    dimension: str
    prompt_style: str
    title_only: bool
    n_trials: int
    prefer_a: int
    prefer_b: int
    tie: int
    inconsistent: int
    consistency_rate: float
    llm_winner: str
    ndcg_winner: str
    coverage_winner: str
    tail_exposure_winner: str
    gini_winner: str
    arp_winner: str
    alignment_to_ndcg: float
    alignment_to_coverage: float
    alignment_to_tail_exposure: float
    alignment_to_gini: float
    alignment_to_arp: float
    per_trial: list[dict]


def _ordered_unique(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _load_result_records(result_dir: Path) -> list[dict]:
    records = []
    for path in sorted(result_dir.glob("*.json")):
        if path.name in {"all_results.json", "summary.json"}:
            continue
        with path.open("r", encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def _avg(records: list[dict], key: str) -> float:
    values = [float(r[key]) for r in records if key in r and r[key] is not None]
    return sum(values) / max(1, len(values))


def _build_summary(records: list[dict], args: argparse.Namespace, elapsed_seconds: float) -> dict:
    datasets = _ordered_unique([r["dataset"] for r in records if r.get("dataset")]) or list(args.datasets)
    dimensions = _ordered_unique([r["dimension"] for r in records if r.get("dimension")]) or list(args.dimensions)
    n_users = max((int(r.get("n_trials", 0)) for r in records), default=args.n_users)
    model_name = args.api_model if args.use_api else args.model_name
    summary = {
        "model_name": model_name,
        "datasets": datasets,
        "dimensions": dimensions,
        "n_users": n_users,
        "seed": args.seed,
        "elapsed_seconds": elapsed_seconds,
        "num_records": len(records),
        "prompt_style": args.prompt_style,
        "title_only": args.title_only,
        "llm_winner_counts": Counter(r["llm_winner"] for r in records),
        "avg_consistency_rate": _avg(records, "consistency_rate"),
        "avg_alignment_to_ndcg": _avg(records, "alignment_to_ndcg"),
        "avg_alignment_to_coverage": _avg(records, "alignment_to_coverage"),
    }
    for key in ["alignment_to_tail_exposure", "alignment_to_gini", "alignment_to_arp"]:
        if any(key in r for r in records):
            summary[f"avg_{key}"] = _avg(records, key)
    return summary


def flip_verdict(verdict: str) -> str:
    return {"A": "B", "B": "A", "TIE": "TIE"}.get(verdict, "UNPARSEABLE")


def metric_winner(
    model_a: str,
    model_b: str,
    metric_a: float | None,
    metric_b: float | None,
    higher_is_better: bool = True,
) -> str:
    if metric_a is None or metric_b is None:
        return "unknown"
    if metric_a == metric_b:
        return "tie"
    if higher_is_better:
        return model_a if metric_a > metric_b else model_b
    return model_a if metric_a < metric_b else model_b


def alignment_score(llm_winner: str, reference_winner: str) -> float:
    if reference_winner == "unknown":
        return 0.0
    if llm_winner == reference_winner:
        return 1.0
    if llm_winner == "tie":
        return 0.5
    return 0.0


def _get_sample_users(dataset: str, model_a: str, model_b: str, n_users: int, seed: int) -> list[int]:
    cache_path = Path(f"/root/new_paper/rec_lists/{dataset}/sample_users_s{seed}.json")
    pair_key = f"{model_a}_vs_{model_b}"
    if cache_path.exists():
        with cache_path.open() as f:
            cache = json.load(f)
        if pair_key in cache:
            return cache[pair_key][:n_users]
    ids_a = set(load_recommendation_user_ids(dataset, model_a, seed))
    ids_b = set(load_recommendation_user_ids(dataset, model_b, seed))
    common_users = sorted(ids_a & ids_b)
    rng = random.Random(seed)
    rng.shuffle(common_users)
    return common_users[:n_users]


def run_pairwise_dimension_eval(
    judge,
    dataset: str,
    model_a: str,
    model_b: str,
    dimension: str,
    n_users: int,
    seed: int,
    title_only: bool = False,
    prompt_style: str = "standard",
) -> PairExperimentResult:
    sample_users = _get_sample_users(dataset, model_a, model_b, n_users, seed)
    sample_set = set(sample_users)

    lists_a = load_recommendations(dataset, model_a, seed, user_subset=sample_set)
    lists_b = load_recommendations(dataset, model_b, seed, user_subset=sample_set)
    user_histories = load_user_histories(dataset, tuple(sample_users))
    needed_items = set()
    for uid in sample_users:
        needed_items.update(int(x) for x in lists_a[uid])
        needed_items.update(int(x) for x in lists_b[uid])
        needed_items.update(user_histories.get(uid, []))
    item_texts = load_item_text_subset(dataset, tuple(sorted(needed_items)), title_only=title_only)
    popularity_bucket_map = load_popularity_bucket_map(dataset) if prompt_style == "popularity_aware" else None

    trials: list[dict] = []
    for uid in sample_users:
        profile = render_user_profile_summary(user_histories.get(uid, []), item_texts)
        list_a_summary = summarize_list_popularity(lists_a[uid], popularity_bucket_map) if popularity_bucket_map else None
        list_b_summary = summarize_list_popularity(lists_b[uid], popularity_bucket_map) if popularity_bucket_map else None
        list_a_text = format_recommendation_list(
            lists_a[uid],
            item_texts,
            popularity_bucket_map=popularity_bucket_map,
        )
        list_b_text = format_recommendation_list(
            lists_b[uid],
            item_texts,
            popularity_bucket_map=popularity_bucket_map,
        )

        sys_fwd, usr_fwd = get_dimension_prompt(
            dimension,
            profile,
            list_a_text,
            list_b_text,
            prompt_style=prompt_style,
            list_a_summary=list_a_summary["text"] if list_a_summary else None,
            list_b_summary=list_b_summary["text"] if list_b_summary else None,
        )
        sys_rev, usr_rev = get_dimension_prompt(
            dimension,
            profile,
            list_b_text,
            list_a_text,
            prompt_style=prompt_style,
            list_a_summary=list_b_summary["text"] if list_b_summary else None,
            list_b_summary=list_a_summary["text"] if list_a_summary else None,
        )
        res_fwd = judge.judge(sys_fwd, usr_fwd, {"user_id": uid, "direction": "forward", "prompt_style": prompt_style})
        res_rev = judge.judge(sys_rev, usr_rev, {"user_id": uid, "direction": "reverse", "prompt_style": prompt_style})

        rev_mapped = flip_verdict(res_rev.verdict)
        consistent = res_fwd.verdict == rev_mapped
        if consistent:
            if res_fwd.verdict == "A":
                effective = model_a
            elif res_fwd.verdict == "B":
                effective = model_b
            else:
                effective = "TIE"
        else:
            effective = "INCONSISTENT"

        trial_payload = {
            "user_id": uid,
            "fwd_verdict": res_fwd.verdict,
            "rev_verdict": res_rev.verdict,
            "rev_mapped": rev_mapped,
            "consistent": consistent,
            "effective_preference": effective,
            "latency_forward_ms": res_fwd.latency_ms,
            "latency_reverse_ms": res_rev.latency_ms,
            "raw_forward": res_fwd.raw_output,
            "raw_reverse": res_rev.raw_output,
        }
        if list_a_summary and list_b_summary:
            trial_payload["list_a_popularity"] = list_a_summary["counts"]
            trial_payload["list_b_popularity"] = list_b_summary["counts"]
            trial_payload["list_a_popularity_summary"] = list_a_summary["text"]
            trial_payload["list_b_popularity_summary"] = list_b_summary["text"]
        trials.append(trial_payload)

    prefer_a = sum(t["effective_preference"] == model_a for t in trials)
    prefer_b = sum(t["effective_preference"] == model_b for t in trials)
    tie = sum(t["effective_preference"] == "TIE" for t in trials)
    inconsistent = sum(t["effective_preference"] == "INCONSISTENT" for t in trials)
    consistency_rate = (len(trials) - inconsistent) / len(trials) if trials else 0.0
    llm_winner = model_a if prefer_a > prefer_b else (model_b if prefer_b > prefer_a else "tie")

    metrics_a = load_metrics(dataset, model_a, seed)
    metrics_b = load_metrics(dataset, model_b, seed)
    ndcg_winner = metric_winner(model_a, model_b, metrics_a.get("ndcg"), metrics_b.get("ndcg"), True)
    coverage_winner = metric_winner(model_a, model_b, metrics_a.get("coverage"), metrics_b.get("coverage"), True)
    tail_exposure_winner = metric_winner(model_a, model_b, metrics_a.get("tail_exposure"), metrics_b.get("tail_exposure"), True)
    gini_winner = metric_winner(model_a, model_b, metrics_a.get("gini"), metrics_b.get("gini"), False)
    arp_winner = metric_winner(model_a, model_b, metrics_a.get("arp"), metrics_b.get("arp"), False)

    return PairExperimentResult(
        dataset=dataset,
        model_a=model_a,
        model_b=model_b,
        dimension=dimension,
        prompt_style=prompt_style,
        title_only=title_only,
        n_trials=len(trials),
        prefer_a=prefer_a,
        prefer_b=prefer_b,
        tie=tie,
        inconsistent=inconsistent,
        consistency_rate=consistency_rate,
        llm_winner=llm_winner,
        ndcg_winner=ndcg_winner,
        coverage_winner=coverage_winner,
        tail_exposure_winner=tail_exposure_winner,
        gini_winner=gini_winner,
        arp_winner=arp_winner,
        alignment_to_ndcg=alignment_score(llm_winner, ndcg_winner),
        alignment_to_coverage=alignment_score(llm_winner, coverage_winner),
        alignment_to_tail_exposure=alignment_score(llm_winner, tail_exposure_winner),
        alignment_to_gini=alignment_score(llm_winner, gini_winner),
        alignment_to_arp=alignment_score(llm_winner, arp_winner),
        per_trial=trials,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dimensions", nargs="+", default=["balanced", "relevance", "diversity", "novelty"])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--pair", type=str, default=None, help="Run only this pair, e.g. dcd_vs_dice")
    parser.add_argument("--n-users", type=int, default=N_USERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--result-subdir", type=str, default="main")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--title-only", action="store_true")
    parser.add_argument("--skip-summary", action="store_true", help="Skip writing all_results/summary (for single-pair mode)")
    parser.add_argument("--use-api", action="store_true", help="Use API instead of local model")
    parser.add_argument("--api-model", type=str, default="qwen-plus", help="API model name (e.g. qwen-plus, moonshot-v1-8k)")
    parser.add_argument("--api-base", type=str, default=None, help="API base URL override")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens for judge output")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--prompt-style", choices=["standard", "popularity_aware"], default="standard")
    args = parser.parse_args()

    out_dir = LLM_EVAL_RESULTS_DIR / args.result_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[run_debiasing_eval] model={args.model_name} api={args.use_api} api_model={args.api_model} "
        f"datasets={args.datasets} dimensions={args.dimensions} n_users={args.n_users} "
        f"result_subdir={args.result_subdir} prompt_style={args.prompt_style} title_only={args.title_only}",
        flush=True,
    )

    judge_cfg = JudgeConfig(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_tokens,
        do_sample=False,
        use_api=args.use_api,
        api_model=args.api_model,
    )
    if args.api_base:
        judge_cfg.api_base = args.api_base
    if args.temperature is not None:
        judge_cfg.temperature = args.temperature
    judge = create_judge(judge_cfg)

    pair_filter = None
    if args.pair:
        parts = args.pair.split("_vs_")
        if len(parts) == 2:
            pair_filter = (parts[0], parts[1])

    new_results = []
    started = time.time()
    for dataset in args.datasets:
        for model_a, model_b in MODEL_PAIRS[dataset]:
            if pair_filter and (model_a, model_b) != pair_filter:
                continue
            for dimension in args.dimensions:
                out_path = out_dir / f"{dataset}_{model_a}_vs_{model_b}_{dimension}.json"
                if out_path.exists():
                    print(f"[run_debiasing_eval] skip {out_path.name} (exists)", flush=True)
                    continue
                print(
                    f"[run_debiasing_eval] running dataset={dataset} pair={model_a}_vs_{model_b} "
                    f"dimension={dimension}",
                    flush=True,
                )
                result = run_pairwise_dimension_eval(
                    judge=judge,
                    dataset=dataset,
                    model_a=model_a,
                    model_b=model_b,
                    dimension=dimension,
                    n_users=args.n_users,
                    seed=args.seed,
                    title_only=args.title_only,
                    prompt_style=args.prompt_style,
                )
                payload = asdict(result)
                new_results.append(payload)
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                print(f"[run_debiasing_eval] wrote {out_path.name}", flush=True)

    if not args.skip_summary:
        all_results = _load_result_records(out_dir)
        summary = _build_summary(all_results, args, time.time() - started)
        with (out_dir / "all_results.json").open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        total_records = len(all_results)
    else:
        total_records = len(new_results)
    print(
        f"[run_debiasing_eval] completed total_records={total_records} new_records={len(new_results)}",
        flush=True,
    )
    judge.cleanup()


if __name__ == "__main__":
    main()
