#!/usr/bin/env python3
"""Scan experiment artifacts and build an availability index for the paper."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


ARTIFACT_ROOTS = [
    Path("/root/autodl-tmp/artifacts"),
    Path("/root/autodl-tmp/artifacts_rerun"),
    Path("/root/autodl-tmp/artifacts_gate1"),
    Path("/root/autodl-tmp/artifacts_gate2"),
]

TARGET_MODELS = [
    "dcd",
    "dice",
    "invcf",
    "pdcl_b05",
    "apdcl_bmax10",
    "paac",
]
TARGET_DATASETS = ["yelp", "amazon"]
TARGET_SEEDS = [42, 43, 44]


@dataclass
class RunRecord:
    artifact_root: str
    run_dir: str
    run_name: str
    dataset: str | None
    seed: int | None
    method: str | None
    normalized_model: str | None
    pd_beta: float | None
    pd_adaptive: bool
    pd_beta_max: float | None
    cl_rate: float | None
    checkpoint_path: str | None
    checkpoint_kind: str | None
    has_checkpoint: bool
    has_config: bool
    has_metrics: bool
    recall: float | None
    ndcg: float | None
    coverage: float | None
    gini: float | None
    arp: float | None
    tail_exposure: float | None


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_seed(run_name: str, config: dict[str, Any]) -> int | None:
    seed = config.get("seed")
    if isinstance(seed, int):
        return seed
    for token in run_name.split("_"):
        if token.startswith("s") and token[1:].isdigit():
            return int(token[1:])
    return None


def first_existing(run_dir: Path, names: list[str]) -> tuple[Path | None, str | None]:
    for name in names:
        path = run_dir / name
        if path.exists():
            return path, name
    return None, None


def normalize_model(run_name: str, config: dict[str, Any]) -> str | None:
    method = str(config.get("method", "")).lower().strip()
    pd_beta = config.get("pd_beta")
    pd_adaptive = bool(config.get("pd_adaptive", False))
    pd_beta_max = config.get("pd_beta_max")

    if method == "paac":
        return "paac"
    if method == "invcf":
        return "invcf"
    if method == "lightgcn_cl":
        return "lightgcn_cl"
    if method == "dice":
        if pd_adaptive:
            if pd_beta_max is not None:
                return f"apdcl_bmax{float(pd_beta_max):.2f}".replace(".", "")
            return "apdcl"
        if pd_beta is not None and float(pd_beta) > 0:
            return f"pdcl_b{float(pd_beta):.2f}".replace(".", "")
        if float(config.get("cl_rate", 0.0)) > 0:
            return "dcd"
        return "dice"

    lowered = run_name.lower()
    if "apdcl" in lowered:
        return "apdcl_bmax10" if "bmax10" in lowered else "apdcl"
    if "pdcl_b05" in lowered:
        return "pdcl_b05"
    return method or None


def canonical_model(model: str | None) -> str | None:
    if model is None:
        return None
    aliases = {
        "pdcl_b050": "pdcl_b05",
        "pdcl_b075": "pdcl_b075",
        "pdcl_b100": "pdcl_b10",
        "apdcl_bmax100": "apdcl_bmax10",
        "apdcl_bmax050": "apdcl_bmax05",
    }
    return aliases.get(model, model)


def score_record(record: RunRecord) -> tuple[int, int, int, int]:
    root_rank = {
        "/root/autodl-tmp/artifacts_gate2": 4,
        "/root/autodl-tmp/artifacts_gate1": 3,
        "/root/autodl-tmp/artifacts_rerun": 2,
        "/root/autodl-tmp/artifacts": 1,
    }.get(record.artifact_root, 0)
    metric_rank = int(record.has_metrics)
    ckpt_rank = int(record.has_checkpoint)
    return (metric_rank, ckpt_rank, root_rank, 1)


def scan_one_run(run_dir: Path, artifact_root: Path) -> RunRecord | None:
    if not run_dir.is_dir():
        return None

    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "test_metrics_merged.json"
    if not config_path.exists() and not metrics_path.exists():
        return None

    config = load_yaml(config_path) if config_path.exists() else {}
    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    checkpoint_path, checkpoint_kind = first_existing(
        run_dir,
        ["best_model.pt", "best_student.pt", "concat_mlp_model.pt"],
    )
    global_metrics = metrics.get("global", metrics.get("test", {}))

    model = canonical_model(normalize_model(run_dir.name, config))
    return RunRecord(
        artifact_root=str(artifact_root),
        run_dir=str(run_dir),
        run_name=run_dir.name,
        dataset=config.get("dataset"),
        seed=parse_seed(run_dir.name, config),
        method=config.get("method"),
        normalized_model=model,
        pd_beta=float(config["pd_beta"]) if config.get("pd_beta") is not None else None,
        pd_adaptive=bool(config.get("pd_adaptive", False)),
        pd_beta_max=float(config["pd_beta_max"]) if config.get("pd_beta_max") is not None else None,
        cl_rate=float(config["cl_rate"]) if config.get("cl_rate") is not None else None,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        checkpoint_kind=checkpoint_kind,
        has_checkpoint=checkpoint_path is not None,
        has_config=config_path.exists(),
        has_metrics=metrics_path.exists(),
        recall=global_metrics.get("recall"),
        ndcg=global_metrics.get("ndcg"),
        coverage=global_metrics.get("coverage"),
        gini=global_metrics.get("gini"),
        arp=global_metrics.get("arp"),
        tail_exposure=global_metrics.get("tail_exposure"),
    )


def choose_best(records: list[RunRecord]) -> RunRecord | None:
    if not records:
        return None
    return sorted(records, key=score_record, reverse=True)[0]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    all_records: list[RunRecord] = []
    for artifact_root in ARTIFACT_ROOTS:
        if not artifact_root.exists():
            continue
        for child in sorted(artifact_root.iterdir()):
            record = scan_one_run(child, artifact_root)
            if record is not None:
                all_records.append(record)

    record_dicts = [asdict(r) for r in all_records]
    write_json(args.output_dir / "artifacts_index.json", record_dicts)
    write_csv(args.output_dir / "artifacts_index.csv", record_dicts)

    grouped: dict[tuple[str, str, int], list[RunRecord]] = {}
    for record in all_records:
        if record.dataset is None or record.seed is None or record.normalized_model is None:
            continue
        grouped.setdefault((record.normalized_model, record.dataset, record.seed), []).append(record)

    selected_rows: list[dict[str, Any]] = []
    target_summary: dict[str, dict[str, dict[str, Any]]] = {}

    for model in TARGET_MODELS:
        target_summary[model] = {}
        for dataset in TARGET_DATASETS:
            dataset_summary: dict[str, Any] = {}
            for seed in TARGET_SEEDS:
                best = choose_best(grouped.get((model, dataset, seed), []))
                key = f"s{seed}"
                dataset_summary[key] = asdict(best) if best else None
                if best:
                    selected_rows.append(
                        {
                            "normalized_model": model,
                            "dataset": dataset,
                            "seed": seed,
                            "run_dir": best.run_dir,
                            "artifact_root": best.artifact_root,
                            "checkpoint_path": best.checkpoint_path,
                            "checkpoint_kind": best.checkpoint_kind,
                            "recall": best.recall,
                            "ndcg": best.ndcg,
                            "coverage": best.coverage,
                            "tail_exposure": best.tail_exposure,
                        }
                    )
            target_summary[model][dataset] = dataset_summary

    write_json(args.output_dir / "target_run_map.json", target_summary)
    write_csv(args.output_dir / "target_run_map.csv", selected_rows)

    availability = []
    for model in TARGET_MODELS:
        for dataset in TARGET_DATASETS:
            row: dict[str, Any] = {"normalized_model": model, "dataset": dataset}
            matched = 0
            for seed in TARGET_SEEDS:
                best = choose_best(grouped.get((model, dataset, seed), []))
                row[f"s{seed}"] = "ok" if best else "missing"
                matched += int(best is not None)
            row["matched_seeds"] = matched
            availability.append(row)
    write_csv(args.output_dir / "availability_matrix.csv", availability)

    summary = {
        "num_scanned_runs": len(all_records),
        "num_target_matches": len(selected_rows),
        "target_models": TARGET_MODELS,
        "target_datasets": TARGET_DATASETS,
        "target_seeds": TARGET_SEEDS,
    }
    write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
