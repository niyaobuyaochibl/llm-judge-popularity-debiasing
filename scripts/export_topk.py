#!/usr/bin/env python3
"""Export top-k recommendation lists from selected checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path("/root")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train_debiasing_baseline import load_data


def build_model(cfg: dict[str, Any], data: dict[str, Any], device: str) -> torch.nn.Module:
    method = cfg["method"]
    if method == "dice":
        from src.models.dice import DICEConfig, DICEModel

        model_cfg = DICEConfig(
            num_users=data["num_users"],
            num_items=data["num_items"],
            embedding_dim=int(cfg.get("embedding_dim", 64)),
            num_layers=int(cfg.get("num_layers", 2)),
            l2_reg=float(cfg.get("l2_reg", 1e-5)),
            disc_weight=float(cfg.get("disc_weight", 0.01)),
            int_weight=float(cfg.get("int_weight", 1.0)),
            con_weight=float(cfg.get("con_weight", 1.0)),
            cl_rate=float(cfg.get("cl_rate", 0.0)),
            cl_temperature=float(cfg.get("cl_temperature", 0.2)),
            eps=float(cfg.get("eps", 0.1)),
            pd_beta=float(cfg.get("pd_beta", 0.0)),
            pd_adaptive=bool(cfg.get("pd_adaptive", False)),
            pd_beta_max=float(cfg.get("pd_beta_max", 0.0)),
            device=device,
        )
        return DICEModel(data["graph"], data["item_pop"], model_cfg).to(device)

    if method == "invcf":
        from src.models.invcf import InvCFConfig, InvCFModel

        model_cfg = InvCFConfig(
            num_users=data["num_users"],
            num_items=data["num_items"],
            embedding_dim=int(cfg.get("embedding_dim", 64)),
            num_layers=int(cfg.get("num_layers", 2)),
            l2_reg=float(cfg.get("l2_reg", 1e-5)),
            inv_weight=float(cfg.get("inv_weight", 0.1)),
            num_envs=int(cfg.get("num_envs", 3)),
            device=device,
        )
        return InvCFModel(data["graph"], data["pop_groups"], model_cfg).to(device)

    if method == "lightgcn_cl":
        from src.models.lightgcn_cl import LightGCNCLConfig, LightGCNCLModel

        model_cfg = LightGCNCLConfig(
            num_users=data["num_users"],
            num_items=data["num_items"],
            embedding_dim=int(cfg.get("embedding_dim", 64)),
            num_layers=int(cfg.get("num_layers", 2)),
            l2_reg=float(cfg.get("l2_reg", 1e-5)),
            cl_rate=float(cfg.get("cl_rate", 0.1)),
            cl_temperature=float(cfg.get("cl_temperature", 0.2)),
            eps=float(cfg.get("eps", 0.1)),
            device=device,
        )
        return LightGCNCLModel(data["graph"], model_cfg).to(device)

    if method == "paac":
        from src.models.paac import PAACConfig, PAACModel

        model_cfg = PAACConfig(
            num_users=data["num_users"],
            num_items=data["num_items"],
            embedding_dim=int(cfg.get("embedding_dim", 64)),
            num_layers=int(cfg.get("num_layers", 2)),
            l2_reg=float(cfg.get("l2_reg", 1e-4)),
            cl_rate=float(cfg.get("cl_rate", 0.2)),
            cl_temperature=float(cfg.get("cl_temperature", 0.2)),
            eps=float(cfg.get("eps", 0.2)),
            sa_rate=float(cfg.get("sa_rate", 1.0)),
            paac_lambda2=float(cfg.get("paac_lambda2", 0.2)),
            paac_gamma=float(cfg.get("paac_gamma", 0.2)),
            device=device,
        )
        return PAACModel(data["graph"], data["item_pop"], model_cfg).to(device)

    raise ValueError(f"Unsupported method: {method}")


def recommend_all(
    model: torch.nn.Module,
    ground_truth: dict[int, list[int]],
    train_graph: Any,
    k: int,
    device: str,
    chunk_size: int = 256,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        user_emb, _, _ = model.compute_user_item_embeddings()
        item_emb = model.fused_items()
        la_bias = getattr(model, "la_bias", None)

    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)
    item_emb_t = item_emb.T.contiguous()
    user_ids = list(ground_truth.keys())
    user_tensor = torch.tensor(user_ids, dtype=torch.long, device=device)

    recs = []
    for start in range(0, len(user_ids), chunk_size):
        end = min(start + chunk_size, len(user_ids))
        idx = user_tensor[start:end]
        scores = torch.index_select(user_emb, 0, idx) @ item_emb_t
        if la_bias is not None:
            scores = scores - la_bias.squeeze(1)

        for local_idx, uid in enumerate(user_ids[start:end]):
            row = scores[local_idx]
            seen = train_graph.get_user_items(int(uid))
            if seen.size > 0:
                seen_t = torch.from_numpy(seen).to(device=device, dtype=torch.long)
                seen_t = seen_t[seen_t < row.shape[0]]
                if seen_t.numel() > 0:
                    row = row.clone()
                    row.index_fill_(0, seen_t, -1e9)
            topk = torch.topk(row, k=min(k, row.shape[0])).indices.cpu().numpy()
            if len(topk) < k:
                topk = np.pad(topk, (0, k - len(topk)), constant_values=-1)
            recs.append(topk)
    return np.stack(recs)


def load_target_run_map(index_dir: Path) -> list[dict[str, Any]]:
    path = index_dir / "target_run_map.csv"
    rows = []
    with path.open("r", encoding="utf-8") as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header is None:
                header = line.split(",")
                continue
            values = line.split(",")
            rows.append(dict(zip(header, values)))
    return rows


def export_one(row: dict[str, Any], output_dir: Path, k: int, device: str) -> dict[str, Any]:
    run_dir = Path(row["run_dir"])
    cfg_path = run_dir / "config.yaml"
    metrics_path = run_dir / "test_metrics_merged.json"
    checkpoint_path = Path(row["checkpoint_path"])

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    data = load_data(cfg)
    model = build_model(cfg, data, device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    rec_matrix = recommend_all(model, data["test_gt"], data["graph"], k=k, device=device)
    user_ids = list(data["test_gt"].keys())

    recommendations = {
        f"user_{uid}": [int(x) for x in rec_matrix[idx].tolist() if int(x) >= 0]
        for idx, uid in enumerate(user_ids)
    }

    payload = {
        "model": row["normalized_model"],
        "dataset": row["dataset"],
        "seed": int(row["seed"]),
        "source_run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics.get("global", {}),
        "recommendations": recommendations,
    }

    ds_dir = output_dir / row["dataset"]
    ds_dir.mkdir(parents=True, exist_ok=True)
    out_path = ds_dir / f"{row['normalized_model']}_s{row['seed']}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return {
        "model": row["normalized_model"],
        "dataset": row["dataset"],
        "seed": int(row["seed"]),
        "output_path": str(out_path),
        "num_users": len(recommendations),
        "recall": payload["metrics"].get("recall"),
        "ndcg": payload["metrics"].get("ndcg"),
        "coverage": payload["metrics"].get("coverage"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_target_run_map(args.index_dir)
    selected = [r for r in rows if int(r["seed"]) == args.seed]

    summaries = []
    for row in selected:
        summaries.append(export_one(row, args.output_dir, k=args.k, device=device))

    with (args.output_dir / f"export_summary_seed{args.seed}.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
