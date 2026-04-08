#!/usr/bin/env python3
"""Build paper-ready traditional-metric tables and figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ["recall", "ndcg", "coverage", "gini", "arp", "tail_exposure"]
MODEL_ORDER = ["dcd", "dice", "invcf", "paac", "pdcl_b05", "apdcl_bmax10"]
MODEL_LABELS = {
    "dcd": "DCD",
    "dice": "DICE",
    "invcf": "InvCF",
    "paac": "PAAC",
    "pdcl_b05": "PDCL",
    "apdcl_bmax10": "A-PDCL",
}


def load_dataset_metrics(path: Path, dataset: str) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for model in MODEL_ORDER:
        metrics = raw[model]
        row = {"dataset": dataset, "model": model, "model_label": MODEL_LABELS[model]}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def write_table_tex(df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Traditional offline metrics for the six recommendation models. Higher is better for Recall, NDCG, Coverage, and Tail Exposure; lower is better for ARP and Gini concentration.}",
        r"\label{tab:traditional_metrics}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Model & Recall & NDCG & Coverage & Gini & ARP & Tail Exposure \\",
        r"\midrule",
    ]
    for dataset in ["yelp", "amazon"]:
        sub = df[df["dataset"] == dataset]
        for _, row in sub.iterrows():
            lines.append(
                f"{dataset.capitalize()} & {row['model_label']} & "
                f"{row['recall']:.4f} & {row['ndcg']:.4f} & {row['coverage']:.4f} & "
                f"{row['gini']:.4f} & {row['arp']:.4f} & {row['tail_exposure']:.6f} \\\\"
            )
        if dataset != "amazon":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_csv(df: pd.DataFrame, out_path: Path) -> None:
    df.to_csv(out_path, index=False)


def write_tradeoff_figure(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    colors = {
        "dcd": "#4c78a8",
        "dice": "#f58518",
        "invcf": "#54a24b",
        "paac": "#e45756",
        "pdcl_b05": "#72b7b2",
        "apdcl_bmax10": "#b279a2",
    }

    for ax, dataset in zip(axes, ["yelp", "amazon"]):
        sub = df[df["dataset"] == dataset]
        for _, row in sub.iterrows():
            ax.scatter(
                row["coverage"],
                row["ndcg"],
                s=120,
                color=colors[row["model"]],
                edgecolors="black",
                linewidths=0.6,
            )
            ax.annotate(
                row["model_label"],
                (row["coverage"], row["ndcg"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )
        ax.set_title(dataset.capitalize())
        ax.set_xlabel("Coverage")
        ax.set_ylabel("NDCG")
        ax.grid(alpha=0.25)

    fig.suptitle("Accuracy-Debiasing Trade-off Across Exported Models", fontsize=12)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    metric_root = Path("/root/new_paper/traditional_metrics")
    paper_root = Path("/root/new_paper/paper")

    df = pd.concat(
        [
            load_dataset_metrics(metric_root / "yelp_metrics.json", "yelp"),
            load_dataset_metrics(metric_root / "amazon_metrics.json", "amazon"),
        ],
        ignore_index=True,
    )
    write_table_csv(df, paper_root / "table_traditional_metrics.csv")
    write_table_tex(df, paper_root / "table_traditional_metrics.tex")
    write_tradeoff_figure(df, paper_root / "fig_tradeoff_ndcg_coverage.png")


if __name__ == "__main__":
    main()
