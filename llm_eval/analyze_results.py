#!/usr/bin/env python3
"""Aggregate judge results and create tables/figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiment_config import FIGURES_DIR, LLM_EVAL_RESULTS_DIR, TABLES_DIR


def load_records(result_dir: Path) -> pd.DataFrame:
    with (result_dir / "all_results.json").open("r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def save_alignment_table(df: pd.DataFrame, out_path: Path) -> None:
    table = (
        df.groupby(["dataset", "dimension"])[["alignment_to_ndcg", "alignment_to_coverage", "consistency_rate"]]
        .mean()
        .reset_index()
    )
    table.to_csv(out_path, index=False)


def plot_consistency(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 4.8))
    sns.barplot(data=df, x="dimension", y="consistency_rate", hue="dataset")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_alignment(df: pd.DataFrame, out_path: Path) -> None:
    melted = df.melt(
        id_vars=["dataset", "dimension"],
        value_vars=["alignment_to_ndcg", "alignment_to_coverage"],
        var_name="metric",
        value_name="alignment",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x="dimension", y="alignment", hue="metric")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-subdir", type=str, default="main")
    args = parser.parse_args()

    result_dir = LLM_EVAL_RESULTS_DIR / args.result_subdir
    table_dir = TABLES_DIR / args.result_subdir
    figure_dir = FIGURES_DIR / args.result_subdir
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = load_records(result_dir)
    save_alignment_table(df, table_dir / "alignment_summary.csv")
    df.to_csv(table_dir / "all_results.csv", index=False)

    plot_consistency(df, figure_dir / "consistency_by_dimension.png")
    plot_alignment(df, figure_dir / "alignment_by_dimension.png")

    summary = {
        "num_records": int(len(df)),
        "avg_consistency_rate": float(df["consistency_rate"].mean()),
        "avg_alignment_to_ndcg": float(df["alignment_to_ndcg"].mean()),
        "avg_alignment_to_coverage": float(df["alignment_to_coverage"].mean()),
    }
    with (table_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
