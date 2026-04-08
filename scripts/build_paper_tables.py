#!/usr/bin/env python3
"""Build paper-friendly CSV/LaTeX tables from repaired judge results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MODEL_LABELS = {
    "dcd": "DCD",
    "dice": "DICE",
    "invcf": "InvCF",
    "paac": "PAAC",
    "pdcl_b05": "PDCL",
    "apdcl_bmax10": "A-PDCL",
    "tie": "Tie",
    "unknown": "Unknown",
}


def pretty_label(value: str) -> str:
    return MODEL_LABELS.get(value, value.replace("_", r"\_"))


def load_records(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def write_pair_table(df: pd.DataFrame, out_csv: Path, out_tex: Path) -> None:
    pair_df = df[
        [
            "dataset",
            "dimension",
            "model_a",
            "model_b",
            "n_trials",
            "prefer_a",
            "prefer_b",
            "tie",
            "inconsistent",
            "consistency_rate",
            "llm_winner",
            "ndcg_winner",
            "coverage_winner",
        ]
    ].copy()
    pair_df["tie_rate"] = pair_df["tie"] / pair_df["n_trials"]
    pair_df.to_csv(out_csv, index=False)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Pair-level outcomes in the repaired main run. Preference counts are computed over 50 users per pair and prompt.}",
        r"\label{tab:pair_outcomes}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllrrrrcll}",
        r"\toprule",
        r"Dataset & Prompt & Model A & Model B & A & B & Tie & Inc. & Cons. & NDCG win & Coverage win \\",
        r"\midrule",
    ]
    for _, row in pair_df.iterrows():
        lines.append(
            f"{row['dataset'].capitalize()} & {row['dimension'].capitalize()} & {pretty_label(row['model_a'])} & {pretty_label(row['model_b'])} & "
            f"{int(row['prefer_a'])} & {int(row['prefer_b'])} & {int(row['tie'])} & {int(row['inconsistent'])} & "
            f"{row['consistency_rate']:.3f} & {pretty_label(row['ndcg_winner'])} & {pretty_label(row['coverage_winner'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}"])
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dimension_table(df: pd.DataFrame, out_tex: Path) -> None:
    summary = (
        df.groupby(["dataset", "dimension"])[["alignment_to_ndcg", "alignment_to_coverage", "consistency_rate"]]
        .mean()
        .reset_index()
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Average judge alignment and consistency in the repaired main run (`phi-2`, seed 42). Alignment values of 0.5 indicate tie-dominated behavior.}",
        r"\label{tab:main_alignment}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Prompt & Align-NDCG & Align-Coverage & Consistency \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['dataset'].capitalize()} & {row['dimension'].capitalize()} & "
            f"{row['alignment_to_ndcg']:.3f} & {row['alignment_to_coverage']:.3f} & {row['consistency_rate']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            f"\\multicolumn{{2}}{{l}}{{Overall average}} & {df['alignment_to_ndcg'].mean():.3f} & {df['alignment_to_coverage'].mean():.3f} & {df['consistency_rate'].mean():.3f} \\\\",
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
        ]
    )
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, required=True)
    args = parser.parse_args()

    df = load_records(args.result_dir / "all_results.json")
    write_pair_table(
        df,
        args.paper_dir / "table_pair_outcomes.csv",
        args.paper_dir / "table_pair_outcomes.tex",
    )
    write_dimension_table(df, args.paper_dir / "table_main_alignment.tex")


if __name__ == "__main__":
    main()
