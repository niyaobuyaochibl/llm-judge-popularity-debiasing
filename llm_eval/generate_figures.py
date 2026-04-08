#!/usr/bin/env python3
"""Generate publication-quality figures for the cross-LLM judge paper."""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

OUT = Path("/root/new_paper/results/analysis/figures")
OUT.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = Path("/root/new_paper/results/analysis/cross_llm_summary.json")
summary = json.loads(SUMMARY_PATH.read_text())

judge_names = summary["judges"]
judge_short = summary.get("judge_short_labels", judge_names)
cloud_judges = summary.get("cloud_judges", [name for name in judge_names if "(local)" not in name])

DISPLAY_LABELS = {
    "Phi-2 (local)": "Phi-2\n(local)",
    "Qwen2.5-7B-Instruct(local)": "Qwen2.5-7B\n(local)",
    "Qwen-Plus": "Qwen\nPlus",
    "DeepSeek-V3": "DeepSeek\nV3",
    "MiniMax-M2.7": "MiniMax\nM2.7",
    "Kimi-K2": "Kimi\nK2",
}
COLOR_MAP = {
    "Phi-2 (local)": "#8B9DC3",
    "Qwen2.5-7B-Instruct(local)": "#F4A259",
    "Qwen-Plus": "#E8575A",
    "DeepSeek-V3": "#5B9BD5",
    "MiniMax-M2.7": "#70AD47",
    "Kimi-K2": "#FFC000",
}

display_labels = [DISPLAY_LABELS.get(name, name) for name in judge_names]
colors = [COLOR_MAP.get(name, "#999999") for name in judge_names]
means = [summary["consistency"][name]["mean"] for name in judge_names]
stds = [summary["consistency"][name]["std"] for name in judge_names]

fig, ax = plt.subplots(figsize=(9.2, 4.8))
x = np.arange(len(judge_names))
bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="black", linewidth=0.6, capsize=5,
              error_kw={"linewidth": 1.2})
ax.set_xticks(x)
ax.set_xticklabels(display_labels)
ax.set_ylabel("Bidirectional Consistency Rate")
ax.set_ylim(0, 1.15)
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(len(judge_names) - 0.05, 0.51, "random baseline", fontsize=8, color="gray", ha="right")
for idx, (bar, mean_value) in enumerate(zip(bars, means)):
    ax.text(bar.get_x() + bar.get_width() / 2.0, mean_value + stds[idx] + 0.03,
            f"{mean_value:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(OUT / "fig_consistency.pdf")
fig.savefig(OUT / "fig_consistency.png")
print("  Saved fig_consistency")

kappa = np.array(summary["cohen_kappa_matrix"])
fig, ax = plt.subplots(figsize=(7.4, 6.1))
mask = np.zeros_like(kappa, dtype=bool)
np.fill_diagonal(mask, True)
kappa_display = np.ma.array(kappa, mask=mask)
im = ax.imshow(kappa_display, cmap="YlOrRd", vmin=-0.1, vmax=1.0, aspect="equal")
ax.set_xticks(range(len(judge_short)))
ax.set_xticklabels(judge_short, rotation=30, ha="right")
ax.set_yticks(range(len(judge_short)))
ax.set_yticklabels(judge_short)
for i in range(len(judge_short)):
    for j in range(len(judge_short)):
        if i == j:
            ax.text(j, i, "1.00", ha="center", va="center", fontsize=10, color="gray")
        else:
            color = "white" if kappa[i][j] > 0.6 else "black"
            ax.text(j, i, f"{kappa[i][j]:.2f}", ha="center", va="center", fontsize=10, color=color)
plt.colorbar(im, ax=ax, shrink=0.8, label="Cohen's κ")
ax.set_title("Pairwise Cohen's κ Between LLM Judges")
fig.savefig(OUT / "fig_kappa_heatmap.pdf")
fig.savefig(OUT / "fig_kappa_heatmap.png")
print("  Saved fig_kappa_heatmap")

from analyze_cross_llm import all_judge_results, JUDGE_DIRS, DATASETS, common_keys

ndcg_align = {}
cov_align = {}
for judge_name in judge_names:
    ndcg_match = 0
    cov_match = 0
    for key in common_keys:
        data = all_judge_results[judge_name].get(key)
        if data:
            ndcg_match += 1 if data["alignment_to_ndcg"] == 1.0 else 0
            cov_match += 1 if data["alignment_to_coverage"] == 1.0 else 0
    ndcg_align[judge_name] = ndcg_match / len(common_keys)
    cov_align[judge_name] = cov_match / len(common_keys)

fig, ax = plt.subplots(figsize=(10.2, 4.8))
x = np.arange(len(judge_names))
width = 0.36
bars_ndcg = ax.bar(x - width / 2, [ndcg_align[name] for name in judge_names], width,
                   label="NDCG alignment", color="#5B9BD5", edgecolor="black", linewidth=0.5)
bars_cov = ax.bar(x + width / 2, [cov_align[name] for name in judge_names], width,
                  label="Coverage alignment", color="#70AD47", edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(judge_short)
ax.set_ylabel("Fraction of Correct Alignments")
ax.set_ylim(0, 1.0)
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend(loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Alignment of LLM Judges to Ground-Truth Metrics")
for bar_group in [bars_ndcg, bars_cov]:
    for bar in bar_group:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f"{height:.0%}",
                ha="center", va="bottom", fontsize=8)
fig.savefig(OUT / "fig_alignment.pdf")
fig.savefig(OUT / "fig_alignment.png")
print("  Saved fig_alignment")

agreement_counts = {"no_majority": 0, "3_of_4": 0, "4_of_4": 0}
for key in common_keys:
    winners = [all_judge_results[judge_name][key]["llm_winner"] for judge_name in cloud_judges]
    max_agree = Counter(winners).most_common(1)[0][1]
    if max_agree == 4:
        agreement_counts["4_of_4"] += 1
    elif max_agree == 3:
        agreement_counts["3_of_4"] += 1
    else:
        agreement_counts["no_majority"] += 1

fig, ax = plt.subplots(figsize=(6.2, 4.2))
labels = ["No majority\n(≤2 agree)", "3 of 4\nagree", "4 of 4\nagree"]
values = [agreement_counts["no_majority"], agreement_counts["3_of_4"], agreement_counts["4_of_4"]]
colors_pie = ["#E8575A", "#FFC000", "#70AD47"]
wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors_pie, autopct="%1.0f%%",
                                  startangle=90, textprops={"fontsize": 11})
for text in autotexts:
    text.set_fontweight("bold")
ax.set_title("Agreement Among 4 API-Based Judges")
fig.savefig(OUT / "fig_agreement_pie.pdf")
fig.savefig(OUT / "fig_agreement_pie.png")
print("  Saved fig_agreement_pie")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for dataset_index, dataset in enumerate(DATASETS):
    ax = axes[dataset_index]
    for judge_index, judge_name in enumerate(judge_names):
        rates = []
        for key in common_keys:
            if key[0] == dataset:
                data = all_judge_results[judge_name].get(key)
                if data:
                    rates.append(data["consistency_rate"])
        if rates:
            ax.bar(judge_index, np.mean(rates), color=colors[judge_index], edgecolor="black", linewidth=0.5)
            ax.errorbar(judge_index, np.mean(rates), yerr=np.std(rates), fmt="none", color="black", capsize=4)
    ax.set_xticks(range(len(judge_names)))
    ax.set_xticklabels(judge_short, rotation=25, ha="right")
    ax.set_title(f"{dataset.capitalize()} Dataset")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axes[0].set_ylabel("Bidirectional Consistency Rate")
fig.suptitle("Consistency Rate by Dataset", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_consistency_by_dataset.pdf")
fig.savefig(OUT / "fig_consistency_by_dataset.png")
print("  Saved fig_consistency_by_dataset")

print(f"\nAll figures saved to {OUT}/")
