#!/usr/bin/env python3
"""Cross-LLM judge analysis across all judges, with table and summary export."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULT_BASE = Path("/root/new_paper/results/llm_eval")

JUDGE_SPECS = [
    {
        "name": "Phi-2 (local)",
        "short": "Phi-2",
        "dirs": [RESULT_BASE / "descnorm3_balanced", RESULT_BASE / "descnorm3_diversity"],
        "is_local": True,
    },
    {
        "name": "Qwen2.5-7B-Instruct(local)",
        "short": "Qwen2.5-7B",
        "dirs": [RESULT_BASE / "crossllm_qwen25_7b_local"],
        "is_local": True,
    },
    {
        "name": "Qwen-Plus",
        "short": "Qwen-Plus",
        "dirs": [RESULT_BASE / "crossllm_qwen_plus"],
        "is_local": False,
    },
    {
        "name": "DeepSeek-V3",
        "short": "DeepSeek-V3",
        "dirs": [RESULT_BASE / "crossllm_deepseek"],
        "is_local": False,
    },
    {
        "name": "MiniMax-M2.7",
        "short": "MiniMax-M2.7",
        "dirs": [RESULT_BASE / "crossllm_minimax"],
        "is_local": False,
    },
    {
        "name": "Kimi-K2",
        "short": "Kimi-K2",
        "dirs": [RESULT_BASE / "crossllm_kimi"],
        "is_local": False,
    },
]

JUDGE_DIRS = {spec["name"]: spec["dirs"] for spec in JUDGE_SPECS}
JUDGE_SHORT = {spec["name"]: spec["short"] for spec in JUDGE_SPECS}
LOCAL_JUDGES = [spec["name"] for spec in JUDGE_SPECS if spec["is_local"]]
CLOUD_JUDGES = [spec["name"] for spec in JUDGE_SPECS if not spec["is_local"]]

DATASETS = ["yelp", "amazon"]
PAIRS = [
    ("dcd", "dice"),
    ("dcd", "apdcl_bmax10"),
    ("dice", "apdcl_bmax10"),
    ("pdcl_b05", "apdcl_bmax10"),
    ("invcf", "paac"),
]
DIMENSIONS = ["balanced", "diversity"]


def load_results(dirs):
    """Load all JSON result files from a list of directories."""
    results = {}
    for directory in dirs:
        for result_file in sorted(directory.glob("*.json")):
            if "all_results" in result_file.name or "summary" in result_file.name:
                continue
            data = json.loads(result_file.read_text())
            key = (data["dataset"], data["model_a"], data["model_b"], data["dimension"])
            results[key] = data
    return results


def fleiss_kappa_multi(ratings_matrix):
    """Compute Fleiss' kappa for multiple raters."""
    n_subjects, n_categories = ratings_matrix.shape
    if n_subjects == 0 or n_categories == 0:
        return 0.0
    raters_per_subject = ratings_matrix.sum(axis=1)[0]
    if raters_per_subject <= 1:
        return 0.0
    category_probs = ratings_matrix.sum(axis=0) / (n_subjects * raters_per_subject)
    per_subject_agreement = (
        np.sum(ratings_matrix ** 2, axis=1) - raters_per_subject
    ) / (raters_per_subject * (raters_per_subject - 1))
    observed = np.mean(per_subject_agreement)
    expected = np.sum(category_probs ** 2)
    if abs(1 - expected) < 1e-10:
        return 1.0
    return (observed - expected) / (1 - expected)


def cohen_kappa(labels1, labels2, categories=None):
    """Cohen's kappa between two raters."""
    if categories is None:
        categories = sorted(set(labels1) | set(labels2))
    cat2idx = {category: idx for idx, category in enumerate(categories)}
    confusion = np.zeros((len(categories), len(categories)), dtype=float)
    for label1, label2 in zip(labels1, labels2):
        if label1 in cat2idx and label2 in cat2idx:
            confusion[cat2idx[label1], cat2idx[label2]] += 1
    total = confusion.sum()
    if total == 0:
        return 0.0
    observed = np.trace(confusion) / total
    expected = np.sum(confusion.sum(axis=0) * confusion.sum(axis=1)) / (total ** 2)
    if abs(1 - expected) < 1e-10:
        return 1.0
    return (observed - expected) / (1 - expected)


def spearman_rank(x, y):
    """Spearman rank correlation."""
    from scipy.stats import spearmanr

    if len(x) < 3:
        return 0.0, 1.0
    rho, p_value = spearmanr(x, y)
    return rho, p_value


def bootstrap_ci(data, stat_func=np.mean, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    data = np.array(data)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        stats.append(stat_func(sample))
    alpha = (1 - ci) / 2
    lo = np.percentile(stats, alpha * 100)
    hi = np.percentile(stats, (1 - alpha) * 100)
    return np.mean(stats), lo, hi


def latex_escape(value):
    text = str(value)
    for old, new in {"&": r"\&", "%": r"\%", "_": r"\_"}.items():
        text = text.replace(old, new)
    return text


def fleiss_for_keys(keys, judge_names, all_results, category_to_idx):
    matrix = np.zeros((len(keys), len(category_to_idx)), dtype=float)
    for subject_idx, key in enumerate(keys):
        for judge_name in judge_names:
            winner = all_results[judge_name][key]["llm_winner"]
            matrix[subject_idx, category_to_idx[winner]] += 1
    return float(fleiss_kappa_multi(matrix))


print("=" * 70)
print("CROSS-LLM JUDGE ANALYSIS")
print("=" * 70)

all_judge_results = {}
for judge_name, judge_dirs in JUDGE_DIRS.items():
    all_judge_results[judge_name] = load_results(judge_dirs)
    print(f"  {judge_name}: {len(all_judge_results[judge_name])} result files loaded")

print("\n" + "=" * 70)
print("1. BIDIRECTIONAL CONSISTENCY RATE")
print("=" * 70)

consistency_table = {}
for judge_name, results in all_judge_results.items():
    rates = [data["consistency_rate"] for data in results.values()]
    mean, lo, hi = bootstrap_ci(rates)
    consistency_table[judge_name] = {
        "mean": float(np.mean(rates)),
        "std": float(np.std(rates)),
        "min": float(np.min(rates)),
        "max": float(np.max(rates)),
        "rates": rates,
        "ci95": [float(lo), float(hi)],
    }
    print(
        f"  {judge_name:28s}: mean={np.mean(rates):.3f} ± {np.std(rates):.3f}  "
        f"[{np.min(rates):.2f}, {np.max(rates):.2f}]"
    )

print("\n  Bootstrap 95% CI for mean consistency:")
for judge_name, info in consistency_table.items():
    lo, hi = info["ci95"]
    print(f"    {judge_name:28s}: {info['mean']:.3f} [{lo:.3f}, {hi:.3f}]")

print("\n" + "=" * 70)
print("2. WINNER DISTRIBUTION PER JUDGE")
print("=" * 70)

winner_distribution = {}
for judge_name, results in all_judge_results.items():
    winners = defaultdict(int)
    for data in results.values():
        winners[data["llm_winner"]] += 1
    winner_distribution[judge_name] = dict(winners)
    total = sum(winners.values())
    print(f"\n  {judge_name}:")
    for winner in sorted(winners):
        count = winners[winner]
        print(f"    {winner:20s}: {count:2d}/{total} ({count / total * 100:.0f}%)")

print("\n" + "=" * 70)
print("3. DETAILED PREFERENCE BREAKDOWN (prefer_a / prefer_b / tie / inconsistent)")
print("=" * 70)

all_keys = sorted(set().union(*(results.keys() for results in all_judge_results.values())))
detail_rows = []
for key in all_keys:
    dataset, model_a, model_b, dimension = key
    row = {"key": key, "dataset": dataset, "pair": f"{model_a}_vs_{model_b}", "dim": dimension}
    for judge_name in JUDGE_DIRS:
        data = all_judge_results[judge_name].get(key)
        if data:
            row[f"{judge_name}_pref_a"] = data["prefer_a"]
            row[f"{judge_name}_pref_b"] = data["prefer_b"]
            row[f"{judge_name}_tie"] = data["tie"]
            row[f"{judge_name}_incon"] = data["inconsistent"]
            row[f"{judge_name}_consist"] = data["consistency_rate"]
            row[f"{judge_name}_winner"] = data["llm_winner"]
        else:
            row[f"{judge_name}_winner"] = "N/A"
    detail_rows.append(row)

print("\n" + "=" * 70)
print("4. DEBIASING SENSITIVITY (alignment to ground-truth metrics)")
print("=" * 70)

ground_truth_metrics = {}
for key in all_keys:
    for judge_name in JUDGE_DIRS:
        data = all_judge_results[judge_name].get(key)
        if data:
            ground_truth_metrics[key] = {
                "ndcg_winner": data["ndcg_winner"],
                "coverage_winner": data["coverage_winner"],
            }
            break

alignment_ndcg = {}
alignment_cov = {}
for judge_name, results in all_judge_results.items():
    ndcg_align = [data.get("alignment_to_ndcg", 0) for data in results.values()]
    cov_align = [data.get("alignment_to_coverage", 0) for data in results.values()]
    alignment_ndcg[judge_name] = ndcg_align
    alignment_cov[judge_name] = cov_align
    print(
        f"  {judge_name:28s}: NDCG-align={np.mean(ndcg_align):.3f}  "
        f"Coverage-align={np.mean(cov_align):.3f}"
    )

print("\n" + "=" * 70)
print("5. CROSS-JUDGE AGREEMENT (Cohen's kappa on winner labels)")
print("=" * 70)

common_keys = sorted(set.intersection(*(set(results.keys()) for results in all_judge_results.values())))
print(f"  Common experiment keys: {len(common_keys)}")

judge_names = list(JUDGE_DIRS.keys())
judge_short_names = [JUDGE_SHORT[judge_name] for judge_name in judge_names]
categories = sorted(
    set(
        all_judge_results[judge_name][key]["llm_winner"]
        for judge_name in judge_names
        for key in common_keys
    )
)

kappa_matrix = np.zeros((len(judge_names), len(judge_names)))
for row_idx, judge_a in enumerate(judge_names):
    for col_idx, judge_b in enumerate(judge_names):
        if row_idx == col_idx:
            kappa_matrix[row_idx][col_idx] = 1.0
            continue
        labels_a = [all_judge_results[judge_a][key]["llm_winner"] for key in common_keys]
        labels_b = [all_judge_results[judge_b][key]["llm_winner"] for key in common_keys]
        kappa_matrix[row_idx][col_idx] = cohen_kappa(labels_a, labels_b, categories)

print("\n  Cohen's Kappa Matrix:")
header = "                              " + "  ".join(f"{name:>16s}" for name in judge_short_names)
print(header)
for row_idx, judge_name in enumerate(judge_names):
    row_str = f"  {judge_name:28s}" + "  ".join(
        f"{kappa_matrix[row_idx][col_idx]:16.3f}" for col_idx in range(len(judge_names))
    )
    print(row_str)

print("\n" + "=" * 70)
print(f"6. FLEISS' KAPPA (all {len(judge_names)} judges)")
print("=" * 70)

category_to_idx = {category: idx for idx, category in enumerate(categories)}
fk = fleiss_for_keys(common_keys, judge_names, all_judge_results, category_to_idx)
print(f"  Fleiss' kappa = {fk:.4f}")

fleiss_by_dimension = {}
for dimension in DIMENSIONS:
    dimension_keys = [key for key in common_keys if key[3] == dimension]
    fleiss_by_dimension[dimension] = fleiss_for_keys(dimension_keys, judge_names, all_judge_results, category_to_idx)
    print(f"  Fleiss' kappa ({dimension:10s}) = {fleiss_by_dimension[dimension]:.4f}")

fleiss_by_dataset = {}
for dataset in DATASETS:
    dataset_keys = [key for key in common_keys if key[0] == dataset]
    fleiss_by_dataset[dataset] = fleiss_for_keys(dataset_keys, judge_names, all_judge_results, category_to_idx)
    print(f"  Fleiss' kappa ({dataset:10s}) = {fleiss_by_dataset[dataset]:.4f}")

print("\n" + "=" * 70)
print("7. SPEARMAN RANK CORRELATION (consistency rate across judges)")
print("=" * 70)

consistency_vectors = {
    judge_name: [all_judge_results[judge_name][key]["consistency_rate"] for key in common_keys]
    for judge_name in judge_names
}

for row_idx, judge_a in enumerate(judge_names):
    for col_idx, judge_b in enumerate(judge_names):
        if col_idx <= row_idx:
            continue
        rho, p_value = spearman_rank(consistency_vectors[judge_a], consistency_vectors[judge_b])
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
        print(f"  {judge_a:28s} vs {judge_b:28s}: rho={rho:.3f}, p={p_value:.4f} {significance}")

print("\n" + "=" * 70)
print("8. SUMMARY TABLE: Dataset × Dimension × Judge")
print("=" * 70)

for dataset in DATASETS:
    for dimension in DIMENSIONS:
        print(f"\n  --- {dataset.upper()} / {dimension.upper()} ---")
        print(f"  {'Pair':<30s}", end="")
        for judge_name in judge_names:
            print(f"  {JUDGE_SHORT[judge_name]:>16s}", end="")
        print()
        for model_a, model_b in PAIRS:
            key = (dataset, model_a, model_b, dimension)
            print(f"  {model_a + '_vs_' + model_b:<30s}", end="")
            for judge_name in judge_names:
                data = all_judge_results[judge_name].get(key)
                label = "N/A" if data is None else f"{data['llm_winner']}({data['consistency_rate']:.0%})"
                print(f"  {label:>16s}", end="")
            print()

print("\n" + "=" * 70)
print("9. LATEX TABLE OUTPUT")
print("=" * 70)

out_dir = Path("/root/new_paper/results/analysis")
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "table_consistency.tex", "w") as handle:
    handle.write("\\begin{table}[htbp]\n\\centering\n")
    handle.write("\\caption{Bidirectional consistency rate of each LLM judge across all pairwise comparisons.}\n")
    handle.write("\\label{tab:consistency}\n")
    handle.write("\\begin{tabular}{lcccc}\n\\hline\n")
    handle.write("Judge & Mean & Std & 95\\% CI & Range \\\\\n\\hline\n")
    for judge_name in judge_names:
        info = consistency_table[judge_name]
        lo, hi = info["ci95"]
        handle.write(
            f"{latex_escape(judge_name)} & {info['mean']:.3f} & {info['std']:.3f} & "
            f"[{lo:.3f}, {hi:.3f}] & [{info['min']:.2f}, {info['max']:.2f}] \\\\\n"
        )
    handle.write("\\hline\n\\end{tabular}\n\\end{table}\n")

with open(out_dir / "table_kappa.tex", "w") as handle:
    handle.write("\\begin{table}[htbp]\n\\centering\n")
    handle.write("\\caption{Pairwise Cohen's $\\kappa$ between LLM judges on winner labels.}\n")
    handle.write("\\label{tab:kappa}\n")
    handle.write("\\resizebox{\\columnwidth}{!}{%\n")
    handle.write("\\begin{tabular}{l" + "c" * len(judge_names) + "}\n\\hline\n")
    handle.write(" & " + " & ".join(latex_escape(short_name) for short_name in judge_short_names) + " \\\\\n\\hline\n")
    for row_idx, short_name in enumerate(judge_short_names):
        values = " & ".join(
            f"{kappa_matrix[row_idx][col_idx]:.3f}" if row_idx != col_idx else "1.000"
            for col_idx in range(len(judge_names))
        )
        handle.write(f"{latex_escape(short_name)} & {values} \\\\\n")
    handle.write("\\hline\n\\end{tabular}%\n}\n\\end{table}\n")

with open(out_dir / "table_main_results.tex", "w") as handle:
    handle.write("\\begin{table*}[htbp]\n\\centering\n\\small\n")
    handle.write(
        f"\\caption{{LLM judge verdicts across {len(judge_names)} judges for each debiasing pair comparison. "
        "Each cell shows the winner (A/B/tie) and bidirectional consistency rate in parentheses. "
        "Bold indicates agreement with NDCG ground truth.}\n"
    )
    handle.write("\\label{tab:main-results}\n")
    handle.write("\\resizebox{\\textwidth}{!}{%\n")
    handle.write("\\begin{tabular}{llll" + "c" * len(judge_names) + "}\n\\hline\n")
    handle.write("Dataset & Pair & Dim. & GT")
    for short_name in judge_short_names:
        handle.write(f" & {latex_escape(short_name)}")
    handle.write(" \\\\\n\\hline\n")
    for dataset in DATASETS:
        for pair_index, (model_a, model_b) in enumerate(PAIRS):
            for dim_index, dimension in enumerate(DIMENSIONS):
                key = (dataset, model_a, model_b, dimension)
                gt = ground_truth_metrics.get(key, {}).get("ndcg_winner", "?")
                dataset_label = dataset.capitalize() if pair_index == 0 and dim_index == 0 else ""
                pair_label = latex_escape(f"{model_a} vs {model_b}") if dim_index == 0 else ""
                handle.write(f"{dataset_label} & {pair_label} & {dimension[:3]} & {latex_escape(gt)}")
                for judge_name in judge_names:
                    data = all_judge_results[judge_name].get(key)
                    if data:
                        cell = latex_escape(f"{data['llm_winner']}({data['consistency_rate']:.0%})")
                        if data["llm_winner"] == gt:
                            cell = f"\\textbf{{{cell}}}"
                        handle.write(f" & {cell}")
                    else:
                        handle.write(" & N/A")
                handle.write(" \\\\\n")
        handle.write("\\hline\n")
    handle.write("\\end{tabular}%\n}\n\\end{table*}\n")

with open(out_dir / "table_fleiss.tex", "w") as handle:
    handle.write("\\begin{table}[htbp]\n\\centering\n")
    handle.write(f"\\caption{{Fleiss' $\\kappa$ measuring agreement among all {len(judge_names)} LLM judges.}}\n")
    handle.write("\\label{tab:fleiss}\n")
    handle.write("\\begin{tabular}{lc}\n\\hline\n")
    handle.write("Scope & Fleiss' $\\kappa$ \\\\\n\\hline\n")
    handle.write(f"Overall & {fk:.4f} \\\\\n")
    for dimension in DIMENSIONS:
        handle.write(f"{dimension.capitalize()} & {fleiss_by_dimension[dimension]:.4f} \\\\\n")
    for dataset in DATASETS:
        handle.write(f"{dataset.capitalize()} & {fleiss_by_dataset[dataset]:.4f} \\\\\n")
    handle.write("\\hline\n\\end{tabular}\n\\end{table}\n")

print(f"\n  LaTeX tables written to {out_dir}/")

summary = {
    "judges": judge_names,
    "judge_short_labels": judge_short_names,
    "local_judges": LOCAL_JUDGES,
    "cloud_judges": CLOUD_JUDGES,
    "n_judges": len(judge_names),
    "n_common_keys": len(common_keys),
    "consistency": {
        judge_name: {
            "mean": consistency_table[judge_name]["mean"],
            "std": consistency_table[judge_name]["std"],
            "min": consistency_table[judge_name]["min"],
            "max": consistency_table[judge_name]["max"],
            "ci95": consistency_table[judge_name]["ci95"],
        }
        for judge_name in judge_names
    },
    "alignment": {
        judge_name: {
            "ndcg_mean": float(np.mean(alignment_ndcg[judge_name])),
            "coverage_mean": float(np.mean(alignment_cov[judge_name])),
        }
        for judge_name in judge_names
    },
    "winner_distribution": winner_distribution,
    "fleiss_kappa_overall": float(fk),
    "fleiss_kappa_by_dimension": fleiss_by_dimension,
    "fleiss_kappa_by_dataset": fleiss_by_dataset,
    "cohen_kappa_matrix": kappa_matrix.tolist(),
    "detail_rows": detail_rows,
}
with open(out_dir / "cross_llm_summary.json", "w") as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)
print(f"  JSON summary written to {out_dir}/cross_llm_summary.json")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
