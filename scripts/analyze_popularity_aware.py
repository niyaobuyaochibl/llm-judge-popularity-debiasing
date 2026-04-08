#!/usr/bin/env python3
"""Compare baseline pairwise judging against a popularity-aware variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path('/root/new_paper/llm_eval')
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_utils import load_metrics
from experiment_config import RESULTS_DIR

KEYS = ["dataset", "model_a", "model_b", "dimension"]
METRICS = {
    "ndcg": True,
    "coverage": True,
    "tail_exposure": True,
    "gini": False,
    "arp": False,
}


def load_records(result_dir: Path) -> pd.DataFrame:
    all_results = result_dir / 'all_results.json'
    if all_results.exists():
        data = json.loads(all_results.read_text())
    else:
        data = []
        for path in sorted(result_dir.glob('*.json')):
            if path.name in {'all_results.json', 'summary.json'}:
                continue
            data.append(json.loads(path.read_text()))
    return pd.DataFrame(data)


def metric_winner(model_a: str, model_b: str, metric_a: float | None, metric_b: float | None, higher_is_better: bool) -> str:
    if metric_a is None or metric_b is None:
        return 'unknown'
    if metric_a == metric_b:
        return 'tie'
    if higher_is_better:
        return model_a if metric_a > metric_b else model_b
    return model_a if metric_a < metric_b else model_b


def alignment_score(llm_winner: str, reference_winner: str) -> float:
    if reference_winner == 'unknown':
        return 0.0
    if llm_winner == reference_winner:
        return 1.0
    if llm_winner == 'tie':
        return 0.5
    return 0.0


def ensure_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for row in df.to_dict(orient='records'):
        metrics_a = load_metrics(row['dataset'], row['model_a'], 42)
        metrics_b = load_metrics(row['dataset'], row['model_b'], 42)
        for metric_name, higher_is_better in METRICS.items():
            winner_key = f'{metric_name}_winner'
            align_key = f'alignment_to_{metric_name}'
            winner = row.get(winner_key)
            if not winner:
                winner = metric_winner(
                    row['model_a'],
                    row['model_b'],
                    metrics_a.get(metric_name),
                    metrics_b.get(metric_name),
                    higher_is_better,
                )
                row[winner_key] = winner
            if align_key not in row:
                row[align_key] = alignment_score(row['llm_winner'], winner)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    out = {}
    for col in ['consistency_rate'] + [f'alignment_to_{metric}' for metric in METRICS]:
        if col in df.columns:
            out[f'{prefix}_{col}'] = float(df[col].mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-subdir', required=True)
    parser.add_argument('--candidate-subdir', required=True)
    parser.add_argument('--output-dir', default=str(RESULTS_DIR / 'analysis' / 'popaware_compare'))
    args = parser.parse_args()

    baseline_dir = RESULTS_DIR / 'llm_eval' / args.baseline_subdir
    candidate_dir = RESULTS_DIR / 'llm_eval' / args.candidate_subdir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = ensure_metric_columns(load_records(baseline_dir))
    candidate = ensure_metric_columns(load_records(candidate_dir))
    if baseline.empty or candidate.empty:
        raise SystemExit('baseline or candidate results are empty')

    merged = baseline.merge(candidate, on=KEYS, suffixes=('_baseline', '_candidate'))
    if merged.empty:
        raise SystemExit('no overlapping records between baseline and candidate')

    for col in ['consistency_rate'] + [f'alignment_to_{metric}' for metric in METRICS]:
        merged[f'delta_{col}'] = merged[f'{col}_candidate'] - merged[f'{col}_baseline']

    pair_csv = output_dir / 'pairwise_delta.csv'
    merged.to_csv(pair_csv, index=False)

    dim_rows = []
    for dimension, group in merged.groupby('dimension'):
        row = {'dimension': dimension, 'n_records': int(len(group))}
        for col in ['consistency_rate'] + [f'alignment_to_{metric}' for metric in METRICS]:
            row[f'baseline_{col}'] = float(group[f'{col}_baseline'].mean())
            row[f'candidate_{col}'] = float(group[f'{col}_candidate'].mean())
            row[f'delta_{col}'] = float(group[f'delta_{col}'].mean())
        dim_rows.append(row)
    dim_df = pd.DataFrame(dim_rows).sort_values('dimension')
    dim_df.to_csv(output_dir / 'dimension_summary.csv', index=False)

    summary = {
        'baseline_subdir': args.baseline_subdir,
        'candidate_subdir': args.candidate_subdir,
        'num_overlapping_records': int(len(merged)),
        **summarize(baseline, 'baseline_avg'),
        **summarize(candidate, 'candidate_avg'),
    }
    for col in ['consistency_rate'] + [f'alignment_to_{metric}' for metric in METRICS]:
        summary[f'delta_avg_{col}'] = float(merged[f'delta_{col}'].mean())
        summary[f'improved_records_{col}'] = int((merged[f'delta_{col}'] > 0).sum())
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    top_cols = ['dataset', 'model_a', 'model_b', 'dimension', 'delta_consistency_rate', 'delta_alignment_to_coverage', 'delta_alignment_to_tail_exposure']
    top_improvements = merged.sort_values(['delta_alignment_to_tail_exposure', 'delta_alignment_to_coverage', 'delta_consistency_rate'], ascending=False)[top_cols].head(10)
    top_regressions = merged.sort_values(['delta_alignment_to_tail_exposure', 'delta_alignment_to_coverage', 'delta_consistency_rate'], ascending=True)[top_cols].head(10)
    top_improvements.to_csv(output_dir / 'top_improvements.csv', index=False)
    top_regressions.to_csv(output_dir / 'top_regressions.csv', index=False)

    lines = []
    lines.append("# Popularity-Aware Comparison")
    lines.append("")
    lines.append(f"Baseline: `{args.baseline_subdir}`  ")
    lines.append(f"Candidate: `{args.candidate_subdir}`  ")
    lines.append(f"Overlapping records: {len(merged)}")
    lines.append("")
    lines.append("## Overall")
    for col in ['consistency_rate'] + [f'alignment_to_{metric}' for metric in METRICS]:
        lines.append(
            f"- `{col}`: baseline {summary[f'baseline_avg_{col}']:.3f}, candidate {summary[f'candidate_avg_{col}']:.3f}, delta {summary[f'delta_avg_{col}']:+.3f}, improved in {summary[f'improved_records_{col}']}/{len(merged)} records"
        )
    lines.append("")
    lines.append("## By Dimension")
    for _, row in dim_df.iterrows():
        lines.append(f"- `{row['dimension']}`: delta consistency {row['delta_consistency_rate']:+.3f}, delta coverage {row['delta_alignment_to_coverage']:+.3f}, delta tail_exposure {row['delta_alignment_to_tail_exposure']:+.3f}")
    (output_dir / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
