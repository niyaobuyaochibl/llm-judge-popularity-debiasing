#!/usr/bin/env python3
"""Analyze completed human sanity check answer sheets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

VALID_VERDICTS = {'A', 'B', 'TIE'}
RESULT_EXCLUDE = {'all_results.json', 'summary.json'}
DEFAULT_JUDGE_DIRS = {
    'phi-2': [
        Path('/root/new_paper/results/llm_eval/main_balanced'),
        Path('/root/new_paper/results/llm_eval/main_diversity'),
    ],
    'Qwen-Plus': [Path('/root/new_paper/results/llm_eval/crossllm_qwen_plus')],
    'DeepSeek-V3': [Path('/root/new_paper/results/llm_eval/crossllm_deepseek')],
    'Kimi-K2': [Path('/root/new_paper/results/llm_eval/crossllm_kimi')],
    'MiniMax-M2.7': [Path('/root/new_paper/results/llm_eval/crossllm_minimax')],
    'Qwen2.5-7B-Instruct(local)': [Path('/root/new_paper/results/llm_eval/crossllm_qwen25_7b_local')],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-dir', type=Path, required=True)
    parser.add_argument('--answers', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--include-default-judge-comparison', action='store_true')
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def load_metadata(package_dir: Path) -> tuple[dict[str, dict], list[dict[str, str]], dict]:
    base_tasks = {}
    with (package_dir / 'metadata' / 'base_tasks.jsonl').open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            base_tasks[row['base_task_id']] = row
    task_map = read_csv(package_dir / 'metadata' / 'annotator_task_map.csv')
    manifest = json.loads((package_dir / 'metadata' / 'package_manifest.json').read_text(encoding='utf-8'))
    return base_tasks, task_map, manifest


def normalize_verdict(raw: str, swap_lists: bool) -> str | None:
    verdict = (raw or '').strip().upper()
    if not verdict:
        return None
    if verdict not in VALID_VERDICTS:
        raise ValueError(f'Invalid verdict: {raw!r}')
    if verdict == 'TIE':
        return 'TIE'
    if not swap_lists:
        return verdict
    return 'A' if verdict == 'B' else 'B'


def fleiss_kappa(rating_rows: list[list[str]]) -> float | None:
    if not rating_rows:
        return None
    n = len(rating_rows[0])
    if n < 2:
        return None
    categories = ['A', 'B', 'TIE']
    matrix = []
    for row in rating_rows:
        if len(row) != n:
            return None
        counts = [row.count(cat) for cat in categories]
        matrix.append(counts)
    total_items = len(matrix)
    p_j = [sum(row[j] for row in matrix) / (total_items * n) for j in range(len(categories))]
    p_i = []
    for row in matrix:
        numerator = sum(count * count for count in row) - n
        p_i.append(numerator / (n * (n - 1)))
    p_bar = sum(p_i) / total_items
    p_e = sum(p * p for p in p_j)
    if p_e == 1:
        return None
    return (p_bar - p_e) / (1 - p_e)


def alignment_score(human_verdict: str, winner: str) -> float:
    if human_verdict == winner:
        return 1.0
    if human_verdict == 'TIE' or winner == 'TIE':
        return 0.5
    return 0.0


def majority_vote(verdicts: list[str]) -> str:
    counts = Counter(verdicts)
    label, votes = counts.most_common(1)[0]
    if votes >= 2:
        return label
    return 'TIE'


def load_answers(answer_paths: list[Path], task_map: list[dict[str, str]]) -> tuple[list[dict], list[str]]:
    map_by_key = {(row['annotator_id'], row['task_code']): row for row in task_map}
    normalized_rows = []
    issues = []
    for answer_path in answer_paths:
        for row in read_csv(answer_path):
            annotator_id = row['annotator_id'].strip()
            task_code = row['task_code'].strip()
            key = (annotator_id, task_code)
            if key not in map_by_key:
                issues.append(f'Unrecognized answer row: {answer_path} {annotator_id} {task_code}')
                continue
            mapping = map_by_key[key]
            swap_lists = bool(int(mapping['swap_lists']))
            verdict = normalize_verdict(row.get('verdict', ''), swap_lists)
            confidence_raw = (row.get('confidence', '') or '').strip()
            confidence = int(confidence_raw) if confidence_raw else None
            normalized_rows.append({
                'annotator_id': annotator_id,
                'task_code': task_code,
                'base_task_id': mapping['base_task_id'],
                'result_stem': mapping['result_stem'],
                'dataset': mapping['dataset'],
                'dimension': mapping['dimension'],
                'model_a': mapping['model_a'],
                'model_b': mapping['model_b'],
                'user_id': int(mapping['user_id']),
                'swap_lists': swap_lists,
                'is_duplicate': bool(int(mapping['is_duplicate'])),
                'verdict_raw': (row.get('verdict', '') or '').strip().upper(),
                'verdict_normalized': verdict,
                'confidence': confidence,
                'notes': row.get('notes', ''),
            })
    return normalized_rows, issues


def compute_duplicate_consistency(rows: list[dict]) -> dict:
    by_annotator_base: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        if row['verdict_normalized'] is None:
            continue
        by_annotator_base[(row['annotator_id'], row['base_task_id'])].append(row['verdict_normalized'])
    per_annotator: dict[str, list[float]] = defaultdict(list)
    for (annotator_id, _), verdicts in by_annotator_base.items():
        if len(verdicts) == 2:
            per_annotator[annotator_id].append(1.0 if verdicts[0] == verdicts[1] else 0.0)
    scores = {annotator_id: (mean(values) if values else None) for annotator_id, values in per_annotator.items()}
    overall_values = [value for value in scores.values() if value is not None]
    return {
        'per_annotator': scores,
        'overall_mean': mean(overall_values) if overall_values else None,
    }


def load_judge_maps() -> dict[str, dict[tuple[str, int], str]]:
    judge_maps: dict[str, dict[tuple[str, int], str]] = {}
    for judge_name, directories in DEFAULT_JUDGE_DIRS.items():
        case_map: dict[tuple[str, int], str] = {}
        for directory in directories:
            if not directory.exists():
                continue
            for path in directory.glob('*.json'):
                if path.name in RESULT_EXCLUDE:
                    continue
                with path.open('r', encoding='utf-8') as f:
                    payload = json.load(f)
                for trial in payload.get('per_trial', []):
                    case_map[(path.stem, int(trial['user_id']))] = str(trial.get('effective_preference', 'MISSING'))
        judge_maps[judge_name] = case_map
    return judge_maps


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.package_dir / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    base_tasks, task_map, manifest = load_metadata(args.package_dir)
    rows, issues = load_answers(args.answers, task_map)

    base_rows = [row for row in rows if not row['is_duplicate']]
    answered_base_rows = [row for row in base_rows if row['verdict_normalized'] is not None]

    ratings_by_task: dict[str, list[str]] = defaultdict(list)
    for row in answered_base_rows:
        ratings_by_task[row['base_task_id']].append(row['verdict_normalized'])

    fleiss = fleiss_kappa([ratings_by_task[task_id] for task_id in sorted(ratings_by_task) if len(ratings_by_task[task_id]) >= 2])
    duplicate_consistency = compute_duplicate_consistency(rows)

    majority_rows = []
    for base_task_id, verdicts in sorted(ratings_by_task.items()):
        task = base_tasks[base_task_id]
        majority = majority_vote(verdicts)
        majority_rows.append({
            'base_task_id': base_task_id,
            'result_stem': task['result_stem'],
            'dataset': task['dataset'],
            'dimension': task['dimension'],
            'user_id': task['user_id'],
            'majority_verdict': majority,
            'n_ratings': len(verdicts),
            'alignment_to_ndcg': alignment_score(majority, task['ndcg_winner']),
            'alignment_to_coverage': alignment_score(majority, task['coverage_winner']),
        })

    judge_agreement = None
    if args.include_default_judge_comparison:
        judge_maps = load_judge_maps()
        judge_agreement = {}
        for judge_name, case_map in judge_maps.items():
            comparable = 0
            agree = 0
            inconsistent = 0
            missing = 0
            for row in majority_rows:
                key = (row['result_stem'], int(row['user_id']))
                judge_label = case_map.get(key)
                if judge_label is None:
                    missing += 1
                    continue
                if judge_label == 'INCONSISTENT':
                    inconsistent += 1
                    continue
                if judge_label not in VALID_VERDICTS:
                    missing += 1
                    continue
                comparable += 1
                if judge_label == row['majority_verdict']:
                    agree += 1
            judge_agreement[judge_name] = {
                'agreement_rate': (agree / comparable) if comparable else None,
                'agree': agree,
                'comparable': comparable,
                'judge_inconsistent': inconsistent,
                'missing': missing,
            }

    report = {
        'package_name': manifest['package_name'],
        'n_base_tasks_expected': manifest['n_base_tasks'],
        'n_base_tasks_answered': len(majority_rows),
        'n_duplicate_base_tasks_expected': manifest['n_duplicates'],
        'fleiss_kappa': fleiss,
        'duplicate_consistency': duplicate_consistency,
        'task_level_alignment': {
            'ndcg_mean': mean([row['alignment_to_ndcg'] for row in majority_rows]) if majority_rows else None,
            'coverage_mean': mean([row['alignment_to_coverage'] for row in majority_rows]) if majority_rows else None,
        },
        'judge_agreement': judge_agreement,
        'issues': issues,
    }

    (output_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (output_dir / 'task_majorities.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['base_task_id', 'result_stem', 'dataset', 'dimension', 'user_id', 'majority_verdict', 'n_ratings', 'alignment_to_ndcg', 'alignment_to_coverage'],
        )
        writer.writeheader()
        writer.writerows(majority_rows)

    md_lines = [
        '# Human Sanity Check Report',
        '',
        f"- Package: `{manifest['package_name']}`",
        f"- Base tasks answered: {len(majority_rows)} / {manifest['n_base_tasks']}",
        f"- Fleiss' kappa: {fleiss if fleiss is not None else 'N/A'}",
        f"- Mean duplicate consistency: {duplicate_consistency['overall_mean'] if duplicate_consistency['overall_mean'] is not None else 'N/A'}",
        f"- Mean alignment to NDCG winner: {report['task_level_alignment']['ndcg_mean'] if report['task_level_alignment']['ndcg_mean'] is not None else 'N/A'}",
        f"- Mean alignment to coverage winner: {report['task_level_alignment']['coverage_mean'] if report['task_level_alignment']['coverage_mean'] is not None else 'N/A'}",
    ]
    if judge_agreement is not None:
        md_lines.extend(['', '## Judge Agreement', ''])
        for judge_name, stats in judge_agreement.items():
            md_lines.append(
                f"- {judge_name}: agreement={stats['agreement_rate'] if stats['agreement_rate'] is not None else 'N/A'}, comparable={stats['comparable']}, judge_inconsistent={stats['judge_inconsistent']}, missing={stats['missing']}"
            )
    if issues:
        md_lines.extend(['', '## Issues', ''])
        md_lines.extend([f'- {issue}' for issue in issues])
    (output_dir / 'report.md').write_text('\n'.join(md_lines) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
