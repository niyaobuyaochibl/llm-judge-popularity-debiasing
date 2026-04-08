#!/usr/bin/env python3
"""Generate a human sanity check package from the debiasing benchmark."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path('/root/new_paper')
LLM_EVAL_ROOT = PROJECT_ROOT / 'llm_eval'
if str(LLM_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_EVAL_ROOT))

from data_utils import (  # type: ignore
    format_recommendation_list,
    load_item_text_subset,
    load_recommendations,
    load_user_histories,
    render_user_profile_summary,
)
from prompts import INSTRUCTIONS  # type: ignore

RESULT_EXCLUDE = {'all_results.json', 'summary.json'}
DISPLAY_LABELS = {
    'balanced': 'Balanced',
    'diversity': 'Diversity',
    'relevance': 'Relevance',
    'novelty': 'Novelty',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source-result-dir',
        type=Path,
        default=PROJECT_ROOT / 'results' / 'llm_eval' / 'crossllm_qwen25_7b_local',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'results' / 'human_eval' / 'human_sanity_check_v1',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-per-config', type=int, default=2)
    parser.add_argument('--n-duplicates', type=int, default=10)
    parser.add_argument('--n-annotators', type=int, default=3)
    parser.add_argument('--history-items', type=int, default=10)
    return parser.parse_args()


def load_result_configs(source_result_dir: Path) -> list[dict]:
    configs = []
    for path in sorted(source_result_dir.glob('*.json')):
        if path.name in RESULT_EXCLUDE:
            continue
        with path.open('r', encoding='utf-8') as f:
            payload = json.load(f)
        if not payload.get('per_trial'):
            continue
        configs.append({
            'result_path': path,
            'result_stem': path.stem,
            'dataset': payload['dataset'],
            'model_a': payload['model_a'],
            'model_b': payload['model_b'],
            'dimension': payload['dimension'],
            'ndcg_winner': payload['ndcg_winner'],
            'coverage_winner': payload['coverage_winner'],
            'per_trial': payload['per_trial'],
        })
    return configs


def sample_base_cases(configs: list[dict], n_per_config: int, seed: int) -> list[dict]:
    cases = []
    for config in configs:
        user_ids = sorted({int(trial['user_id']) for trial in config['per_trial']})
        if len(user_ids) < n_per_config:
            raise ValueError(f"Not enough user cases in {config['result_stem']}: {len(user_ids)} < {n_per_config}")
        rng = random.Random(f"{seed}:{config['result_stem']}")
        selected = rng.sample(user_ids, n_per_config)
        for idx, user_id in enumerate(selected, start=1):
            cases.append({
                'base_task_id': f"{config['result_stem']}__u{user_id}",
                'result_stem': config['result_stem'],
                'dataset': config['dataset'],
                'model_a': config['model_a'],
                'model_b': config['model_b'],
                'dimension': config['dimension'],
                'user_id': int(user_id),
                'sample_index_within_config': idx,
                'ndcg_winner': config['ndcg_winner'],
                'coverage_winner': config['coverage_winner'],
            })
    return cases


def allocate_duplicate_ids(base_cases: list[dict], n_duplicates: int, seed: int) -> list[str]:
    if n_duplicates <= 0:
        return []
    by_dimension: dict[str, list[str]] = defaultdict(list)
    for case in base_cases:
        by_dimension[case['dimension']].append(case['base_task_id'])
    dims = sorted(by_dimension)
    if not dims:
        return []
    rng = random.Random(f'duplicates:{seed}')
    counts = {dim: n_duplicates // len(dims) for dim in dims}
    for dim in dims[: n_duplicates % len(dims)]:
        counts[dim] += 1
    selected: list[str] = []
    for dim in dims:
        pool = sorted(by_dimension[dim])
        take = min(counts[dim], len(pool))
        selected.extend(rng.sample(pool, take))
    if len(selected) < n_duplicates:
        remaining = [case['base_task_id'] for case in base_cases if case['base_task_id'] not in selected]
        extra = min(n_duplicates - len(selected), len(remaining))
        selected.extend(rng.sample(sorted(remaining), extra))
    return selected


def build_task_corpus(base_cases: list[dict], history_items: int, seed: int) -> list[dict]:
    needed_users: dict[str, set[int]] = defaultdict(set)
    needed_models: dict[tuple[str, str], None] = {}
    recs_cache: dict[tuple[str, str], dict[int, list[int]]] = {}

    for case in base_cases:
        needed_users[case['dataset']].add(case['user_id'])
        needed_models[(case['dataset'], case['model_a'])] = None
        needed_models[(case['dataset'], case['model_b'])] = None

    for dataset, model in sorted(needed_models):
        recs_cache[(dataset, model)] = load_recommendations(dataset, model, seed)

    histories_cache = {
        dataset: load_user_histories(dataset, tuple(sorted(user_ids)), max_items=history_items)
        for dataset, user_ids in needed_users.items()
    }

    item_ids_by_dataset: dict[str, set[int]] = defaultdict(set)
    for case in base_cases:
        dataset = case['dataset']
        user_id = case['user_id']
        item_ids_by_dataset[dataset].update(histories_cache[dataset].get(user_id, []))
        item_ids_by_dataset[dataset].update(recs_cache[(dataset, case['model_a'])][user_id])
        item_ids_by_dataset[dataset].update(recs_cache[(dataset, case['model_b'])][user_id])

    item_texts_cache = {
        dataset: load_item_text_subset(dataset, tuple(sorted(item_ids)))
        for dataset, item_ids in item_ids_by_dataset.items()
    }

    corpus = []
    for case in base_cases:
        dataset = case['dataset']
        user_id = case['user_id']
        item_texts = item_texts_cache[dataset]
        recs_a = recs_cache[(dataset, case['model_a'])][user_id]
        recs_b = recs_cache[(dataset, case['model_b'])][user_id]
        histories = histories_cache[dataset].get(user_id, [])
        corpus.append({
            **case,
            'user_profile': render_user_profile_summary(histories, item_texts),
            'list_a_items': recs_a,
            'list_b_items': recs_b,
            'list_a_text': format_recommendation_list(recs_a, item_texts),
            'list_b_text': format_recommendation_list(recs_b, item_texts),
            'instruction_text': INSTRUCTIONS[case['dimension']],
        })
    return corpus


def format_task_markdown(task_code: str, task: dict, displayed_list_a: str, displayed_list_b: str) -> str:
    focus = DISPLAY_LABELS[task['dimension']]
    lines = [
        f"## Task {task_code}",
        "",
        f"- Focus: {focus}",
        f"- Record your answer in `answers_template.csv` for task code `{task_code}`.",
        "- Choose exactly one verdict: `A`, `B`, or `TIE`.",
        "- Confidence: integer `1` to `5`.",
        "",
        "### User Profile",
        task['user_profile'],
        "",
        "### List A",
        displayed_list_a,
        "",
        "### List B",
        displayed_list_b,
        "",
        "### Instruction",
        task['instruction_text'],
    ]
    return "\n".join(lines)


def format_task_html(task_code: str, task: dict, displayed_list_a: str, displayed_list_b: str) -> str:
    focus = DISPLAY_LABELS[task['dimension']]
    return f"""
    <section class=\"task\">
      <h2>Task {html.escape(task_code)}</h2>
      <p><strong>Focus:</strong> {html.escape(focus)}</p>
      <p><strong>Answer sheet row:</strong> {html.escape(task_code)}</p>
      <p><strong>Allowed verdicts:</strong> A, B, or TIE</p>
      <p><strong>Confidence:</strong> 1 to 5</p>
      <h3>User Profile</h3>
      <pre>{html.escape(task['user_profile'])}</pre>
      <h3>List A</h3>
      <pre>{html.escape(displayed_list_a)}</pre>
      <h3>List B</h3>
      <pre>{html.escape(displayed_list_b)}</pre>
      <h3>Instruction</h3>
      <p>{html.escape(task['instruction_text'])}</p>
    </section>
    """.strip()


def build_annotator_tasks(corpus: list[dict], duplicate_ids: list[str], annotator_index: int, seed: int) -> list[dict]:
    rng = random.Random(f'annotator:{annotator_index}:{seed}')
    base_by_id = {task['base_task_id']: task for task in corpus}
    rows = []
    swap_lookup: dict[str, bool] = {}

    for task in corpus:
        swap = rng.choice([False, True])
        swap_lookup[task['base_task_id']] = swap
        rows.append({
            'base_task_id': task['base_task_id'],
            'is_duplicate': False,
            'swap_lists': swap,
        })

    for base_task_id in duplicate_ids:
        rows.append({
            'base_task_id': base_task_id,
            'is_duplicate': True,
            'swap_lists': not swap_lookup[base_task_id],
        })

    rng.shuffle(rows)
    packaged = []
    for idx, row in enumerate(rows, start=1):
        task = base_by_id[row['base_task_id']]
        task_code = f'T{idx:02d}'
        if row['swap_lists']:
            displayed_list_a = task['list_b_text']
            displayed_list_b = task['list_a_text']
        else:
            displayed_list_a = task['list_a_text']
            displayed_list_b = task['list_b_text']
        packaged.append({
            **row,
            'task_code': task_code,
            'displayed_list_a': displayed_list_a,
            'displayed_list_b': displayed_list_b,
            'task': task,
        })
    return packaged


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_package(output_dir: Path, corpus: list[dict], duplicate_ids: list[str], n_annotators: int, seed: int, source_result_dir: Path) -> None:
    metadata_dir = output_dir / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)

    base_csv_rows = []
    with (metadata_dir / 'base_tasks.jsonl').open('w', encoding='utf-8') as f:
        for task in corpus:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')
            base_csv_rows.append({
                'base_task_id': task['base_task_id'],
                'result_stem': task['result_stem'],
                'dataset': task['dataset'],
                'dimension': task['dimension'],
                'model_a': task['model_a'],
                'model_b': task['model_b'],
                'user_id': task['user_id'],
                'ndcg_winner': task['ndcg_winner'],
                'coverage_winner': task['coverage_winner'],
            })
    write_csv(
        metadata_dir / 'base_tasks.csv',
        base_csv_rows,
        ['base_task_id', 'result_stem', 'dataset', 'dimension', 'model_a', 'model_b', 'user_id', 'ndcg_winner', 'coverage_winner'],
    )

    annotator_task_rows = []
    for annotator_index in range(1, n_annotators + 1):
        annotator_id = f'annotator_{annotator_index:02d}'
        annotator_dir = output_dir / annotator_id
        annotator_dir.mkdir(parents=True, exist_ok=True)
        packaged = build_annotator_tasks(corpus, duplicate_ids, annotator_index, seed)

        markdown_chunks = [
            f'# Human Sanity Check Task Booklet ({annotator_id})',
            '',
            'Use `answers_template.csv` to record your final verdicts.',
            'Allowed verdicts: `A`, `B`, `TIE`.',
            'Confidence must be an integer from `1` to `5`.',
            '',
        ]
        html_chunks = [
            '<!doctype html>',
            '<html lang="en">',
            '<head>',
            '  <meta charset="utf-8">',
            f'  <title>Human Sanity Check {annotator_id}</title>',
            '  <style>',
            '    body { font-family: Arial, sans-serif; line-height: 1.5; margin: 32px auto; max-width: 960px; padding: 0 24px; }',
            '    pre { white-space: pre-wrap; background: #f7f7f7; padding: 12px; border-radius: 8px; }',
            '    .task { border-top: 2px solid #ddd; padding-top: 24px; margin-top: 24px; }',
            '  </style>',
            '</head>',
            '<body>',
            f'  <h1>Human Sanity Check ({html.escape(annotator_id)})</h1>',
            '  <p>Use <code>answers_template.csv</code> to record your final verdicts. Allowed verdicts: <code>A</code>, <code>B</code>, <code>TIE</code>. Confidence must be an integer from <code>1</code> to <code>5</code>.</p>',
        ]

        answer_rows = []
        for row in packaged:
            task = row['task']
            markdown_chunks.append(format_task_markdown(row['task_code'], task, row['displayed_list_a'], row['displayed_list_b']))
            markdown_chunks.append('')
            markdown_chunks.append('---')
            markdown_chunks.append('')
            html_chunks.append(format_task_html(row['task_code'], task, row['displayed_list_a'], row['displayed_list_b']))

            annotator_task_rows.append({
                'annotator_id': annotator_id,
                'task_code': row['task_code'],
                'base_task_id': task['base_task_id'],
                'result_stem': task['result_stem'],
                'dataset': task['dataset'],
                'dimension': task['dimension'],
                'model_a': task['model_a'],
                'model_b': task['model_b'],
                'user_id': task['user_id'],
                'swap_lists': int(row['swap_lists']),
                'is_duplicate': int(row['is_duplicate']),
            })
            answer_rows.append({
                'annotator_id': annotator_id,
                'task_code': row['task_code'],
                'verdict': '',
                'confidence': '',
                'notes': '',
            })

        html_chunks.extend(['</body>', '</html>'])
        (annotator_dir / 'tasks.md').write_text('\n'.join(markdown_chunks), encoding='utf-8')
        (annotator_dir / 'tasks.html').write_text('\n'.join(html_chunks), encoding='utf-8')
        write_csv(
            annotator_dir / 'answers_template.csv',
            answer_rows,
            ['annotator_id', 'task_code', 'verdict', 'confidence', 'notes'],
        )

    write_csv(
        metadata_dir / 'annotator_task_map.csv',
        annotator_task_rows,
        ['annotator_id', 'task_code', 'base_task_id', 'result_stem', 'dataset', 'dimension', 'model_a', 'model_b', 'user_id', 'swap_lists', 'is_duplicate'],
    )

    manifest = {
        'package_name': output_dir.name,
        'seed': seed,
        'source_result_dir': str(source_result_dir),
        'n_base_tasks': len(corpus),
        'n_duplicates': len(duplicate_ids),
        'n_annotators': n_annotators,
        'tasks_per_annotator': len(corpus) + len(duplicate_ids),
        'dimensions': sorted({task['dimension'] for task in corpus}),
        'datasets': sorted({task['dataset'] for task in corpus}),
        'duplicate_base_task_ids': duplicate_ids,
    }
    (metadata_dir / 'package_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    organizer_notes = dedent(
        f"""
        # Organizer Notes

        Package root: `{output_dir}`

        Share only these files with annotators:
        - `annotator_01/tasks.html` and `annotator_01/answers_template.csv`
        - `annotator_02/tasks.html` and `annotator_02/answers_template.csv`
        - `annotator_03/tasks.html` and `annotator_03/answers_template.csv`
        - `annotator_instructions.md`

        Keep `metadata/` private. It contains the hidden answer mapping used for analysis.

        Suggested workflow:
        1. Ask each annotator to work independently.
        2. Ask them to fill `answers_template.csv` and return it as `answers_completed.csv`.
        3. After collecting all answer sheets, run:

           `python /root/new_paper/scripts/analyze_human_sanity_check.py --package-dir {output_dir} --answers <path1> <path2> <path3>`
        """
    ).strip()
    (output_dir / 'organizer_notes.md').write_text(organizer_notes + '\n', encoding='utf-8')

    annotator_instructions = dedent(
        """
        # Annotator Instructions

        Please judge each task independently.

        Rules:
        1. Use only the information shown in the task booklet.
        2. Ignore whether a list appears on the left/right or as List A/List B.
        3. Choose `TIE` only when the two lists are genuinely indistinguishable under the stated focus.
        4. Do not use search engines or external tools.
        5. Do not discuss the tasks with other annotators.

        Task types:
        - `Balanced`: consider relevance, diversity, novelty, and overall usefulness.
        - `Diversity`: consider only diversity and breadth of covered interests.

        Answer sheet:
        - `verdict`: must be `A`, `B`, or `TIE`
        - `confidence`: integer `1` to `5`
        - `notes`: optional short justification
        """
    ).strip()
    (output_dir / 'annotator_instructions.md').write_text(annotator_instructions + '\n', encoding='utf-8')

    readme = dedent(
        f"""
        # Human Sanity Check Package

        This package contains a small human evaluation set derived from the existing recommendation debiasing benchmark.

        Summary:
        - Base cases: {len(corpus)}
        - Reverse-order duplicates per annotator: {len(duplicate_ids)}
        - Annotators: {n_annotators}
        - Tasks per annotator: {len(corpus) + len(duplicate_ids)}
        - Source result directory: `{source_result_dir}`

        Key files:
        - `annotator_instructions.md`
        - `organizer_notes.md`
        - `metadata/package_manifest.json`
        - `metadata/annotator_task_map.csv`
        - `annotator_XX/tasks.html`
        - `annotator_XX/answers_template.csv`
        """
    ).strip()
    (output_dir / 'README.md').write_text(readme + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    configs = load_result_configs(args.source_result_dir)
    base_cases = sample_base_cases(configs, args.n_per_config, args.seed)
    duplicate_ids = allocate_duplicate_ids(base_cases, args.n_duplicates, args.seed)
    corpus = build_task_corpus(base_cases, args.history_items, args.seed)
    write_package(args.output_dir, corpus, duplicate_ids, args.n_annotators, args.seed, args.source_result_dir)
    print(json.dumps({
        'output_dir': str(args.output_dir),
        'n_configs': len(configs),
        'n_base_tasks': len(corpus),
        'n_duplicates': len(duplicate_ids),
        'n_annotators': args.n_annotators,
        'tasks_per_annotator': len(corpus) + len(duplicate_ids),
    }, indent=2))


if __name__ == '__main__':
    main()
