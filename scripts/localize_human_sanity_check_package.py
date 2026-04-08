#!/usr/bin/env python3
# Create a Chinese-localized human sanity check package for participant distribution.

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

DIMENSION_ZH = {
    'Balanced': '综合判断',
    'Diversity': '多样性',
    'Relevance': '相关性',
    'Novelty': '新颖性',
}

INSTRUCTION_ZH = {
    'Compare the two lists by considering relevance, diversity, novelty, and overall usefulness.': '请综合比较两份推荐列表，考虑相关性、多样性、新颖性和整体实用性。',
    'Compare the two lists based only on diversity and breadth of covered interests.': '请只根据多样性和兴趣覆盖广度来比较两份推荐列表。',
    'Compare the two lists based only on relevance to the user profile.': '请只根据与用户画像的相关性来比较两份推荐列表。',
    'Compare the two lists based only on novelty, surprise, and discovery potential.': '请只根据新颖性、惊喜感和探索价值来比较两份推荐列表。',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def localize_tasks_md(text: str) -> str:
    replacements = {
        'Human Sanity Check Task Booklet': '人工复核任务册',
        'Use `answers_template.csv` to record your final verdicts.': '请在 `answers_template.csv` 中填写你的最终判断。',
        'Allowed verdicts: `A`, `B`, `TIE`.': '可选判断结果：`A`、`B`、`TIE`（平局）。',
        'Confidence must be an integer from `1` to `5`.': '置信度请填写 `1` 到 `5` 的整数。',
        '## Task ': '## 任务 ',
        '- Focus: ': '- 评价维度：',
        '### User Profile': '### 用户画像',
        'This user has previously interacted with:': '该用户过去互动过：',
        '### List A': '### 候选列表 A',
        '### List B': '### 候选列表 B',
        '### Instruction': '### 评价说明',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace(
        '- Record your answer in `answers_template.csv` for task code `',
        '- 请在 `answers_template.csv` 中找到任务编号 `',
    )
    text = text.replace(
        '`.
- Choose exactly one verdict: `A`, `B`, or `TIE`.',
        '` 并填写答案。
- 只能选择一个结果：`A`、`B` 或 `TIE`（平局）。',
    )
    text = text.replace(
        '- Confidence: integer `1` to `5`.',
        '- 置信度：填写 `1` 到 `5` 的整数。',
    )

    for en, zh in DIMENSION_ZH.items():
        text = text.replace(f'- 评价维度：{en}', f'- 评价维度：{zh}')
    for en, zh in INSTRUCTION_ZH.items():
        text = text.replace(en, zh)
    return text


def localize_tasks_html(text: str) -> str:
    replacements = {
        'Human Sanity Check': '人工复核任务',
        'Use <code>answers_template.csv</code> to record your final verdicts. Allowed verdicts: <code>A</code>, <code>B</code>, <code>TIE</code>. Confidence must be an integer from <code>1</code> to <code>5</code>.': '请在 <code>answers_template.csv</code> 中填写你的最终判断。可选结果为 <code>A</code>、<code>B</code>、<code>TIE</code>（平局）。置信度请填写 <code>1</code> 到 <code>5</code> 的整数。',
        '<h2>Task ': '<h2>任务 ',
        '<strong>Focus:</strong>': '<strong>评价维度：</strong>',
        '<strong>Answer sheet row:</strong>': '<strong>答题表行号：</strong>',
        '<strong>Allowed verdicts:</strong> A, B, or TIE': '<strong>可选结果：</strong> A、B、TIE（平局）',
        '<strong>Confidence:</strong> 1 to 5': '<strong>置信度：</strong> 1 到 5',
        '<h3>User Profile</h3>': '<h3>用户画像</h3>',
        '<h3>List A</h3>': '<h3>候选列表 A</h3>',
        '<h3>List B</h3>': '<h3>候选列表 B</h3>',
        '<h3>Instruction</h3>': '<h3>评价说明</h3>',
        'This user has previously interacted with:': '该用户过去互动过：',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    for en, zh in DIMENSION_ZH.items():
        text = text.replace(f'</strong> {en}</p>', f'</strong> {zh}</p>')
    for en, zh in INSTRUCTION_ZH.items():
        text = text.replace(en, zh)
    return text


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + '
', encoding='utf-8')


def build_root_docs(output_dir: Path) -> None:
    write_text(
        output_dir / 'README.md',
        '''# 人工复核任务包

这是一套给中文参与者使用的人工 sanity check 分发包。

摘要：
- 基础题目：40
- 每位参与者的反转重复题：10
- 参与者人数：3
- 每位参与者题量：50

关键文件：
- `annotator_instructions.md`
- `给参与者的转发文案.txt`
- `distribution/`
- `annotator_XX/tasks.html`
- `annotator_XX/answers_template.csv`
''',
    )
    write_text(
        output_dir / 'annotator_instructions.md',
        '''# 参与者说明

请独立完成每一道题。

规则：
1. 只使用任务册中给出的信息，不要使用搜索引擎或外部工具。
2. 忽略列表出现的位置，不要因为它是 A 或 B 就产生偏好。
3. 只有在两份列表确实难以区分时才选择 `TIE`（平局）。
4. 不要和其他参与者讨论题目。
5. 请直接在 `answers_template.csv` 中填写答案。

题型说明：
- `综合判断`：综合考虑相关性、多样性、新颖性和整体实用性。
- `多样性`：只考虑多样性和兴趣覆盖广度。

答题表字段：
- `verdict`：填写 `A`、`B` 或 `TIE`
- `confidence`：填写 `1` 到 `5` 的整数
- `notes`：可选，写一句简短理由
''',
    )
    write_text(
        output_dir / 'organizer_notes.md',
        '''# 组织者说明

这份目录是中文版分发包。分发时只需要发送 `distribution/` 里的独立压缩包。

不要发送：
- `metadata/`
- 本说明文件

收回三位参与者填写好的 CSV 后，继续使用原来的统计命令分析即可。
''',
    )
    write_text(
        output_dir / '给参与者的转发文案.txt',
        '''你好，这是一份推荐列表人工评测任务。

请你打开压缩包里的 `tasks.html`，按顺序阅读每道题，并在 `answers_template.csv` 中填写：
- `verdict`：A / B / TIE
- `confidence`：1 到 5
- `notes`：可选，简单写一句理由

注意：
1. 只根据任务页里提供的信息判断。
2. 不要因为列表在左边还是右边就产生偏好。
3. 只有两份列表确实分不出来时才选 TIE。
4. 完成后把填写好的 CSV 发回给我。

谢谢。
''',
    )


def build_distribution(output_dir: Path) -> None:
    dist_dir = output_dir / 'distribution'
    dist_dir.mkdir(parents=True, exist_ok=True)
    shared_files = [
        output_dir / 'annotator_instructions.md',
        output_dir / '给参与者的转发文案.txt',
    ]
    for idx in range(1, 4):
        annotator_id = f'annotator_{idx:02d}'
        annotator_dir = output_dir / annotator_id
        zip_path = dist_dir / f'{annotator_id}_中文版任务包.zip'
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for shared in shared_files:
                zf.write(shared, arcname=shared.name)
            for rel_name in ['tasks.html', 'tasks.md', 'answers_template.csv']:
                src = annotator_dir / rel_name
                zf.write(src, arcname=f'{annotator_id}/{rel_name}')
    write_text(
        dist_dir / 'README.txt',
        '''中文版人工复核分发包

请一人发送一个压缩包，不要混发。
不要发送 metadata 或 organizer 文件。

- annotator_01: annotator_01_中文版任务包.zip
- annotator_02: annotator_02_中文版任务包.zip
- annotator_03: annotator_03_中文版任务包.zip
''',
    )


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir
    output_dir = args.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copytree(source_dir / 'metadata', output_dir / 'metadata')

    for idx in range(1, 4):
        annotator_id = f'annotator_{idx:02d}'
        src_dir = source_dir / annotator_id
        dst_dir = output_dir / annotator_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_dir / 'answers_template.csv', dst_dir / 'answers_template.csv')
        write_text(dst_dir / 'tasks.md', localize_tasks_md((src_dir / 'tasks.md').read_text(encoding='utf-8')))
        write_text(dst_dir / 'tasks.html', localize_tasks_html((src_dir / 'tasks.html').read_text(encoding='utf-8')))

    build_root_docs(output_dir)
    build_distribution(output_dir)
    print(output_dir)


if __name__ == '__main__':
    main()
