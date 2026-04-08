# LLM Judge Validation for Popularity Debiasing

This repository contains the code, manuscript, parsed evaluation outputs, and analysis artifacts for the paper:

**Can Large Language Model Judges Recognize Popularity Debiasing? A Multi-Judge Benchmark and Validation Protocol for Large-Scale Recommender Evaluation**

## Repository contents

- `llm_eval/`: evaluation pipeline, prompts, and judge orchestration code
- `scripts/`: analysis, result-repair, and paper-asset generation scripts
- `paper/`: current Journal of Big Data manuscript source and compiled PDF
- `traditional_metrics/`: offline metric summaries for Yelp and Amazon
- `index/`: artifact and run index files used during the study
- `results/analysis/`: aggregated cross-judge and popularity-aware comparison reports
- `results/human_eval/analysis_distinct_v2/`: aggregate human sanity-check analysis outputs
- `results/llm_eval/`: sanitized parsed verdict files with raw model text removed

## Data-release note

This public repository includes derived evaluation artifacts, analysis outputs, and sanitized parsed verdict files. It does not include large raw recommendation-list exports or private API credentials. The underlying Yelp and Amazon datasets remain available from their original public sources cited in the manuscript.

## Reproducibility note

The parsed judge outputs included here are sanitized for public release by removing raw generation text while preserving verdicts, consistency labels, alignment fields, and latency metadata used in the paper's analyses.
