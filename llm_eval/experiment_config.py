"""Experiment configuration for the debiasing judge paper."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path("/root/new_paper")
RESULTS_DIR = PROJECT_ROOT / "results"
LLM_EVAL_RESULTS_DIR = RESULTS_DIR / "llm_eval"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

DEFAULT_MODEL_7B = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MODEL_14B = "Qwen/Qwen2.5-14B-Instruct"

SEED = 42
N_USERS = 50
DATASETS = ["yelp", "amazon"]

MODEL_PAIRS = {
    "yelp": [
        ("dcd", "dice"),
        ("dcd", "apdcl_bmax10"),
        ("dice", "apdcl_bmax10"),
        ("pdcl_b05", "apdcl_bmax10"),
        ("invcf", "paac"),
    ],
    "amazon": [
        ("dcd", "dice"),
        ("dcd", "apdcl_bmax10"),
        ("dice", "apdcl_bmax10"),
        ("pdcl_b05", "apdcl_bmax10"),
        ("invcf", "paac"),
    ],
}
