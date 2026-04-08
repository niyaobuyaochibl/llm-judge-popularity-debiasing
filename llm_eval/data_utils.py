"""Data loading helpers for the debiasing judge experiments."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path("/root/new_paper")
REC_LISTS_DIR = PROJECT_ROOT / "rec_lists"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_ROOTS = {
    "yelp": Path("/root/autodl-tmp/yelp-parquet"),
    "amazon": Path("/root/autodl-tmp/amazon-processed"),
}


def parse_user_key(key: str) -> int:
    if key.startswith("user_"):
        return int(key.split("_", 1)[1])
    return int(key)


def format_item_text(dataset: str, row: pd.Series, title_only: bool = False) -> str:
    if dataset == "yelp":
        text = str(row.get("description", "")).strip()
        if title_only and "|" in text:
            return text.split("|", 1)[0].strip()
        return text or f"Business #{row.name}"

    title = str(row.get("title_str", "")).strip()
    category = str(row.get("category_str", "")).strip()
    description = str(row.get("description_str", "")).strip()
    if title_only:
        return title or f"Book #{row.name}"
    if category and category.lower() != "nan":
        return f"{title} | {category}"
    if description and description.lower() != "nan":
        desc = " ".join(description.split())
        return f"{title} | {desc[:240]}"
    return title or f"Book #{row.name}"


@lru_cache(maxsize=8)
def load_item_texts(dataset: str, title_only: bool = False) -> dict[int, str]:
    meta_path = DATA_ROOTS[dataset] / "item_metadata.parquet"
    df = pd.read_parquet(meta_path)
    texts: dict[int, str] = {}
    if dataset == "yelp":
        for idx, row in df.iterrows():
            item_id = row.get("item_id", f"item_{idx}")
            if isinstance(item_id, str) and item_id.startswith("item_"):
                item_idx = int(item_id.split("_", 1)[1])
            else:
                item_idx = int(idx)
            texts[item_idx] = format_item_text(dataset, row, title_only=title_only)
    else:
        for idx, row in df.iterrows():
            item_idx = int(row.get("item_idx", idx))
            texts[item_idx] = format_item_text(dataset, row, title_only=title_only)
    return texts


def load_item_text_subset(dataset: str, item_ids: tuple[int, ...], title_only: bool = False) -> dict[int, str]:
    if dataset != "amazon":
        all_texts = load_item_texts(dataset, title_only=title_only)
        return {item_id: all_texts.get(item_id, f"Item #{item_id}") for item_id in item_ids}

    meta_path = DATA_ROOTS[dataset] / "item_metadata.parquet"
    df = pd.read_parquet(
        meta_path,
        columns=["item_idx", "title_str", "category_str", "description_str"],
        filters=[("item_idx", "in", list(item_ids))],
    )
    texts: dict[int, str] = {}
    for idx, row in df.iterrows():
        item_idx = int(row.get("item_idx", idx))
        texts[item_idx] = format_item_text(dataset, row, title_only=title_only)
    for item_id in item_ids:
        texts.setdefault(item_id, f"Book #{item_id}")
    return texts


@lru_cache(maxsize=4)
def load_train_interactions(dataset: str) -> pd.DataFrame:
    path = DATA_ROOTS[dataset] / "train_interactions.parquet"
    schema_names = [f.name for f in pq.read_schema(str(path))]
    if {"user_idx", "item_idx"}.issubset(schema_names):
        df = pd.read_parquet(path, columns=["user_idx", "item_idx"])
        return df.rename(columns={"user_idx": "user_id", "item_idx": "item_id"})

    df = pd.read_parquet(path, columns=["user_id", "item_id"])
    if dataset == "yelp":
        df["user_id"] = df["user_id"].astype(str).str.replace("user_", "", regex=False).astype(int)
        df["item_id"] = df["item_id"].astype(str).str.replace("item_", "", regex=False).astype(int)
        return df

    with (DATA_ROOTS[dataset] / "user_mapping.json").open("r", encoding="utf-8") as f:
        user_map = json.load(f)
    with (DATA_ROOTS[dataset] / "item_mapping.json").open("r", encoding="utf-8") as f:
        item_map = json.load(f)
    df["user_id"] = df["user_id"].map(user_map).astype(int)
    df["item_id"] = df["item_id"].map(item_map).astype(int)
    return df


def build_user_profile_summary(
    user_id: int,
    dataset: str,
    item_texts: dict[int, str],
    max_items: int = 10,
) -> str:
    train_df = load_train_interactions(dataset)
    user_rows = train_df[train_df["user_id"] == user_id]
    items = user_rows["item_id"].astype(int).tolist()[:max_items]
    if not items:
        return "This user has no recorded interaction history."
    lines = [f"- {item_texts.get(item, f'Item #{item}')}" for item in items]
    return "This user has previously interacted with:\n" + "\n".join(lines)


def load_user_histories(dataset: str, user_ids: tuple[int, ...], max_items: int = 10) -> dict[int, list[int]]:
    user_list = list(user_ids)
    if not user_list:
        return {}

    path = DATA_ROOTS[dataset] / "train_interactions.parquet"
    schema_names = [f.name for f in pq.read_schema(str(path))]
    if {"user_idx", "item_idx"}.issubset(schema_names):
        df = pd.read_parquet(
            path,
            columns=["user_idx", "item_idx"],
            filters=[("user_idx", "in", user_list)],
        )
        df = df.rename(columns={"user_idx": "user_id", "item_idx": "item_id"})
    elif dataset == "yelp":
        df = pd.read_parquet(
            path,
            columns=["user_id", "item_id"],
            filters=[("user_id", "in", [f"user_{uid}" for uid in user_list])],
        )
        df["user_id"] = df["user_id"].astype(str).str.replace("user_", "", regex=False).astype(int)
        df["item_id"] = df["item_id"].astype(str).str.replace("item_", "", regex=False).astype(int)
    else:
        with (DATA_ROOTS[dataset] / "user_mapping.json").open("r", encoding="utf-8") as f:
            user_map = json.load(f)
        with (DATA_ROOTS[dataset] / "item_mapping.json").open("r", encoding="utf-8") as f:
            item_map = json.load(f)
        rev_user = {v: k for k, v in user_map.items()}
        orig_ids = []
        for uid in user_list:
            orig = rev_user.get(uid)
            if orig is not None:
                orig_ids.append(orig)
        df = pd.read_parquet(
            path,
            columns=["user_id", "item_id"],
            filters=[("user_id", "in", orig_ids)] if orig_ids else None,
        )
        df["user_id"] = df["user_id"].map(user_map).astype(int)
        df["item_id"] = df["item_id"].map(item_map).astype(int)
        df = df[df["user_id"].isin(user_list)]

    histories: dict[int, list[int]] = {}
    for user_id, group in df.groupby("user_id", sort=False):
        histories[int(user_id)] = group["item_id"].astype(int).tolist()[:max_items]
    return histories


def render_user_profile_summary(item_ids: list[int], item_texts: dict[int, str]) -> str:
    if not item_ids:
        return "This user has no recorded interaction history."
    lines = [f"- {item_texts.get(item, f'Item #{item}')}" for item in item_ids]
    return "This user has previously interacted with:\n" + "\n".join(lines)


def format_recommendation_list(
    item_ids: list[int],
    item_texts: dict[int, str],
    numbered: bool = True,
    popularity_bucket_map: dict[int, str] | None = None,
) -> str:
    lines = []
    for idx, item_id in enumerate(item_ids):
        prefix = f"{idx + 1}. " if numbered else "- "
        item_text = item_texts.get(int(item_id), f"Item #{item_id}")
        if popularity_bucket_map is not None:
            bucket = popularity_bucket_map.get(int(item_id), "unknown").upper()
            item_text = f"{item_text} [Popularity: {bucket}]"
        lines.append(prefix + item_text)
    return "\n".join(lines)


def load_rec_payload(dataset: str, model: str, seed: int) -> dict[str, Any]:
    path = REC_LISTS_DIR / dataset / f"{model}_s{seed}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_recommendations(dataset: str, model: str, seed: int, user_subset: set[int] | None = None) -> dict[int, list[int]]:
    path = REC_LISTS_DIR / dataset / f"{model}_s{seed}.json"
    if user_subset is not None:
        import ijson
        recs: dict[int, list[int]] = {}
        with path.open("rb") as f:
            for key, value in ijson.kvitems(f, "recommendations"):
                uid = parse_user_key(key)
                if uid in user_subset:
                    recs[uid] = [int(x) for x in value]
                    if len(recs) >= len(user_subset):
                        break
        return recs
    payload = load_rec_payload(dataset, model, seed)
    recs_raw = payload["recommendations"]
    return {parse_user_key(key): [int(x) for x in value] for key, value in recs_raw.items()}


def load_recommendation_user_ids(dataset: str, model: str, seed: int) -> list[int]:
    import ijson
    path = REC_LISTS_DIR / dataset / f"{model}_s{seed}.json"
    user_ids: list[int] = []
    with path.open("rb") as f:
        in_recs = False
        depth = 0
        for prefix, event, value in ijson.parse(f):
            if prefix == "" and event == "map_key" and value == "recommendations":
                in_recs = True
                continue
            if in_recs:
                if event == "start_map" and depth == 0:
                    depth = 1
                    continue
                if depth == 1 and event == "map_key":
                    user_ids.append(parse_user_key(value))
                elif event == "start_map":
                    depth += 1
                elif event == "start_array":
                    depth += 1
                elif event in ("end_map", "end_array"):
                    depth -= 1
                    if depth == 0:
                        break
    return user_ids


def load_metrics(dataset: str, model: str, seed: int) -> dict[str, Any]:
    path = REC_LISTS_DIR / dataset / f"{model}_s{seed}.json"
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("metrics", {})


@lru_cache(maxsize=4)
def load_popularity_groups(dataset: str) -> dict[str, list[int]]:
    path = DATA_ROOTS[dataset] / "popularity_groups.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=4)
def load_popularity_bucket_map(dataset: str) -> dict[int, str]:
    groups = load_popularity_groups(dataset)
    bucket_map: dict[int, str] = {}
    for bucket, item_ids in groups.items():
        for item_id in item_ids:
            bucket_map[int(item_id)] = bucket
    return bucket_map


def summarize_list_popularity(
    item_ids: list[int],
    popularity_bucket_map: dict[int, str] | None,
) -> dict[str, dict | str] | None:
    if popularity_bucket_map is None:
        return None
    counts = {"head": 0, "mid": 0, "tail": 0, "unknown": 0}
    for item_id in item_ids:
        bucket = popularity_bucket_map.get(int(item_id), "unknown")
        counts[bucket] = counts.get(bucket, 0) + 1
    total = max(1, len(item_ids))
    text = (
        f"Head: {counts['head']}/{total}; "
        f"Mid: {counts['mid']}/{total}; "
        f"Tail: {counts['tail']}/{total}"
    )
    return {"counts": counts, "text": text}
