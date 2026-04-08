"""Prompt templates adapted from the TKDE/IPM judge projects."""

from __future__ import annotations


SYSTEM_PROMPTS = {
    "balanced": (
        "You are an expert recommendation system evaluator. "
        "Judge which recommendation list better serves the user based on relevance, "
        "diversity, novelty, and overall quality."
    ),
    "relevance": (
        "You are an expert recommendation system evaluator. "
        "Judge which list is more relevant to the user's demonstrated interests. "
        "Ignore diversity and novelty."
    ),
    "diversity": (
        "You are an expert recommendation system evaluator. "
        "Judge which list is more diverse in categories, topics, styles, or types. "
        "Ignore relevance."
    ),
    "novelty": (
        "You are an expert recommendation system evaluator. "
        "Judge which list is more novel, discovery-oriented, and less obvious. "
        "Ignore relevance unless needed to break ties."
    ),
}


INSTRUCTIONS = {
    "balanced": (
        "Compare the two lists by considering relevance, diversity, novelty, and overall usefulness."
    ),
    "relevance": (
        "Compare the two lists based only on relevance to the user profile."
    ),
    "diversity": (
        "Compare the two lists based only on diversity and breadth of covered interests."
    ),
    "novelty": (
        "Compare the two lists based only on novelty, surprise, and discovery potential."
    ),
}


POPULARITY_AWARE_SYSTEM_PROMPTS = {
    "balanced": (
        "You are an expert recommendation system evaluator. "
        "Each recommended item is annotated with a popularity bucket (HEAD, MID, or TAIL). "
        "Judge which list better serves the user while also providing healthier long-tail exposure "
        "and less over-reliance on HEAD items."
    ),
    "relevance": (
        "You are an expert recommendation system evaluator. "
        "Each recommended item is annotated with a popularity bucket (HEAD, MID, or TAIL). "
        "Judge which list is more relevant to the user's demonstrated interests, and use the popularity "
        "labels only as a secondary signal when relevance is otherwise similar."
    ),
    "diversity": (
        "You are an expert recommendation system evaluator. "
        "Each recommended item is annotated with a popularity bucket (HEAD, MID, or TAIL). "
        "Judge which list is more diverse and less popularity-concentrated, rewarding broader MID/TAIL exposure."
    ),
    "novelty": (
        "You are an expert recommendation system evaluator. "
        "Each recommended item is annotated with a popularity bucket (HEAD, MID, or TAIL). "
        "Judge which list is more novel and discovery-oriented, treating plausible MID/TAIL exposure as positive evidence."
    ),
}


POPULARITY_AWARE_INSTRUCTIONS = {
    "balanced": (
        "Compare the two lists by considering relevance, diversity, novelty, overall usefulness, and whether the list avoids unnecessary dependence on HEAD items in favor of MID/TAIL exposure."
    ),
    "relevance": (
        "Compare the two lists based primarily on relevance to the user profile. If relevance is close, prefer the list that avoids unnecessary concentration on HEAD items."
    ),
    "diversity": (
        "Compare the two lists based on diversity and breadth, using the popularity labels to prefer broader MID/TAIL exposure and lower concentration on HEAD items."
    ),
    "novelty": (
        "Compare the two lists based on novelty, surprise, and discovery potential, using the popularity labels as explicit evidence of long-tail discovery."
    ),
}


USER_TEMPLATE = """Given a user's interaction history and two recommendation lists from different systems, determine which list is better.

## User Profile
{user_profile}

## List A
{list_a}

## List B
{list_b}

## Instructions
{instruction}

Respond with ONLY "A", "B", or "TIE".

Your verdict:"""


USER_TEMPLATE_POPULARITY_AWARE = """Given a user's interaction history and two recommendation lists from different systems, determine which list is better.

## User Profile
{user_profile}

## Popularity Guidance
Each recommended item is labeled as HEAD, MID, or TAIL.
- HEAD: very popular or frequently exposed items
- MID: moderately popular items
- TAIL: niche or under-exposed items

## List A Summary
{list_a_summary}

## List A
{list_a}

## List B Summary
{list_b_summary}

## List B
{list_b}

## Instructions
{instruction}

Use the popularity labels explicitly when making the decision.
Respond with ONLY "A", "B", or "TIE".

Your verdict:"""


def get_dimension_prompt(
    dimension: str,
    user_profile: str,
    list_a: str,
    list_b: str,
    prompt_style: str = "standard",
    list_a_summary: str | None = None,
    list_b_summary: str | None = None,
) -> tuple[str, str]:
    if prompt_style == "standard":
        system = SYSTEM_PROMPTS[dimension]
        user = USER_TEMPLATE.format(
            user_profile=user_profile,
            list_a=list_a,
            list_b=list_b,
            instruction=INSTRUCTIONS[dimension],
        )
        return system, user

    if prompt_style == "popularity_aware":
        system = POPULARITY_AWARE_SYSTEM_PROMPTS[dimension]
        user = USER_TEMPLATE_POPULARITY_AWARE.format(
            user_profile=user_profile,
            list_a_summary=list_a_summary or "Head: unknown; Mid: unknown; Tail: unknown",
            list_a=list_a,
            list_b_summary=list_b_summary or "Head: unknown; Mid: unknown; Tail: unknown",
            list_b=list_b,
            instruction=POPULARITY_AWARE_INSTRUCTIONS[dimension],
        )
        return system, user

    raise ValueError(f"Unsupported prompt_style: {prompt_style}")
