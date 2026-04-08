"""Description normalization utilities adapted from DescNorm."""

from __future__ import annotations

import re


def normalize_item_text(text: str, mode: str = "title_only") -> str:
    text = " ".join(str(text).split())
    if mode == "identity":
        return text
    if mode == "title_only":
        if "|" in text:
            return text.split("|", 1)[0].strip()
        if ":" in text:
            return text.split(":", 1)[0].strip()
        return text
    if mode == "strip_richness":
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"\[[^\]]*\]", "", text)
        if "|" in text:
            left, _, right = text.partition("|")
            right = " ".join(right.split()[:12])
            return f"{left.strip()} | {right}".strip()
        return text
    raise ValueError(f"Unsupported DescNorm mode: {mode}")
