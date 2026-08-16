from __future__ import annotations

import math
from typing import Iterable

from .types import DatasetCandidate

# Local catalog used when Hub is slow/unavailable and for score boosting.
LOCAL_CATALOG: list[dict] = [
    {
        "id": "ylecun/mnist",
        "tags": ["mnist", "digit", "image", "vision"],
        "downloads": 80000,
    },
    {
        "id": "zalando-datasets/fashion_mnist",
        "tags": ["fashion-mnist", "clothing", "image", "vision"],
        "downloads": 30000,
    },
    {
        "id": "cifar10",
        "tags": ["cifar", "cifar10", "image", "vision"],
        "downloads": 20000,
    },
]


def _tokens(text: str) -> set[str]:
    import re

    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def _score(dataset_id: str, tags: Iterable[str], downloads: int | None, domains: list[str], queries: list[str]) -> tuple[float, str]:
    blob = (dataset_id + " " + " ".join(tags)).lower()
    hay = _tokens(blob)
    domain_hits = sum(1 for d in domains if d.lower() in blob)
    qtoks: set[str] = set()
    for q in queries:
        qtoks |= _tokens(q)
    overlap = len(hay & qtoks)
    dl = math.log1p(downloads or 0)
    score = 2.5 * domain_hits + 1.2 * overlap + 0.35 * dl
    reasons = []
    if domain_hits:
        reasons.append(f"domain_hits={domain_hits}")
    if overlap:
        reasons.append(f"query_overlap={overlap}")
    if downloads:
        reasons.append(f"downloads={downloads}")
    return score, ", ".join(reasons) or "baseline"


def discover_datasets(
    queries: list[str],
    domains: list[str],
    limit: int = 6,
) -> list[DatasetCandidate]:
    candidates: dict[str, DatasetCandidate] = {}

    for row in LOCAL_CATALOG:
        score, reason = _score(row["id"], row["tags"], row.get("downloads"), domains, queries)
        candidates[row["id"]] = DatasetCandidate(
            id=row["id"],
            score=round(score, 3),
            downloads=row.get("downloads"),
            tags=list(row["tags"]),
            reason=reason + " [catalog]",
        )

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        seen_ids: set[str] = set()
        for q in queries[:4]:
            for info in api.list_datasets(search=q, sort="downloads", limit=8):
                if info.id in seen_ids:
                    continue
                seen_ids.add(info.id)
                tags = list(getattr(info, "tags", None) or [])
                downloads = getattr(info, "downloads", None)
                score, reason = _score(info.id, tags, downloads, domains, queries)
                prev = candidates.get(info.id)
                if prev is None or score > prev.score:
                    candidates[info.id] = DatasetCandidate(
                        id=info.id,
                        score=round(score, 3),
                        downloads=downloads,
                        tags=tags[:12],
                        reason=reason + " [hub]",
                    )
    except Exception as exc:  # noqa: BLE001 — agent must degrade gracefully
        # Keep catalog results; orchestrator records the error.
        _ = exc

    ranked = sorted(candidates.values(), key=lambda c: c.score, reverse=True)
    return ranked[:limit]
