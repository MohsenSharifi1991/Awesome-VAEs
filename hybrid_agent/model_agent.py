from __future__ import annotations

import math
from typing import Iterable

from .types import ModelCandidate

# Models known to load via transformers AutoModel for this demo path.
PREFERRED_LOADABLE = {
    "pszmk/mnist-vae-latent2": {
        "tags": ["vae", "mnist", "transformers", "safetensors", "cpu-friendly"],
        "downloads": 8,
        "reason": "known AutoModel + safetensors MNIST VAE (CPU-safe)",
        "priority": 2.5,
    },
    "uday9k/Gaussian_MNIST_VAE": {
        "tags": ["vae", "mnist", "pytorch"],
        "downloads": 10,
        "reason": "custom MNIST VAE weights on Hub",
        "priority": 0.5,
    },
}


def _tokens(text: str) -> set[str]:
    import re

    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def _score(
    model_id: str,
    tags: Iterable[str],
    downloads: int | None,
    domains: list[str],
    queries: list[str],
    pipeline_tag: str | None,
) -> tuple[float, str]:
    blob = (model_id + " " + " ".join(tags) + " " + (pipeline_tag or "")).lower()
    domain_hits = sum(1 for d in domains if d.lower() in blob)
    qtoks: set[str] = set()
    for q in queries:
        qtoks |= _tokens(q)
    overlap = len(_tokens(blob) & qtoks)
    vae_bonus = 2.0 if ("vae" in blob or "autoencoder" in blob) else 0.0
    load_bonus = 0.0
    tagset = {t.lower() for t in tags}
    if "transformers" in tagset or "safetensors" in tagset or "pytorch" in blob:
        load_bonus = 1.5
    if model_id in PREFERRED_LOADABLE:
        load_bonus += 3.0
    dl = math.log1p(downloads or 0)
    score = 2.2 * domain_hits + 1.0 * overlap + vae_bonus + load_bonus + 0.25 * dl
    reasons = []
    if domain_hits:
        reasons.append(f"domain_hits={domain_hits}")
    if vae_bonus:
        reasons.append("vae_match")
    if load_bonus:
        reasons.append(f"loadability={load_bonus:.1f}")
    if downloads:
        reasons.append(f"downloads={downloads}")
    return score, ", ".join(reasons) or "baseline"


def discover_models(
    queries: list[str],
    domains: list[str],
    limit: int = 6,
) -> list[ModelCandidate]:
    candidates: dict[str, ModelCandidate] = {}

    for mid, meta in PREFERRED_LOADABLE.items():
        score, reason = _score(mid, meta["tags"], meta.get("downloads"), domains, queries, None)
        priority = float(meta.get("priority", 1.0))
        candidates[mid] = ModelCandidate(
            id=mid,
            score=round(score + 1.0 + priority, 3),
            downloads=meta.get("downloads"),
            tags=list(meta["tags"]),
            reason=meta["reason"] + "; " + reason,
            loadable=True,
        )

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        searches = list(dict.fromkeys(queries + ["mnist vae", "variational autoencoder", "vae"]))
        seen: set[str] = set()
        for q in searches[:5]:
            for info in api.list_models(search=q, sort="downloads", limit=10):
                if info.id in seen:
                    continue
                seen.add(info.id)
                tags = list(getattr(info, "tags", None) or [])
                downloads = getattr(info, "downloads", None)
                pipeline_tag = getattr(info, "pipeline_tag", None)
                score, reason = _score(info.id, tags, downloads, domains, queries, pipeline_tag)
                # Soft filter: keep VAE-ish or domain-matching models
                blob = (info.id + " " + " ".join(tags)).lower()
                if not any(k in blob for k in ("vae", "autoencoder", "mnist", "generative")):
                    continue
                prev = candidates.get(info.id)
                if prev is None or score > prev.score:
                    candidates[info.id] = ModelCandidate(
                        id=info.id,
                        score=round(score, 3),
                        downloads=downloads,
                        tags=tags[:12],
                        pipeline_tag=pipeline_tag,
                        reason=reason + " [hub]",
                        loadable=info.id in PREFERRED_LOADABLE,
                    )
    except Exception as exc:  # noqa: BLE001
        _ = exc

    ranked = sorted(candidates.values(), key=lambda c: c.score, reverse=True)
    return ranked[:limit]
