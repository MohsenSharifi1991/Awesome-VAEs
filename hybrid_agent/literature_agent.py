from __future__ import annotations

import math
import re
from pathlib import Path

from .types import PaperHit

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

YEAR_RE = re.compile(r"^##\s+(\d{4})\s*$")
LINK_RE = re.compile(r"(https?://\S+)")


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def search_literature(queries: list[str], limit: int = 8) -> list[PaperHit]:
    if not README.exists():
        return []

    query_tokens: set[str] = set()
    for q in queries:
        query_tokens |= _tokenize(q)
    # Always bias toward VAE core terms from this repo
    query_tokens |= {"vae", "variational", "autoencoder", "disentanglement"}

    year: str | None = None
    scored: list[PaperHit] = []

    for line in README.read_text(encoding="utf-8", errors="ignore").splitlines():
        ym = YEAR_RE.match(line.strip())
        if ym:
            year = ym.group(1)
            continue
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls = LINK_RE.findall(line)
        if not urls:
            continue
        title = LINK_RE.sub("", line).strip(" .;-")
        if len(title) < 8:
            continue
        tokens = _tokenize(title)
        overlap = len(tokens & query_tokens)
        if overlap == 0:
            continue
        # Prefer denser keyword matches; slight recency bump
        year_bonus = 0.0
        if year and year.isdigit():
            year_bonus = max(0.0, (int(year) - 2015) / 20.0)
        score = overlap + 0.35 * math.log1p(overlap) + year_bonus
        scored.append(
            PaperHit(
                title=title[:200],
                url=urls[0].rstrip(").,]"),
                year=year,
                score=round(score, 3),
                snippet=line[:240],
            )
        )

    scored.sort(key=lambda p: p.score, reverse=True)
    # de-dupe by title prefix
    uniq: list[PaperHit] = []
    seen: set[str] = set()
    for p in scored:
        key = p.title.lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
        if len(uniq) >= limit:
            break
    return uniq
