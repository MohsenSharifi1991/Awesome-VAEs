from __future__ import annotations

import re

from .types import Plan, Stage

INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("vae", re.compile(r"\b(vae|variational\s+autoencoder|auto-?encoder)\b", re.I)),
    ("disentanglement", re.compile(r"\bdisentangl", re.I)),
    ("generation", re.compile(r"\b(generat|sample|synthesize|latent)\b", re.I)),
    ("reconstruction", re.compile(r"\b(reconstr|encode|decode|compress)\b", re.I)),
    ("mnist", re.compile(r"\b(mnist|digit)\b", re.I)),
    ("fashion", re.compile(r"\bfashion[- ]?mnist\b", re.I)),
    ("celeba", re.compile(r"\b(celeba|face)\b", re.I)),
    ("cifar", re.compile(r"\bcifar\b", re.I)),
]


DOMAIN_PRIORS = {
    "mnist": ["mnist", "digit", "binarized"],
    "fashion": ["fashion-mnist", "fashion_mnist", "clothing"],
    "celeba": ["celeba", "faces", "portrait"],
    "cifar": ["cifar", "cifar10"],
}


def extract_intents(task: str) -> list[str]:
    found = [name for name, pat in INTENT_PATTERNS if pat.search(task)]
    if "vae" not in found:
        found.insert(0, "vae")  # repo prior: Awesome-VAEs
    if "generation" not in found and "reconstruction" not in found:
        found.append("generation")
        found.append("reconstruction")
    # de-dupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for x in found:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_search_queries(task: str, intents: list[str]) -> list[str]:
    queries = [task.strip()]
    base = []
    if "vae" in intents:
        base.append("vae")
        base.append("variational autoencoder")
    if "mnist" in intents:
        base.append("mnist vae")
    if "fashion" in intents:
        base.append("fashion mnist vae")
    if "celeba" in intents:
        base.append("celeba vae")
    if "cifar" in intents:
        base.append("cifar vae")
    if "disentanglement" in intents:
        base.append("disentanglement vae")
    for q in base:
        if q.lower() not in {x.lower() for x in queries}:
            queries.append(q)
    return queries[:6]


def preferred_domains(intents: list[str]) -> list[str]:
    domains: list[str] = []
    for key, vals in DOMAIN_PRIORS.items():
        if key in intents:
            domains.extend(vals)
    if not domains:
        domains = ["mnist", "digit"]  # safe default for CPU demo
    return domains


def plan_task(task: str) -> Plan:
    intents = extract_intents(task)
    stages = [
        Stage.PLAN,
        Stage.LITERATURE,
        Stage.DATASETS,
        Stage.MODELS,
        Stage.INFERENCE,
        Stage.REPORT,
    ]
    return Plan(
        task=task.strip(),
        intents=intents,
        stages=stages,
        search_queries=build_search_queries(task, intents),
        preferred_domains=preferred_domains(intents),
        notes=(
            "Hybrid plan: symbolic intent extraction + fixed research DAG; "
            "agents adapt ranking and may fall back during inference."
        ),
    )
