from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Stage(str, Enum):
    PLAN = "plan"
    LITERATURE = "literature"
    DATASETS = "datasets"
    MODELS = "models"
    INFERENCE = "inference"
    REPORT = "report"


@dataclass
class PaperHit:
    title: str
    url: str
    year: str | None
    score: float
    snippet: str = ""


@dataclass
class DatasetCandidate:
    id: str
    score: float
    downloads: int | None = None
    tags: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class ModelCandidate:
    id: str
    score: float
    downloads: int | None = None
    tags: list[str] = field(default_factory=list)
    pipeline_tag: str | None = None
    reason: str = ""
    loadable: bool | None = None


@dataclass
class InferenceResult:
    model_id: str
    dataset_id: str
    device: str
    mode: str
    artifact_paths: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class Plan:
    task: str
    intents: list[str]
    stages: list[Stage]
    search_queries: list[str]
    preferred_domains: list[str]
    notes: str = ""


@dataclass
class RunReport:
    task: str
    plan: Plan
    papers: list[PaperHit]
    datasets: list[DatasetCandidate]
    models: list[ModelCandidate]
    inference: InferenceResult | None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
