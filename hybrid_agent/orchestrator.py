from __future__ import annotations

import json
from pathlib import Path

from .dataset_agent import discover_datasets
from .inference_agent import run_inference
from .literature_agent import search_literature
from .model_agent import discover_models
from .planner import plan_task
from .types import InferenceResult, RunReport, Stage


def run_pipeline(task: str, out_dir: str | Path = "runs/latest") -> RunReport:
    """Hybrid control loop: symbolic plan, then tool agents, then inference."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    plan = plan_task(task)
    (out / "plan.json").write_text(
        json.dumps(
            {
                "task": plan.task,
                "intents": plan.intents,
                "stages": [s.value for s in plan.stages],
                "search_queries": plan.search_queries,
                "preferred_domains": plan.preferred_domains,
                "notes": plan.notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    papers = []
    datasets = []
    models = []
    inference: InferenceResult | None = None

    if Stage.LITERATURE in plan.stages:
        try:
            papers = search_literature(plan.search_queries, domains=plan.preferred_domains, limit=8)
            (out / "literature.json").write_text(
                json.dumps([p.__dict__ for p in papers], indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"literature: {exc}")

    if Stage.DATASETS in plan.stages:
        try:
            datasets = discover_datasets(plan.search_queries, plan.preferred_domains, limit=6)
            (out / "datasets.json").write_text(
                json.dumps([d.__dict__ for d in datasets], indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"datasets: {exc}")

    if Stage.MODELS in plan.stages:
        try:
            models = discover_models(plan.search_queries, plan.preferred_domains, limit=6)
            (out / "models.json").write_text(
                json.dumps([m.__dict__ for m in models], indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"models: {exc}")

    if Stage.INFERENCE in plan.stages:
        top_model = next((m for m in models if m.loadable), models[0] if models else None)
        top_dataset = datasets[0] if datasets else None
        if top_model is None or top_dataset is None:
            errors.append("inference: missing model or dataset candidate")
        else:
            try:
                raw = run_inference(
                    model_id=top_model.id,
                    dataset_id=top_dataset.id,
                    out_dir=out / "inference",
                )
                inference = InferenceResult(
                    model_id=raw["model_id"],
                    dataset_id=raw["dataset_id"],
                    device=raw["device"],
                    mode=raw["mode"],
                    artifact_paths=raw["artifact_paths"],
                    metrics=raw["metrics"],
                    notes=raw["notes"],
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"inference: {exc}")

    report = RunReport(
        task=task,
        plan=plan,
        papers=papers,
        datasets=datasets,
        models=models,
        inference=inference,
        errors=errors,
    )
    (out / "report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    # Human-readable summary
    lines = [
        f"# Hybrid Agent Run",
        f"",
        f"**Task:** {task}",
        f"",
        f"## Plan",
        f"- Intents: {', '.join(plan.intents)}",
        f"- Domains: {', '.join(plan.preferred_domains)}",
        f"- Queries: {', '.join(plan.search_queries)}",
        f"",
        f"## Literature ({len(papers)})",
    ]
    for p in papers[:5]:
        lines.append(f"- ({p.year}) {p.title} — {p.url} [score={p.score}]")
    lines += ["", f"## Datasets ({len(datasets)})"]
    for d in datasets:
        lines.append(f"- `{d.id}` score={d.score} — {d.reason}")
    lines += ["", f"## Models ({len(models)})"]
    for m in models:
        lines.append(f"- `{m.id}` score={m.score} loadable={m.loadable} — {m.reason}")
    lines += ["", "## Inference"]
    if inference:
        lines.append(f"- Model: `{inference.model_id}`")
        lines.append(f"- Dataset: `{inference.dataset_id}`")
        lines.append(f"- Device: `{inference.device}`")
        lines.append(f"- Metrics: {inference.metrics}")
        lines.append(f"- Artifacts:")
        for a in inference.artifact_paths:
            lines.append(f"  - `{a}`")
        lines.append(f"- Notes: {inference.notes}")
    else:
        lines.append("- No inference result")
    if errors:
        lines += ["", "## Errors"]
        for e in errors:
            lines.append(f"- {e}")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
