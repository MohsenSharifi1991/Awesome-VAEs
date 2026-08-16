# Hybrid Agentic System for VAE Research Workflows

## Goal

Given a natural-language research/ML task, autonomously:

1. Decompose the task into a plan
2. Retrieve related literature from Awesome-VAEs
3. Discover suitable datasets (Hugging Face Hub)
4. Discover suitable models (Hugging Face Hub)
5. Run cloud inference and persist artifacts

## Why “hybrid”

The system is hybrid along three axes:

| Axis | Symbolic / deterministic | Agentic / adaptive |
|------|--------------------------|--------------------|
| Control | Rule-based planner emits a fixed DAG of stages | Orchestrator can skip, retry, or fall back per stage results |
| Retrieval | Keyword extraction + heuristic ranking | Live Hub queries + literature search over the curated paper list |
| Execution | Explicit inference contract (reconstruct / sample) | Model-specific loaders with remote-code trust + CPU fallback |

This avoids a fully opaque LLM-only loop while still being tool-using and adaptive.

## Architecture

```
                    ┌─────────────────────┐
   user task ──────►│  Symbolic Planner   │
                    │  (intent + DAG)     │
                    └─────────┬───────────┘
                              │ Plan
                    ┌─────────▼───────────┐
                    │    Orchestrator     │
                    │  (hybrid control)   │
                    └──┬────┬────┬────┬───┘
                       │    │    │    │
           literature  │    │    │    │ inference
                       ▼    ▼    ▼    ▼
                 Awesome  HF   HF   Torch /
                 -VAEs   Data Models Hub run
```

### Agents

- **Planner** — maps task text to intent tags (`vae`, `mnist`, `generation`, …) and a stage list.
- **LiteratureAgent** — keyword search over `README.md` (Awesome-VAEs).
- **DatasetAgent** — Hugging Face `list_datasets` + local catalog boosts.
- **ModelAgent** — Hugging Face `list_models` with VAE-oriented queries and ranking.
- **InferenceAgent** — loads the top feasible model, pulls a dataset sample, runs reconstruction and/or latent sampling, writes images + JSON report.

### Ranking

Candidates are scored with transparent heuristics:

- keyword overlap with the task
- Hub download counts (log-scaled)
- task/domain priors (e.g. MNIST + VAE)
- loadability hints (`transformers`, `safetensors`, `pytorch`)

## Cloud execution

This package is intended to run inside a cloud agent / VM (no local GPU required). Inference uses CPU by default; CUDA is used when present.

### Inference fallbacks

1. Load the top-ranked Hub model with `transformers.AutoModel`
2. If load fails **or** latent means are collapsed, try a known CPU-safe Hub MNIST VAE
3. If still unusable, train a local `TinyVAE` on the discovered dataset (few epochs, CPU) and run reconstruction + prior sampling

## CLI

```bash
python3 -m hybrid_agent "Generate MNIST digits with a VAE and reconstruct samples"
python3 -m hybrid_agent --task "Fashion-MNIST VAE reconstruction" --out runs/demo
```
