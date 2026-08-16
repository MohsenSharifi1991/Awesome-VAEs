from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> None:
    """Save a [N,1,H,W] or [N,H,W] tensor batch as a simple PNG grid via torchvision."""
    from torchvision.utils import save_image

    if images.ndim == 3:
        images = images.unsqueeze(1)
    images = images.detach().cpu().float()
    # binarized logits → probs if outside [0,1]
    if images.min() < 0 or images.max() > 1:
        images = torch.sigmoid(images)
    save_image(images, str(path), nrow=nrow, padding=2)


def _load_mnist_batch(dataset_id: str, n: int, device: torch.device) -> tuple[torch.Tensor, str]:
    """Return flattened [N,784] float tensors in [0,1], plus resolved dataset name."""
    # Prefer torchvision for reliability; map Hub ids to torchvision datasets.
    from torchvision import datasets, transforms

    root = Path.home() / ".cache" / "hybrid_agent" / "data"
    root.mkdir(parents=True, exist_ok=True)
    tfm = transforms.ToTensor()

    name = dataset_id.lower()
    if "fashion" in name:
        ds = datasets.FashionMNIST(root=str(root), train=False, download=True, transform=tfm)
        resolved = "zalando-datasets/fashion_mnist"
    else:
        ds = datasets.MNIST(root=str(root), train=False, download=True, transform=tfm)
        resolved = "ylecun/mnist"

    xs = []
    for i in range(min(n, len(ds))):
        x, _ = ds[i]
        xs.append(x.view(-1))
    batch = torch.stack(xs, dim=0).to(device)
    return batch, resolved


def _load_pszmk_vae(model_id: str, device: torch.device) -> Any:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model


def _reconstruct_and_sample(model: Any, batch: torch.Tensor, n_sample: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Works with pszmk/mnist-vae-latent2 style models (encoder + decoder submodules)."""
    metrics: dict[str, float] = {}
    with torch.no_grad():
        if hasattr(model, "encode"):
            mean, log_std = model.encode(batch)
        elif hasattr(model, "encoder"):
            enc = model.encoder
            mean, log_std = enc.encode(batch) if hasattr(enc, "encode") else enc(batch)
        else:
            raise RuntimeError("Unsupported model interface for encode")

        if hasattr(model, "decode"):
            recon = model.decode(mean)
        elif hasattr(model, "forward_latent_positions"):
            recon = model.forward_latent_positions(mean)
        elif hasattr(model, "decoder"):
            dec = model.decoder
            recon = dec.forward_latent_positions(mean) if hasattr(dec, "forward_latent_positions") else dec(mean)
        else:
            raise RuntimeError("Unsupported model interface for decode")

        recon_prob = torch.sigmoid(recon) if recon.min() < 0 or recon.max() > 1 else recon
        metrics["recon_mse"] = round(F.mse_loss(recon_prob, batch).item(), 6)

        latent_dim = mean.shape[-1]
        z_prior = torch.randn(n_sample, latent_dim, device=batch.device)
        if hasattr(model, "decode"):
            samples = model.decode(z_prior)
        elif hasattr(model, "forward_latent_positions"):
            samples = model.forward_latent_positions(z_prior)
        else:
            dec = model.decoder
            samples = dec.forward_latent_positions(z_prior) if hasattr(dec, "forward_latent_positions") else dec(z_prior)

    return recon_prob.view(-1, 1, 28, 28), samples.view(-1, 1, 28, 28), metrics


def run_inference(
    model_id: str,
    dataset_id: str,
    out_dir: Path,
    n_recon: int = 16,
    n_sample: int = 16,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device()
    notes: list[str] = []

    batch, resolved_ds = _load_mnist_batch(dataset_id, n_recon, device)
    notes.append(f"Loaded evaluation batch from {resolved_ds} via torchvision")

    # Prefer known loadable AutoModel path
    if model_id == "pszmk/mnist-vae-latent2" or "mnist-vae" in model_id.lower():
        try:
            model = _load_pszmk_vae("pszmk/mnist-vae-latent2", device)
            used_model = "pszmk/mnist-vae-latent2"
            notes.append("Loaded via transformers.AutoModel(trust_remote_code=True)")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load preferred VAE: {exc}") from exc
    else:
        # Attempt AutoModel; if it fails, fall back to preferred MNIST VAE.
        try:
            model = _load_pszmk_vae(model_id, device)
            used_model = model_id
            notes.append("Loaded requested model via AutoModel")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Requested model load failed ({exc}); falling back to pszmk/mnist-vae-latent2")
            model = _load_pszmk_vae("pszmk/mnist-vae-latent2", device)
            used_model = "pszmk/mnist-vae-latent2"

    # Binarize slightly for this MNIST VAE training convention
    batch_bin = (batch > 0.5).float()
    recon, samples, metrics = _reconstruct_and_sample(model, batch_bin, n_sample)

    recon_path = out_dir / "reconstructions.png"
    sample_path = out_dir / "samples.png"
    _save_grid(recon, recon_path)
    # samples may be logits
    _save_grid(samples, sample_path)

    # Also dump a few latent means for inspection
    with torch.no_grad():
        if hasattr(model, "encode"):
            mean, log_std = model.encode(batch_bin)
        else:
            mean, log_std = model.encoder(batch_bin)
    latent_path = out_dir / "latent_means.json"
    latent_path.write_text(
        json.dumps(
            {
                "latent_dim": int(mean.shape[-1]),
                "means": mean.detach().cpu().tolist(),
                "log_stds": log_std.detach().cpu().tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    artifact_paths = [str(recon_path), str(sample_path), str(latent_path)]
    result = {
        "model_id": used_model,
        "dataset_id": resolved_ds,
        "requested_model_id": model_id,
        "requested_dataset_id": dataset_id,
        "device": str(device),
        "mode": "reconstruction+prior_sample",
        "artifact_paths": artifact_paths,
        "metrics": metrics,
        "notes": "; ".join(notes),
    }
    (out_dir / "inference.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
