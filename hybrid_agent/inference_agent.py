from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .tiny_vae import TinyVAE, train_tiny_vae


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> None:
    from torchvision.utils import save_image

    if images.ndim == 3:
        images = images.unsqueeze(1)
    images = images.detach().cpu().float()
    if images.min() < 0 or images.max() > 1:
        images = torch.sigmoid(images)
    save_image(images, str(path), nrow=nrow, padding=2)


def _mnist_loaders(dataset_id: str, batch_size: int = 128, n_eval: int = 16):
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    root = Path.home() / ".cache" / "hybrid_agent" / "data"
    root.mkdir(parents=True, exist_ok=True)
    tfm = transforms.ToTensor()
    name = dataset_id.lower()
    if "fashion" in name:
        train_ds = datasets.FashionMNIST(root=str(root), train=True, download=True, transform=tfm)
        test_ds = datasets.FashionMNIST(root=str(root), train=False, download=True, transform=tfm)
        resolved = "zalando-datasets/fashion_mnist"
    else:
        train_ds = datasets.MNIST(root=str(root), train=True, download=True, transform=tfm)
        test_ds = datasets.MNIST(root=str(root), train=False, download=True, transform=tfm)
        resolved = "ylecun/mnist"

    # Keep training light for cloud CPU demos.
    train_subset = Subset(train_ds, list(range(min(8000, len(train_ds)))))
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_x = torch.stack([test_ds[i][0].view(-1) for i in range(n_eval)], dim=0)
    return train_loader, eval_x, resolved


def _try_load_hub_vae(model_id: str, device: torch.device) -> Any | None:
    try:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        model.to(device)
        model.eval()
        return model
    except Exception:
        return None


def _hub_encode_decode(model: Any, batch: torch.Tensor, n_sample: int):
    with torch.no_grad():
        if hasattr(model, "encode"):
            mean, log_std = model.encode(batch)
        elif hasattr(model, "encoder"):
            enc = model.encoder
            mean, log_std = enc.encode(batch) if hasattr(enc, "encode") else enc(batch)
        else:
            raise RuntimeError("no encode")

        def decode(z: torch.Tensor) -> torch.Tensor:
            if hasattr(model, "decode"):
                return model.decode(z)
            if hasattr(model, "forward_latent_positions"):
                return model.forward_latent_positions(z)
            dec = model.decoder
            return dec.forward_latent_positions(z) if hasattr(dec, "forward_latent_positions") else dec(z)

        recon = decode(mean)
        z_prior = torch.randn(n_sample, mean.shape[-1], device=batch.device)
        samples = decode(z_prior)
    return mean, log_std, recon, samples


def _latent_collapsed(mean: torch.Tensor) -> bool:
    return bool(mean.std(dim=0).mean().item() < 1e-4)


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

    train_loader, eval_x, resolved_ds = _mnist_loaders(dataset_id, n_eval=n_recon)
    eval_x = eval_x.to(device)
    notes.append(f"Resolved dataset {resolved_ds}")

    used_model = model_id
    metrics: dict[str, Any] = {}
    mode = "hub_reconstruction+prior_sample"

    hub_model = _try_load_hub_vae(model_id, device)
    use_local = hub_model is None
    if hub_model is not None:
        notes.append(f"Loaded Hub model `{model_id}` via AutoModel")
        try:
            mean, log_std, recon, samples = _hub_encode_decode(hub_model, (eval_x > 0.5).float(), n_sample)
            if _latent_collapsed(mean):
                notes.append("Detected collapsed Hub latents; switching to local TinyVAE training")
                use_local = True
            else:
                recon_prob = torch.sigmoid(recon) if recon.min() < 0 or recon.max() > 1 else recon
                metrics["recon_mse"] = round(F.mse_loss(recon_prob, eval_x).item(), 6)
                mean_out, log_std_out = mean, log_std
                recon_img = recon_prob.view(-1, 1, 28, 28)
                sample_img = samples.view(-1, 1, 28, 28)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Hub inference failed ({exc}); switching to local TinyVAE")
            use_local = True
    else:
        notes.append(f"Hub load failed for `{model_id}`; training local TinyVAE on discovered dataset")

    if use_local:
        # Prefer the known-good CPU Hub model first if the requested one failed/collapsed.
        fallback_id = "pszmk/mnist-vae-latent2"
        if model_id != fallback_id:
            alt = _try_load_hub_vae(fallback_id, device)
            if alt is not None:
                try:
                    mean, log_std, recon, samples = _hub_encode_decode(alt, (eval_x > 0.5).float(), n_sample)
                    if not _latent_collapsed(mean):
                        used_model = fallback_id
                        notes.append(f"Used fallback Hub model `{fallback_id}`")
                        recon_prob = torch.sigmoid(recon) if recon.min() < 0 or recon.max() > 1 else recon
                        metrics["recon_mse"] = round(F.mse_loss(recon_prob, eval_x).item(), 6)
                        mean_out, log_std_out = mean, log_std
                        recon_img = recon_prob.view(-1, 1, 28, 28)
                        sample_img = samples.view(-1, 1, 28, 28)
                        use_local = False
                        mode = "hub_fallback_reconstruction+prior_sample"
                except Exception:
                    pass

    if use_local:
        model, stats = train_tiny_vae(train_loader, device=device, epochs=3, latent_dim=16)
        used_model = f"local:TinyVAE(trained_on={resolved_ds})"
        mode = "local_train+reconstruction+prior_sample"
        notes.append(
            f"Trained TinyVAE epochs={stats.epochs} final_loss={stats.final_loss} samples={stats.samples_seen}"
        )
        with torch.no_grad():
            mean_out, logvar = model.encode(eval_x)
            recon_logits = model.decode(mean_out)
            recon_prob = torch.sigmoid(recon_logits)
            z_prior = torch.randn(n_sample, model.latent_dim, device=device)
            sample_logits = model.decode(z_prior)
            metrics["recon_mse"] = round(F.mse_loss(recon_prob, eval_x).item(), 6)
            metrics["train_final_loss"] = stats.final_loss
            mean_out, log_std_out = mean_out, 0.5 * logvar
            recon_img = recon_prob.view(-1, 1, 28, 28)
            sample_img = sample_logits.view(-1, 1, 28, 28)
        ckpt = out_dir / "tiny_vae.pt"
        torch.save({"state_dict": model.state_dict(), "latent_dim": model.latent_dim}, ckpt)
        notes.append(f"Saved checkpoint {ckpt}")

    recon_path = out_dir / "reconstructions.png"
    sample_path = out_dir / "samples.png"
    _save_grid(recon_img, recon_path)
    _save_grid(sample_img, sample_path)

    latent_path = out_dir / "latent_means.json"
    latent_path.write_text(
        json.dumps(
            {
                "latent_dim": int(mean_out.shape[-1]),
                "mean_std": float(mean_out.std(dim=0).mean().item()),
                "means": mean_out.detach().cpu().tolist()[:16],
                "log_stds": log_std_out.detach().cpu().tolist()[:16],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    artifact_paths = [str(recon_path), str(sample_path), str(latent_path)]
    if (out_dir / "tiny_vae.pt").exists():
        artifact_paths.append(str(out_dir / "tiny_vae.pt"))

    result = {
        "model_id": used_model,
        "dataset_id": resolved_ds,
        "requested_model_id": model_id,
        "requested_dataset_id": dataset_id,
        "device": str(device),
        "mode": mode,
        "artifact_paths": artifact_paths,
        "metrics": metrics,
        "notes": "; ".join(notes),
    }
    (out_dir / "inference.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
