from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyVAE(nn.Module):
    """Compact MLP VAE for 28x28 grayscale digits (CPU-friendly)."""

    def __init__(self, latent_dim: int = 16, hidden: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(784, hidden)
        self.fc21 = nn.Linear(hidden, latent_dim)
        self.fc22 = nn.Linear(hidden, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden)
        self.fc4 = nn.Linear(hidden, 784)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc3(z))
        return self.fc4(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_logits: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld) / x.size(0)


@dataclass
class TrainStats:
    epochs: int
    final_loss: float
    samples_seen: int


def train_tiny_vae(
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 3,
    latent_dim: int = 16,
    lr: float = 1e-3,
) -> tuple[TinyVAE, TrainStats]:
    model = TinyVAE(latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    samples_seen = 0
    last = 0.0
    for _ in range(epochs):
        total = 0.0
        n = 0
        for batch, _ in train_loader:
            batch = batch.view(batch.size(0), -1).to(device)
            opt.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            opt.step()
            total += loss.item() * batch.size(0)
            n += batch.size(0)
            samples_seen += batch.size(0)
        last = total / max(n, 1)
    model.eval()
    return model, TrainStats(epochs=epochs, final_loss=round(last, 4), samples_seen=samples_seen)
