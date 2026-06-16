import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoderBlock(nn.Module):
    def __init__(self, in_dim, sparse_dim):
        super().__init__()
        self.encoder = nn.Linear(in_dim, sparse_dim)
        self.decoder = nn.Linear(sparse_dim, in_dim)
        self.rms_norm = nn.RMSNorm(sparse_dim)

    def forward(self, x, lam=0.05):
        h = F.relu(self.encoder(x))
        f = self.rms_norm(h)
        reconstructed = self.decoder(f)

        recon_loss = F.mse_loss(reconstructed, x)
        l1_loss = h.abs().mean()
        loss = recon_loss + lam * l1_loss

        return reconstructed, f, loss
