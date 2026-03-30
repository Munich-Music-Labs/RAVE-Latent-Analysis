import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoderBlock(nn.Module):
    def __init__(self, in_dim, sparse_dim, rho=0.05):
        super().__init__()

        self.encoder = nn.Linear(in_dim, sparse_dim)
        self.decoder = nn.Linear(sparse_dim, in_dim)

        self.rho = rho


    def forward(self, l):
        # Sparse autoencoding
        sparse_code = F.relu(self.encoder(l))
        reconstructed = self.decoder(sparse_code)

        # Losses
        l1_loss = sparse_code.abs().mean()

        rho_hat = sparse_code.mean(dim=0)
        eps = 1e-8
        rho_hat = torch.clamp(rho_hat, eps, 1 - eps)
        rho = torch.full_like(rho_hat, self.rho)

        kl_loss = torch.sum(
            rho * torch.log(rho / rho_hat) +
            (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        )

        return reconstructed, sparse_code, l1_loss, kl_loss