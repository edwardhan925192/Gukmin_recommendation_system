import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# kernel function
class LocalKernel(nn.Module):
    def __init__(self):
        super(LocalKernel, self).__init__()

    def forward(self, u, v):
        # Euclidean distance
        dist = torch.norm(u - v, p=2, dim=2)
        hat = torch.clamp(1. - dist**2, min=0.)
        return hat

# ====================== main goal ======================== #
# Knowing which inputs should work with which hidden dimensions.
class KernelLayer(nn.Module):
    def __init__(self, n_in, n_hid, n_dim, lambda_s, lambda_2, activation=F.sigmoid):
        super(KernelLayer, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_dim = n_dim
        self.lambda_s = lambda_s
        self.lambda_2 = lambda_2
        self.activation = activation

        self.W = nn.Parameter(torch.randn(n_in, n_hid))
        self.u = nn.Parameter(torch.randn(n_in, 1, n_dim) * 1e-3)


        self.v = nn.Parameter(torch.randn(1, n_hid, n_dim) * 1e-3)
        self.b = nn.Parameter(torch.randn(n_hid))
        self.local_kernel = LocalKernel()

    def forward(self, x):

        print(x.shape)
        # decoder
        w_hat = self.local_kernel(self.u, self.v)

        # Regularization terms
        sparse_reg_term = self.lambda_s * torch.norm(w_hat, p=2)
        l2_reg_term = self.lambda_2 * torch.norm(self.W, p=2)

        W_eff = self.W * w_hat  # Local kernelised weight matrix

        y = torch.mm(x, W_eff) + self.b
        y = self.activation(y)

        return y, sparse_reg_term + l2_reg_term
