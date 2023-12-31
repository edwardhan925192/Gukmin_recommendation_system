import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# n_kernel = number of users
class GlobalKernel(nn.Module):
    def __init__(self, gk_size, dot_scale):
        super(GlobalKernel, self).__init__()
        self.gk_size = gk_size
        self.dot_scale = dot_scale
        self.conv_kernel = None

    def forward(self, input):

        # ======================== components of convolution is made with average pooled items ============ #
        avg_pooling = torch.mean(input, dim=0)  # Item (axis=0) based average pooling
        avg_pooling = avg_pooling.unsqueeze(0)  # (1, m) shape

        # Get n_kernel from the shape of avg_pooling
        n_kernel = avg_pooling.shape[1]

        # Initialize conv_kernel if it's not already
        # ========================== convolution layer =========================== #
        if self.conv_kernel is None:
            self.conv_kernel = nn.Parameter(torch.randn(n_kernel, self.gk_size**2) * 0.1)

        gk = torch.mm(avg_pooling, self.conv_kernel) * self.dot_scale  # Scaled dot product

        # Reshape to [1, 1, gk_size, gk_size]
        gk = gk.view(1, 1, self.gk_size, self.gk_size)

        return gk


class GlobalConv(nn.Module):
    def __init__(self):
        super(GlobalConv, self).__init__()

    def forward(self, input, W):
        # Reshape to [1, 1, input.shape[0], input.shape[1]]
        input = input.unsqueeze(0).unsqueeze(1)
        conv2d = F.relu(F.conv2d(input, W, stride=1, padding=1))
        return conv2d.squeeze(0).squeeze(1)
