import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, in_channels, cross_attention_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Conv2d(256, cross_attention_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, cross_attention_dim),
            nn.ReLU()
        )