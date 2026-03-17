import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from Encoder import Encoder

class RecurrentCrossAttnBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, cross_attention_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.MultiheadAttention(),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.MultiheadAttention(),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.GRUCell(),
        )

    def forward(self, X):
        X = self.block(X)
        return X

class RecurrentSelfAttnBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.MultiheadAttention(),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.MultiheadAttention(),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.GRUCell(),
        )

    def forward(self, X):
        X = self.block(X)
        return X

class DiffusionModel(nn.Module):

    def __init__(self):
        super().__init__()


class RasterPipeline:

    def __init__(self):
        self.cross_attention_dim = 0
        self.size = 256

        self.context_encoder = Encoder(in_channels=1, cross_attention_dim=self.cross_attention_dim)
        self.borehole_encoder = Encoder(in_channels=2, cross_attention_dim=self.cross_attention_dim)

        self.borehole_cross_attn = nn.MultiheadAttention()

    def encode_context(self, elevation, magnetic, gravity):
        context = torch.stack([elevation, magnetic, gravity])
        context_encoded = self.context_encoder(context)

        return context_encoded



