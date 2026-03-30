import torch
import torch.nn as nn

class Embedder(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.SiLU()
        )

    def forward(self, X):
        X = self.embedder(X)
        return X

