import torch.nn as nn

class ContextEncoder(nn.Module):

    def __init__(self, in_channels, cross_attention_dim=512, seq_len=64):
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
            nn.ReLU(),
        )

        size = int(seq_len ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d((size, size))

    def forward(self, X):
        X = self.encoder(X)
        X = self.pool(X)

        B, D, H, W = X.shape
        X = X.view(B, D, H * W)
        X = X.permute(0, 2, 1)

        return X