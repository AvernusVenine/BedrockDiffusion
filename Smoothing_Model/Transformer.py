import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, X):
        X = self.proj(X).flatten(2).transpose(1, 2)
        return X


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, mod_res, base_res):
        super().__init__()

        self.G = (mod_res / base_res)
        self.embed_dim = embed_dim

        div_term = 1.0 / (10000.0 ** (torch.arange(0, embed_dim // 2).float() / embed_dim))

        self.register_buffer('div_term', div_term)

    def forward(self, X, H, W):
        row_pos = torch.arange(H, device=X.device, dtype=X.dtype)
        col_pos = torch.arange(W, device=X.device, dtype=X.dtype)
        dt = self.div_term.to(device=X.device, dtype=X.dtype)

        row_enc = torch.sin(self.G * row_pos.unsqueeze(1) * dt.unsqueeze(0))
        col_enc = torch.cos(self.G * col_pos.unsqueeze(1) * dt.unsqueeze(0))

        row_enc = row_enc.unsqueeze(1).expand(H, W, -1)
        col_enc = col_enc.unsqueeze(0).expand(H, W, -1)

        pe = torch.cat([row_enc, col_enc], dim=-1)
        pe = pe.reshape(1, H * W, self.embed_dim)

        return X + pe

class TransformerCrossBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, X, ctx):
        ctx = self.norm1(ctx).float()
        X_f = self.norm2(X).float()
        X = X + self.cross_attn(X_f, ctx, ctx)[0].to(X.dtype)
        X = X + self.mlp(self.norm3(X))
        return X


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, X):
        X_norm = self.norm1(X).float()
        X = X + self.attn(X_norm, X_norm, X_norm)[0]
        X = X + self.mlp(self.norm2(X))
        return X

class SmoothingTransformer(nn.Module):
    def __init__(self, patch_size, embed_dim, num_heads, depth, mlp_dim):
        super().__init__()

        bedrock_res = 30

        self.embed_dim = embed_dim
        self.patch_size = patch_size

        ###--- Bedrock Embedding ---###
        self.patch_embedding = PatchEmbedding(patch_size, 1, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, bedrock_res, bedrock_res)

        ###--- Transformer Architecture ---###
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, X):
        B = X.shape[0]

        H = X.shape[2] // self.patch_size
        W = X.shape[3] // self.patch_size

        X = self.patch_embedding(X)
        X = self.pos_encoding(X, H, W)

        for block in self.encoder_blocks:
            X = block(X)

        X = X.permute(0, 2, 1).reshape(B, self.embed_dim, H, W)

        X = self.upsample(X)
        X = self.refine(X)

        return X