import math
import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock
from construction.Encoder import AvgPoolFilter2D

class PatchEmbedding(nn.Module):
    """
    IN:  (B x C x N x N)
    OUT: (B x (N/patch_size)**2 x embed_dim)
    """
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, X):
        X = self.proj(X).flatten(2).transpose(1, 2)
        return X


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super().__init__()

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

        pe = torch.zeros(1, seq_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, X):
        X = X + self.pe[:, :X.size(1)]
        return X


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, X, c):
        X = X + self.cross_attn(self.norm1(X), c, c)[0]
        X_norm = self.norm2(X)
        X = X + self.attn(X_norm, X_norm, X_norm)[0]
        X = X + self.mlp(self.norm3(X))

        return X

class BedrockTransformer(nn.Module):
    def __init__(self, raster_size, patch_size, embed_dim, num_heads, depth, mlp_dim):
        super().__init__()

        ###--- Condition Encoder --- ###
        self.pool = AvgPoolFilter2D(16, 16, 0, filter_value=-1)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, embed_dim),
            nn.ReLU()
        )

        ###--- Transformer Architecture ---###
        self.patch_embedding = PatchEmbedding(patch_size, 1, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, (raster_size // patch_size) ** 2)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1)
        )

        ###--- Refining ---###
        self.refine = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, X, c):
        X = self.patch_embedding(X)
        X = self.pos_encoding(X)

        c = self.pool(c)
        c = self.encoder(c)

        B, C, H, W = c.shape
        c = c.reshape(B, C, H*W).permute(0, 2, 1)

        for block in self.transformer_blocks:
            X = block(X, c)

        B, seq_len, embed_dim = X.shape
        N = int(seq_len ** 0.5)
        X = X.permute(0, 2, 1).reshape(B, embed_dim, N, N)
        X = self.upsample(X)

        X = self.refine(X)
        return X

class ConditionTransformer(nn.Module):

    def __init__(self, cross_attention_dim, num_heads):
        super().__init__()

        head_dim = cross_attention_dim // num_heads

        # Formation information transformer
        self.transformer1 = BasicTransformerBlock(
            dim=cross_attention_dim,
            num_attention_heads=num_heads,
            attention_head_dim=head_dim,
            cross_attention_dim=cross_attention_dim,
        )
        # Boreholes transformer
        self.transformer2 = BasicTransformerBlock(
            dim=cross_attention_dim,
            num_attention_heads=num_heads,
            attention_head_dim=head_dim,
            cross_attention_dim=cross_attention_dim,
        )
        # Quaternary transformer
        self.transformer3 = BasicTransformerBlock(
            dim=cross_attention_dim,
            num_attention_heads=num_heads,
            attention_head_dim=head_dim,
            cross_attention_dim=cross_attention_dim,
        )

    def forward(self, X, embedded_info, encoded_boreholes, encoded_quaternary):

        X = self.transformer1(X, encoder_hidden_states=embedded_info)
        X = self.transformer2(X, encoder_hidden_states=encoded_boreholes)
        X = self.transformer3(X, encoder_hidden_states=encoded_quaternary)

        return X