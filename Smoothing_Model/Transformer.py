import torch
import torch.nn as nn

class CNNPatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        #- Feature Extraction -#
        self.conv1 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )

        #- Splits into patches and projects -#
        self.proj = nn.Conv2d(128, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)

        ###--- Residual Skip ---###
        self.residual = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=1),
            nn.BatchNorm2d(128),
        )

    def forward(self, X):
        residual = self.residual(X)

        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)

        X = X + residual

        X = self.proj(X).flatten(2).transpose(1, 2)
        return X

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

        self.embed_dim = embed_dim
        self.patch_size = patch_size

        ###--- Bedrock Embedding ---###
        self.patch_embedding = CNNPatchEmbedding(patch_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, 1, 1)

        ###--- Elevation Embedding ---###
        self.elev_patch_embedding = CNNPatchEmbedding(patch_size, embed_dim)
        self.elev_pos_encoding = PositionalEncoding(embed_dim, 1, 1)

        ###--- Transformer Architecture ---###
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])

        ###--- Elevation Skip Connection ---###
        self.elev_skip = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, 64, kernel_size=3, padding=0)
        )

        ###--- Bedrock Upsampling Head ---###
        self.upsample = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(embed_dim, 128 * 16, kernel_size=3, padding=0),
            nn.PixelShuffle(4),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(128, 64 * 16, kernel_size=3, padding=0),
            nn.PixelShuffle(4),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(64, 32 * 16, kernel_size=3, padding=0),
            nn.PixelShuffle(4),
            nn.GroupNorm(8, 32),
            nn.SiLU()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

        #- Initialize upscale weights to avoid checkerboard patterns -#
        for layer in self.upsample:
            if isinstance(layer, nn.Conv2d):
                weight = layer.weight.data
                out_channels, in_channels, H, W = weight.shape
                sub = torch.zeros(out_channels // 16, in_channels, H, W)
                nn.init.kaiming_normal_(sub)
                weight = sub.repeat_interleave(16, dim=0)
                layer.weight.data.copy_(weight)

    def forward(self, elev, terra, mask=None):
        B, D = elev.shape[0], self.embed_dim

        H = elev.shape[2] // self.patch_size
        W = elev.shape[3] // self.patch_size

        ###--- Elevation skip connection ---###
        e_skip = self.elev_skip(elev)

        ###--- Embed inputs and apply positional encodings ---###
        terra = self.patch_embedding(terra)
        terra = self.pos_encoding(terra, H, W)

        if mask is not None:
            terra = terra[~mask].reshape(B, -1, D)

        elev = self.elev_patch_embedding(elev)
        elev = self.elev_pos_encoding(elev, H, W)

        n_elev = elev.shape[1]
        encoder_input = torch.concatenate([elev, terra], dim=1)

        for block in self.encoder_blocks:
            encoder_input = block(encoder_input)

        #- Extract encoded elevation and apply skip connection -#
        elev = encoder_input[:, :n_elev, :]
        elev = elev.permute(0, 2, 1).reshape(B, self.embed_dim, H, W)
        elev = self.upsample(elev)

        elev = torch.concatenate([elev, e_skip], dim=1)
        elev = self.refine(elev)

        return elev