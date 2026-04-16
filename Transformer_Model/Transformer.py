import math
import torch
import torch.nn as nn

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
    def __init__(self, embed_dim, H, W, mod_res, base_res):
        super().__init__()

        G = (mod_res / base_res)

        div_term = 1.0 / (10000.0 ** (torch.arange(0, embed_dim // 2).float() / embed_dim))

        row_pos = torch.arange(H).float()
        col_pos = torch.arange(W).float()

        row_enc = torch.sin(G * row_pos.unsqueeze(1) * div_term.unsqueeze(0))
        col_enc = torch.cos(G * col_pos.unsqueeze(1) * div_term.unsqueeze(0))

        pe = torch.zeros(H, W, embed_dim)
        pe[:, 0::2] = row_enc
        pe[:, 1::2] = col_enc

        self.register_buffer('pe', pe.reshape(1, H*W, embed_dim))
        self.register_buffer('div_term', div_term)
        self.G = G
        self.embed_dim = embed_dim

    def encode_coords(self, coords):
        B, K, _ = coords.shape

        row = coords[:, :, 0].float()
        col = coords[:, :, 1].float()
        dt = self.div_term

        row_enc = torch.sin(self.G * row.unsqueeze(-1) * dt)
        col_enc = torch.cos(self.G * col.unsqueeze(-1) * dt)

        out = torch.zeros(B, K, self.embed_dim, device=dt.device, dtype=dt.dtype)
        out[:, :, 0::2] = row_enc
        out[:, :, 1::2] = col_enc

        return out

    def forward(self, X):
        X = X + self.pe[:, :X.size(1)]
        return X

class TransformerCrossBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.norm1   = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, X, ctx):
        ctx = self.norm1(ctx)

        X = X + self.cross_attn(self.norm2(X), ctx, ctx)[0]
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
        X_norm = self.norm1(X)
        X = X + self.attn(X_norm, X_norm, X_norm)[0]
        X = X + self.mlp(self.norm2(X))
        return X


class BedrockTransformer(nn.Module):
    def __init__(self, raster_size, patch_size, embed_dim, num_heads, encoder_depth, decoder_depth, mlp_dim):
        super().__init__()

        self.embed_dim = embed_dim
        self.H = int(raster_size // patch_size)
        self.W = int(raster_size // patch_size)

        ###--- Bedrock Queries ---###
        self.bedrock_query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.bedrock_query_token, std=0.02)

        self.query_pos_encoding = PositionalEncoding(embed_dim, self.H, self.W, patch_size, patch_size)

        ###--- Borehole Embedding ---###
        #- (Existence, Elevation Top, Elevation Bot) -#
        self.borehole_projection = nn.Linear(3, embed_dim)
        self.borehole_pos_encoding = PositionalEncoding(embed_dim, raster_size, raster_size, 1, patch_size)

        ###--- AlphaEarth Embedding ---###
        self.ae_patch_embedding = PatchEmbedding(patch_size, 64, embed_dim)
        self.ae_pos_encoding = PositionalEncoding(embed_dim, self.H, self.W, patch_size, patch_size)

        ###--- Elevation Embedding ---###
        self.elev_patch_embedding = PatchEmbedding(patch_size, 1, embed_dim)
        self.elev_pos_encoding = PositionalEncoding(embed_dim, self.H, self.W, patch_size, patch_size)

        ###--- Transformer Architecture ---###
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(encoder_depth)
        ])
        self.decoder_blocks = nn.ModuleList([
            TransformerCrossBlock(embed_dim, num_heads, mlp_dim) for _ in range(decoder_depth)
        ])

        ###--- Bedrock Surface Elevation Head ---###
        self.elev_decoder = TransformerCrossBlock(embed_dim, num_heads, mlp_dim)
        self.elev_upsample = nn.Sequential(
            nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1)
        )
        self.elev_refine = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

        ###--- Bedrock Existence Head ---###
        self.exist_decoder = TransformerCrossBlock(embed_dim, num_heads, mlp_dim)
        self.exist_upsample = nn.Sequential(
            nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1)
        )
        self.exist_refine = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def apply_mask(self, tokens, B, mask=None):
        if mask is None:
            return tokens
        return tokens[~mask].reshape(B, -1, self.embed_dim)

    def forward(self, elev, bh, ae, elev_mask=None, ae_mask=None):
        B, D = elev.shape[0], self.embed_dim

        ###--- Embed inputs and apply positional encodings ---###
        elev = self.elev_patch_embedding(elev)
        elev = self.elev_pos_encoding(elev)

        bh_tokens = self.borehole_projection(bh[:, :, :3])
        bh = bh_tokens + self.borehole_pos_encoding.encode_coords(bh[:, :, 3:])

        ae = self.ae_patch_embedding(ae)
        ae = self.ae_pos_encoding(ae)

        ###--- Apply masks if in pretraining ---###
        elev = self.apply_mask(elev, B, elev_mask)
        ae = self.apply_mask(ae, B, ae_mask)

        ###--- Encoder Blocks ---###
        encoder_input = torch.concatenate([elev, ae, bh], dim=1)

        for block in self.encoder_blocks:
            encoder_input = block(encoder_input)

        ###--- Bedrock Elevation Query Tokens ---###
        queries = self.bedrock_query_token.expand(B, self.H * self.W, self.embed_dim)
        queries = self.query_pos_encoding(queries)

        ###--- Decoder Blocks ---###
        for block in self.decoder_blocks:
            queries = block(queries, encoder_input)

        ###--- Bedrock Surface Elevation Head ---###
        elev_queries = self.elev_decoder(queries, encoder_input)
        elev_queries = elev_queries.permute(0, 2, 1).reshape(B, self.embed_dim, self.H, self.W)

        elev_out = self.elev_upsample(elev_queries)
        elev_out = self.elev_refine(elev_out)

        ###--- Existence Head ---###
        exist_queries = self.exist_decoder(queries, encoder_input)
        exist_queries = exist_queries.permute(0, 2, 1).reshape(B, self.embed_dim, self.H, self.W)

        exist_out = self.exist_upsample(exist_queries)
        exist_out = self.exist_refine(exist_out)

        return elev_out, exist_out