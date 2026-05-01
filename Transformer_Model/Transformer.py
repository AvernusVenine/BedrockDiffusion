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
    def __init__(self, embed_dim, H, W, mod_res, base_res):
        super().__init__()
    
        self.G = (mod_res / base_res)
        self.embed_dim = embed_dim
        
        div_term = 1.0 / (10000.0 ** (torch.arange(0, embed_dim // 2).float() / embed_dim))
    
        self.register_buffer('div_term', div_term)

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

    def forward(self, X, H, W):
        
        row_pos = torch.arange(H, device=X.device, dtype=X.dtype)
        col_pos = torch.arange(W, device=X.device, dtype=X.dtype)
        dt = self.div_term.to(device=X.device, dtype=X.dtype)
        
        row_enc = torch.sin(self.G * row_pos.unsqueeze(1) * dt.unsqueeze(0))
        col_enc = torch.cos(self.G * col_pos.unsqueeze(1) * dt.unsqueeze(0))
        
        row_enc = row_enc.unsqueeze(1).expand(H, W, -1)
        col_enc = col_enc.unsqueeze(0).expand(H, W, -1)
        
        pe = torch.cat([row_enc, col_enc], dim=-1)
        pe = pe.reshape(1, H*W, self.embed_dim)
        
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
        self.norm1   = nn.LayerNorm(embed_dim)
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


class BedrockTransformer(nn.Module):
    def __init__(self, raster_size, patch_size, embed_dim, num_heads, encoder_depth, decoder_depth, mlp_dim):
        super().__init__()

        bedrock_res = 30
        ae_res = 30
        elev_res = 30

        self.embed_dim = embed_dim
        self.raster_size = raster_size
        self.patch_size = patch_size

        ###--- Bedrock Queries ---###
        self.bedrock_query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.bedrock_query_token, std=0.02)

        self.query_pos_encoding = PositionalEncoding(embed_dim, raster_size, raster_size, bedrock_res, bedrock_res)

        ###--- Borehole Embedding ---###
        #- (Existence, Elevation Top, Elevation Bot) -#
        self.borehole_projection = nn.Linear(3, embed_dim)
        self.borehole_pos_encoding = PositionalEncoding(embed_dim, raster_size, raster_size, bedrock_res, bedrock_res)

        ###--- AlphaEarth Embedding ---###
        self.ae_patch_embedding = PatchEmbedding(patch_size, 64, embed_dim)
        self.ae_pos_encoding = PositionalEncoding(embed_dim, raster_size, raster_size, ae_res, bedrock_res)

        ###--- Elevation Embedding ---###
        self.elev_patch_embedding = PatchEmbedding(patch_size, 1, embed_dim)
        self.elev_pos_encoding = PositionalEncoding(embed_dim, raster_size, raster_size, elev_res, bedrock_res)

        ###--- Transformer Architecture ---###
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(encoder_depth)
        ])
        self.decoder_blocks = nn.ModuleList([
            TransformerCrossBlock(embed_dim, num_heads, mlp_dim) for _ in range(decoder_depth)
        ])

        ###--- Bedrock Surface Elevation Head ---###
        self.elev_decoder_blocks = nn.ModuleList([
            TransformerCrossBlock(embed_dim, num_heads, mlp_dim) for _ in range(decoder_depth)
        ])
        self.elev_upsample = nn.Sequential(
            nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False),
            nn.ReplicationPad2d(1),
            nn.Conv2d(embed_dim, 1, kernel_size=3, padding=0)
        )
        self.elev_refine = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(1, 32, kernel_size=5, padding=0),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ReplicationPad2d(2),
            nn.Conv2d(32, 32, kernel_size=5, padding=0),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, 1, kernel_size=3, padding=0)
        )

    def apply_mask(self, tokens, B, mask=None):
        if mask is None:
            return tokens
        return tokens[~mask].reshape(B, -1, self.embed_dim)

    def forward(self, elev, bh, ae):
        B, D = elev.shape[0], self.embed_dim
        
        H = elev.shape[2] // self.patch_size
        W = elev.shape[3] // self.patch_size

        ###--- Embed inputs and apply positional encodings ---###
        elev = self.elev_patch_embedding(elev)
        elev = self.elev_pos_encoding(elev, H, W)

        bh_tokens = self.borehole_projection(bh[:, :, :3])
        bh = bh_tokens + self.borehole_pos_encoding.encode_coords(bh[:, :, 3:])

        ae = self.ae_patch_embedding(ae)
        ae = self.ae_pos_encoding(ae, H, W)

        ###--- Encoder Blocks ---###
        encoder_input = torch.concatenate([elev, ae, bh], dim=1)

        for block in self.encoder_blocks:
            encoder_input = block(encoder_input)

        ###--- Bedrock Elevation Query Tokens ---###
        patch_grid = (self.raster_size // self.patch_size) ** 2
        queries = self.bedrock_query_token.expand(B, patch_grid, self.embed_dim)
        queries = self.query_pos_encoding(queries, H, W)

        ###--- Bedrock Surface Elevation Head ---###
        elev_queries = queries
        for block in self.elev_decoder_blocks:
            elev_queries = block(elev_queries, encoder_input)
        elev_queries = elev_queries.permute(0, 2, 1).reshape(B, self.embed_dim, self.raster_size // self.patch_size, self.raster_size // self.patch_size)

        elev_out = self.elev_upsample(elev_queries)
        elev_out = self.elev_refine(elev_out)

        return elev_out