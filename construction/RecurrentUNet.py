import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from Encoder import Encoder
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    UNetMidBlock2DCrossAttn,
)
from diffusers.models.embeddings import Timesteps, TimestepEmbedding

class SpatialGRU(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=channels,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(
            in_features=channels * 2,
            out_features=channels,
            bias=False,
        )

    def forward(self, X, h):
        B, C, H, W = X.shape
        dtype = X.dtype

        seq = X.permute(0, 2, 3, 1).reshape(B, H * W, C)

        #bfloat16 does not work with GRU for some reason...
        seq = seq.float()
        h = h.float()

        out, h_new = self.gru(seq, h)

        out = out.to(dtype)
        h_new = h_new.to(dtype)

        out = self.proj(out)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return out, h_new

class RecurrentUNet(nn.Module):

    C0, C1, C2, C3 = 128, 256, 512, 512
    TIME_EMBEDDING_DIM = 512
    CROSS_ATTENTION_DIM = 768
    ATTENTION_HEADS = 8

    def __init__(self):
        super().__init__()

        # Time Embedding
        self.time_proj = Timesteps(
            self.C0,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=self.C0,
            time_embed_dim=self.TIME_EMBEDDING_DIM,
            act_fn='silu',
        )

        #=============================
        # SHARED ENCODER
        #=============================

        # Initial
        self.pre_down = nn.Sequential(
            nn.Conv2d(1, self.C0, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.C0, self.C0, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, self.C0),
            nn.SiLU(),
        )

        # Down Block 0
        self.down0 = CrossAttnDownBlock2D(
            in_channels=self.C0,
            out_channels=self.C1,
            temb_channels=self.TIME_EMBEDDING_DIM,
            num_layers=2,
            cross_attention_dim=self.CROSS_ATTENTION_DIM,
            num_attention_heads=self.ATTENTION_HEADS,
            add_downsample=True,
        )
        self.gru0 = SpatialGRU(self.C1)

        # Down Block 1
        self.down1 = CrossAttnDownBlock2D(
            in_channels=self.C1,
            out_channels=self.C2,
            temb_channels=self.TIME_EMBEDDING_DIM,
            num_layers=2,
            cross_attention_dim=self.CROSS_ATTENTION_DIM,
            num_attention_heads=self.ATTENTION_HEADS,
            add_downsample=True
        )
        self.gru1 = SpatialGRU(self.C2)

        # Down Block 2
        self.down2 = CrossAttnDownBlock2D(
            in_channels=self.C2,
            out_channels=self.C3,
            temb_channels=self.TIME_EMBEDDING_DIM,
            num_layers=2,
            cross_attention_dim=self.CROSS_ATTENTION_DIM,
            num_attention_heads=self.ATTENTION_HEADS,
            add_downsample=False
        )
        self.gru2 = SpatialGRU(self.C3)

        # Mid Block
        self.mid = UNetMidBlock2DCrossAttn(
            in_channels=self.C3,
            temb_channels=self.TIME_EMBEDDING_DIM,
            cross_attention_dim=self.CROSS_ATTENTION_DIM,
            num_attention_heads=self.ATTENTION_HEADS
        )
        self.gru3 = SpatialGRU(self.C3)

        #=============================
        # NOISE DECODER
        #=============================

        # Up 0
        self.noise_up0 = CrossAttnUpBlock2D(
            in_channels=self.C3,
            out_channels=self.C3,
            prev_output_channel=self.C3,
            temb_channels=self.TIME_EMBEDDING_DIM,
            num_layers=2,
            cross_attention_dim=self.CROSS_ATTENTION_DIM,
            num_attention_heads=self.ATTENTION_HEADS,
            add_upsample=True
        )
        self.noise_gru4 = SpatialGRU(self.C3)

        # Up 1
        self.noise_up1 = CrossAttnUpBlock2D(
            in_channels=self.C2,
            out_channels=self.C1,
            prev_output_channel=self.C3,
            temb_channels=self.TIME_EMBEDDING_DIM,
            num_layers=3,
            cross_attention_dim=self.CROSS_ATTENTION_DIM,
            num_attention_heads=self.ATTENTION_HEADS,
            add_upsample=True
        )
        self.noise_gru5 = SpatialGRU(self.C1)

        # Up 2
        self.noise_up2 = CrossAttnUpBlock2D(
            in_channels=self.C1,
            out_channels=self.C0,
            prev_output_channel=self.C1,
            temb_channels=self.TIME_EMBEDDING_DIM,
            num_layers=3,
            cross_attention_dim=self.CROSS_ATTENTION_DIM,
            num_attention_heads=self.ATTENTION_HEADS,
            add_upsample=True
        )
        self.noise_gru6 = SpatialGRU(self.C0)

        # Output
        self.noise_out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.C0, self.C0, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.C0),
            nn.SiLU(),
            nn.Conv2d(self.C0, 1, kernel_size=3, padding=1)
        )

        #=============================
        # MASK DECODER
        #=============================

        # Up 0
        self.mask_up0 = CrossAttnUpBlock2D(
            in_channels=self.C3,
            out_channels=self.C3,
            prev_output_channel=self.C3,
            temb_channels=self.TIME_EMBEDDING_DIM,
            num_layers=2,
            cross_attention_dim=self.CROSS_ATTENTION_DIM,
            num_attention_heads=self.ATTENTION_HEADS,
            add_upsample=True
        )
        self.mask_gru4 = SpatialGRU(self.C3)

        # Up 1
        self.mask_up1 = CrossAttnUpBlock2D(
            in_channels=self.C2,
            out_channels=self.C1,
            prev_output_channel=self.C3,
            temb_channels=self.TIME_EMBEDDING_DIM,
            num_layers=3,
            cross_attention_dim=self.CROSS_ATTENTION_DIM,
            num_attention_heads=self.ATTENTION_HEADS,
            add_upsample=True
        )
        self.mask_gru5 = SpatialGRU(self.C1)

        # Up 2
        self.mask_up2 = CrossAttnUpBlock2D(
            in_channels=self.C1,
            out_channels=self.C0,
            prev_output_channel=self.C1,
            temb_channels=self.TIME_EMBEDDING_DIM,
            num_layers=3,
            cross_attention_dim=self.CROSS_ATTENTION_DIM,
            num_attention_heads=self.ATTENTION_HEADS,
            add_upsample=True
        )
        self.mask_gru6 = SpatialGRU(self.C0)

        # Output
        self.mask_out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.C0, self.C0, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.C0),
            nn.SiLU(),
            nn.Conv2d(self.C0, 1, kernel_size=3, padding=1)
        )

    def init_gru_states(self, batch_size, device, dtype):
        return {
            'gru0': torch.zeros(2, batch_size, self.C1, device=device, dtype=dtype),
            'gru1': torch.zeros(2, batch_size, self.C2, device=device, dtype=dtype),
            'gru2': torch.zeros(2, batch_size, self.C3, device=device, dtype=dtype),
            'gru3': torch.zeros(2, batch_size, self.C3, device=device, dtype=dtype),
            'noise_gru4': torch.zeros(2, batch_size, self.C3, device=device, dtype=dtype),
            'noise_gru5': torch.zeros(2, batch_size, self.C1, device=device, dtype=dtype),
            'noise_gru6': torch.zeros(2, batch_size, self.C0, device=device, dtype=dtype),
            'mask_gru4': torch.zeros(2, batch_size, self.C3, device=device, dtype=dtype),
            'mask_gru5': torch.zeros(2, batch_size, self.C1, device=device, dtype=dtype),
            'mask_gru6': torch.zeros(2, batch_size, self.C0, device=device, dtype=dtype),
        }

    def enable_gradient_checkpointing(self):
        for module in [
            self.down0, self.down1, self.down2,
            self.mid,
            self.noise_up0, self.noise_up1, self.noise_up2,
            self.mask_up0, self.mask_up1, self.mask_up2,
        ]:
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True

    def forward(self, sample, timestep, encoder_hidden_states, gru_states = None):
        # Time embedding
        emb = self.time_proj(timestep).to(dtype=sample.dtype)
        emb = self.time_embedding(emb)

        #=============================
        # SHARED ENCODER
        #=============================

        h = self.pre_down(sample)
        h, res0 = self.down0(h, emb, encoder_hidden_states=encoder_hidden_states)
        h, gru_states['gru0'] = self.gru0(h, gru_states.get('gru0'))

        h, res1 = self.down1(h, emb, encoder_hidden_states=encoder_hidden_states)
        h, gru_states['gru1'] = self.gru1(h, gru_states.get('gru1'))

        h, res2 = self.down2(h, emb, encoder_hidden_states=encoder_hidden_states)
        h, gru_states['gru2'] = self.gru2(h, gru_states.get('gru2'))

        h = self.mid(h, emb, encoder_hidden_states=encoder_hidden_states)
        h, gru_states['gru3'] = self.gru3(h, gru_states.get('gru3'))

        #=============================
        # NOISE DECODER
        #=============================

        n = h
        n = self.noise_up0(n, res2, temb=emb, encoder_hidden_states=encoder_hidden_states)
        n, gru_states['noise_gru4'] = self.noise_gru4(n, gru_states.get('noise_gru4'))

        n = self.noise_up1(n, res1, temb=emb, encoder_hidden_states=encoder_hidden_states)
        n, gru_states['noise_gru5'] = self.noise_gru5(n, gru_states.get('noise_gru5'))

        n = self.noise_up2(n, res0, temb=emb, encoder_hidden_states=encoder_hidden_states)
        n, gru_states['noise_gru6'] = self.noise_gru6(n, gru_states.get('noise_gru6'))

        predicted_noise = self.noise_out(n)

        #=============================
        # MASK DECODER
        #=============================

        m = h
        m = self.mask_up0(m, res2, temb=emb, encoder_hidden_states=encoder_hidden_states)
        m, gru_states['mask_gru4'] = self.mask_gru4(m, gru_states.get('mask_gru4'))

        m = self.mask_up1(m, res1, temb=emb, encoder_hidden_states=encoder_hidden_states)
        m, gru_states['mask_gru5'] = self.mask_gru5(m, gru_states.get('mask_gru5'))

        m = self.mask_up2(m, res0, temb=emb, encoder_hidden_states=encoder_hidden_states)
        m, gru_states['mask_gru6'] = self.mask_gru6(m, gru_states.get('mask_gru6'))

        predicted_mask = self.mask_out(m)

        return predicted_noise, predicted_mask, gru_states
