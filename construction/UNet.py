import torch.nn as nn
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    UNetMidBlock2DCrossAttn,
)
from diffusers.models.embeddings import Timesteps, TimestepEmbedding


class UNet(nn.Module):

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

        # Down Block 2 (no downsample)
        self.down2 = CrossAttnDownBlock2D(
            in_channels=self.C2,
            out_channels=self.C3,
            temb_channels=self.TIME_EMBEDDING_DIM,
            num_layers=2,
            cross_attention_dim=self.CROSS_ATTENTION_DIM,
            num_attention_heads=self.ATTENTION_HEADS,
            add_downsample=False
        )

        # Mid Block
        self.mid = UNetMidBlock2DCrossAttn(
            in_channels=self.C3,
            temb_channels=self.TIME_EMBEDDING_DIM,
            cross_attention_dim=self.CROSS_ATTENTION_DIM,
            num_attention_heads=self.ATTENTION_HEADS
        )

        # =============================
        # NOISE DECODER
        # =============================

        # Up 0
        self.noise_up0 = CrossAttnUpBlock2D(
            in_channels=self.C2,
            out_channels=self.C3,
            prev_output_channel=self.C3,
            temb_channels=self.TIME_EMBEDDING_DIM,
            num_layers=3,
            cross_attention_dim=self.CROSS_ATTENTION_DIM,
            num_attention_heads=self.ATTENTION_HEADS,
            add_upsample=True
        )

        # Up 1
        self.noise_up1 = CrossAttnUpBlock2D(
            in_channels=self.C1,
            out_channels=self.C2,
            prev_output_channel=self.C3,
            temb_channels=self.TIME_EMBEDDING_DIM,
            num_layers=3,
            cross_attention_dim=self.CROSS_ATTENTION_DIM,
            num_attention_heads=self.ATTENTION_HEADS,
            add_upsample=True
        )

        # Up 2
        self.noise_up2 = CrossAttnUpBlock2D(
            in_channels=self.C0,
            out_channels=self.C1,
            prev_output_channel=self.C2,
            temb_channels=self.TIME_EMBEDDING_DIM,
            num_layers=3,
            cross_attention_dim=self.CROSS_ATTENTION_DIM,
            num_attention_heads=self.ATTENTION_HEADS,
            add_upsample=False
        )

        # Output
        self.noise_out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.C1, self.C1, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.C1),
            nn.SiLU(),
            nn.Conv2d(self.C1, 1, kernel_size=3, padding=1)
        )

        #=============================
        # MASK DECODER
        #=============================

        # Up 0
        self.mask_up0 = CrossAttnUpBlock2D(
            in_channels=self.C2,
            out_channels=self.C3,
            prev_output_channel=self.C3,
            temb_channels=self.TIME_EMBEDDING_DIM,
            num_layers=3,
            cross_attention_dim=self.CROSS_ATTENTION_DIM,
            num_attention_heads=self.ATTENTION_HEADS,
            add_upsample=True
        )

        # Up 1
        self.mask_up1 = CrossAttnUpBlock2D(
            in_channels=self.C1,
            out_channels=self.C2,
            prev_output_channel=self.C3,
            temb_channels=self.TIME_EMBEDDING_DIM,
            num_layers=3,
            cross_attention_dim=self.CROSS_ATTENTION_DIM,
            num_attention_heads=self.ATTENTION_HEADS,
            add_upsample=True
        )

        # Up 2
        self.mask_up2 = CrossAttnUpBlock2D(
            in_channels=self.C0,
            out_channels=self.C1,
            prev_output_channel=self.C2,
            temb_channels=self.TIME_EMBEDDING_DIM,
            num_layers=3,
            cross_attention_dim=self.CROSS_ATTENTION_DIM,
            num_attention_heads=self.ATTENTION_HEADS,
            add_upsample=False
        )

        # Output
        self.mask_out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.C1, self.C1, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.C1),
            nn.SiLU(),
            nn.Conv2d(self.C1, 1, kernel_size=3, padding=1)
        )

    def enable_gradient_checkpointing(self):
        for module in [
            self.down0, self.down1, self.down2,
            self.mid,
            self.noise_up0, self.noise_up1, self.noise_up2,
            self.mask_up0, self.mask_up1, self.mask_up2,
        ]:
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True

    def forward(self, sample, timestep, encoder_hidden_states):
        # Time embedding
        emb = self.time_proj(timestep).to(dtype=sample.dtype)
        emb = self.time_embedding(emb)

        #=============================
        # SHARED ENCODER
        #=============================

        h = self.pre_down(sample)
        h0 = h

        # Down blocks
        h, res0 = self.down0(h, emb, encoder_hidden_states=encoder_hidden_states)
        h, res1 = self.down1(h, emb, encoder_hidden_states=encoder_hidden_states)
        h, res2 = self.down2(h, emb, encoder_hidden_states=encoder_hidden_states)

        # Mid block
        h = self.mid(h, emb, encoder_hidden_states=encoder_hidden_states)

        all_res = (h0,) + res0 + res1 + res2

        up0_res = all_res[6:]
        up1_res = all_res[3:6]
        up2_res = all_res[0:3]

        #=============================
        # NOISE DECODER
        #=============================

        n = h

        # Up blocks
        n = self.noise_up0(n, up0_res, temb=emb, encoder_hidden_states=encoder_hidden_states)
        n = self.noise_up1(n, up1_res, temb=emb, encoder_hidden_states=encoder_hidden_states)
        n = self.noise_up2(n, up2_res, temb=emb, encoder_hidden_states=encoder_hidden_states)

        predicted_noise = self.noise_out(n)

        #=============================
        # MASK DECODER
        #=============================

        m = h

        # Up blocks
        m = self.mask_up0(m, up0_res, temb=emb, encoder_hidden_states=encoder_hidden_states)
        m = self.mask_up1(m, up1_res, temb=emb, encoder_hidden_states=encoder_hidden_states)
        m = self.mask_up2(m, up2_res, temb=emb, encoder_hidden_states=encoder_hidden_states)

        predicted_mask = self.mask_out(m)

        return predicted_noise, predicted_mask