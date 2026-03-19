import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
import pandas as pd
import Data
from Data import BedrockDataset
from ContextEncoder import ContextEncoder

class ExpandedUNet(nn.Module):
    def __init__(self, unet, n_formations):
        super().__init__()
        self.unet = unet
        self.n_formations = n_formations
        self._bottleneck = None

        #Hook to capture the output of the midblock from unet
        self.unet.mid_block.register_forward_hook(self._hook)

        self.existence_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, n_formations, kernel_size=4, stride=2, padding=1)
        )

    def _hook(self, module, input, output):
        self._bottleneck = output

    def forward(self, sample, timesteps, encoder_hidden_states):
        noise_pred = self.unet(
            sample,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        existence_logits = self.existence_decoder(self._bottleneck)

        return noise_pred, existence_logits


def sanitise_input(data):
    """
    Replaces all nan elevation values with the highest elevation value in at a given location, effectively setting
    the formation thickness to 0 meaning it does not exist
    :param data: Data tensor
    :return: Sanitised data tensor
             Existence tensor
    """
    data = data.numpy()

    existence = (~np.isnan(data)).astype(np.float32)

    data = np.where(np.isnan(data), 0.0, data)

    return torch.from_numpy(data), torch.from_numpy(existence)

def train_model(data_path, save_path, max_epochs=15, lr=1e-3):
    """
    Trains the diffusion model
    :param data_path: Data path
    :param save_path: Save path
    :param max_epochs: Maximum number of epochs to train
    :param lr: Initial learning rate
    :return: 
    """
    rasters, elevation = Data.load_rasters(data_path)
    data, scaler = Data.create_data(rasters, elevation, count=1000) # (B x N x N x C)

    data[:, :, :, :len(rasters)], existence = sanitise_input(data[:, :, :, :len(rasters)])

    dataset = BedrockDataset(data[:, :, :, :len(rasters)], data[:, :, :, len(rasters):], existence, scaler)

    train_size = int(0.8 * len(dataset))
    test_size = int(len(dataset) - train_size)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    context_encoder = ContextEncoder(
        in_channels=2*len(rasters) + 1,
        cross_attention_dim=512,
        seq_len=64,
    ).to(device)
    context_encoder = torch.compile(context_encoder)

    unet = UNet2DConditionModel(
        sample_size=200,
        in_channels=len(rasters),
        out_channels=len(rasters),
        cross_attention_dim=512,
        down_block_types=(
            'CrossAttnDownBlock2D',
            'CrossAttnDownBlock2D',
            'DownBlock2D',
        ),
        up_block_types=(
            'UpBlock2D',
            'CrossAttnUpBlock2D',
            'CrossAttnUpBlock2D',
        ),
        block_out_channels=(128, 256, 512),
        layers_per_block=2,
        attention_head_dim=8,
        norm_num_groups=32,
    ).to(device)
    unet.enable_gradient_checkpointing()

    model = ExpandedUNet(unet, n_formations=len(rasters)).to(device)
    model = torch.compile(model)

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(context_encoder.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    best_loss = np.inf

    loss_dict = {'train': [], 'test': []}

    for epoch in range(max_epochs):
        print(f'Epoch {epoch+1}')

        model.train()
        context_encoder.train()

        train_loss = 0.0
        for data, context, existence, boreholes, bh_existence in train_loader:
            data = data.permute(0, 3, 1, 2).to(device)
            context = context.permute(0, 3, 1, 2).to(device)
            existence = existence.permute(0, 3, 1, 2).to(device)
            boreholes = boreholes.permute(0, 3, 1, 2).to(device)
            bh_existence = bh_existence.permute(0, 3, 1, 2).to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                context_input = torch.cat([context, boreholes, bh_existence], dim=1)
                encoder_hidden_states = context_encoder(context_input)

                noise = torch.randn(data.shape, device=device)
                timesteps = torch.randint(0, 1000, (data.shape[0], ), device=device, dtype=torch.long)

                data_t = scheduler.add_noise(data, noise, timesteps)

                predicted_noise, existence_logits = model(data_t, timesteps, encoder_hidden_states=encoder_hidden_states)

                noise_loss = (existence * F.mse_loss(predicted_noise, noise, reduction='none')).mean()

                snr_weights = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                existence_loss = (
                    snr_weights * F.binary_cross_entropy_with_logits(
                        existence_logits, existence, reduction='none'
                    )
                ).mean()

                loss = noise_loss + 0.1 * existence_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f'Train Loss: {train_loss / len(train_loader)}')
        loss_dict['train'].append(train_loss / len(train_loader))

        model.eval()
        context_encoder.eval()
        test_loss = 0.0

        with torch.no_grad():

            for data, context, existence, boreholes, bh_existence in test_loader:
                data = data.permute(0, 3, 1, 2).to(device)
                context = context.permute(0, 3, 1, 2).to(device)
                existence = existence.permute(0, 3, 1, 2).to(device)
                boreholes = boreholes.permute(0, 3, 1, 2).to(device)
                bh_existence = bh_existence.permute(0, 3, 1, 2).to(device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    context_input = torch.cat([context, boreholes, bh_existence], dim=1)
                    encoder_hidden_states = context_encoder(context_input)

                    noise = torch.randn(data.shape, device=device)
                    timesteps = torch.randint(0, 1000, (data.shape[0],), device=device, dtype=torch.long)

                    data_t = scheduler.add_noise(data, noise, timesteps)

                    predicted_noise, existence_logits = model(data_t, timesteps, encoder_hidden_states=encoder_hidden_states)

                    noise_loss = (existence * F.mse_loss(predicted_noise, noise, reduction='none')).mean()

                    snr_weights = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                    existence_loss = (
                            snr_weights * F.binary_cross_entropy_with_logits(
                        existence_logits, existence, reduction='none'
                    )
                    ).mean()

                    loss = noise_loss + 0.1 * existence_loss

                test_loss += loss.item()

        test_loss = test_loss / len(test_loader)

        print(f'Test Loss: {test_loss}')
        loss_dict['test'].append(test_loss)

        pd.DataFrame(loss_dict).to_csv(f'{save_path}_loss.csv')

        if test_loss < best_loss:
            best_loss = test_loss

            torch.save(
                {
                    'epoch': epoch+1,
                    'model': model._orig_mod.unet.state_dict(),
                    'existence_decoder': model._orig_mod.existence_decoder.state_dict(),
                    'context_encoder': context_encoder._orig_mod.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': best_loss,
                    'n_formations': len(rasters),
                },
                f'{save_path}.mdl'
            )

        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    'epoch': epoch+1,
                    'model': model._orig_mod.unet.state_dict(),
                    'existence_decoder': model._orig_mod.existence_decoder.state_dict(),
                    'context_encoder': context_encoder._orig_mod.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': test_loss,
                    'n_formations': len(rasters),
                },
                f'{save_path}_epoch{epoch + 1:04d}.mdl'
            )