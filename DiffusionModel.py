import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import random_split, DataLoader
from diffusers import UNet2DConditionModel, DDPMScheduler
import Data
from Data import BedrockDataset
from ContextEncoder import ContextEncoder

def sanitise_input(data):
    """
    Replaces all nan elevation values with the highest elevation value in at a given location, effectively setting
    the formation thickness to 0 meaning it does not exist
    :param data: Data tensor
    :return: Sanitised data tensor
    """
    data = data.numpy()

    elevation_max = np.nanmax(data, axis=-1, keepdims=True)
    data = np.where(np.isnan(data), elevation_max, data)

    return torch.from_numpy(data)

def train_model(data_path, save_path, max_epochs=15, lr=1e-3):
    rasters, elevation = Data.load_rasters(data_path)
    data, scaler = Data.create_data(rasters, elevation, count=1000)

    data = sanitise_input(data)

    dataset = BedrockDataset(data[:, :, :, :len(rasters)], data[:, :, :, len(rasters):], scaler)

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

    model = UNet2DConditionModel(
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
    model.enable_gradient_checkpointing()

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(context_encoder.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    best_loss = np.inf

    for epoch in range(max_epochs):
        print(f'Epoch {epoch+1}')

        model.train()
        context_encoder.train()

        train_loss = 0.0
        for data, context, boreholes, existence in train_loader:
            data = data.permute(0, 3, 1, 2).to(device)
            context = context.permute(0, 3, 1, 2).to(device)
            boreholes = boreholes.permute(0, 3, 1, 2).to(device)
            existence = existence.permute(0, 3, 1, 2).to(device)

            context_input = torch.cat([context, boreholes, existence], dim=1)
            encoder_hidden_states = context_encoder(context_input)

            noise = torch.randn(data.shape, device=device)
            timesteps = torch.randint(0, 1000, (data.shape[0], ), device=device, dtype=torch.long)

            data_t = scheduler.add_noise(data, noise, timesteps)

            predicted_noise = model(data_t, timesteps, encoder_hidden_states=encoder_hidden_states).sample

            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f'Train Loss: {train_loss / len(train_loader)}')

        model.eval()
        context_encoder.eval()
        test_loss = 0.0

        with torch.no_grad():

            for data, context, boreholes, existence in test_loader:
                data = data.permute(0, 3, 1, 2).to(device)
                context = context.permute(0, 3, 1, 2).to(device)
                boreholes = boreholes.permute(0, 3, 1, 2).to(device)
                existence = existence.permute(0, 3, 1, 2).to(device)

                context_input = torch.cat([context, boreholes, existence], dim=1)
                encoder_hidden_states = context_encoder(context_input)

                noise = torch.randn(data.shape, device=device)
                timesteps = torch.randint(0, 1000, (data.shape[0],), device=device, dtype=torch.long)

                data_t = scheduler.add_noise(data, noise, timesteps)

                predicted_noise = model(data_t, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                test_loss += F.mse_loss(predicted_noise, noise)

        print(f'Test Loss: {test_loss / len(test_loader)}')

        if test_loss < best_loss:
            best_loss = test_loss

            torch.save(
                {
                    'epoch': epoch+1,
                    'model': model.state_dict(),
                    'context_encoder': context_encoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': best_loss,
                    'n_formations': len(rasters),
                },
                save_path
            )