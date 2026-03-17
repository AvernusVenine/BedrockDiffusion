import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
import pandas as pd
import Data
from Data import BedrockDataset
from ContextEncoder import ContextEncoder
import joblib

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
    """
    Trains the diffusion model
    :param data_path: Data path
    :param save_path: Save path
    :param max_epochs: Maximum number of epochs to train
    :param lr: Initial learning rate
    :return: 
    """
    rasters, elevation = Data.load_rasters(data_path)
    data, scaler = Data.create_data(rasters, elevation, count=1000)

    data[:, :, :, :len(rasters)] = sanitise_input(data[:, :, :, :len(rasters)])

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
    context_encoder = torch.compile(context_encoder)

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
    model = torch.compile(model)

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

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
        for data, context, boreholes, existence in train_loader:
            data = data.permute(0, 3, 1, 2).to(device)
            context = context.permute(0, 3, 1, 2).to(device)
            boreholes = boreholes.permute(0, 3, 1, 2).to(device)
            existence = existence.permute(0, 3, 1, 2).to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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
        loss_dict['train'].append(train_loss / len(train_loader))

        model.eval()
        context_encoder.eval()
        test_loss = 0.0

        with torch.no_grad():

            for data, context, boreholes, existence in test_loader:
                data = data.permute(0, 3, 1, 2).to(device)
                context = context.permute(0, 3, 1, 2).to(device)
                boreholes = boreholes.permute(0, 3, 1, 2).to(device)
                existence = existence.permute(0, 3, 1, 2).to(device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    context_input = torch.cat([context, boreholes, existence], dim=1)
                    encoder_hidden_states = context_encoder(context_input)

                    noise = torch.randn(data.shape, device=device)
                    timesteps = torch.randint(0, 1000, (data.shape[0],), device=device, dtype=torch.long)

                    data_t = scheduler.add_noise(data, noise, timesteps)

                    predicted_noise = model(data_t, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                    loss = F.mse_loss(predicted_noise, noise)

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
                    'model': model._orig_mod.state_dict(),
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
                    'epoch': epoch + 1,
                    'model': model._orig_mod.state_dict(),
                    'context_encoder': context_encoder._orig_mod.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': test_loss,
                    'n_formations': len(rasters),
                },
                f'{save_path}_epoch{epoch + 1:0d}.mdl'
            )

def get_random_sample(data_path):
    rasters, elevation = Data.load_rasters(data_path)
    data, scaler = Data.create_data(rasters, elevation, count=1)

    data[:, :, :, :len(rasters)] = sanitise_input(data[:, :, :, :len(rasters)])

    dataset = BedrockDataset(data[:, :, :, :len(rasters)], data[:, :, :, len(rasters):], scaler)

    return dataset[0]

def generate(data, model_path, scaler_path, save_path, num_steps=100, seed=0):
    elevation = data[1]
    boreholes = data[2]
    existence = data[3]

    model_dict = torch.load(model_path, map_location='cpu')
    scaler = joblib.load(scaler_path)

    #TODO: Swap back to CUDA on higher end hardware
    device = torch.device('cpu')

    state_dict = model_dict['model']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model = UNet2DConditionModel(
        sample_size=200,
        in_channels=model_dict['n_formations'],
        out_channels=model_dict['n_formations'],
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
    model.load_state_dict(state_dict)
    model.eval()

    state_dict = model_dict['context_encoder']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    context_encoder = ContextEncoder(
        in_channels=2*model_dict['n_formations'] + 1,
        cross_attention_dim=512,
        seq_len=64,
    ).to(device)
    context_encoder.load_state_dict(state_dict)
    context_encoder.eval()

    elevation = elevation.permute(2, 0, 1).unsqueeze(0).to(device)
    boreholes = boreholes.permute(2, 0, 1).unsqueeze(0).to(device)
    existence = existence.permute(2, 0, 1).unsqueeze(0).to(device)

    context_input = torch.cat([elevation, boreholes, existence], dim=1)

    with torch.no_grad():
        encoder_hidden_states = context_encoder(context_input)

        scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
        scheduler.set_timesteps(num_steps)

        torch.manual_seed(seed)
        sample = torch.randn((1, model_dict['n_formations'], 200, 200), device=device)

        idx = 0
        for t in scheduler.timesteps:
            t_batch = t.unsqueeze(0).to(device)

            if device.type == 'cuda':
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    predicted_noise = model(sample, t_batch, encoder_hidden_states=encoder_hidden_states).sample
            else:
                predicted_noise = model(sample, t_batch, encoder_hidden_states=encoder_hidden_states).sample

            sample = scheduler.step(predicted_noise, t, sample).prev_sample

            idx += 1
            print(f'Step {idx}')

    generated = sample.squeeze(0).cpu().numpy()

    C, H, W = generated.shape
    generated = scaler.inverse_transform(generated.reshape(-1, 1)).reshape(C, H, W)

    np.save(save_path, generated)

    return generated