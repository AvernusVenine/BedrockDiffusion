from diffusers import DDPMScheduler
import numpy as np
import Data
from Data import BedrockDataset
from construction.Encoder import Encoder
from construction.Embedder import Embedder
from construction.UNet import UNet
from construction.Transformer import ConditionTransformer
from construction.FormationInfo import FORMATION_INFO_DIM
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import pandas as pd
import joblib

def sanitise_input(data, context):

    for point in data:

        # Deletes formations when they are entirely np.nan and thus do not exist in the data chunk
        keys_to_drop = [k for k, v in point.items() if np.all(np.isnan(v))]
        for k in keys_to_drop:
            del point[k]

        for k, v in point.items():
            existence = ~np.isnan(v)
            elevation = np.nan_to_num(v)

            point[k] = np.stack([existence, elevation], axis=0)

    context = [context[idx] for idx in range(len(data)) if data[idx] != {}]
    data = [data[idx] for idx in range(len(data)) if data[idx] != {}]

    return data, context

def train(data_path, save_path, lr=1e-4, max_epochs=100, load_model=False):
    raster_size = 256
    cross_attention_dim = 768
    seq_len = 64
    num_timesteps = 2000
    num_geophysical_channels = 1

    # =============================
    # DATA PREPROCESSING
    # =============================

    rasters, context = Data.load_rasters(data_path, order=['ogcm'])
    data, context, scaler_dict = Data.create_data(rasters, context, count=1500, size=raster_size)

    for k, v in scaler_dict.items():
        joblib.dump(v, f'{save_path}_{k}.scl')

    data, context = sanitise_input(data, context)

    # TODO: Remove after testing.  Need to see if geophysical data is more than just noise for paleozoic
    context = [{k: v for k, v in c.items() if k == 'elevation'} for c in context]

    dataset = BedrockDataset(data, context, scaler_dict, raster_size)

    train_size = int(0.8 * len(dataset))
    test_size = int(len(dataset) - train_size)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # =============================
    # MODEL CONSTRUCTION
    # =============================

    model_dict = None
    if load_model:
        model_dict = torch.load(f'{save_path}.mdl')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    geo_context_encoder = Encoder(in_channels=num_geophysical_channels, cross_attention_dim=cross_attention_dim, seq_len=seq_len).to(device)
    if load_model:
        geo_context_encoder.load_state_dict(model_dict['geo_context_encoder'])
    geo_context_encoder = torch.compile(geo_context_encoder)

    unet = UNet().to(device)
    if load_model:
        unet.load_state_dict(model_dict['unet'])
    unet.enable_gradient_checkpointing()
    unet = torch.compile(unet)

    formation_embedder = Embedder(in_features=FORMATION_INFO_DIM, out_features=cross_attention_dim).to(device)
    if load_model:
        formation_embedder.load_state_dict(model_dict['formation_embedder'])
    formation_embedder = torch.compile(formation_embedder)

    borehole_encoder = Encoder(in_channels=4, cross_attention_dim=cross_attention_dim, seq_len=seq_len).to(device)
    if load_model:
        borehole_encoder.load_state_dict(model_dict['borehole_encoder'])
    borehole_encoder = torch.compile(borehole_encoder)

    quaternary_encoder = Encoder(in_channels=4, cross_attention_dim=cross_attention_dim, seq_len=seq_len).to(device)
    if load_model:
        quaternary_encoder.load_state_dict(model_dict['quaternary_encoder'])
    quaternary_encoder = torch.compile(quaternary_encoder)

    condition_transformer = ConditionTransformer(cross_attention_dim=cross_attention_dim, num_heads=16).to(device)
    if load_model:
        condition_transformer.load_state_dict(model_dict['condition_transformer'])
    condition_transformer = torch.compile(condition_transformer)

    optimizer = torch.optim.AdamW(
        list(geo_context_encoder.parameters()) + list(unet.parameters()) + list(formation_embedder.parameters()) +
        list(borehole_encoder.parameters()) + list(condition_transformer.parameters()) + list(
            quaternary_encoder.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    if load_model:
        optimizer.load_state_dict(model_dict['optimizer'])

    scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)

    scheduler = DDPMScheduler(num_train_timesteps=num_timesteps, beta_schedule='squaredcos_cap_v2')
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    best_loss = np.inf
    if load_model:
        best_loss = model_dict['loss']

    loss_dict = {'train': [], 'test': []}

    for epoch in range(max_epochs):
        print(f'Epoch {epoch+1}')

        # =============================
        # TEST LOOP
        # =============================

        geo_context_encoder.train()
        unet.train()
        formation_embedder.train()
        borehole_encoder.train()
        condition_transformer.train()
        quaternary_encoder.train()

        train_loss = 0.0
        for rasters, existence, context, boreholes, formation_info, quaternary in train_loader:
            # rasters:        (B, F, N, N)
            # existence:      (B, F, N, N)
            # context:        (B, 7, N, N)
            # boreholes:      (B, F, 4, N, N)
            # formation_info: (B, F, M)
            # quaternary:     (B, 4, N, N)

            rasters = rasters.to(device, dtype=torch.bfloat16)
            existence = existence.to(device, dtype=torch.bfloat16)
            context = context.to(device, dtype=torch.bfloat16)
            boreholes = boreholes.to(device, dtype=torch.bfloat16)
            formation_info = formation_info.to(device, dtype=torch.bfloat16)
            quaternary = quaternary.to(device, dtype=torch.bfloat16)

            # Flatten and interleave different formations
            B, C, N, _ = rasters.shape

            rasters = rasters.view(B * C, N, N).unsqueeze(1) # (B * F, N, N)
            existence = existence.view(B * C, N, N)          # (B * F, N, N)
            boreholes = boreholes.view(B * C, 4, N, N)       # (B * F, 4, N, N)
            formation_info = formation_info.view(B * C, -1)  # (B * F, M)

            context = context.repeat_interleave(C, dim=0)       # (B * F, 7, N, N)
            quaternary = quaternary.repeat_interleave(C, dim=0) # (B * F, 4, N, N)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                encoded_context = geo_context_encoder(context)
                embedded_info = formation_embedder(formation_info).unsqueeze(1)
                encoded_boreholes = borehole_encoder(boreholes)
                encoded_quaternary = quaternary_encoder(quaternary)

                encoder_hidden_states = condition_transformer(
                    encoded_context,
                    embedded_info,
                    encoded_boreholes,
                    encoded_quaternary
                )

                B = rasters.shape[0]

                noise = torch.randn(rasters.shape, device=device)
                timesteps = torch.randint(0, 1000, (B,), device=device).long()

                noisy_rasters = scheduler.add_noise(rasters, noise, timesteps)

                predicted_noise, predicted_mask = unet(
                    sample=noisy_rasters,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states
                )

                noise_loss = (existence * F.mse_loss(predicted_noise.squeeze(1), noise.squeeze(1), reduction='none')).mean()

                snr_weights = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                existence_loss = (
                    snr_weights * F.binary_cross_entropy_with_logits(
                        predicted_mask.squeeze(1), existence, reduction='none'
                    )
                ).mean()

                loss = noise_loss + 0.5 * existence_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f'Train Loss: {train_loss / len(train_loader)}')
        loss_dict['train'].append(train_loss / len(train_loader))

        # =============================
        # TEST LOOP
        # =============================

        geo_context_encoder.eval()
        unet.eval()
        formation_embedder.eval()
        borehole_encoder.eval()
        condition_transformer.eval()
        quaternary_encoder.eval()

        test_loss = 0.0

        with torch.no_grad():
            for rasters, existence, context, boreholes, formation_info, quaternary in test_loader:
                # rasters:        (B, F, N, N)
                # existence:      (B, F, N, N)
                # context:        (B, 7, N, N)
                # boreholes:      (B, F, 4, N, N)
                # formation_info: (B, F, M)
                # quaternary:     (B, 4, N, N)

                rasters = rasters.to(device, dtype=torch.bfloat16)
                existence = existence.to(device, dtype=torch.bfloat16)
                context = context.to(device, dtype=torch.bfloat16)
                boreholes = boreholes.to(device, dtype=torch.bfloat16)
                formation_info = formation_info.to(device, dtype=torch.bfloat16)
                quaternary = quaternary.to(device, dtype=torch.bfloat16)

                # Flatten and interleave different formations
                B, C, N, _ = rasters.shape

                rasters = rasters.view(B * C, N, N).unsqueeze(1)  # (B * F, N, N)
                existence = existence.view(B * C, N, N)  # (B * F, N, N)
                boreholes = boreholes.view(B * C, 4, N, N)  # (B * F, 4, N, N)
                formation_info = formation_info.view(B * C, -1)  # (B * F, M)

                context = context.repeat_interleave(C, dim=0)  # (B * F, 7, N, N)
                quaternary = quaternary.repeat_interleave(C, dim=0)  # (B * F, 4, N, N)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    encoded_context = geo_context_encoder(context)
                    embedded_info = formation_embedder(formation_info).unsqueeze(1)
                    encoded_boreholes = borehole_encoder(boreholes)
                    encoded_quaternary = quaternary_encoder(quaternary)

                    encoder_hidden_states = condition_transformer(
                        encoded_context,
                        embedded_info,
                        encoded_boreholes,
                        encoded_quaternary
                    )

                    B = rasters.shape[0]

                    noise = torch.randn(rasters.shape, device=device)
                    timesteps = torch.randint(0, 1000, (B,), device=device).long()

                    noisy_rasters = scheduler.add_noise(rasters, noise, timesteps)

                    predicted_noise, predicted_mask = unet(
                        sample=noisy_rasters,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states
                    )

                    noise_loss = (existence * F.mse_loss(predicted_noise.squeeze(1), noise.squeeze(1), reduction='none')).mean()

                    snr_weights = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                    existence_loss = (
                            snr_weights * F.binary_cross_entropy_with_logits(
                        predicted_mask.squeeze(1), existence, reduction='none'
                    )
                    ).mean()

                    loss = noise_loss + 0.5 * existence_loss

                test_loss += loss.item()

        test_loss = test_loss / len(test_loader)

        print(f'Test Loss: {test_loss}')
        loss_dict['test'].append(test_loss)

        scheduler_lr.step()

        pd.DataFrame(loss_dict).to_csv(f'{save_path}_loss.csv')

        if test_loss < best_loss:
            best_loss = test_loss

            torch.save(
                {
                    'epoch': epoch + 1,
                    'unet': unet._orig_mod.state_dict(),
                    'geo_context_encoder': geo_context_encoder._orig_mod.state_dict(),
                    'formation_embedder': formation_embedder._orig_mod.state_dict(),
                    'borehole_encoder': borehole_encoder._orig_mod.state_dict(),
                    'quaternary_encoder': quaternary_encoder._orig_mod.state_dict(),
                    'condition_transformer': condition_transformer._orig_mod.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': best_loss,
                },
                f'{save_path}.mdl'
            )

        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    'epoch': epoch + 1,
                    'unet': unet._orig_mod.state_dict(),
                    'geo_context_encoder': geo_context_encoder._orig_mod.state_dict(),
                    'formation_embedder': formation_embedder._orig_mod.state_dict(),
                    'borehole_encoder': borehole_encoder._orig_mod.state_dict(),
                    'quaternary_encoder': quaternary_encoder._orig_mod.state_dict(),
                    'condition_transformer': condition_transformer._orig_mod.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': test_loss,
                },
                f'{save_path}_epoch{epoch + 1:04d}.mdl'
            )