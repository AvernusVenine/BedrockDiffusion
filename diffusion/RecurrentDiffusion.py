from diffusers import DDPMScheduler
import numpy as np
import Data
from Data import BedrockDataset, collate_fn
from construction.Encoder import Encoder
from construction.Embedder import Embedder
from construction.RecurrentUNet import RecurrentUNet
from construction.Transformer import ConditionTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import pandas as pd
from construction.FormationInfo import FORMATION_INFO_DIM
import joblib

def sanitise_input(data):

    for point in data:

        # Deletes formations when they are entirely np.nan and thus do not exist in the data chunk
        keys_to_drop = [k for k, v in point.items() if np.all(np.isnan(v))]
        for k in keys_to_drop:
            del point[k]

        for k, v in point.items():
            existence = ~np.isnan(v)
            elevation = np.nan_to_num(v)

            point[k] = np.stack([existence, elevation], axis=0)

    return data


def train(data_path, save_path, lr=1e-4, max_epochs=15):
    raster_size = 128
    cross_attention_dim = 768
    seq_len = 64

    # =============================
    # DATA PREPROCESSING
    # =============================

    rasters, context = Data.load_rasters(data_path)
    data, context, scaler_dict = Data.create_data(rasters, context, count=1000, size=raster_size)

    for k, v in scaler_dict.items():
        joblib.dump(v, f'{save_path}_{k}.scl')

    data = sanitise_input(data)

    dataset = BedrockDataset(data, context, scaler_dict, raster_size)

    train_size = int(0.8 * len(dataset))
    test_size = int(len(dataset) - train_size)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True,
                             collate_fn=collate_fn)

    # =============================
    # MODEL CONSTRUCTION
    # =============================

    torch._dynamo.config.verbose = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    geo_context_encoder = Encoder(in_channels=7, cross_attention_dim=cross_attention_dim, seq_len=seq_len).to(device)
    geo_context_encoder = torch.compile(geo_context_encoder)

    unet = RecurrentUNet().to(device)
    unet.enable_gradient_checkpointing()
    unet = torch.compile(unet)

    formation_embedder = Embedder(in_features=FORMATION_INFO_DIM, out_features=cross_attention_dim).to(device)
    formation_embedder = torch.compile(formation_embedder)

    borehole_encoder = Encoder(in_channels=4, cross_attention_dim=cross_attention_dim, seq_len=seq_len).to(device)
    borehole_encoder = torch.compile(borehole_encoder)

    quaternary_encoder = Encoder(in_channels=4, cross_attention_dim=cross_attention_dim, seq_len=seq_len).to(device)
    quaternary_encoder = torch.compile(quaternary_encoder)

    condition_transformer = ConditionTransformer(cross_attention_dim=cross_attention_dim, num_heads=16).to(device)
    condition_transformer = torch.compile(condition_transformer)

    optimizer = torch.optim.AdamW(
        list(geo_context_encoder.parameters()) + list(unet.parameters()) + list(formation_embedder.parameters()) +
        list(borehole_encoder.parameters()) + list(condition_transformer.parameters()) + list(quaternary_encoder.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # TODO: Change maximum number of formations as cap increases
    max_f = Data.MAX_FORMATIONS

    best_loss = np.inf

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
        for rasters, existence, context, boreholes, formation_info, quaternary, mask in train_loader:
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
            mask = mask.to(device, dtype=torch.bfloat16)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # encoded_context:    (B, seq_len, cross_attention_dim)
                # encoded_quaternary: (B, seq_len, cross_attention_dim)
                encoded_context = geo_context_encoder(context)
                encoded_quaternary = quaternary_encoder(quaternary)

                # Loop through all formations and create conditions
                B = rasters.shape[0]

                conditions = torch.zeros(size=(B, max_f, seq_len, cross_attention_dim), device=device)

                for idx in range(max_f):
                    if not mask[:, idx].any():
                        continue

                    embedded_info = formation_embedder(formation_info[:, idx, :])
                    embedded_info = embedded_info.unsqueeze(1)

                    encoded_boreholes = borehole_encoder(boreholes[:, idx, :, :, :])

                    cond = condition_transformer(
                        encoded_context,
                        embedded_info,
                        encoded_boreholes,
                        encoded_quaternary,
                    )

                    conditions[:, idx] = cond * mask[:, idx].float().view(B, 1, 1)

                gru_states = unet.init_gru_states(B, device, dtype=torch.bfloat16)
                predicted_noises = torch.zeros_like(rasters)
                predicted_masks = torch.zeros_like(rasters)

                noise = torch.randn(rasters.shape, device=device)
                timesteps = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device).long()

                rasters_flat = rasters.view(B, max_f, rasters.shape[-2], rasters.shape[-1])
                noise_flat = noise.view(B, max_f, rasters.shape[-2], rasters.shape[-1])

                noisy_rasters = scheduler.add_noise(rasters_flat, noise_flat, timesteps)

                for idx in range(max_f):
                    if not mask[:, idx].any():
                        continue

                    noise_pred, mask_pred, gru_states = unet(
                        sample=noisy_rasters[:, idx].unsqueeze(1),
                        timestep=timesteps,
                        encoder_hidden_states=conditions[:, idx],
                        gru_states=gru_states
                    )

                    predicted_noises[:, idx] = noise_pred.squeeze(1)
                    predicted_masks[:, idx] = mask_pred.squeeze(1)

                pad_mask = mask.float().view(B, max_f, 1, 1)
                snr_weights = alphas_cumprod[timesteps].view(B, 1, 1, 1)

                noise_loss = (
                    pad_mask * existence * F.mse_loss(predicted_noises, noise, reduction='none')
                ).sum() / (pad_mask * existence).sum().clamp(min=1)

                existence_loss = (
                    pad_mask * snr_weights * F.binary_cross_entropy_with_logits(
                        predicted_masks, existence, reduction='none'
                    )
                ).sum() / pad_mask.sum().clamp(min=1)

                loss = noise_loss + 0.1 * existence_loss

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
            for rasters, existence, context, boreholes, formation_info, quaternary, mask in test_loader:
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
                mask = mask.to(device, dtype=torch.bfloat16)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # encoded_context:    (B, seq_len, cross_attention_dim)
                    # encoded_quaternary: (B, seq_len, cross_attention_dim)
                    encoded_context = geo_context_encoder(context)
                    encoded_quaternary = quaternary_encoder(quaternary)

                    # Loop through all formations and create conditions
                    B = rasters.shape[0]
                    conditions = torch.zeros(size=(B, max_f, seq_len, cross_attention_dim), device=device)

                    for idx in range(max_f):
                        if not mask[:, idx].any():
                            continue

                        embedded_info = formation_embedder(formation_info[:, idx, :])
                        embedded_info = embedded_info.unsqueeze(1)

                        encoded_boreholes = borehole_encoder(boreholes[:, idx, :, :, :])

                        cond = condition_transformer(
                            encoded_context,
                            embedded_info,
                            encoded_boreholes,
                            encoded_quaternary,
                        )

                        conditions[:, idx] = cond * mask[:, idx].float().view(B, 1, 1)

                    gru_states = unet.init_gru_states(B, device, dtype=torch.bfloat16)
                    predicted_noises = torch.zeros_like(rasters)
                    predicted_masks = torch.zeros_like(rasters)

                    noise = torch.randn(rasters.shape, device=device)
                    timesteps = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device).long()

                    rasters_flat = rasters.view(B, max_f, rasters.shape[-2], rasters.shape[-1])
                    noise_flat = noise.view(B, max_f, rasters.shape[-2], rasters.shape[-1])

                    noisy_rasters = scheduler.add_noise(rasters_flat, noise_flat, timesteps)

                    for idx in range(max_f):
                        if not mask[:, idx].any():
                            continue

                        noise_pred, mask_pred, gru_states = unet(
                            sample=noisy_rasters[:, idx].unsqueeze(1),
                            timestep=timesteps,
                            encoder_hidden_states=conditions[:, idx],
                            gru_states=gru_states
                        )

                        predicted_noises[:, idx] = noise_pred.squeeze(1)
                        predicted_masks[:, idx] = mask_pred.squeeze(1)

                    pad_mask = mask.float().view(B, max_f, 1, 1)
                    snr_weights = alphas_cumprod[timesteps].view(B, 1, 1, 1)

                    noise_loss = (
                        pad_mask * existence * F.mse_loss(predicted_noises, noise, reduction='none')
                    ).sum() / (pad_mask * existence).sum().clamp(min=1)

                    existence_loss = (
                        pad_mask * snr_weights * F.binary_cross_entropy_with_logits(
                            predicted_masks, existence, reduction='none'
                        )
                    ).sum() / pad_mask.sum().clamp(min=1)

                    loss = noise_loss + 0.1 * existence_loss

                test_loss += loss.item()

        test_loss = test_loss / len(test_loader)

        print(f'Test Loss" {test_loss}')
        loss_dict['test'].append(test_loss)

        pd.DataFrame(loss_dict).to_csv(f'{save_path}_loss.csv')

        if test_loss < best_loss:
            best_loss = test_loss

            torch.save(
                {
                    'epoch': epoch+1,
                    'unet': unet._orig_mod.state_dict(),
                    'geo_context_encoder': geo_context_encoder._orig_mod.state_dict(),
                    'formation_embedder': formation_embedder._orig_mod.state_dict(),
                    'borehole_encoder': borehole_encoder._orig_mod.state_dict(),
                    'quaternary_encoder': quaternary_encoder._orig_mod.state_dict(),
                    'condition_transformer': condition_transformer._orig_mod.state_dict(),
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
                    'unet': unet._orig_mod.state_dict(),
                    'geo_context_encoder': geo_context_encoder._orig_mod.state_dict(),
                    'formation_embedder': formation_embedder._orig_mod.state_dict(),
                    'borehole_encoder': borehole_encoder._orig_mod.state_dict(),
                    'quaternary_encoder': quaternary_encoder._orig_mod.state_dict(),
                    'condition_transformer': condition_transformer._orig_mod.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': test_loss,
                    'n_formations': len(rasters),
                },
                f'{save_path}_epoch{epoch + 1:04d}.mdl'
            )