import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import joblib
import Data
from Transformer_Model.Transformer import BedrockTransformer
from Smoothing_Model.Transformer import SmoothingTransformer
import random

def augment_batch(elevation, top_rasters, base_rasters, alphaearth):

    k = random.randint(0, 3)
    flip = random.random() < 0.5

    elevation = torch.rot90(elevation, k=k, dims=(-2, -1))
    top_rasters = torch.rot90(top_rasters, k=k, dims=(-2, -1))
    base_rasters = torch.rot90(base_rasters, k=k, dims=(-2, -1))
    alphaearth = torch.rot90(alphaearth, k=k, dims=(-2, -1))

    if flip:
        elevation = torch.flip(elevation, dims=(-1,))
        top_rasters = torch.flip(top_rasters, dims=(-1,))
        base_rasters = torch.flip(base_rasters, dims=(-1,))
        alphaearth = torch.flip(alphaearth, dims=(-1,))

    return elevation, top_rasters, base_rasters, alphaearth


def prepare_batch(elevation, top_rasters, base_rasters, alphaearth, device):
    elevation, top_rasters, base_rasters, alphaearth = augment_batch(elevation, top_rasters, base_rasters, alphaearth)

    B, F, _, H, W = elevation.shape

    elevation = elevation[:, 0].unsqueeze(1).expand(B, F, 1, H, W).reshape(B * F, 1, H, W)
    alphaearth = alphaearth[:, 0].unsqueeze(1).expand(B, F, 64, H, W).reshape(B * F, 64, H, W)

    B, F, _, H, W = top_rasters.shape

    top_rasters = top_rasters.reshape(B * F, 1, H, W)
    base_rasters = base_rasters.reshape(B * F, 1, H, W)

    elevation = elevation.to(device, dtype=torch.bfloat16)
    alphaearth = alphaearth.to(device, dtype=torch.bfloat16)
    top_rasters = top_rasters.to(device, dtype=torch.bfloat16)

    return elevation, top_rasters, alphaearth

def train(data_path, save_path, lr=1e-4, max_epochs=100):
    data_count = 500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # =============================
    # LOADING TERRA
    # =============================

    model_dict = torch.load(f'{save_path}.mdl')

    raster_size = model_dict['raster_size']
    patch_size = model_dict['patch_size']
    embed_dim = model_dict['embed_dim']
    encoder_depth = model_dict['encoder_depth']
    decoder_depth = model_dict['decoder_depth']
    mlp_dim = model_dict['mlp_dim']

    terra = BedrockTransformer(
        raster_size,
        patch_size,
        embed_dim,
        num_heads=8,
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        mlp_dim=mlp_dim
    ).to(device)
    terra.load_state_dict(model_dict['model'])

    for param in terra.parameters():
        param.requires_grad = False

    terra.eval()

    print('TERRA loaded')

    # =============================
    # SMOOTHING MODEL
    # =============================

    max_size = 512
    patch_size = 16
    embed_dim = 256
    mlp_dim = 512
    depth = 6

    print('Loading rasters')

    formations = [
        'kwnd', 'dlp', 'dspl', 'omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh', 'opod',
        'cp', 'cjdn', 'cstl', 'ctcg', 'cwoc', 'cecr', 'cmts'
    ]

    counties = [Data.CountySource(p, formations) for p in data_path]

    scaler_dict = {'elevation': joblib.load(f'{save_path}_elevation.scl')}

    dataset = Data.MultiCountyDataset(counties, scaler_dict, data_count, max_size)

    test_dataset = dataset.split_test(int(data_count * 0.1))
    train_dataset = dataset

    test_dataset.generate_indices()

    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=0)

    print('Constructing smoothing model...')

    model = SmoothingTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=8,
        depth=depth,
        mlp_dim=mlp_dim
    ).to(device)
    print(' constructed smoother')

    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=10
    )

    patience = 0

    best_loss = np.inf
    loss_dict = {'train': [], 'test': []}

    for epoch in range(max_epochs):
        print(f'Epoch {epoch+1}')

        # =============================
        # TRAIN LOOP
        # =============================

        model.train()

        train_loss = 0.0
        train_dataset.generate_indices()

        for elevation, top_rasters, base_rasters, alphaearth in train_loader:

            elevation, top_rasters, alphaearth = prepare_batch(elevation, top_rasters, base_rasters, alphaearth, device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

                ###--- Generate raw TERRA predictions ---###
                with torch.no_grad:
                    B, C, H, W = top_rasters.shape
                    n = max_size // raster_size

                    terra_pred = torch.zeros(B, C, H, W, device=device)

                    for i in range(n):
                        for j in range(n):
                            count = np.random.randint(1, 101)

                            r0, r1 = i * raster_size, (i+1) * raster_size
                            c0, c1 = j * raster_size, (j+1) * raster_size

                            elev_tile = elevation[:, :, r0:r1, c0:c1]
                            t_tile = top_rasters[:, :, r0:r1, c0:c1]
                            b_tile = base_rasters[:, :, r0:r1, c0:c1]
                            ae_tile = alphaearth[:, :, r0:r1, c0:c1]

                            boreholes = [train_dataset.select_boreholes(top, base, count=count, size=raster_size) for top, base in zip(t_tile, b_tile)]
                            boreholes = torch.stack(boreholes).reshape(B, count, 5)
                            boreholes = boreholes.to(device, dtype=torch.bfloat16)

                            tile_pred = terra(elev_tile, boreholes, ae_tile)
                            terra_pred[:, :, r0:r1, c0:c1] = tile_pred

                optimizer.zero_grad()

                ###--- Randomly crop the input to allow for varying sizes ---###
                num = (max_size - raster_size) // patch_size
                chosen = np.random.randint(0, num + 1)
                target_size = raster_size + chosen * patch_size

                max_offset = max_size - target_size
                r0 = np.random.randint(0, max_offset + 1) if max_offset > 0 else 0
                c0 = np.random.randint(0, max_offset + 1) if max_offset > 0 else 0

                terra_pred = terra_pred[:, :, r0:r0+target_size, c0:c0+target_size]
                top_rasters = top_rasters[:, :, r0:r0+target_size, c0:c0+target_size]

                smoothed = model(terra_pred)

                loss = F.mse_loss(smoothed, top_rasters)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        print(f'Train Loss: {train_loss / len(train_loader)}')
        loss_dict['train'].append(train_loss / len(train_loader))

        # =============================
        # TEST LOOP
        # =============================

        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for elevation, top_rasters, base_rasters, alphaearth in test_loader:
                elevation, top_rasters, alphaearth = prepare_batch(elevation, top_rasters, base_rasters, alphaearth,
                                                                   device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

                    ###--- Generate raw TERRA predictions ---###
                    B, C, H, W = top_rasters.shape
                    n = max_size // raster_size

                    terra_pred = torch.zeros(B, C, H, W, device=device)

                    for i in range(n):
                        for j in range(n):
                            count = np.random.randint(1, 101)

                            r0, r1 = i * raster_size, (i + 1) * raster_size
                            c0, c1 = j * raster_size, (j + 1) * raster_size

                            elev_tile = elevation[:, :, r0:r1, c0:c1]
                            t_tile = top_rasters[:, :, r0:r1, c0:c1]
                            b_tile = base_rasters[:, :, r0:r1, c0:c1]
                            ae_tile = alphaearth[:, :, r0:r1, c0:c1]

                            boreholes = [train_dataset.select_boreholes(top, base, count=count) for top, base in
                                         zip(t_tile, b_tile)]
                            boreholes = torch.stack(boreholes).reshape(B, count, 5)

                            tile_pred = terra(elev_tile, boreholes, ae_tile)
                            terra_pred[:, :, r0:r1, c0:c1] = tile_pred

                    num = (max_size - raster_size) // patch_size
                    chosen = np.random.randint(0, num + 1)
                    target_size = raster_size + chosen * patch_size

                    max_offset = max_size - target_size
                    r0 = np.random.randint(0, max_offset + 1) if max_offset > 0 else 0
                    c0 = np.random.randint(0, max_offset + 1) if max_offset > 0 else 0

                    terra_pred = terra_pred[:, :, r0:r0 + target_size, c0:c0 + target_size]
                    top_rasters = top_rasters[:, :, r0:r0 + target_size, c0:c0 + target_size]

                    smoothed = model(terra_pred)

                    loss = F.mse_loss(smoothed, top_rasters)

                test_loss += loss.item()

        scheduler.step(test_loss)

        print(f'Test Loss: {test_loss / len(test_loader)}')
        loss_dict['test'].append(test_loss / len(test_loader))

        pd.DataFrame(loss_dict).to_csv(f'{save_path}_smooth_loss.csv')

        if test_loss < best_loss:
            best_loss = test_loss
            patience = 0

            torch.save(
                {
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'loss': best_loss,
                    'patch_size': patch_size,
                    'embed_dim': embed_dim,
                    'depth': depth,
                    'mlp_dim': mlp_dim,
                },
                f'{save_path}_smoother.mdl'
            )

        else:
            if patience > 15:
                return

            patience += 1