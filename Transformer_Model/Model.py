import numpy as np
import Data
from Transformer_Model.Transformer import BedrockTransformer
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import pandas as pd
import joblib
import matplotlib.pyplot as plt

class BoreholeScheduler:
    def __init__(self, max_count_range, min_count_range, warmup_epochs):
        self.max_lo, self.max_hi = max_count_range
        self.min_lo, self.min_hi = min_count_range

        self.warmup_epochs = warmup_epochs
        self.rng = np.random.default_rng()

    def sample(self, epoch):
        t = min(epoch / self.warmup_epochs, 1.0)
    
        current_max = int(round(self.max_hi - t * (self.max_hi - self.max_lo)))
        current_min = int(round(self.min_hi - t * (self.min_hi - self.min_lo)))
        current_min = min(current_min, current_max - 1)
    
        return int(self.rng.integers(current_min, current_max + 1))

def prepare_batch(elevation, top_rasters, base_rasters, alphaearth, dataset, bh_count, device):
    B, F, _, H, W = elevation.shape

    elevation = elevation[:, 0].unsqueeze(1).expand(B, F, 1, H, W).reshape(B*F, 1, H, W)
    alphaearth = alphaearth[:, 0].unsqueeze(1).expand(B, F, 64, H, W).reshape(B*F, 64, H, W)

    B, F, _, H, W = top_rasters.shape

    boreholes = [dataset.select_boreholes(top, base, count=bh_count) for top, base in zip(top_rasters, base_rasters)]
    boreholes = torch.stack(boreholes).reshape(B*F, bh_count, 5)

    top_rasters = top_rasters.reshape(B*F, 1, H, W)
    base_rasters = base_rasters.reshape(B*F, 1, H, W)

    thickness = (top_rasters - base_rasters).nan_to_num(0.0)
    existence = (thickness > 0).float()

    elevation = elevation.to(device, dtype=torch.bfloat16)
    alphaearth = alphaearth.to(device, dtype=torch.bfloat16)
    existence = existence.to(device, dtype=torch.bfloat16)
    top_rasters = top_rasters.to(device, dtype=torch.bfloat16)
    boreholes = boreholes.to(device, dtype=torch.bfloat16)

    return elevation, top_rasters, existence, alphaearth, boreholes

def test(data_path, model_path, save_path, count=20):
    model_dict = torch.load(f'{model_path}.mdl')

    raster_size = model_dict['raster_size']
    patch_size = model_dict['patch_size']
    embed_dim = model_dict['embed_dim']
    encoder_depth = model_dict['encoder_depth']
    decoder_depth = model_dict['decoder_depth']
    mlp_dim = model_dict['mlp_dim']

    print('Model loaded')

    formations = [
        'kwnd', 'dlp', 'dspl', 'omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh', 'opod',
        'cp', 'cjdn', 'cstl', 'ctcg', 'cwoc', 'cecr', 'cmts'
    ]

    counties = [Data.CountySource(p, formations) for p in data_path]

    scaler_dict = {'elevation': joblib.load(f'{model_path}_elevation.scl')}

    dataset = Data.MultiCountyDataset(counties, scaler_dict, 5, raster_size, 1.0)
    dataset.generate_indices()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state = {k: v for k, v in model_dict['model'].items() if not k.endswith('pe')}

    model = BedrockTransformer(
        raster_size,
        patch_size,
        embed_dim,
        num_heads=8,
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        mlp_dim=mlp_dim
    ).to(device)
    model.load_state_dict(state, strict=False)

    model.eval()

    elevation, top_rasters, base_rasters, alphaearth, formations = dataset.get_full(0)

    elevation = elevation.unsqueeze(0)
    top_rasters = top_rasters.unsqueeze(0)
    base_rasters = base_rasters.unsqueeze(0)
    alphaearth = alphaearth.unsqueeze(0)

    elevation, top_rasters, base_rasters, alphaearth, boreholes = prepare_batch(
        elevation, top_rasters, base_rasters, alphaearth, dataset, count, device
    )

    print('Evaluating...')

    with torch.no_grad():
        elevation = elevation.to(device, dtype=torch.float32)
        alphaearth = alphaearth.to(device, dtype=torch.float32)
        boreholes = boreholes.to(device, dtype=torch.float32)

        predicted = model(elevation, boreholes, alphaearth)

    print(' Done')

    elevation = elevation.squeeze(1).cpu().float().numpy()
    predicted = predicted.squeeze(1).cpu().float().numpy()
    top_rasters = top_rasters.squeeze(1).cpu().float().numpy()

    c_elev = counties[0].elevation
    c_elev = scaler_dict['elevation'].transform(c_elev.reshape(-1, 1)).reshape(c_elev.shape)
    elev_vmin, elev_vmax = np.nanmin(c_elev), np.nanmax(c_elev)

    all_elevs = np.concatenate([predicted.ravel(), top_rasters.ravel()])
    vmin, vmax = np.nanmin(all_elevs), np.nanmax(all_elevs)

    print('Creating figures...')

    for idx in range(len(predicted)):
        fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

        plt.title(f'{formations[idx].capitalize()}')

        axes[0].imshow(elevation[idx], cmap='terrain', vmin=elev_vmin, vmax=elev_vmax)
        axes[0].set_title('Elevation')

        axes[1].imshow(predicted[idx], cmap='terrain', vmin=vmin, vmax=vmax)
        axes[1].set_title('Predicted')

        axes[2].imshow(top_rasters[idx], cmap='terrain', vmin=vmin, vmax=vmax)
        axes[2].set_title('Ground Truth')

        plt.savefig(f'{save_path}_{formations[idx]}.png')
        plt.close()

        print(f' {formations[idx]}')

    #- Use next raster to remove places where the thickness is near 0, signifying the formation does not exist -#
    for idx in range(len(predicted) - 1):
        pred_mask = (predicted[idx] - predicted[idx+1]) > .01
        true_mask = (top_rasters[idx] - top_rasters[idx+1]) > 0

        predicted[idx] = np.where(pred_mask, predicted[idx], np.nan)
        top_rasters[idx] = np.where(true_mask, top_rasters[idx], np.nan)

    for idx in range(len(predicted)):
        fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

        plt.title(f'{formations[idx].capitalize()}')

        axes[0].imshow(elevation[idx], cmap='terrain', vmin=elev_vmin, vmax=elev_vmax)
        axes[0].set_title('Elevation')

        axes[1].imshow(predicted[idx], cmap='terrain', vmin=vmin, vmax=vmax)
        axes[1].set_title('Predicted')

        axes[2].imshow(top_rasters[idx], cmap='terrain', vmin=vmin, vmax=vmax)
        axes[2].set_title('Ground Truth')

        plt.savefig(f'{save_path}_{formations[idx]}_masked.png')
        plt.close()

        print(f' {formations[idx]}')

def train(data_path, save_path, lr=1e-4, max_epochs=100):
    raster_size = 64
    patch_size = 16
    embed_dim = 512
    mlp_dim = 1024
    data_count = 3000
    encoder_depth = 6
    decoder_depth = 4

    # =============================
    # DATA PREPROCESSING
    # =============================

    print('Loading rasters')

    formations = [
        'kwnd', 'dlp', 'dspl', 'omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh', 'opod',
        'cp', 'cjdn', 'cstl', 'ctcg', 'cwoc', 'cecr', 'cmts'
    ]

    counties = [Data.CountySource(p, formations) for p in data_path]

    scaler_dict = Data.create_global_scaler_dict(counties)

    print('Generating scalers')

    for k, v in scaler_dict.items():
        joblib.dump(v, f'{save_path}_{k}.scl')

    # =============================
    # DATASET CONSTRUCTION
    # =============================

    dataset = Data.MultiCountyDataset(counties, scaler_dict, data_count, raster_size)

    test_dataset = dataset.split_test(int(data_count * 0.1))
    train_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size=6, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=6, num_workers=0)

    test_dataset.generate_indices()

    # =============================
    # MODEL CONSTRUCTION
    # =============================

    print('Constructing Model...')

    bh_scheduler = BoreholeScheduler([50, 200], [1, 25], 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BedrockTransformer(
        raster_size,
        patch_size,
        embed_dim,
        num_heads=8,
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        mlp_dim=mlp_dim,
    ).to(device)
    print(' constructed TERRA')

    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=lr,
        weight_decay=1e-4,
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
            count = bh_scheduler.sample(epoch)

            elevation, top_rasters, existence, alphaearth, boreholes = prepare_batch(
                elevation, top_rasters, base_rasters, alphaearth, train_dataset, count, device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                predicted_elevation = model(elevation, boreholes, alphaearth)

                loss = F.mse_loss(predicted_elevation, top_rasters)

            optimizer.zero_grad()
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
                count = bh_scheduler.sample(epoch)

                elevation, top_rasters, existence, alphaearth, boreholes = prepare_batch(
                    elevation, top_rasters, base_rasters, alphaearth, train_dataset, count, device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    predicted_elevation = model(elevation, boreholes, alphaearth)

                    loss = F.mse_loss(predicted_elevation, top_rasters)

                test_loss += loss.item()

        print(f'Test Loss: {test_loss / len(test_loader)}')
        loss_dict['test'].append(test_loss / len(test_loader))

        pd.DataFrame(loss_dict).to_csv(f'{save_path}_loss.csv')

        if test_loss < best_loss:
            best_loss = test_loss
            patience = 0

            torch.save(
                {
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': best_loss,
                    'raster_size': raster_size,
                    'patch_size': patch_size,
                    'embed_dim': embed_dim,
                    'encoder_depth': encoder_depth,
                    'decoder_depth': decoder_depth,
                    'mlp_dim': mlp_dim,
                },
                f'{save_path}.mdl'
            )
        else:
            if patience > 15:
                return

            patience += 1