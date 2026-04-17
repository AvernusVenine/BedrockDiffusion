import numpy as np
import Data
from Transformer_Model.Transformer import BedrockTransformer
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import pandas as pd
import joblib
import matplotlib.pyplot as plt

class MaskScheduler:
    def __init__(self, num_tokens, pretrain_epochs, min_ratio, max_ratio):
        self.num_tokens = num_tokens
        self.pretrain_epochs = pretrain_epochs
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.rng = np.random.default_rng()

    def _sample_mask(self, B, device):
        ratio = self.rng.uniform(self.min_ratio, self.max_ratio)
        num_masked = int(round(self.num_tokens * ratio))

        mask = torch.zeros(B, self.num_tokens, dtype=torch.bool, device=device)
        for i in range(B):
            idx = torch.randperm(self.num_tokens, device=device)[:num_masked]
            mask[i, idx] = True
        return mask

    def sample(self, epoch, B, device):
        if epoch >= self.pretrain_epochs:
            return None, None

        elev_mask = self._sample_mask(B, device)
        ae_mask = self._sample_mask(B, device)
        return elev_mask, ae_mask


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

        return int(self.rng.integers(current_min, current_max + 1))

def prepare_batch(elevation, top_rasters, base_rasters, alphaearth, dataset, bh_count, device):
    B, F, _, H, W = elevation.shape

    elevation = elevation[:, 0].unsqueeze(1).expand(B, F, 1, H, W).reshape(B*F, 1, H, W)
    alphaearth = alphaearth[:, 0].unsqueeze(1).expand(B, F, 64, H, W).reshape(B*F, 64, H, W)

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

def test(data_path, model_path, save_path, bh_count=10, formations=None):
    model_dict = torch.load(f'{model_path}.mdl')

    raster_size = model_dict['raster_size']
    patch_size = model_dict['patch_size']
    embed_dim = model_dict['embed_dim']
    encoder_depth = model_dict['encoder_depth']
    decoder_depth = model_dict['decoder_depth']
    mlp_dim = model_dict['mlp_dim']

    if formations is None:
        formations = ['kwnd', 'dlp', 'dspl', 'omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh',
                      'opod', 'cp', 'cjdn', 'cstl', 'ctcg', 'cwoc', 'cecr', 'cmts']

    counties = [Data.CountySource(data_path, formations)]
    scaler_dict = {'elevation': joblib.load(f'{model_path}_elevation.scl')}

    dataset = Data.MultiCountyDataset(counties, scaler_dict, 500, raster_size, 1.0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BedrockTransformer(
        raster_size,
        patch_size,
        embed_dim,
        num_heads=8,
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        mlp_dim=mlp_dim
    ).to(device)
    model.load_state_dict(model_dict['model'])

    dataset.generate_indices()

    elevation, top_rasters, base_rasters, alphaearth = dataset.get_full(0)

    boreholes = [dataset.select_boreholes(top, base, count=bh_count) for top, base in zip(top_rasters, base_rasters)]
    boreholes = torch.stack(boreholes)

    elevation = elevation.to(device, dtype=torch.float32)
    alphaearth = alphaearth.to(device, dtype=torch.float32)
    boreholes = boreholes.to(device, dtype=torch.float32)

    predicted_elevation, predicted_existence = model(elevation, boreholes, alphaearth)

    predicted_elevation = predicted_elevation.cpu().float().numpy()
    predicted_existence = predicted_existence.cpu().float().numpy()

    existence = ((top_rasters - base_rasters) > 0.0).cpu().float().numpy()

    predicted_elevation = scaler_dict['elevation'].inverse_transform(predicted_elevation.reshape(-1, 1)).reshape(predicted_elevation.shape)
    top_rasters = scaler_dict['elevation'].inverse_transform(top_rasters.reshape(-1, 1)).reshape(elevation.shape)
    elevation = scaler_dict['elevation'].inverse_transform(elevation.reshape(-1, 1)).reshape(elevation.shape)

    joblib.dump({
        'surface_elevation': elevation,
        'predicted_elevation': predicted_elevation,
        'predicted_existence': predicted_existence,
        'actual_elevation': top_rasters,
        'actual_existence': existence
    }, save_path)


def train(data_path, save_path, lr=1e-4, max_epochs=100, load_model=False):
    raster_size = 64
    patch_size = 16
    embed_dim = 512
    mlp_dim = 1024
    data_count = 5000
    encoder_depth = 6
    decoder_depth = 3

    # =============================
    # DATA PREPROCESSING
    # =============================

    print('LOADING RASTERS')

    formations = ['kwnd', 'dlp', 'dspl', 'omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh', 'opod',
                 'cp', 'cjdn', 'cstl', 'ctcg', 'cwoc', 'cecr', 'cmts']

    counties = [Data.CountySource(p, formations) for p in data_path]

    scaler_dict = Data.create_global_scaler_dict(counties)

    print('GENERATING SCALERS')

    for k, v in scaler_dict.items():
        joblib.dump(v, f'{save_path}_{k}.scl')

    # =============================
    # DATASET CONSTRUCTION
    # =============================

    dataset = Data.MultiCountyDataset(counties, scaler_dict, data_count, raster_size, 2.0)

    test_dataset = dataset.split_test()
    train_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # =============================
    # MODEL CONSTRUCTION
    # =============================

    bh_scheduler = BoreholeScheduler([10, 100], [1, 50], 100)

    num_tokens = (raster_size // patch_size) ** 2
    mask_scheduler = MaskScheduler(num_tokens, 100, .5, .5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dict = None
    if load_model:
        model_dict = torch.load(f'{save_path}.mdl')

        raster_size = model_dict['raster_size']
        patch_size = model_dict['patch_size']
        embed_dim = model_dict['embed_dim']
        encoder_depth = model_dict['encoder_depth']
        decoder_depth = model_dict['decoder_depth']
        mlp_dim = model_dict['mlp_dim']

    model = BedrockTransformer(
        raster_size,
        patch_size,
        embed_dim,
        num_heads=8,
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        mlp_dim=mlp_dim,
    ).to(device)

    if load_model:
        model.load_state_dict(model_dict['model'])

    model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    if load_model:
        optimizer.load_state_dict(model_dict['optimizer'])

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

            elevation, top_rasters, existence, alphaearth, boreholes = prepare_batch(elevation, top_rasters,
                    base_rasters, alphaearth, train_dataset, count, device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                elev_mask, ae_mask = mask_scheduler.sample(epoch, elevation.shape[0], device)

                data = [elevation, boreholes, alphaearth, elev_mask, ae_mask, top_rasters, base_rasters]
                for idx in range(len(data)):
                    if (data[idx] == np.nan).any():
                        print(f'NaN found in {idx}')

                predicted_elevation, predicted_existence = model(elevation, boreholes, alphaearth, elev_mask, ae_mask)

                elevation_loss = F.mse_loss(predicted_elevation, top_rasters)
                existence_loss = F.binary_cross_entropy_with_logits(predicted_existence, existence)

                loss = elevation_loss + 0.25*existence_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f'Train Loss: {train_loss / len(train_loader)}')
        loss_dict['train'].append(train_loss / len(train_loader))

        # =============================
        # TEST LOOP
        # =============================

        model.eval()

        test_loss = 0.0
        test_dataset.generate_indices()

        with torch.no_grad():
            for elevation, top_rasters, base_rasters, alphaearth in test_loader:
                count = bh_scheduler.sample(epoch)

                elevation, top_rasters, existence, alphaearth, boreholes = prepare_batch(elevation, top_rasters,
                        base_rasters, alphaearth, test_dataset, count, device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    elev_mask, ae_mask = mask_scheduler.sample(epoch, elevation.shape[0], device)

                    predicted_elevation, predicted_existence = model(elevation, boreholes, alphaearth, elev_mask, ae_mask)

                    elevation_loss = F.mse_loss(predicted_elevation, top_rasters)
                    existence_loss = F.binary_cross_entropy_with_logits(predicted_existence, existence)

                    loss = elevation_loss + 0.25 * existence_loss

                test_loss += loss.item()

        print(f'Test Loss: {test_loss / len(test_loader)}')
        loss_dict['test'].append(test_loss / len(test_loader))

        pd.DataFrame(loss_dict).to_csv(f'{save_path}_loss.csv')

        if epoch == 99:
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model': model._orig_mod.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': test_loss,
                    'raster_size': raster_size,
                    'patch_size': patch_size,
                    'embed_dim': embed_dim,
                    'encoder_depth': encoder_depth,
                    'decoder_depth': decoder_depth,
                    'mlp_dim': mlp_dim,
                },
                f'{save_path}_pretrain.mdl'
            )

        if test_loss < best_loss:
            best_loss = test_loss

            torch.save(
                {
                    'epoch': epoch + 1,
                    'model': model._orig_mod.state_dict(),
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