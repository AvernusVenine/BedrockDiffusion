import numpy as np
import Data
from Transformer_Model.Transformer import BedrockTransformer
from Transformer_Model import FormationInfo
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

def prepare_batch(elevation, top_rasters, base_rasters, alphaearth, magnetic, formation_info, dataset, bh_count, device):
    B, F, _, H, W = elevation.shape

    elevation = elevation[:, 0].unsqueeze(1).expand(B, F, 1, H, W).reshape(B*F, 1, H, W)
    alphaearth = alphaearth[:, 0].unsqueeze(1).expand(B, F, 64, H, W).reshape(B*F, 64, H, W)
    magnetic = magnetic[:, 0].unsqueeze(1).expand(B, F, 5, H, W).reshape(B*F, 5, H, W)
    formation_info = formation_info[:, 0].unsqueeze(1).expand(B, F, FormationInfo.FORMATION_INFO_DIM, H, W).reshape(
    B*F, FormationInfo.FORMATION_INFO_DIM, H, W)

    B, F, _, H, W = top_rasters.shape

    boreholes = [dataset.select_boreholes(top, base, count=bh_count) for top, base in zip(top_rasters, base_rasters)]
    boreholes = torch.stack(boreholes).reshape(B*F, bh_count, 5)

    top_rasters = top_rasters.reshape(B*F, 1, H, W)
    base_rasters = base_rasters.reshape(B*F, 1, H, W)

    thickness = (top_rasters - base_rasters).nan_to_num(0.0)
    existence = (thickness > 0).float()

    elevation = elevation.to(device, dtype=torch.bfloat16)
    alphaearth = alphaearth.to(device, dtype=torch.bfloat16)
    magnetic = magnetic.to(device, dtype=torch.bfloat16)
    existence = existence.to(device, dtype=torch.bfloat16)
    top_rasters = top_rasters.to(device, dtype=torch.bfloat16)
    boreholes = boreholes.to(device, dtype=torch.bfloat16)

    return elevation, top_rasters, existence, alphaearth, magnetic, formation_info, boreholes

def test(data_path, model_path, save_path):
    raster_size = 64
    patch_size = 16
    embed_dim = 768
    mlp_dim = 1024
    depth = 6

    rasters, context = Data.load_rasters(data_path)

    scaler = joblib.load(f'{model_path}_elevation.scl')
    scaler_dict = {'elevation': scaler}

    valid_mask = np.zeros(context['elevation'].shape, dtype=bool)
    for sparse in rasters.values():
        valid_mask[sparse['rows'], sparse['cols']] = True

    patches = Data.find_valid_patches(valid_mask, raster_size, raster_size)

    dataset = Data.TransformerDataset(rasters, context, scaler_dict, patches, 1, 256)
    dataset.generate_indices()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dict = torch.load(f'{model_path}.mdl')

    model = BedrockTransformer(
        raster_size,
        patch_size,
        embed_dim,
        num_heads=8,
        depth=depth,
        mlp_dim=mlp_dim,
    ).to(device)
    model.load_state_dict(model_dict['model'])

    model.eval()

    rasters, context, boreholes = dataset[0]

    with torch.no_grad():
        context = context.to(device, dtype=torch.float32)
        boreholes = boreholes.to(device, dtype=torch.float32)

        predicted = model(context, boreholes)

    context = context.squeeze(1).cpu().float().numpy()
    predicted = predicted.squeeze(1).cpu().float().numpy()
    rasters = rasters.squeeze(1).numpy()
    boreholes = boreholes[:, 2].cpu().float().numpy()

    predicted[np.isnan(rasters)] = np.nan

    all_values = np.concatenate([context.ravel(), predicted.ravel(), boreholes.ravel()])
    vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)

    fig, axes = plt.subplots(len(rasters), 4, figsize=(20, 5*len(rasters)))

    for row in range(len(predicted)):
        axes[row, 0].imshow(context[row], cmap='terrain', vmin=vmin, vmax=vmax)
        axes[row, 0].set_title('Elevation')

        axes[row, 1].imshow(predicted[row], cmap='terrain', vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f'Predicted')

        axes[row, 2].imshow(rasters[row], cmap='terrain', vmin=vmin, vmax=vmax)
        axes[row, 2].set_title(f'Actual')

        axes[row, 3].imshow(boreholes[row] > 0, cmap='binary')
        axes[row, 3].set_title('Boreholes')

    plt.savefig(f'{save_path}.png')
    plt.close()

def train(data_path, save_path, lr=1e-4, max_epochs=100):
    raster_size = 80
    patch_size = 16
    mag_patch_size = 8
    embed_dim = 512
    mlp_dim = 1024
    data_count = 3000
    encoder_depth = 6
    decoder_depth = 4

    # =============================
    # DATA PREPROCESSING
    # =============================

    print('Loading Rasters')

    formations = [
        'kwnd', 'dlp', 'dspl', 'omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh', 'opod',
        'cp', 'cjdn', 'cstl', 'ctcg', 'cwoc', 'cecr', 'cmts'
    ]

    counties = [Data.CountySource(p, formations) for p in data_path]

    scaler_dict = Data.create_global_scaler_dict(counties)

    print('Generating Scalers')

    for k, v in scaler_dict.items():
        joblib.dump(v, f'{save_path}_{k}.scl')

    # =============================
    # DATASET CONSTRUCTION
    # =============================

    dataset = Data.MultiCountyDataset(counties, scaler_dict, data_count, raster_size, 2.0)

    test_dataset = dataset.split_test(int(data_count * 0.1))
    train_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size=6, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=6, num_workers=0)

    test_dataset.generate_indices()

    # =============================
    # MODEL CONSTRUCTION
    # =============================

    print('Constructing Model...')

    bh_scheduler = BoreholeScheduler([20, 100], [1, 20], 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BedrockTransformer(
        raster_size,
        patch_size,
        mag_patch_size,
        embed_dim,
        num_heads=8,
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        mlp_dim=mlp_dim,
    ).to(device)
    print(' constructed TERRA')
    
    #model = torch.compile(model)
    
    #print(' compiled TERRA')

    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

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

        for elevation, top_rasters, base_rasters, alphaearth, magnetic, formation_info in train_loader:
            count = bh_scheduler.sample(epoch)

            elevation, top_rasters, existence, alphaearth, magnetic, formation_info, boreholes = prepare_batch(
                elevation, top_rasters, base_rasters, alphaearth, magnetic, formation_info, train_dataset, count, device)

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
            for elevation, top_rasters, base_rasters, alphaearth, magnetic, formation_info in test_loader:
                count = bh_scheduler.sample(epoch)

                elevation, top_rasters, existence, alphaearth, magnetic, formation_info, boreholes = prepare_batch(
                    elevation, top_rasters, base_rasters, alphaearth, magnetic, formation_info, train_dataset, count,
                    device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    predicted_elevation = model(elevation, boreholes, alphaearth)

                    loss = F.mse_loss(predicted_elevation, top_rasters)

                test_loss += loss.item()

        print(f'Test Loss: {test_loss / len(test_loader)}')
        loss_dict['test'].append(test_loss / len(test_loader))
        

        pd.DataFrame(loss_dict).to_csv(f'{save_path}_loss.csv')

        if epoch == 49:
            print('Pretraining completed')
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
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