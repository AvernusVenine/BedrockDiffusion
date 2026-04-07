import numpy as np
import Data
from construction.Transformer import BedrockTransformer
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import pandas as pd
import joblib
import matplotlib.pyplot as plt

def test_transformer(data_path, model_path, save_path, formations):
    raster_size = 256
    patch_size = 16
    embed_dim = 768
    data_count = 1

    rasters, context = Data.load_rasters(data_path)
    scaler_dict = Data.create_scaler_dict(rasters, context)

    context = {'elevation': context['elevation']}

    data, ctx = Data.select_data_patches(rasters, context, data_count, raster_size, fill_nan=True)

    filtered_data = []
    filtered_ctx = []

    for patch, c in zip(data, ctx):
        filtered = {k: v for k, v in patch.items() if k in formations}
        if filtered:
            for formation in filtered.values():
                filtered_data.append(formation)
                filtered_ctx.append(c['elevation'])

    dataset = Data.TransformerDataset(filtered_data, filtered_ctx, scaler_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dict = torch.load(model_path)

    model = BedrockTransformer(
        raster_size,
        patch_size,
        embed_dim,
        num_heads=8,
        depth=6,
        mlp_dim=1024,
    ).to(device)
    model.load_state_dict(model_dict['model'])

    results = []

    for rasters, context, boreholes in dataset:
        rasters = torch.from_numpy(rasters)
        context = torch.from_numpy(context)

        model.eval()

        with torch.no_grad():
            context = context.to(device, dtype=torch.float32).reshape(1, 1, raster_size, raster_size)
            boreholes = boreholes.to(device, dtype=torch.float32).reshape(1, 1, raster_size, raster_size)

            predicted = model(context, boreholes)

            results.append({
                'elevation': context.squeeze().cpu().float().numpy(),
                'predicted': predicted.squeeze().cpu().float().numpy(),
                'actual': rasters.numpy(),
                'boreholes': boreholes.squeeze().cpu().float().numpy()
            })

    all_values = np.concatenate([
        np.stack([r['elevation'], r['predicted'], r['actual']]).ravel()
        for r in results
    ])
    vmin, vmax = all_values.min(), all_values.max()

    fig, axes = plt.subplots(len(results), 4, figsize=(20, 5*len(results)))

    if len(results) == 1:
        axes = axes[np.newaxis, :]

    for row, r in enumerate(results):
        axes[row, 0].imshow(r['elevation'], cmap='terrain', vmin=vmin, vmax=vmax)
        axes[row, 0].set_title('Elevation')

        axes[row, 1].imshow(r['predicted'], cmap='terrain', vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f'{formations[row]} Predicted')

        axes[row, 2].imshow(r['actual'], cmap='terrain', vmin=vmin, vmax=vmax)
        axes[row, 2].set_title(f'{formations[row]} Actual')

        axes[row, 3].imshow(r['boreholes'] > 0, cmap='binary')
        axes[row, 3].set_title('Boreholes')

    plt.savefig(f'{save_path}.png')
    plt.close()

def train_transformer(data_path, save_path, lr=1e-4, max_epochs=100):
    raster_size = 256
    patch_size = 8
    embed_dim = 768
    mlp_dim = 1024
    data_count = 7500
    depth = 10
    formations = ['omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh', 'opod']

    # =============================
    # DATA PREPROCESSING
    # =============================

    rasters, context = Data.load_rasters(data_path)
    scaler_dict = Data.create_scaler_dict(rasters, context)

    for k, v in scaler_dict.items():
        joblib.dump(v, f'{save_path}_{k}.scl')

    valid_mask = np.zeros(list(rasters.values())[0].shape, dtype=bool)

    for v in rasters.values():
        mask = ~np.isnan(v)
        valid_mask = mask | valid_mask

    ###--- Test Dataset ---###
    ###--- Remove test data sections from the continuous dataset ---###
    patches = Data.find_valid_patches(valid_mask, raster_size, raster_size)

    test_size = int(data_count * 0.1)

    indices = np.random.choice(len(patches), size=test_size, replace=False)
    test_indices = [patches[i] for i in indices]

    for x, y in test_indices:
        valid_mask[x:x+raster_size, y:y+raster_size] = False

    test_dataset = Data.TransformerDataset(rasters, context, scaler_dict, test_indices, test_size, raster_size)

    ###--- Train Dataset ---###
    patches = Data.find_valid_patches(valid_mask, raster_size, raster_size)
    train_dataset = Data.TransformerDataset(rasters, context, scaler_dict, patches, data_count, raster_size)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # =============================
    # MODEL CONSTRUCTION
    # =============================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BedrockTransformer(
        raster_size,
        patch_size,
        embed_dim,
        num_heads=8,
        depth=depth,
        mlp_dim=mlp_dim,
    ).to(device)
    model = torch.compile(model)

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
        for rasters, context, boreholes in train_loader:
            rasters = rasters.to(device, dtype=torch.bfloat16)
            context = context.to(device, dtype=torch.bfloat16)
            boreholes = boreholes.to(device, dtype=torch.bfloat16)

            B, C, H, W = rasters.shape

            rasters = rasters.reshape(B * C, 1, H, W)
            boreholes = boreholes.reshape(B * C, 1 , H, W)
            context = context.reshape(B * C, 1, H, W)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                predicted = model(context, boreholes)

                loss = F.mse_loss(predicted, rasters)

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

        with torch.no_grad():
            for rasters, context, boreholes in test_loader:
                rasters = rasters.to(device, dtype=torch.bfloat16)
                context = context.to(device, dtype=torch.bfloat16)
                boreholes = boreholes.to(device, dtype=torch.bfloat16)

                B, C, H, W = rasters.shape

                rasters = rasters.reshape(B * C, 1, H, W)
                boreholes = boreholes.reshape(B * C, 1, H, W)
                context = context.reshape(B * C, 1, H, W)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    predicted = model(context, boreholes)

                    loss = F.mse_loss(predicted, rasters)

                test_loss += loss.item()

        print(f'Test Loss: {test_loss / len(train_loader)}')
        loss_dict['test'].append(test_loss / len(train_loader))

        pd.DataFrame(loss_dict).to_csv(f'{save_path}_loss.csv')

        if test_loss < best_loss:
            best_loss = test_loss

            torch.save(
                {
                    'epoch': epoch + 1,
                    'model': model._orig_mod.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': best_loss,
                    'max_size': raster_size,
                    'patch_size': patch_size,
                    'depth': depth,
                    'mlp_dim': mlp_dim,
                },
                f'{save_path}.mdl'
            )