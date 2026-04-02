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
    data_count = 5

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

    rasters, context, boreholes = dataset[0]

    rasters = torch.from_numpy(rasters)
    context = torch.from_numpy(context)

    model.eval()

    with torch.no_grad():
        context = context.to(device, dtype=torch.float32).reshape(1, 1, raster_size, raster_size)
        boreholes = boreholes.to(device, dtype=torch.float32).reshape(1, 1, raster_size, raster_size)

        predicted = model(context, boreholes)

    predicted = predicted.squeeze().cpu().float().numpy()
    actual = rasters.numpy()
    elevation = context.squeeze().cpu().float().numpy()
    borehole_map = boreholes.squeeze().cpu().float().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(elevation, cmap='terrain')
    axes[0].set_title('Elevation')

    axes[1].imshow(predicted, cmap='terrain')
    axes[1].set_title('Predicted')

    im2 = axes[2].imshow(actual, cmap='terrain')
    axes[2].set_title('Actual')
    plt.colorbar(im2, ax=axes[2])

    axes[3].imshow(borehole_map != -1, cmap='binary')
    axes[3].set_title('Boreholes')

    plt.savefig(f'{save_path}.png')

def train_transformer(data_path, save_path, lr=1e-4, max_epochs=100):
    raster_size = 256
    patch_size = 16
    embed_dim = 768
    data_count = 1500
    formations = ['omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh', 'opod']

    # =============================
    # DATA PREPROCESSING
    # =============================

    rasters, context = Data.load_rasters(data_path)
    scaler_dict = Data.create_scaler_dict(rasters, context)

    for k, v in scaler_dict.items():
        joblib.dump(v, f'{save_path}_{k}.scl')

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

    train_size = int(0.8 * len(dataset))
    test_size = int(len(dataset) - train_size)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

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
        depth=6,
        mlp_dim=1024,
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
            rasters = rasters.to(device, dtype=torch.bfloat16).unsqueeze(1)
            context = context.to(device, dtype=torch.bfloat16).unsqueeze(1)
            boreholes = boreholes.to(device, dtype=torch.bfloat16).unsqueeze(1)

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
                rasters = rasters.to(device, dtype=torch.bfloat16).unsqueeze(1)
                context = context.to(device, dtype=torch.bfloat16).unsqueeze(1)
                boreholes = boreholes.to(device, dtype=torch.bfloat16).unsqueeze(1)

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
                },
                f'{save_path}.mdl'
            )