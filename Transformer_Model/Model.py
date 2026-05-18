import numpy as np
import Data
from Transformer_Model.Transformer import BedrockTransformer
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import time

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

def augment_batch(elevation, top_rasters, base_rasters, alphaearth, rng):
    k = int(rng.integers(0, 3))
    flip = rng.random() < 0.5

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

def prepare_batch(elevation, top_rasters, base_rasters, alphaearth, dataset, bh_count, device, rng, augment=True, size=None):
    if augment:
        elevation, top_rasters, base_rasters, alphaearth = augment_batch(elevation, top_rasters, base_rasters, alphaearth, rng)

    B, F, _, H, W = elevation.shape

    elevation = elevation[:, 0].unsqueeze(1).expand(B, F, 1, H, W).reshape(B*F, 1, H, W)
    alphaearth = alphaearth[:, 0].unsqueeze(1).expand(B, F, 64, H, W).reshape(B*F, 64, H, W)

    B, F, _, H, W = top_rasters.shape

    boreholes = [dataset.select_boreholes(top, base, count=bh_count, rng=rng, size=size) for top, base in zip(top_rasters, base_rasters)]
    boreholes = torch.stack(boreholes).reshape(B*F, bh_count, 5)

    top_rasters = top_rasters.reshape(B*F, 1, H, W)
    base_rasters = base_rasters.reshape(B*F, 1, H, W)

    thickness = (top_rasters - base_rasters).nan_to_num(0.0)
    existence = (thickness > 0).float()

    elevation = elevation.to(device, dtype=torch.float32)
    alphaearth = alphaearth.to(device, dtype=torch.float32)
    existence = existence.to(device, dtype=torch.float32)
    top_rasters = top_rasters.to(device, dtype=torch.float32)
    boreholes = boreholes.to(device, dtype=torch.float32)

    return elevation, top_rasters, existence, alphaearth, boreholes

def test(data_path, model_path, save_path, count=20, total_size=None, seed=0, raster_size=None):
    model_dict = torch.load(f'{model_path}.mdl')

    if raster_size is None:
        raster_size = model_dict['raster_size']
    patch_size = model_dict['patch_size']
    embed_dim = model_dict['embed_dim']
    encoder_depth = model_dict['depth']
    mlp_dim = model_dict['mlp_dim']

    print('Model loaded')

    rng = np.random.default_rng(seed)

    formations = [
        'kwnd', 'dlp', 'dspl', 'omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh', 'opod',
        'cp', 'cjdn', 'cstl', 'ctcg', 'cwoc', 'cecr', 'cmts'
    ]

    counties = [Data.CountySource(p, formations) for p in data_path]

    scaler_dict = {'elevation': joblib.load(f'{model_path}_elevation.scl')}

    dataset = Data.MultiCountyDataset(counties, scaler_dict, 1, total_size, 1.0)
    dataset.generate_indices(rng)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state = {k: v for k, v in model_dict['model'].items() if not k.endswith('pe')}

    model = BedrockTransformer(
        raster_size,
        patch_size,
        embed_dim,
        num_heads=8,
        depth=encoder_depth,
        mlp_dim=mlp_dim
    ).to(device)
    model.load_state_dict(state)

    model.eval()

    elevation, top_rasters, base_rasters, alphaearth, formations = dataset.get_full(0)

    elevation = elevation.unsqueeze(0)
    top_rasters = top_rasters.unsqueeze(0)
    base_rasters = base_rasters.unsqueeze(0)
    alphaearth = alphaearth.unsqueeze(0)

    elevation, top_rasters, _, alphaearth, boreholes = prepare_batch(
        elevation, top_rasters, base_rasters, alphaearth, dataset, count, device, rng
    )

    base_rasters = base_rasters.reshape(elevation.shape).to(device, dtype=torch.float32)

    B, C, H, W = top_rasters.shape
    predicted = torch.zeros(B, C, H, W, device=device)

    n = total_size // raster_size

    print('Evaluating...')
    t = time.time()

    for i in range(n):
        for j in range(n):
            r0, r1 = i * raster_size, (i + 1) * raster_size
            c0, c1 = j * raster_size, (j + 1) * raster_size

            elev_tile = elevation[:, :, r0:r1, c0:c1]
            t_tile = top_rasters[:, :, r0:r1, c0:c1].unsqueeze(1)
            b_tile = base_rasters[:, :, r0:r1, c0:c1].unsqueeze(1)
            ae_tile = alphaearth[:, :, r0:r1, c0:c1]

            boreholes = [dataset.select_boreholes(top, base, count=count, size=raster_size, rng=rng) for top, base in zip(t_tile, b_tile)]
            boreholes = torch.stack(boreholes).reshape(B, count, 5)
            boreholes = boreholes.to(device, dtype=torch.float32)

            with torch.no_grad():
                tile_pred = model(elev_tile, boreholes, ae_tile)

            predicted[:, :, r0:r1, c0:c1] = tile_pred

    print(' Done')
    print(time.time() - t)

    elevation = elevation.squeeze(1).cpu().float().numpy()
    predicted = predicted.squeeze(1).cpu().float().numpy()
    top_rasters = top_rasters.squeeze(1).cpu().float().numpy()

    thickness = np.zeros_like(top_rasters)
    true_thickness = np.zeros_like(top_rasters)

    for idx in range(len(predicted)-1):
        t = predicted[idx, :, :] - predicted[idx+1, :, :]
        thickness[idx] = np.where(t < 1e-3, np.nan, t)

        t = top_rasters[idx, :, :] - top_rasters[idx+1, :, :]
        true_thickness[idx] = np.where(t < 1e-3, np.nan, t)

    thickness[len(predicted)-1, :, :] = predicted[len(predicted)-1, :, :]

    c_elev = counties[0].elevation
    c_elev = scaler_dict['elevation'].transform(c_elev.reshape(-1, 1)).reshape(c_elev.shape)
    elev_vmin, elev_vmax = np.nanmin(c_elev), np.nanmax(c_elev)

    all_elevs = np.concatenate([predicted.ravel(), top_rasters.ravel()])
    vmin, vmax = np.nanmin(all_elevs), np.nanmax(all_elevs)

    all_thck = np.concatenate([thickness.ravel(), true_thickness.ravel()])
    tmin, tmax = np.nanmin(all_thck), np.nanmax(all_thck)

    print('Creating figures...')

    for idx in range(len(predicted)):
        fig, axes = plt.subplots(ncols=5, figsize=(25, 5))

        plt.title(f'{formations[idx].capitalize()}')

        axes[0].imshow(elevation[idx], cmap='terrain', vmin=elev_vmin, vmax=elev_vmax)
        axes[0].set_title('Elevation')

        axes[1].imshow(predicted[idx], cmap='terrain', vmin=vmin, vmax=vmax)
        axes[1].set_title('Predicted')

        axes[2].imshow(top_rasters[idx], cmap='terrain', vmin=vmin, vmax=vmax)
        axes[2].set_title('Ground Truth')

        axes[3].imshow(thickness[idx], cmap='Blues', vmin=tmin, vmax=tmax)
        axes[3].set_title('Predicted Thickness')

        axes[4].imshow(true_thickness[idx], cmap='Blues', vmin=tmin, vmax=tmax)
        axes[4].set_title('True Thickness')

        plt.savefig(f'{save_path}_{formations[idx]}.png')
        plt.close()

        print(f' {formations[idx]}')

def test_existence(data_path, model_path, save_path, count=20, total_size=None, seed=0, raster_size=None):
    model_dict = torch.load(f'{model_path}.mdl')

    if raster_size is None:
        raster_size = model_dict['raster_size']
    patch_size = model_dict['patch_size']
    embed_dim = model_dict['embed_dim']
    encoder_depth = model_dict['depth']
    mlp_dim = model_dict['mlp_dim']

    print('Model loaded')

    rng = np.random.default_rng(seed)

    formations = [
        'kwnd', 'dlp', 'dspl', 'omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh', 'opod',
        'cp', 'cjdn', 'cstl', 'ctcg', 'cwoc', 'cecr', 'cmts'
    ]

    counties = [Data.CountySource(p, formations) for p in data_path]

    scaler_dict = {'elevation': joblib.load(f'{model_path}_elevation.scl')}

    dataset = Data.MultiCountyDataset(counties, scaler_dict, 1, total_size, 1.0)
    dataset.generate_indices(rng)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state = {k: v for k, v in model_dict['model'].items() if not k.endswith('pe')}

    model = BedrockTransformer(
        raster_size,
        patch_size,
        embed_dim,
        num_heads=8,
        depth=encoder_depth,
        mlp_dim=mlp_dim
    ).to(device)
    model.load_state_dict(state)

    model.eval()

    elevation, top_rasters, base_rasters, alphaearth, formations = dataset.get_full(0)

    elevation = elevation.unsqueeze(0)
    top_rasters = top_rasters.unsqueeze(0)
    base_rasters = base_rasters.unsqueeze(0)
    alphaearth = alphaearth.unsqueeze(0)

    elevation, top_rasters, existence, alphaearth, boreholes = prepare_batch(
        elevation, top_rasters, base_rasters, alphaearth, dataset, count, device, rng
    )

    base_rasters = base_rasters.reshape(elevation.shape).to(device, dtype=torch.float32)

    B, C, H, W = top_rasters.shape
    predicted = torch.zeros(B, C, H, W, device=device)

    n = total_size // raster_size

    print('Evaluating...')
    t = time.time()

    for i in range(n):
        for j in range(n):
            r0, r1 = i * raster_size, (i + 1) * raster_size
            c0, c1 = j * raster_size, (j + 1) * raster_size

            elev_tile = elevation[:, :, r0:r1, c0:c1]
            t_tile = top_rasters[:, :, r0:r1, c0:c1].unsqueeze(1)
            b_tile = base_rasters[:, :, r0:r1, c0:c1].unsqueeze(1)
            ae_tile = alphaearth[:, :, r0:r1, c0:c1]

            boreholes = [dataset.select_boreholes(top, base, count=count, size=raster_size, rng=rng) for top, base in zip(t_tile, b_tile)]
            boreholes = torch.stack(boreholes).reshape(B, count, 5)
            boreholes = boreholes.to(device, dtype=torch.float32)

            with torch.no_grad():
                tile_pred = model(elev_tile, boreholes, ae_tile)

            predicted[:, :, r0:r1, c0:c1] = tile_pred

    print(' Done')
    print(time.time() - t)

    elevation = elevation.squeeze(1).cpu().float().numpy()
    predicted = predicted.squeeze(1).cpu().float().numpy()
    existence = existence.squeeze(1).cpu().float().numpy()

    predicted = (predicted >= 0.75)

    c_elev = counties[0].elevation
    c_elev = scaler_dict['elevation'].transform(c_elev.reshape(-1, 1)).reshape(c_elev.shape)
    elev_vmin, elev_vmax = np.nanmin(c_elev), np.nanmax(c_elev)

    print('Creating figures...')

    for idx in range(len(predicted)):
        fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

        plt.title(f'{formations[idx].capitalize()}')

        axes[0].imshow(elevation[idx], cmap='terrain', vmin=elev_vmin, vmax=elev_vmax)
        axes[0].set_title('Elevation')

        axes[1].imshow(predicted[idx], cmap='binary')
        axes[1].set_title('Predicted')

        axes[2].imshow(existence[idx], cmap='binary')
        axes[2].set_title('Ground Truth')

        plt.savefig(f'{save_path}_{formations[idx]}.png')
        plt.close()

        print(f' {formations[idx]}')

def test_center(data_path, model_path, save_path, count=20, total_size=None, seed=0, raster_size=None):
    model_dict = torch.load(f'{model_path}.mdl')

    if raster_size is None:
        raster_size = model_dict['raster_size']
    patch_size = model_dict['patch_size']
    embed_dim = model_dict['embed_dim']
    encoder_depth = model_dict['depth']
    mlp_dim = model_dict['mlp_dim']

    print('Model loaded')

    rng = np.random.default_rng(seed)

    formations = [
        'kwnd', 'dlp', 'dspl', 'omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh', 'opod',
        'cp', 'cjdn', 'cstl', 'ctcg', 'cwoc', 'cecr', 'cmts'
    ]

    counties = [Data.CountySource(p, formations) for p in data_path]

    scaler_dict = {'elevation': joblib.load(f'{model_path}_elevation.scl')}

    dataset = Data.MultiCountyDataset(counties, scaler_dict, 1, total_size, 1.0)
    dataset.generate_indices(rng)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state = {k: v for k, v in model_dict['model'].items() if not k.endswith('pe')}

    model = BedrockTransformer(
        raster_size,
        patch_size,
        embed_dim,
        num_heads=8,
        depth=encoder_depth,
        mlp_dim=mlp_dim
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    elevation, top_rasters, base_rasters, alphaearth, formations = dataset.get_full(0)

    elevation   = elevation.unsqueeze(0)
    top_rasters = top_rasters.unsqueeze(0)
    base_rasters = base_rasters.unsqueeze(0)
    alphaearth  = alphaearth.unsqueeze(0)

    elevation, top_rasters, _, alphaearth, boreholes = prepare_batch(
        elevation, top_rasters, base_rasters, alphaearth, dataset, count, device, rng
    )

    base_rasters = base_rasters.reshape(elevation.shape).to(device, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Tiling geometry
    # ------------------------------------------------------------------ #
    crop_size = 64          # pixels we keep from each tile's center
    overlap   = 16          # pixels of soft blending overlap between crops
    margin    = (raster_size - crop_size) // 2   # context border discarded per side
    stride    = crop_size - overlap               # 48 — distance between tile origins

    B, C, H, W = top_rasters.shape

    # Valid output region (same shrinkage as before: one margin on each side)
    out_H = H - 2 * margin
    out_W = W - 2 * margin

    # ------------------------------------------------------------------ #
    # 1-D tent (triangle) weight: rises from 0 at edges to 1 at center.
    # Over a crop_size window with `overlap` soft ramp on each side,
    # overlapping tents sum to ~1 everywhere, giving a smooth blend.
    # ------------------------------------------------------------------ #
    def make_weight_1d(size, ramp):
        w = torch.ones(size, dtype=torch.float32)
        ramp_vals = torch.linspace(0, 1, ramp + 2)[1:-1]   # exclude 0 and 1 endpoints
        w[:ramp]  = ramp_vals
        w[-ramp:] = ramp_vals.flip(0)
        return w

    wy = make_weight_1d(crop_size, overlap).to(device)   # (crop_size,)
    wx = make_weight_1d(crop_size, overlap).to(device)   # (crop_size,)
    weight_2d = wy[:, None] * wx[None, :]                # (crop_size, crop_size)

    # ------------------------------------------------------------------ #
    # Helper: run one full pass of tiled inference and accumulate into
    # weighted sum + weight buffers.
    # offset_r / offset_c shift the tile grid within the input space.
    # ------------------------------------------------------------------ #
    def run_pass(offset_r, offset_c, accum, accum_w):
        """
        Tiles start at (offset_r + i*stride, offset_c + j*stride) in input coords.
        Their center-crop maps to output coords after subtracting `margin`.
        Only tiles whose input window and output window fall fully in-bounds
        are processed.
        """
        i = 0
        while True:
            in_r0 = offset_r + i * stride
            in_r1 = in_r0 + raster_size
            if in_r1 > H:
                break

            # Where this crop's top edge lands in output space
            out_r0 = in_r0 + margin - margin   # == in_r0  (before global margin shift)
            # Remap to output coords: output pixel 0 == input pixel `margin`
            out_r0 = in_r0 + margin - margin    # simplifies to in_r0
            # Re-derive cleanly:
            crop_in_r0 = in_r0 + margin         # first kept input row
            out_r0     = crop_in_r0 - margin    # subtract the global output offset
            out_r1     = out_r0 + crop_size

            if out_r0 < 0 or out_r1 > out_H:
                i += 1
                continue

            j = 0
            while True:
                in_c0 = offset_c + j * stride
                in_c1 = in_c0 + raster_size
                if in_c1 > W:
                    break

                crop_in_c0 = in_c0 + margin
                out_c0     = crop_in_c0 - margin
                out_c1     = out_c0 + crop_size

                if out_c0 < 0 or out_c1 > out_W:
                    j += 1
                    continue

                elev_tile = elevation  [:, :, in_r0:in_r1, in_c0:in_c1]
                t_tile    = top_rasters[:, :, in_r0:in_r1, in_c0:in_c1].unsqueeze(1)
                b_tile    = base_rasters[:, :, in_r0:in_r1, in_c0:in_c1].unsqueeze(1)
                ae_tile   = alphaearth [:, :, in_r0:in_r1, in_c0:in_c1]

                bh = [
                    dataset.select_boreholes(top, base, count=count, size=raster_size, rng=rng)
                    for top, base in zip(t_tile, b_tile)
                ]
                bh = torch.stack(bh).reshape(B, count, 5).to(device, dtype=torch.float32)

                with torch.no_grad():
                    tile_pred = model(elev_tile, bh, ae_tile)

                # Center crop
                center = tile_pred[:, :, margin:margin + crop_size, margin:margin + crop_size]

                # Weighted accumulation  (weight_2d broadcast over B and C)
                accum  [:, :, out_r0:out_r1, out_c0:out_c1] += center * weight_2d
                accum_w[:, :, out_r0:out_r1, out_c0:out_c1] += weight_2d

                j += 1
            i += 1

    # ------------------------------------------------------------------ #
    # Two passes: normal grid and shifted grid (shift by stride//2)
    # ------------------------------------------------------------------ #
    accum   = torch.zeros(B, C, out_H, out_W, device=device)
    accum_w = torch.zeros(B, C, out_H, out_W, device=device)

    shift = stride // 2   # 24 pixels — half a stride

    print('Evaluating (pass 1 — normal grid)...')
    t0 = time.time()
    run_pass(0,     0,     accum, accum_w)

    print('Evaluating (pass 2 — shifted grid)...')
    run_pass(shift, shift, accum, accum_w)

    # Normalize: divide accumulated predictions by accumulated weights.
    # Guard against any uncovered pixels (weight == 0) — shouldn't happen
    # with two passes but safe to handle.
    predicted = torch.where(accum_w > 0, accum / accum_w, torch.zeros_like(accum))

    print(' Done')
    print(time.time() - t0)

    # Crop ground-truth and elevation to the same output extent
    elevation   = elevation  [:, :, margin:margin + out_H, margin:margin + out_W]
    top_rasters = top_rasters[:, :, margin:margin + out_H, margin:margin + out_W]

    elevation   = elevation.squeeze(1).cpu().float().numpy()
    predicted   = predicted.squeeze(1).cpu().float().numpy()
    top_rasters = top_rasters.squeeze(1).cpu().float().numpy()

    thickness      = np.zeros_like(top_rasters)
    true_thickness = np.zeros_like(top_rasters)

    for idx in range(len(predicted) - 1):
        t = predicted[idx] - predicted[idx + 1]
        thickness[idx] = np.where(t < 5e-3, np.nan, t)

        t = top_rasters[idx] - top_rasters[idx + 1]
        true_thickness[idx] = np.where(t < 0, np.nan, t)

    thickness[len(predicted) - 1] = predicted[len(predicted) - 1]

    c_elev = counties[0].elevation
    c_elev = scaler_dict['elevation'].transform(c_elev.reshape(-1, 1)).reshape(c_elev.shape)
    elev_vmin, elev_vmax = np.nanmin(c_elev), np.nanmax(c_elev)

    all_elevs = np.concatenate([predicted.ravel(), top_rasters.ravel()])
    vmin, vmax = np.nanmin(all_elevs), np.nanmax(all_elevs)

    all_thck = np.concatenate([thickness.ravel(), true_thickness.ravel()])
    tmin, tmax = np.nanmin(all_thck), np.nanmax(all_thck)

    print('Creating figures...')

    for idx in range(len(predicted)):
        fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
        plt.title(f'{formations[idx].capitalize()}')

        axes[0].imshow(elevation[idx],      cmap='terrain', vmin=elev_vmin, vmax=elev_vmax)
        axes[0].set_title('Elevation')

        axes[1].imshow(predicted[idx],      cmap='terrain', vmin=vmin, vmax=vmax)
        axes[1].set_title('Predicted')

        axes[2].imshow(top_rasters[idx],    cmap='terrain', vmin=vmin, vmax=vmax)
        axes[2].set_title('Ground Truth')

        plt.savefig(f'{save_path}_{formations[idx]}.png')
        plt.close()

        print(f' {formations[idx]}')

def test_sw(data_path, model_path, save_path, count=20, total_size=None, seed=0, raster_size=None):
    model_dict = torch.load(f'{model_path}.mdl')

    if raster_size is None:
        raster_size = model_dict['raster_size']
    patch_size    = model_dict['patch_size']
    embed_dim     = model_dict['embed_dim']
    encoder_depth = model_dict['depth']
    mlp_dim       = model_dict['mlp_dim']

    print('Model loaded')

    rng = np.random.default_rng(seed)

    formations = [
        'kwnd', 'dlp', 'dspl', 'omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh', 'opod',
        'cp', 'cjdn', 'cstl', 'ctcg', 'cwoc', 'cecr', 'cmts'
    ]

    counties    = [Data.CountySource(p, formations) for p in data_path]
    scaler_dict = {'elevation': joblib.load(f'{model_path}_elevation.scl')}

    dataset = Data.MultiCountyDataset(counties, scaler_dict, 1, total_size, 1.0)
    dataset.generate_indices(rng)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state = {k: v for k, v in model_dict['model'].items() if not k.endswith('pe')}

    model = BedrockTransformer(
        raster_size, patch_size, embed_dim,
        num_heads=8, depth=encoder_depth, mlp_dim=mlp_dim
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    elevation, top_rasters, base_rasters, alphaearth, formations = dataset.get_full(0)

    elevation    = elevation.unsqueeze(0)
    top_rasters  = top_rasters.unsqueeze(0)
    base_rasters = base_rasters.unsqueeze(0)
    alphaearth   = alphaearth.unsqueeze(0)

    elevation, top_rasters, _, alphaearth, boreholes = prepare_batch(
        elevation, top_rasters, base_rasters, alphaearth, dataset, count, device, rng
    )

    base_rasters = base_rasters.reshape(elevation.shape).to(device, dtype=torch.float32)

    B, C, H, W = top_rasters.shape

    overlap = 16
    stride  = raster_size - overlap

    def make_weight_1d(size, ramp):
        w = torch.ones(size, dtype=torch.float32)
        ramp_vals = torch.linspace(0, 1, ramp + 2)[1:-1]
        w[:ramp]  = ramp_vals
        w[-ramp:] = ramp_vals.flip(0)
        return w

    wy = make_weight_1d(raster_size, overlap).to(device)
    wx = make_weight_1d(raster_size, overlap).to(device)
    weight_2d = wy[:, None] * wx[None, :]

    accum   = torch.zeros(B, C, H, W, device=device)
    accum_w = torch.zeros(B, C, H, W, device=device)

    def run_pass(offset_r, offset_c):
        max_r1, max_c1 = 0, 0
        i = 0
        while True:
            r0 = offset_r + i * stride
            r1 = r0 + raster_size
            if r1 > H:
                break

            j = 0
            while True:
                c0 = offset_c + j * stride
                c1 = c0 + raster_size
                if c1 > W:
                    break

                elev_tile = elevation   [:, :, r0:r1, c0:c1]
                t_tile    = top_rasters [:, :, r0:r1, c0:c1].unsqueeze(1)
                b_tile    = base_rasters[:, :, r0:r1, c0:c1].unsqueeze(1)
                ae_tile   = alphaearth  [:, :, r0:r1, c0:c1]

                bh = [
                    dataset.select_boreholes(top, base, count=count, size=raster_size, rng=rng)
                    for top, base in zip(t_tile, b_tile)
                ]
                bh = torch.stack(bh).reshape(B, count, 5).to(device, dtype=torch.float32)

                with torch.no_grad():
                    tile_pred = model(elev_tile, bh, ae_tile)

                accum  [:, :, r0:r1, c0:c1] += tile_pred * weight_2d
                accum_w[:, :, r0:r1, c0:c1] += weight_2d

                max_c1 = max(max_c1, c1)
                j += 1

            max_r1 = max(max_r1, r1)
            i += 1

        return max_r1, max_c1

    shift = stride // 2

    print('Evaluating (pass 1 — normal grid)...')
    t0 = time.time()
    max_r1, max_c1 = run_pass(0, 0)

    print('Evaluating (pass 2 — shifted grid)...')
    # Shifted pass may cover slightly less; take the min so every output
    # pixel is guaranteed to have weight from both passes
    r1_b, c1_b = run_pass(shift, shift)
    max_r1 = min(max_r1, r1_b)
    max_c1 = min(max_c1, c1_b)

    predicted = accum[:, :, :max_r1, :max_c1] / accum_w[:, :, :max_r1, :max_c1]

    print(' Done')
    print(time.time() - t0)

    elevation   = elevation  [:, :, :max_r1, :max_c1].squeeze(1).cpu().float().numpy()
    top_rasters = top_rasters[:, :, :max_r1, :max_c1]
    predicted   = predicted.squeeze(1).cpu().float().numpy()
    top_rasters = top_rasters.squeeze(1).cpu().float().numpy()

    thickness      = np.zeros_like(top_rasters)
    true_thickness = np.zeros_like(top_rasters)

    for idx in range(len(predicted) - 1):
        t = predicted[idx] - predicted[idx + 1]
        thickness[idx] = np.where(t < 5e-2, np.nan, t)

        t = top_rasters[idx] - top_rasters[idx + 1]
        true_thickness[idx] = np.where(t <= 1e-5, np.nan, t)

    thickness[len(predicted) - 1] = predicted[len(predicted) - 1]

    c_elev = counties[0].elevation
    c_elev = scaler_dict['elevation'].transform(c_elev.reshape(-1, 1)).reshape(c_elev.shape)
    elev_vmin, elev_vmax = np.nanmin(c_elev), np.nanmax(c_elev)

    all_elevs = np.concatenate([predicted.ravel(), top_rasters.ravel()])
    vmin, vmax = np.nanmin(all_elevs), np.nanmax(all_elevs)

    diff = np.abs(predicted - top_rasters)
    tmin, tmax = np.nanmin(diff), np.nanmax(diff)

    print(tmax)

    print('Creating figures...')

    for idx in range(len(predicted)):
        fig, axes = plt.subplots(ncols=4, figsize=(20, 5))
        plt.title(f'{formations[idx].capitalize()}')

        axes[0].imshow(elevation[idx],      cmap='terrain', vmin=elev_vmin, vmax=elev_vmax)
        axes[0].set_title('Elevation')

        axes[1].imshow(predicted[idx],      cmap='terrain', vmin=vmin, vmax=vmax)
        axes[1].set_title('Predicted')

        axes[2].imshow(top_rasters[idx],    cmap='terrain', vmin=vmin, vmax=vmax)
        axes[2].set_title('Ground Truth')

        axes[3].imshow(diff[idx],      cmap='Blues', vmin=tmin, vmax=tmax)
        axes[3].set_title('Difference')

        plt.savefig(f'{save_path}_{formations[idx]}.png')
        plt.close()

        print(f' {formations[idx]}')

def train(data_path, save_path, lr=1e-4, max_epochs=100, seed=0, load=False):
    if load:
        model_dict = torch.load(f'{save_path}.mdl')

        raster_size = model_dict['raster_size']
        patch_size = model_dict['patch_size']
        embed_dim = model_dict['embed_dim']
        mlp_dim = model_dict['mlp_dim']
        encoder_depth = model_dict['depth']
    else:
        model_dict = None

        raster_size = 128
        patch_size = 16
        embed_dim = 512
        mlp_dim = 1024
        encoder_depth = 12

    data_count = 3000

    bh_max = 200
    bh_min = 20

    rng = np.random.default_rng(seed)

    # =============================
    # DATA PREPROCESSING
    # =============================

    print('Loading rasters')

    formations = [
        'kwnd', 'dlp', 'dspl', 'omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh', 'opod',
        'cp', 'cjdn', 'cstl', 'ctcg', 'cwoc', 'cecr', 'cmts'
    ]

    counties = [Data.CountySource(p, formations) for p in data_path]

    if load:
        scaler_dict = {'elevation': joblib.load(f'{save_path}_elevation.scl')}
    else:
        scaler_dict = Data.create_global_scaler_dict(counties)

        print('Generating scalers')

        for k, v in scaler_dict.items():
            joblib.dump(v, f'{save_path}_{k}.scl')

    # =============================
    # DATASET CONSTRUCTION
    # =============================

    dataset = Data.MultiCountyDataset(counties, scaler_dict, data_count, raster_size)

    test_dataset = dataset.split_test(int(data_count * 0.1), rng)
    train_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=0)

    # =============================
    # MODEL CONSTRUCTION
    # =============================

    print('Constructing Model...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BedrockTransformer(
        raster_size,
        patch_size,
        embed_dim,
        num_heads=8,
        depth=encoder_depth,
        mlp_dim=mlp_dim,
    ).to(device)
    if load:
        model.load_state_dict(model_dict['model'])

    print('     constructed TERRA')

    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    if load:
        optimizer.load_state_dict(model_dict['optimizer'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    test_dataset.generate_indices(rng)

    patience = 0
    best_loss = np.inf

    if load:
        best_loss = float(model_dict['loss'])

    loss_dict = {'train': [], 'test': []}

    for epoch in range(max_epochs):
        print(f'Epoch {epoch+1}')

        # =============================
        # TRAIN LOOP
        # =============================

        model.train()

        train_loss = 0.0
        train_dataset.generate_indices(rng)

        for elevation, top_rasters, base_rasters, alphaearth in train_loader:
            count = int(rng.integers(bh_min, bh_max + 1))

            elevation, top_rasters, existence, alphaearth, boreholes = prepare_batch(
                elevation, top_rasters, base_rasters, alphaearth, train_dataset, count, device, rng)

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
            for idx, (elevation, top_rasters, base_rasters, alphaearth) in enumerate(test_loader):
                count = int(rng.integers(bh_min, bh_max + 1))

                elevation, top_rasters, existence, alphaearth, boreholes = prepare_batch(
                    elevation, top_rasters, base_rasters, alphaearth, train_dataset, count, device, rng, augment=False)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    predicted_elevation = model(elevation, boreholes, alphaearth)

                    loss = F.mse_loss(predicted_elevation, top_rasters)

                test_loss += loss.item()

        scheduler.step(test_loss)

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
                    'depth': encoder_depth,
                    'mlp_dim': mlp_dim,
                },
                f'{save_path}.mdl'
            )
        else:
            if patience > 20:
                return

            patience += 1


def train_existence(data_path, save_path, lr=1e-4, max_epochs=100, seed=0, load=False):
    if load:
        model_dict = torch.load(f'{save_path}.mdl')

        raster_size = model_dict['raster_size']
        patch_size = model_dict['patch_size']
        embed_dim = model_dict['embed_dim']
        mlp_dim = model_dict['mlp_dim']
        encoder_depth = model_dict['depth']
    else:
        model_dict = None

        raster_size = 128
        patch_size = 16
        embed_dim = 512
        mlp_dim = 1024
        encoder_depth = 12

    data_count = 3000

    bh_max = 200
    bh_min = 20

    rng = np.random.default_rng(seed)

    # =============================
    # DATA PREPROCESSING
    # =============================

    print('Loading rasters')

    formations = [
        'kwnd', 'dlp', 'dspl', 'omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh', 'opod',
        'cp', 'cjdn', 'cstl', 'ctcg', 'cwoc', 'cecr', 'cmts'
    ]

    counties = [Data.CountySource(p, formations) for p in data_path]

    if load:
        scaler_dict = {'elevation': joblib.load(f'{save_path}_elevation.scl')}
    else:
        scaler_dict = Data.create_global_scaler_dict(counties)

        print('Generating scalers')

        for k, v in scaler_dict.items():
            joblib.dump(v, f'{save_path}_{k}.scl')

    # =============================
    # DATASET CONSTRUCTION
    # =============================

    dataset = Data.MultiCountyDataset(counties, scaler_dict, data_count, raster_size)

    test_dataset = dataset.split_test(int(data_count * 0.1), rng)
    train_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=0)

    # =============================
    # MODEL CONSTRUCTION
    # =============================

    print('Constructing Model...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BedrockTransformer(
        raster_size,
        patch_size,
        embed_dim,
        num_heads=8,
        depth=encoder_depth,
        mlp_dim=mlp_dim,
    ).to(device)
    if load:
        model.load_state_dict(model_dict['model'])

    print('     constructed TERRA')

    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    if load:
        optimizer.load_state_dict(model_dict['optimizer'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    test_dataset.generate_indices(rng)

    patience = 0
    best_loss = np.inf

    if load:
        best_loss = float(model_dict['loss'])

    loss_dict = {'train': [], 'test': []}

    for epoch in range(max_epochs):
        print(f'Epoch {epoch + 1}')

        # =============================
        # TRAIN LOOP
        # =============================

        model.train()

        train_loss = 0.0
        train_dataset.generate_indices(rng)

        for elevation, top_rasters, base_rasters, alphaearth in train_loader:
            count = int(rng.integers(bh_min, bh_max + 1))

            elevation, top_rasters, existence, alphaearth, boreholes = prepare_batch(
                elevation, top_rasters, base_rasters, alphaearth, train_dataset, count, device, rng)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                predicted_existence= model(elevation, boreholes, alphaearth)

                loss = F.binary_cross_entropy_with_logits(predicted_existence, existence)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        print(f'Train Loss: {train_loss / (len(train_loader) * 4)}')
        loss_dict['train'].append(train_loss / (len(train_loader) * 4))

        # =============================
        # TEST LOOP
        # =============================

        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for idx, (elevation, top_rasters, base_rasters, alphaearth) in enumerate(test_loader):
                count = int(rng.integers(bh_min, bh_max + 1))

                elevation, top_rasters, existence, alphaearth, boreholes = prepare_batch(
                    elevation, top_rasters, base_rasters, alphaearth, train_dataset, count, device, rng, augment=False)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    predicted_existence = model(elevation, boreholes, alphaearth)

                    loss = F.binary_cross_entropy_with_logits(predicted_existence, existence)

                test_loss += loss.item()

        scheduler.step(test_loss)

        print(f'Test Loss: {test_loss / (len(test_loader) * 4)}')
        loss_dict['test'].append(test_loss / (len(test_loader) * 4))

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
                    'depth': encoder_depth,
                    'mlp_dim': mlp_dim,
                },
                f'{save_path}.mdl'
            )
        else:
            if patience > 20:
                return

            patience += 1


def fine_tune(data_path, save_path, lr=1e-6, max_epochs=100, seed=0):
    model_dict = torch.load(f'{save_path}.mdl')

    raster_size = model_dict['raster_size']
    patch_size = model_dict['patch_size']
    embed_dim = model_dict['embed_dim']
    mlp_dim = model_dict['mlp_dim']
    encoder_depth = model_dict['depth']

    data_count = 1000

    bh_max = 500
    bh_min = 50

    rng = np.random.default_rng(seed)

    print('Loading rasters')

    formations = [
        'kwnd', 'dlp', 'dspl', 'omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh', 'opod',
        'cp', 'cjdn', 'cstl', 'ctcg', 'cwoc', 'cecr', 'cmts'
    ]

    counties = [Data.CountySource(p, formations) for p in data_path]

    scaler_dict = {'elevation': joblib.load(f'{save_path}_elevation.scl')}

    # =============================
    # DATASET CONSTRUCTION
    # =============================

    dataset = Data.MultiCountyDataset(counties, scaler_dict, data_count, raster_size, max_f=8)

    test_dataset = dataset.split_test(int(data_count * 0.1), rng)
    train_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=0)

    # =============================
    # MODEL CONSTRUCTION
    # =============================

    print('Constructing Model...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BedrockTransformer(
        raster_size,
        patch_size,
        embed_dim,
        num_heads=8,
        depth=encoder_depth,
        mlp_dim=mlp_dim,
        use_checkpoint=True
    ).to(device)
    model.load_state_dict(model_dict['model'])

    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    print('     loaded TERRA')

    test_dataset.generate_indices(rng)

    patience = 0
    best_loss = float(model_dict['loss'])

    loss_dict = {'train': [], 'test': []}

    for epoch in range(max_epochs):
        print(f'Epoch {epoch+1}')

        # =============================
        # TRAIN LOOP
        # =============================

        model.train()

        train_loss = 0.0
        train_dataset.generate_indices(rng)

        for elevation, top_rasters, base_rasters, alphaearth in train_loader:
            count = int(rng.integers(bh_min, bh_max + 1))

            elevation, top_rasters, existence, alphaearth, boreholes = prepare_batch(
                    elevation, top_rasters, base_rasters, alphaearth, train_dataset, count, device, rng)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                predicted_elevation = model(elevation, boreholes, alphaearth)

                loss = F.mse_loss(predicted_elevation, top_rasters)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        print(f'Train Loss: {train_loss / (len(train_loader) * 8)}')
        loss_dict['train'].append(train_loss / (len(train_loader) * 8))

        # =============================
        # TEST LOOP
        # =============================

        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for idx, (elevation, top_rasters, base_rasters, alphaearth) in enumerate(test_loader):
                count = int(rng.integers(bh_min, bh_max + 1))

                elevation, top_rasters, existence, alphaearth, boreholes = prepare_batch(
                    elevation, top_rasters, base_rasters, alphaearth, train_dataset, count, device, rng, augment=False)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    predicted_elevation = model(elevation, boreholes, alphaearth)

                    loss = F.mse_loss(predicted_elevation, top_rasters)

                test_loss += loss.item()

        print(f'Test Loss: {test_loss / (len(test_loader) * 8)}')
        loss_dict['test'].append(test_loss / (len(test_loader) * 8))

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
                    'depth': encoder_depth,
                    'mlp_dim': mlp_dim,
                },
                f'{save_path}.mdl'
            )
        else:
            if patience > 20:
                return

            patience += 1