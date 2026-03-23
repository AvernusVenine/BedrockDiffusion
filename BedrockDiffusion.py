from diffusers import DDPMScheduler
import numpy as np
import Data
from Data import BedrockDataset, collate_fn
from Encoder import Encoder
from Embedder import Embedder
from RecurrentUNet import RecurrentUNet
import torch
import torch.nn as nn

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
    raster_size = 256
    cross_attention_dim = 768

    # =============================
    # DATA PREPROCESSING
    # =============================

    rasters, context = Data.load_rasters(data_path)
    data, context, scaler_dict = Data.create_data(rasters, context, count=1000)

    data = sanitise_input(data)

    dataset = BedrockDataset(data, context, scaler_dict, raster_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    geo_context_encoder = Encoder(in_channels=3, cross_attention_dim=cross_attention_dim).to(device)
    geo_context_encoder = torch.compile(geo_context_encoder)

    unet = RecurrentUNet().to(device)
    unet.enable_gradient_checkpointing()
    unet = torch.compile(unet)

    formation_embedder = Embedder(in_features=1, out_features=cross_attention_dim).to(device)
    formation_embedder = torch.compile(formation_embedder)

    borehole_encoder = Encoder(in_channels=4, cross_attention_dim=cross_attention_dim).to(device)
    borehole_encoder = torch.compile(borehole_encoder)

    optimizer = torch.optim.AdamW(
        list(geo_context_encoder.parameters()) + list(unet.parameters()) + list(formation_embedder.parameters()) +
        list(borehole_encoder.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    best_loss = np.inf

    loss_dict = {'train': [], 'test': []}

    for epoch in range(max_epochs):
        print(f'Epoch {epoch+1}')

        geo_context_encoder.train()
        unet.train()
        formation_embedder.train()
        borehole_encoder.train()

        train_loss = 0.0
        for _ in _:

            pass

