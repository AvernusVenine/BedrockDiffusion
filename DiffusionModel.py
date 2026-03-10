import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from diffusers import UNet2DConditionModel, DDPMScheduler
import Data
from Data import BedrockDataset

def create_existence_mask(data):

    masks = []
    for idx in range(int(data.shape[1]/2)):
        top = torch.isnan(data[:, 2*idx, :, :])
        base = torch.isnan(data[:, 2*idx + 1, :, :])

        existence = ~(top & base)
        masks.append(existence.float().unsqueeze(1))

    return torch.cat(masks, dim=1)

def sanitise_input(data):
    """
    Replaces all nan elevation values with the highest elevation value in at a given location, effectively setting
    the formation thickness to 0 meaning it does not exist
    :param data: Data tensor
    :return: Sanitised data tensor
    """
    elevation_max = np.nanmax(data, axis=-1, keepdims=True)

    data = np.where(np.isnan(data), elevation_max, data)

    return torch.tensor(data)

def train_model(data_path, save_path, max_epochs=15, lr=1e-3):

    rasters, elevation = Data.load_rasters(data_path)
    data, scaler = Data.create_data(rasters, elevation, count=1000)

    data = sanitise_input(data)

    dataset = BedrockDataset(data[:, :, :, :len(rasters)], data[:, :, :, len(rasters):])

    train_size = int(0.8 * len(dataset))
    test_size = int(len(dataset) - train_size)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet2DConditionModel(
        sample_size=200,
        in_channels=2*len(top_rasters),
        out_channels=3*len(top_rasters),
    )

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

    for epoch in range(max_epochs):
        print(f'Epoch {epoch+1}')

        model.train()

        train_loss = 0
        for data, context in train_loader:
            data = data.to(device)
            context = data.to(device)

            noise = torch.randn(data.shape, device=device)
            timestamps = torch.randint(
                0,
            )

        pass