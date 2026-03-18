import numpy as np
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
import pandas as pd
import Data
from Data import BedrockDataset
from ContextEncoder import ContextEncoder
import joblib
import matplotlib.pyplot as plt
import DiffusionModel

def graph_loss(loss_path):
    """
    Graphs the loss
    :param loss_path:
    :return:
    """
    df = pd.read_csv(loss_path)
    df = df.set_index('Unnamed: 0')

    df.plot()
    plt.show()

def get_random_sample(data_path):
    """
    Selects a random sample from data, mostly used to generate an output from the model based on it
    :param data_path: Data path
    :return: Dataset containing one sample
    """
    rasters, elevation = Data.load_rasters(data_path)
    data, scaler = Data.create_data(rasters, elevation, count=1)

    data[:, :, :, :len(rasters)] = DiffusionModel.sanitise_input(data[:, :, :, :len(rasters)])

    dataset = BedrockDataset(data[:, :, :, :len(rasters)], data[:, :, :, len(rasters):], scaler)

    return dataset

def generate(dataset, model_path, scaler_path, save_path, num_steps=100, seed=0, count=100):
    """
    Generates a model output based on a given sample and seed
    :param dataset: Sample dataset
    :param model_path: Model path
    :param scaler_path: Scaler path
    :param save_path: Output save path
    :param num_steps: Number of timesteps to take
    :param seed: Optional integer seed
    :param count: Optional integer borehole count
    :return: Model output
    """
    data = dataset[0]

    elevation = data[1]
    boreholes, existence = dataset.select_boreholes(0, count=count)

    model_dict = torch.load(model_path, map_location='cpu')
    scaler = joblib.load(scaler_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dict = model_dict['model']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model = UNet2DConditionModel(
        sample_size=200,
        in_channels=model_dict['n_formations'],
        out_channels=model_dict['n_formations'],
        cross_attention_dim=512,
        down_block_types=(
            'CrossAttnDownBlock2D',
            'CrossAttnDownBlock2D',
            'DownBlock2D',
        ),
        up_block_types=(
            'UpBlock2D',
            'CrossAttnUpBlock2D',
            'CrossAttnUpBlock2D',
        ),
        block_out_channels=(128, 256, 512),
        layers_per_block=2,
        attention_head_dim=8,
        norm_num_groups=32,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    state_dict = model_dict['context_encoder']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    context_encoder = ContextEncoder(
        in_channels=2*model_dict['n_formations'] + 1,
        cross_attention_dim=512,
        seq_len=64,
    ).to(device)
    context_encoder.load_state_dict(state_dict)
    context_encoder.eval()

    elevation = elevation.permute(2, 0, 1).unsqueeze(0).to(device)
    boreholes = boreholes.permute(2, 0, 1).unsqueeze(0).to(device)
    existence = existence.permute(2, 0, 1).unsqueeze(0).to(device)

    context_input = torch.cat([elevation, boreholes, existence], dim=1)

    with torch.no_grad():
        encoder_hidden_states = context_encoder(context_input)

        scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
        scheduler.set_timesteps(num_steps)

        torch.manual_seed(seed)
        sample = torch.randn((1, model_dict['n_formations'], 200, 200), device=device)

        idx = 0
        for t in scheduler.timesteps:
            t_batch = t.unsqueeze(0).to(device)

            if device.type == 'cuda':
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    predicted_noise = model(sample, t_batch, encoder_hidden_states=encoder_hidden_states).sample
            else:
                predicted_noise = model(sample, t_batch, encoder_hidden_states=encoder_hidden_states).sample

            sample = scheduler.step(predicted_noise, t, sample).prev_sample

            idx += 1
            print(f'Step {idx}')

    generated = sample.squeeze(0).cpu().numpy()

    C, H, W = generated.shape
    generated = scaler.inverse_transform(generated.reshape(-1, 1)).reshape(C, H, W)

    np.save(save_path, generated)

    return generated

def sanitise_rasters(data):
    """
    Sanitises a given set of rasters by comparing formation thickness on a given order, and if the thickness is below
    some epsilon the formation does not exist in that area and is replaced with nan
    :param data: Raster data
    :return: Sanitised rasters
    """
    order = [12, 8, 11, 10, 9, 7, 13, 16, 15, 14, 1, 4, 5, 6, 0, 2, 3]
    epsilon = 5.0

    data = data.copy()

    for idx in range(len(order) - 1):
        upper = order[idx]
        lower = order[idx+1]

        thickness = data[upper] - data[lower]
        existence = thickness <= epsilon

        data[upper, existence] = np.nan

    return data

def plot_rasters(output_path, data_path, elevation_path, borehole_path, scaler_path):
    """
    Plots rasters to compare model outputs and ground truth
    :param output_path: Model output path
    :param data_path: Ground truth path
    :param elevation_path: Elevation path
    :param borehole_path: Borehole path
    :param scaler_path: Scaler path
    :return:
    """
    output = np.load(output_path) #(num_formation x 200 x 200)
    data = np.load(data_path) #(200 x 200 x num_formation)
    elevation = np.load(elevation_path) #(200 x 200 x 1)
    boreholes = np.load(borehole_path) #(200 x 200 x num_formation)

    scaler = joblib.load(scaler_path)

    data = np.transpose(data, (2, 0, 1))
    C, H, W = data.shape
    data = scaler.inverse_transform(data.reshape(-1, 1)).reshape(C, H, W)

    boreholes = np.transpose(boreholes, (2, 0, 1))
    C, H, W = boreholes.shape
    boreholes = scaler.inverse_transform(boreholes.reshape(-1, 1)).reshape(C, H, W)

    boreholes = boreholes.astype(float)
    boreholes[boreholes == 0] = np.nan

    output = sanitise_rasters(output)

    bh_cmap = plt.get_cmap('viridis').copy()
    bh_cmap.set_bad(color='white')

    fig, axes = plt.subplots(
        output.shape[0] + 1,
        3,
        figsize=(12, (output.shape[0] + 1) * 3),
        constrained_layout=True,
    )

    col_titles = ['Predicted', 'Truth', 'Boreholes']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, pad=8)

    for f in range(output.shape[0]):
        vmin = min(np.nanmin(output[f]), np.nanmin(data[f]))
        vmax = max(np.nanmax(output[f]), np.nanmax(data[f]))

        im = axes[f, 0].imshow(output[f], cmap='viridis', vmin=vmin, vmax=vmax, origin='upper')
        fig.colorbar(im, ax=axes[f, 0], fraction=0.046, pad=0.04)

        im = axes[f, 1].imshow(data[f], cmap='viridis', vmin=vmin, vmax=vmax, origin='upper')
        fig.colorbar(im, ax=axes[f, 1], fraction=0.046, pad=0.04)

        im = axes[f, 2].imshow(boreholes[f], cmap='viridis', vmin=vmin, vmax=vmax, origin='upper')
        fig.colorbar(im, ax=axes[f, 2], fraction=0.046, pad=0.04)

        for col in range(3):
            axes[f, col].set_xticks([])
            axes[f, col].set_yticks([])

    elev_row = output.shape[0]

    for col in range(3):
        axes[elev_row, col].set_visible(False)

    elev_ax = fig.add_subplot(output.shape[0] + 1, 1, output.shape[0] + 1)
    im = elev_ax.imshow(elevation, cmap='terrain', origin='upper')
    fig.colorbar(im, ax=elev_ax, fraction=0.015, pad=0.04)
    elev_ax.set_title('Surface Elevation', fontsize=11)
    elev_ax.set_xticks([])
    elev_ax.set_yticks([])

    plt.show()