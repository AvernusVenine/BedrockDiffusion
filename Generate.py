import numpy as np
import torch
from diffusers import DDIMScheduler
import pandas as pd
import Data
from Data import BedrockDataset
import joblib
import matplotlib.pyplot as plt
import StandardDiffusion
from construction.Encoder import Encoder
from construction.Embedder import Embedder
from construction.UNet import UNet
from construction.Transformer import ConditionTransformer
from construction.FormationInfo import FORMATION_INFO_DIM

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

def get_random_sample(data_path, size, order=None):
    """
    Selects a random sample from data, mostly used to generate an output from the model based on it
    :param data_path: Data path
    :param size: Raster size
    :param order: Optional bedrock ordering list
    :return: Dataset containing one sample
    """
    rasters, context = Data.load_rasters(data_path, order=order)
    data, ctx, scaler_dict = Data.create_data(rasters, context, count=1, size=size)

    data, ctx = StandardDiffusion.sanitise_input(data, ctx)

    #TODO: Remove after testing just elevation
    ctx = [{k: v for k, v in c.items() if k == 'elevation'} for c in ctx]

    dataset = BedrockDataset(data, ctx, scaler_dict, size)

    return dataset

def generate(dataset, model_path, save_path, num_steps=100, seed=0, count=100):
    """
    Generates a model output based on a given sample and seed
    :param dataset: Sample dataset
    :param model_path: Model path
    :param save_path: Output save path
    :param num_steps: Number of timesteps to take
    :param seed: Optional integer seed
    :param count: Optional integer borehole count
    :return: Model output
    """
    cross_attention_dim = 768
    seq_len = 64

    scaler = joblib.load(f'{model_path}_elevation.scl')

    rasters, existence, context, boreholes, formation_info, quaternary = dataset[0]
    boreholes, quaternary = dataset.select_boreholes(0, seed=seed, count=count)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dict = torch.load(f'{model_path}.mdl', map_location='cpu')

    geo_context_encoder = Encoder(in_channels=1, cross_attention_dim=cross_attention_dim, seq_len=seq_len).to(device)
    geo_context_encoder.load_state_dict(model_dict['geo_context_encoder'])
    geo_context_encoder.eval()

    unet = UNet().to(device)
    unet.load_state_dict(model_dict['unet'])
    unet.eval()

    formation_embedder = Embedder(in_features=FORMATION_INFO_DIM, out_features=cross_attention_dim).to(device)
    formation_embedder.load_state_dict(model_dict['formation_embedder'])
    formation_embedder.eval()

    borehole_encoder = Encoder(in_channels=4, cross_attention_dim=cross_attention_dim, seq_len=seq_len).to(device)
    borehole_encoder.load_state_dict(model_dict['borehole_encoder'])
    borehole_encoder.eval()

    quaternary_encoder = Encoder(in_channels=4, cross_attention_dim=cross_attention_dim, seq_len=seq_len).to(device)
    quaternary_encoder.load_state_dict(model_dict['quaternary_encoder'])
    quaternary_encoder.eval()

    condition_transformer = ConditionTransformer(cross_attention_dim=cross_attention_dim, num_heads=16).to(device)
    condition_transformer.load_state_dict(model_dict['condition_transformer'])
    condition_transformer.eval()

    rasters = rasters.to(device, dtype=torch.bfloat16)
    existence = existence.to(device, dtype=torch.bfloat16)
    context = context.to(device, dtype=torch.bfloat16).unsqueeze(0)
    boreholes = boreholes.to(device, dtype=torch.bfloat16)
    formation_info = formation_info.to(device, dtype=torch.bfloat16)
    quaternary = quaternary.to(device, dtype=torch.bfloat16).unsqueeze(0)

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            encoded_context = geo_context_encoder(context)
            embedded_info = formation_embedder(formation_info).unsqueeze(1)
            encoded_boreholes = borehole_encoder(boreholes)
            encoded_quaternary = quaternary_encoder(quaternary)

            encoder_hidden_states = condition_transformer(
                encoded_context,
                embedded_info,
                encoded_boreholes,
                encoded_quaternary
            )

            scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
            scheduler.set_timesteps(num_steps)

            torch.manual_seed(seed)

            sample = torch.randn((1, 1, 256, 256), device=device)
            mask = None

            idx = 0
            for t in scheduler.timesteps:
                t_batch = t.unsqueeze(0).to(device)

                predicted_noise, predicted_mask = unet(sample, t_batch, encoder_hidden_states=encoder_hidden_states)

                sample = scheduler.step(predicted_noise, t, sample).prev_sample
                mask = predicted_mask

                idx += 1
                print(f'Step {idx}')

        # =============================
        # PREDICTED OUTPUT
        # =============================

        generated = sample.squeeze().cpu().float().numpy()
        H, W = generated.shape
        generated = scaler.inverse_transform(generated.reshape(-1, 1)).reshape(H, W)

        predicted_existence = (torch.sigmoid(mask.squeeze()) > 0.5).cpu().numpy()
        generated_masked = np.where(predicted_existence, generated, np.nan)

        np.save(f'{save_path}_predicted_elevation.npy', generated_masked)

        # =============================
        # TRUE RASTERS
        # =============================

        true_elev = rasters[0].cpu().float().numpy()
        H, W = true_elev.shape
        true_elev = scaler.inverse_transform(true_elev.reshape(-1, 1)).reshape(H, W)

        true_existence = existence.cpu().float().numpy().astype(bool)
        true_masked = np.where(true_existence, true_elev, np.nan)

        np.save(f'{save_path}_true_elevation.npy', true_masked)

        # =============================
        # CONTEXT AND BOREHOLES
        # =============================

        np.save(f'{save_path}_context.npy', context.squeeze(0).cpu().float().numpy())
        np.save(f'{save_path}_boreholes.npy', boreholes.cpu().float().numpy())

        return generated_masked, true_masked


def plot_rasters(save_path, model_path, plot_save_path):
    """
    Loads and plots predicted and true formation elevations side by side with the surface elevation map.
    NaN values (non-existent formation) are shown in white.
    :param save_path: Path used when saving generation outputs
    :param model_path: Model path used to load the elevation scaler
    :param plot_save_path: Optional path to save the figure
    """
    generated_masked = np.load(f'{save_path}_predicted_elevation.npy')
    true_masked = np.load(f'{save_path}_true_elevation.npy')
    context = np.load(f'{save_path}_context.npy')
    boreholes = np.load(f'{save_path}_boreholes.npy')

    scaler = joblib.load(f'{model_path}_elevation.scl')

    shape = context[0].shape
    surface_elev = scaler.inverse_transform(
        context[0].reshape(-1, 1)
    ).reshape(shape)

    borehole_exists = boreholes[0, 3, :, :]
    borehole_y, borehole_x = np.where(borehole_exists > 0)

    combined = np.concatenate([
        generated_masked[~np.isnan(generated_masked)],
        true_masked[~np.isnan(true_masked)]
    ])
    vmin, vmax = combined.min(), combined.max()

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='white')

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    im0 = axes[0].imshow(true_masked[0], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title('True formation elevation')
    axes[0].axis('off')

    im1 = axes[1].imshow(generated_masked, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title('Predicted formation elevation')
    axes[1].axis('off')

    fig.colorbar(im1, ax=axes[:2], orientation='vertical', fraction=0.02, pad=0.02, label='Elevation (m)')

    axes[2].imshow(surface_elev, cmap='terrain')
    axes[2].scatter(borehole_x, borehole_y, s=2, c='red', linewidths=0)
    axes[2].set_title(f'Elevation and Borehole locations (n={len(borehole_x)})')
    axes[2].axis('off')

    """im4 = axes[3].imshow(context[1], cmap='RdBu_r')
    axes[3].set_title('Magnetic')
    axes[3].axis('off')
    fig.colorbar(im4, ax=axes[3], orientation='vertical', fraction=0.02, pad=0.02, label='Magnetic (scaled)')

    im5 = axes[4].imshow(context[5], cmap='RdBu_r')
    axes[4].set_title('Gravity')
    axes[4].axis('off')
    fig.colorbar(im5, ax=axes[4], orientation='vertical', fraction=0.02, pad=0.02, label='Gravity (scaled)')"""

    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=150, bbox_inches='tight')

    plt.close()