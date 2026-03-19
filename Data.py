import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class BedrockDataset(Dataset):
    """
    Dataset containing the subsurface rasters and the context to generate them.
    Context takes the form of a (B x N x N x C) array where C corresponds with the following:
        0: Elevation map
    """
    def __init__(self, data, context, existence, scaler):
        self.data = data
        self.context = context
        self.existence = existence
        self.scaler = scaler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        boreholes, bh_existence = self.select_boreholes(idx)
        return self.data[idx], self.context[idx], self.existence[idx], boreholes, bh_existence

    def select_boreholes(self, idx, seed=None, count=None):
        """
        Selects (0-300) random points with a drilled depth based off a normal distribution
        :param idx: Data index
        :param seed: Optional integer seed for numpy
        :param count: Optional borehole count rather than randomizing it
        :return: Known formation elevation tensor,
                 Tensor signifying knowledge of a formations elevation
        """
        rng = np.random.default_rng(seed)

        if count is None:
            count = rng.integers(0, 301)

        holes = torch.zeros((200, 200, self.data.shape[3]), dtype=torch.float32)
        existence = torch.zeros((200, 200, self.data.shape[3]), dtype=torch.float32)

        for _ in range(count):
            x = rng.integers(0, 200)
            y = rng.integers(0, 200)

            z = rng.standard_normal() * 175.0 + 265.0
            z = z / self.scaler.scale_[0]
            z = self.context[idx, x, y, 0] - z

            for jdx in range(self.data.shape[3]):

                elevation = self.data[idx, x, y, jdx]

                if elevation >= z:
                    holes[x, y, jdx] = elevation
                    existence[x, y, jdx] = 1.0

        return holes, existence

def load_rasters(path):
    """
    Loads all rasters stored as numpy arrays at a given path
    :param path: Data path
    :return: List of formation elevation rasters as numpy arrays, Elevation raster as a numpy array
    """
    order = ['omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh', 'opod', 'cjdn', 'cstl', 'ctcg',
             'cwoc', 'cecr', 'cmts', 'undiff']

    rasters = [np.load(f'{path}/{idx}_top.npy') for idx in order]
    elevation = np.load(f'{path}/elevation.npy')

    return rasters, elevation

def create_data(rasters, elevation, count=100, size=200):
    """
    Selects K random NxN pieces of land and compresses them into individual data pieces
    :param rasters: Rasters numpy array
    :param elevation: Elevation raster numpy array
    :param count: Amount of data to generate
    :param size: Resolution of data to generate
    :return: Data tensor,
             Scaler
    """
    rng = np.random.default_rng()

    scaler = scale_rasters(np.concatenate([rasters, [elevation]]))
    shape = rasters[0].shape

    rasters = [scaler.transform(idx.reshape(-1, 1)).reshape(shape)[1000:-1000, 1000:-1000] for idx in rasters]
    elevation = scaler.transform(elevation.reshape(-1, 1)).reshape(shape)[1000:-1000, 1000:-1000]

    data = []

    for _ in range(count):
        x = rng.integers(0, rasters[0].shape[0] - size)
        y = rng.integers(0, rasters[0].shape[1] - size)

        arr = np.full((size, size, len(rasters) + 1), np.nan)

        for idx in range(len(rasters)):

            arr[:, :, idx] = rasters[idx][x:x+size, y:y+size]

        """Last channel of map should be elevation, possibly add geophysical data"""
        arr[:, :, arr.shape[2] - 1] = elevation[x:x+size, y:y+size]

        data.append(arr)

    data = torch.from_numpy(np.array(data, dtype=np.float32))

    return data, scaler

def scale_rasters(rasters):
    """
    Helper function to scale rasters onto a normal scale
    :param rasters: Rasters tensor
    :return: Scaler
    """
    scaler = StandardScaler()
    scaler.fit(rasters.reshape(-1, 1))

    return scaler