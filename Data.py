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
    def __init__(self, data, context, scaler):
        self.data = data
        self.context = context
        self.scaler = scaler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        boreholes, existence = self.select_boreholes(idx)
        return self.data[idx], self.context[idx], boreholes, existence

    def select_boreholes(self, idx):
        """
        Selects (0-25) random points with a drilled depth based off a normal distribution
        :param idx: Data index
        :return: Known formation elevation tensor,
                 Tensor signifying knowledge of a formations elevation
        """
        count = np.random.randint(low=0, high=26)

        holes = torch.zeros((200, 200, self.data.shape[3]), dtype=torch.float32)
        existence = torch.zeros((200, 200, self.data.shape[3]), dtype=torch.float32)

        for _ in range(count):
            x = np.random.randint(low=0, high=200)
            y = np.random.randint(low=0, high=200)

            z = np.random.randn() * 175.0 + 265.0
            z = z / self.scaler.scale_[0]
            z = self.context[idx, x, y, 0] - z

            for jdx in range(self.data.shape[3]):

                elevation = self.data[idx, x, y, jdx]

                if elevation >= z:
                    holes[x, y, jdx] = elevation
                    existence[x, y, jdx] = 1.0

        return holes, existence

def load_rasters(path, undiff_prefix='cmts'):
    """
    Loads all rasters stored as numpy arrays at a given path
    :param path: Data path
    :param undiff_prefix: Formation that borders very old undifferentiated bedrock
    :return: List of formation elevation rasters as numpy arrays, Elevation raster as a numpy array
    """
    path = Path(path).resolve()

    files = [f for f in path.glob(f'**/*_top.npy') if f.is_file()]

    rasters = [np.load(f) for f in files]
    undiff = np.load(f'{path}/{undiff_prefix}_base.npy')

    rasters.append(undiff)

    elevation = np.load(f'{path}/elevation.npy')

    return rasters, elevation

def create_data(rasters, elevation, count=100, size=200):
    """
    Selects N random 200x200 pieces of land and compresses them into individual data pieces
    :param rasters: Rasters numpy array
    :param elevation: Elevation raster numpy array
    :param count: Amount of data to generate
    :param size: Resolution of data to generate
    :return: Data tensor,
             Scaler
    """

    scaler = scaler_rasters(np.concatenate([rasters, [elevation]]))
    shape = rasters[0].shape

    rasters = [scaler.transform(idx.reshape(-1, 1)).reshape(shape)[1000:-1000, 1000:-1000] for idx in rasters]
    elevation = scaler.transform(elevation.reshape(-1, 1)).reshape(shape)[1000:-1000, 1000:-1000]

    data = []

    for _ in range(count):
        x = np.random.randint(low=0, high=rasters[0].shape[0] - size)
        y = np.random.randint(low=0, high=rasters[0].shape[1] - size)

        arr = np.full((size, size, len(rasters) + 1), np.nan)

        for idx in range(len(rasters)):

            arr[:, :, idx] = rasters[idx][x:x+size, y:y+size]

        """Last channel of map should be elevation, possibly add geophysical data"""
        arr[:, :, arr.shape[2] - 1] = elevation[x:x+size, y:y+size]

        data.append(arr)

    data = torch.from_numpy(np.array(data, dtype=np.float32))

    return data, scaler

def scaler_rasters(rasters):
    scaler = StandardScaler()
    scaler.fit(rasters.reshape(-1, 1))

    return scaler
