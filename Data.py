import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class BedrockDataset(Dataset):
    """
    Dataset containing the subsurface rasters and the context to generate them
    """

    FORMATION_KEYS = ['omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp',
                      'opsh', 'opod', 'cjdn', 'cstl', 'ctcg', 'cwoc', 'cecr', 'cmts', 'undiff']

    CONTEXT_KEYS = ['elevation', 'magnetic', 'magnetic_1st', 'magnetic_2nd',
                    'magnetic_tilt', 'gravity', 'gravity_2nd']

    def __init__(self, data, context, scaler_dict, size):
        self.data = data
        self.context = context
        self.scaler_dict = scaler_dict
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rasters = torch.stack(
            [torch.tensor(self.data[idx][k][1], dtype=torch.float32) for k in self.FORMATION_KEYS]
        )

        boreholes = self.select_boreholes(idx)

        context = torch.stack(
            [torch.tensor(self.context[idx][k], dtype=torch.float32) for k in self.CONTEXT_KEYS]
        )

        formation_info = None

        return rasters, context, boreholes, formation_info

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

        formation_keys = list(self.data[idx].keys())
        n_formations = len(formation_keys)

        if count is None:
            count = rng.integers(0, 301)

        out = torch.zeros((n_formations, 4, self.size, self.size), dtype=torch.float32)

        elevation_scaler = self.scaler_dict['elevation']

        for _ in range(count):
            x = rng.integers(0, self.size)
            y = rng.integers(0, self.size)

            z = rng.standard_normal() * 175.0 + 265.0
            z = z / elevation_scaler.scale_[0]

            z = self.context[idx]['elevation'][x, y] - z

            for jdx, k in enumerate(formation_keys):
                # Mark borehole existence
                out[jdx, 3, x, y] = 1.0

                # Skip if formation did not exist at this point
                if not self.data[idx][k][0, x, y]:
                    continue

                top_elev = self.data[idx][1, x, y]

                if top_elev >= z:
                    # Bottom elevation is the top of the next formation if it exists and within range
                    if jdx + 1 < n_formations:
                        next_k = formation_keys[jdx + 1]
                        next_valid = self.data[idx][next_k][0, x, y]
                        next_elev = self.data[idx][next_k][1, x, y]

                        if next_valid and next_elev >= z:
                            bot_elev = next_elev
                        else:
                            bot_elev = z
                    else:
                        bot_elev = z

                    out[jdx, 0, x, y] = top_elev
                    out[jdx, 1, x, y] = bot_elev
                    out[jdx, 2, x, y] = 1.0

        return out

def collate_fn(batch):
    """
    Helper function to pad recurrent inputs
    :param batch: Data batch
    :return: Padded batch
    """
    padded = pad_sequence(batch, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(seq) for seq in batch])

    return padded, lengths

def load_rasters(path):
    """
    Loads all rasters stored as numpy arrays at a given path
    :param path: Data path
    :return: Dictionary of formation elevation rasters as numpy arrays,
             Dictionary of geophysical context rasters as numpy arrays
    """
    order = ['omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh', 'opod', 'cjdn', 'cstl', 'ctcg',
             'cwoc', 'cecr', 'cmts', 'undiff']

    rasters = {idx: np.load(f'{path}/{idx}_top.npy') for idx in order}
    context = {
        'elevation': np.load(f'{path}/elevation.npy'),
        'magnetic': np.load(f'{path}/magnetic.npy'),
        'magnetic_1st': np.load(f'{path}/magnetic_1st.npy'),
        'magnetic_2nd': np.load(f'{path}/magnetic_2nd.npy'),
        'magnetic_tilt': np.load(f'{path}/magnetic_tilt.npy'),
        'gravity': np.load(f'{path}/gravity.npy'),
        'gravity_2nd': np.load(f'{path}/gravity_2nd.npy')
    }

    return rasters, context

def create_data(rasters, context, count=100, size=200):
    """
    Selects K random NxN pieces of land and compresses them into individual data pieces
    :param rasters: Rasters dictionary
    :param context: Geophysical context dictionary
    :param count: Amount of data to generate
    :param size: Resolution of data to generate
    :return: Data list,
             Scaler dictionary
    """

    # =============================
    # DATA SCALING
    # =============================

    scaler_dict = {}

    elevation_scaler = StandardScaler()
    elevation_scaler.fit(
        np.concatenate(
            [v.reshape(-1) for v in rasters.values()] + [context['elevation'].reshape(-1)]
        ).reshape(-1, 1)
    )

    shape = rasters['undiff'].shape

    rasters = {k: elevation_scaler.transform(v.reshape(-1, 1)).reshape(shape)[1000:-1000, 1000:-1000] for k, v in rasters.items()}
    context['elevation'] = elevation_scaler.transform(context['elevation'].reshape(-1, 1)).reshape(shape)[1000:-1000, 1000:-1000]

    scaler_dict['elevation'] = elevation_scaler

    context_keys = ['magnetic', 'magnetic_1st', 'magnetic_2nd', 'magnetic_tilt', 'gravity', 'gravity_2nd']

    for key in context_keys:
        scaler = StandardScaler()
        context[key] = scaler.fit_transform(context[key].reshape(-1, 1)).reshape(shape)[1000:-1000, 1000:-1000]
        scaler_dict[key] = scaler

    # =============================
    # DATA RANDOM SELECTION
    # =============================
    rng = np.random.default_rng()

    shape = rasters['undiff'].shape

    data = []
    data_context = []

    for _ in range(count):
        x = rng.integers(0, shape[0] - size)
        y = rng.integers(0, shape[1] - size)

        point = {}
        point_context = {}

        for k, v in rasters.items():
            point[k] = v[x:x+size, y:y+size]

        for k, v in context.items():
            point_context[k] = v[x:x+size, y:y+size]

        data.append(point)
        data_context.append(point_context)

    return data, data_context, scaler_dict