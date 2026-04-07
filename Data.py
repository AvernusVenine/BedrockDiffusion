import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from construction.FormationInfo import FORMATION_ENCODINGS
import copy

MAX_FORMATIONS = 17

class TransformerDataset(Dataset):
    def __init__(self, data, context, scaler_dict, patches, count, size):
        self.data = data
        self.context = context
        self.scaler_dict = scaler_dict
        self.patches = patches
        self.count = count
        self.size = size

        self.indices = None

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        if idx == 0:
            self.generate_indices()

        x, y = self.patches[self.indices[idx]]

        rasters = np.array([r[x:x+self.size, y:y+self.size] for r in self.data.values()])
        rasters = np.array([r for r in rasters if ~np.isnan(r).all()])

        for idx in range(len(rasters) - 1):
            mask = np.isnan(rasters[idx])

            if not mask.any():
                continue

            for next_raster in rasters[idx+1:]:
                mask = np.isnan(rasters[idx])

                if not mask.any():
                    break

                rasters[idx] = np.where(mask, next_raster, rasters[idx])

        boreholes = self.select_boreholes(rasters)

        rasters = self.scaler_dict['elevation'].transform(rasters.reshape(-1, 1)).reshape(rasters.shape)
        rasters = torch.from_numpy(rasters)

        context = self.context['elevation'][x:x+self.size, y:y+self.size]
        context = self.scaler_dict['elevation'].transform(context.reshape(-1, 1)).reshape(context.shape)
        context = torch.from_numpy(context).unsqueeze(0).repeat(rasters.shape[0], 1, 1)

        return rasters, context, boreholes

    def generate_indices(self):
        rng = np.random.default_rng()
        self.indices = rng.choice(len(self.patches), size=self.count, replace=False)

    def select_boreholes(self, rasters, seed=None, count=None):
        rng = np.random.default_rng(seed)

        if count is None:
            count = rng.integers(10, 301)

        out = torch.full(rasters.shape, np.nan, dtype=torch.float32)

        for _ in range(count):
            x = rng.integers(0, self.size)
            y = rng.integers(0, self.size)

            out[:, x, y] = torch.from_numpy(rasters[:, x, y])

        out = self.scaler_dict['borehole'].transform(out.reshape(-1, 1)).reshape(out.shape)
        out[np.isnan(out)] = -1.0

        out = torch.from_numpy(out)
        return out

class GeophysicalDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class BedrockDataset(Dataset):
    """
    Dataset containing the subsurface rasters and the context to generate them
    """

    def __init__(self, data, context, scaler_dict, size):
        self.data = data
        self.context = context
        self.scaler_dict = scaler_dict
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rasters = []
        for key in self.data[idx].keys():
            tensor = torch.tensor(self.data[idx][key], dtype=torch.float32)
            tensor = self.scaler_dict['elevation'].transform(tensor.reshape(-1, 1)).reshape(tensor.shape)
            rasters.append(tensor)
        rasters = torch.tensor(rasters)

        context = []
        for key in self.context[idx].keys():
            tensor = torch.tensor(self.context[idx][key], dtype=torch.float32)
            tensor = self.scaler_dict[key].transform(tensor.reshape(-1, 1)).reshape(tensor.shape)
            context.append(tensor)
        context = torch.tensor(context)

        boreholes = self.select_boreholes(idx)

        return rasters, context, boreholes

    # TODO: Reimplement random z values, currently ignored for testings sake
    def select_boreholes(self, idx, seed=None, count=None):
        """
        Selects (0-300) random points with a drilled depth based off a normal distribution
        :param idx: Data index
        :param seed: Optional integer seed for numpy
        :param count: Optional borehole count rather than randomizing it
        :return: Formation elevation tensor,
        """
        rng = np.random.default_rng(seed)

        formation_keys = list(self.data[idx].keys())
        n_formations = len(formation_keys)

        if count is None:
            count = rng.integers(0, 301)

        out = torch.full((n_formations, self.size, self.size), torch.nan, dtype=torch.float32)

        for _ in range(count):
            x = rng.integers(0, self.size)
            y = rng.integers(0, self.size)

            for jdx, k in enumerate(formation_keys):
                out[jdx, x, y] = float(self.data[idx][k][x, y])

        out = self.scaler_dict['borehole'].transform(out.reshape(-1, 1)).reshape(out.shape)
        out[np.isnan(out)] = -1.0

        out = torch.from_numpy(out)
        return out

def find_valid_patches(mask, N, M):
    H, W = mask.shape

    invalid = (1 - mask).astype(np.float32)
    integral = np.cumsum(np.cumsum(invalid, axis=0), axis=1)

    bottom_right = integral[N-1:H, M-1:W]
    top = np.zeros_like(bottom_right)
    left = np.zeros_like(bottom_right)
    top_left = np.zeros_like(bottom_right)

    top[1:, :] = integral[:H-N, M-1:W]
    left[:, 1:] = integral[N-1:H, :W-M]
    top_left[1:, 1:] = integral[:H-N, :W-M]

    invalid_counts = bottom_right - top - left + top_left
    rows, cols = np.where(invalid_counts == 0)

    return list(zip(rows.tolist(), cols.tolist()))

def collate_fn(batch):
    """
    Helper function to pad recurrent inputs
    :param batch: List of (rasters, context, boreholes, formation_info) tuples
    :return: Padded batch
    """
    rasters, existence, contexts, boreholes, formation_infos, quats = zip(*batch)

    padded_rasters = []
    padded_existence = []
    padded_boreholes = []
    padded_infos = []
    masks = []

    for r, e, b, i in zip(rasters, existence, boreholes, formation_infos):
        f = r.shape[0]
        pad = MAX_FORMATIONS - f

        padded_rasters.append(
            torch.cat([r, torch.zeros(pad, *r.shape[1:])], dim=0)
        )
        padded_existence.append(
            torch.cat([e, torch.zeros(pad, *e.shape[1:])], dim=0)
        )
        padded_boreholes.append(
            torch.cat([b, torch.zeros(pad, *b.shape[1:])], dim=0)
        )
        padded_infos.append(
            torch.cat([i, torch.zeros(pad, i.shape[1])], dim=0)
        )

        masks.append(
            torch.cat([torch.ones(f, dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)])
        )

    return (
        torch.stack(padded_rasters),
        torch.stack(padded_existence),
        torch.stack(contexts),
        torch.stack(padded_boreholes),
        torch.stack(padded_infos),
        torch.stack(quats),
        torch.stack(masks),
    )

def load_rasters(path, order=None, fill_nan=False):
    """
    Loads all rasters stored as numpy arrays at a given path
    :param path: Data path
    :param order: Optional formation loading order
    :param fill_nan: Whether to fill nan values with the next non-nan elevation
    :return: Dictionary of formation elevation rasters as numpy arrays,
             Dictionary of geophysical context rasters as numpy arrays
    """
    if order is None:
        order = ['kwnd', 'dlp', 'dspl', 'omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh', 'opod',
                 'cp', 'cjdn', 'cstl', 'ctcg', 'cwoc', 'cecr', 'cmts']

    rasters = {idx: np.load(f'{path}/{idx}_top.npy')  for idx in order}

    if fill_nan:
        for idx, key in enumerate(order[:-1]):
            mask = np.isnan(rasters[key])
            if not mask.any():
                continue

            for next_key in order[:idx+1]:
                new_mask = mask & np.isnan(rasters[key])
                if not new_mask.any():
                    break
                rasters[key] = np.where(new_mask, rasters[next_key], rasters[key])

    context = {
        'elevation': np.load(f'{path}/elevation.npy'),
        #'magnetic': np.load(f'{path}/magnetic.npy')[1000:-1000, 1000:-1000] ,
        #'magnetic_1st': np.load(f'{path}/magnetic_1st.npy')[1000:-1000, 1000:-1000] ,
        #'magnetic_2nd': np.load(f'{path}/magnetic_2nd.npy')[1000:-1000, 1000:-1000] ,
        #'magnetic_tilt': np.load(f'{path}/magnetic_tilt.npy')[1000:-1000, 1000:-1000] ,
        #'gravity': np.load(f'{path}/gravity.npy')[1000:-1000, 1000:-1000] ,
        #'gravity_2nd': np.load(f'{path}/gravity_2nd.npy')[1000:-1000, 1000:-1000]
    }

    return rasters, context

def select_data_patches(rasters, context, count, size, fill_nan=True):
    rng = np.random.default_rng()
    shape = context['elevation'].shape
    order = list(rasters.keys())

    data = []
    ctx = []

    for _ in range(count):
        x = rng.integers(0, shape[0] - size)
        y = rng.integers(0, shape[1] - size)

        patch = {k: v[x:x + size, y:y + size] for k, v in rasters.items()}

        if fill_nan:
            patch = {k: v for k, v in patch.items() if not np.all(np.isnan(v))}

            patch_order = [k for k in order if k in patch]
            for idx, key in enumerate(patch_order[:-1]):
                mask = np.isnan(patch[key])
                if not mask.any():
                    continue

                for next_key in patch_order[idx+1:]:
                    new_mask = mask & np.isnan(patch[key])
                    if not new_mask.any():
                        break
                    patch[key] = np.where(new_mask, patch[next_key], patch[key])


        data.append(patch)
        ctx.append({k: v[x:x + size, y:y + size] for k, v in context.items()})

    return data, ctx

def create_scaler_dict(rasters, context):
    scaler_dict = {}

    elevation_scaler = StandardScaler()
    elevation_scaler.fit(
        np.concatenate(
            [v.reshape(-1) for v in rasters.values()] + [context['elevation'].reshape(-1)]
        ).reshape(-1, 1)
    )
    scaler_dict['elevation'] = elevation_scaler

    borehole_scaler = MinMaxScaler()
    borehole_scaler.fit(np.array([-500, 1500]).reshape(-1, 1))
    scaler_dict['borehole'] = borehole_scaler

    for key in context.keys():
        if key == 'elevation':
            continue

        scaler = StandardScaler()
        scaler.fit(context[key].reshape(-1, 1))
        scaler_dict[key] = scaler

    return scaler_dict

### --- DEPRECIATED --- ###
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

    borehole_scaler = MinMaxScaler()
    borehole_scaler.fit([v.reshape(-1) for v in rasters.values()])

    shape = context['elevation'].shape

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

    shape = context['elevation'].shape

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