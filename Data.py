import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, Sampler
import os
import warnings
from Transformer_Model import FormationInfo

warnings.simplefilter("error", RuntimeWarning)

class CountySource:
    def __init__(self, path, order, btype='paleozoic'):
        self.path = path
        self.order = order
        self.btype = btype

        self._rasters = None
        self._elevation = None
        self._magnetic = None
        self._magnetic_fvd = None
        self._magnetic_svd = None
        self._magnetic_tlt = None
        self._magnetic_ana = None
        self._gravity = None
        self._gravity_svd = None
        self._alphaearth = None
        self._patches = None

    @property
    def rasters(self):
        if self._rasters is None:
            self._load_rasters()
        return self._rasters

    @property
    def elevation(self):
        if self._elevation is None:
            self._load_rasters()
        return self._elevation

    @property
    def magnetic(self):
        if self._magnetic is None:
            self._load_rasters()
        return self._magnetic

    @property
    def magnetic_fvd(self):
        if self._magnetic_fvd is None:
            self._load_rasters()
        return self._magnetic_fvd

    @property
    def magnetic_svd(self):
        if self._magnetic_svd is None:
            self._load_rasters()
        return self._magnetic_svd

    @property
    def magnetic_tlt(self):
        if self._magnetic_tlt is None:
            self._load_rasters()
        return self._magnetic_tlt

    @property
    def magnetic_ana(self):
        if self._magnetic_ana is None:
            self._load_rasters()
        return self._magnetic_ana

    @property
    def gravity(self):
        if self._gravity is None:
            self._load_rasters()
        return self._gravity

    @property
    def gravity_svd(self):
        if self._gravity_svd is None:
            self._load_rasters()
        return self._gravity_svd

    @property
    def alphaearth(self):
        if self._alphaearth is None:
            self._load_rasters()
        return self._alphaearth

    def get_patches(self, raster_size):
        if self._patches is None:
            valid_mask = np.zeros(next(iter(self.rasters.values()))['shape'], dtype=bool)
            for sparse in self.rasters.values():
                valid_mask[sparse['rows'], sparse['cols']] = True
            
            self._patches = find_valid_patches(valid_mask, raster_size, raster_size)

        return self._patches

    def _load_rasters(self):
        print(f'Loading rasters for {self.path} with type {self.btype}...')

        depth_to_bdrk = None
        if self.btype == 'precambrian':
            depth_to_bdrk = np.load(os.path.join(self.path, f'depth_to_bdrk.npy'))

        rasters = {}
        for formation in self.order:
            top_path = os.path.join(self.path, f'{formation}_top.npy')

            if not os.path.isfile(top_path):
                continue

            if self.btype == 'paleozoic':
                rasters[f'{formation}_top'] = to_sparse(np.load(top_path))

                base_path = os.path.join(self.path, f'{formation}_base.npy')
                rasters[f'{formation}_base'] = to_sparse(np.load(base_path))

            #- Need to artificially create the base layer of the precambrian as MGS does not interpolate it -#
            if self.btype == 'precambrian':
                existence = ~np.isnan(np.load(top_path))

                offset = np.random.uniform(100, 500, size=depth_to_bdrk.shape)

                base = np.copy(depth_to_bdrk)
                base[existence] = base[existence] - offset[existence]

                rasters[f'{formation}_top'] = to_sparse(depth_to_bdrk)
                rasters[f'{formation}_base'] = to_sparse(base)

        self._rasters = rasters
        
        print(f'    rasters loaded')
        
        self._elevation = np.load(os.path.join(self.path, 'elevation.npy'))

        print(f'    elevation loaded')

        self._magnetic = np.load(os.path.join(self.path, 'magnetic.npy'))
        self._magnetic_fvd = np.load(os.path.join(self.path, 'magnetic_fvd.npy'))
        self._magnetic_svd = np.load(os.path.join(self.path, 'magnetic_svd.npy'))
        self._magnetic_tlt = np.load(os.path.join(self.path, 'magnetic_tlt.npy'))
        self._magnetic_ana = np.load(os.path.join(self.path, 'magnetic_ana.npy'))
        #self._gravity = np.load(os.path.join(self.path, 'gravity.npy'))
        #self._gravity_svd = np.load(os.path.join(self.path, 'gravity_svd.npy'))

        print(f'    geophysical loaded')

        self._alphaearth = np.load(os.path.join(self.path, 'alphaearth_2023.npy'))
        
        print(f'    alphaearth loaded')

    def is_loaded(self):
        return self._rasters is not None

class PatchSampler(Sampler):
    """Sampler that yields (county_idx, patch_idx) tuples from the dataset's
    pre-generated index list.  Because the sampler runs in the **main process**,
    it always sees the latest indices even with persistent DataLoader workers."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        yield from self.dataset.indices

    def __len__(self):
        return len(self.dataset.indices)


class MultiCountyDataset(Dataset):
    def __init__(self, counties, scaler_dict, count, raster_size, btype='paleozoic', temperature=2.0):
        self.counties = counties
        self.scaler_dict = scaler_dict
        self.count = count
        self.size = raster_size
        self.temperature = temperature
        self.btype = btype

        self._county_patches = None
        self._county_weights = None

        self.indices = None

    def __len__(self):
        return self.count

    def get_full(self, idx):
        if isinstance(idx, tuple):
            county_idx, patch_idx = idx
        else:
            county_idx, patch_idx = self.indices[idx]
        county = self.counties[county_idx]
        patches = self._county_patches[county_idx]
        x, y = patches[patch_idx]

        rasters = county.rasters
        scaler = self.scaler_dict['elevation']

        top_rasters = np.array([
            patch_from_sparse(s, x, y, self.size)
            for k, s in rasters.items() if str(k).endswith('top')
        ])
        base_rasters = np.array([
            patch_from_sparse(s, x, y, self.size)
            for k, s in rasters.items() if str(k).endswith('base')
        ])

        ###--- Drop all formations absent from patch ---###
        present = ~np.array([np.isnan(r).all() for r in top_rasters])
        top_rasters = top_rasters[present]
        base_rasters = base_rasters[present]

        ###--- Fill NaNs using next valid (deeper) formation ---###
        for i in range(len(top_rasters) - 1):
            nan_mask = np.isnan(top_rasters[i])
            if not nan_mask.any():
                continue
            for j in range(i + 1, len(top_rasters)):
                still_nan = nan_mask & np.isnan(top_rasters[i])
                if not still_nan.any():
                    break
                top_rasters[i] = np.where(still_nan, top_rasters[j], top_rasters[i])
                base_rasters[i] = np.where(still_nan, top_rasters[j], base_rasters[i])

        top_rasters[-1] = np.nan_to_num(np.nanmean(top_rasters[-1]))
        base_rasters[-1] = np.nan_to_num(np.nanmean(base_rasters[-1]))

        top_rasters = scaler.transform(top_rasters.reshape(-1, 1)).reshape(top_rasters.shape)
        base_rasters = scaler.transform(base_rasters.reshape(-1, 1)).reshape(base_rasters.shape)

        top_rasters = torch.from_numpy(top_rasters).unsqueeze(1)
        base_rasters = torch.from_numpy(base_rasters).unsqueeze(1)

        elevation = county.elevation[x:x + self.size, y:y + self.size]
        elevation = scaler.transform(elevation.reshape(-1, 1)).reshape(elevation.shape)
        elevation = (
            torch.from_numpy(elevation)
            .unsqueeze(0)
            .repeat(top_rasters.shape[0], 1, 1)
            .unsqueeze(1)
        )

        alphaearth = county.alphaearth[:, x:x + self.size, y:y + self.size]
        alphaearth = (
            torch.from_numpy(alphaearth)
            .unsqueeze(0)
            .repeat(top_rasters.shape[0], 1, 1, 1)
        )

        if self.btype == 'paleozoic':
            return elevation, top_rasters, base_rasters, alphaearth
        elif self.btype == 'precambrian':
            magnetic = county.magnetic[x:x + self.size, y:y + self.size]
            magnetic = (
                torch.from_numpy(magnetic)
                .unsqueeze(0)
                .repeat(top_rasters.shape[0], 1, 1)
                .unsqueeze(1)
            )
            gravity = county.gravity[x:x + self.size, y:y + self.size]
            gravity = (
                torch.from_numpy(gravity)
                .unsqueeze(0)
                .repeat(top_rasters.shape[0], 1, 1)
                .unsqueeze(1)
            )

            return elevation, top_rasters, alphaearth, magnetic, gravity

        return None

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            county_idx, patch_idx = idx
        else:
            county_idx, patch_idx = self.indices[idx]
        county = self.counties[county_idx]
        patches = self._county_patches[county_idx]
        x, y = patches[patch_idx]

        rasters = county.rasters

        top_rasters = np.array([
            patch_from_sparse(s, x, y, self.size)
            for k, s in rasters.items() if str(k).endswith('top')
        ])
        base_rasters = np.array([
            patch_from_sparse(s, x, y, self.size)
            for k, s in rasters.items() if str(k).endswith('base')
        ])

        formation_info = np.array([
            FormationInfo.FORMATION_ENCODINGS[k.split('_')[0]] for k in rasters.keys()
        ])

        ###--- Drop all formations absent from patch ---###
        present = ~np.array([np.isnan(r).all() for r in top_rasters])
        top_rasters = top_rasters[present]
        base_rasters = base_rasters[present]

        top_mean = np.nanmean(top_rasters[-1])
        base_mean = np.nanmean(base_rasters[-1])
        
        top_rasters[-1] = np.where(np.isnan(top_rasters[-1]), top_mean, top_rasters[-1])
        base_rasters[-1] = np.where(np.isnan(base_rasters[-1]), base_mean, base_rasters[-1])

        ###--- Fill NaNs using next valid (deeper) formation ---###
        for i in range(len(top_rasters) - 1):
            for j in range(i + 1, len(top_rasters)):
                still_nan = np.isnan(top_rasters[i])
                if not still_nan.any():
                    break
                top_rasters[i] = np.where(still_nan, top_rasters[j], top_rasters[i])
                base_rasters[i] = np.where(still_nan, top_rasters[j], base_rasters[i])

        top_rasters = self.scaler_dict['elevation'].transform(top_rasters.reshape(-1, 1)).reshape(top_rasters.shape)
        base_rasters = self.scaler_dict['elevation'].transform(base_rasters.reshape(-1, 1)).reshape(base_rasters.shape)

        top_rasters = torch.from_numpy(top_rasters).unsqueeze(1)
        base_rasters = torch.from_numpy(base_rasters).unsqueeze(1)

        elevation = county.elevation[x:x + self.size, y:y + self.size].astype(np.float32)
        elevation = np.where(np.isnan(elevation), np.nanmean(elevation), elevation)
        elevation = self.scaler_dict['elevation'].transform(elevation.reshape(-1, 1)).reshape(elevation.shape)
        elevation = (
            torch.from_numpy(elevation)
            .unsqueeze(0)
            .repeat(top_rasters.shape[0], 1, 1)
            .unsqueeze(1)
        )

        alphaearth = county.alphaearth[:, x:x + self.size, y:y + self.size]
        alphaearth = (
            torch.from_numpy(alphaearth)
            .unsqueeze(0)
            .repeat(top_rasters.shape[0], 1, 1, 1)
        )

        formation_info = (
            torch.from_numpy(formation_info)
            .unsqueeze(1)
        )

        magnetic = county.magnetic[x:x+self.size, y:y+self.size]
        magnetic = np.where(np.isnan(magnetic), np.nanmean(magnetic), magnetic)
        magnetic = self.scaler_dict['magnetic'].transform(magnetic.reshape(-1, 1)).reshape(magnetic.shape)

        magnetic_fvd = county.magnetic_fvd[x:x+self.size, y:y+self.size]
        magnetic_fvd = np.where(np.isnan(magnetic_fvd), np.nanmean(magnetic_fvd), magnetic_fvd)
        magnetic_fvd = self.scaler_dict['magnetic_fvd'].transform(magnetic_fvd.reshape(-1, 1)).reshape(magnetic_fvd.shape)

        magnetic_svd = county.magnetic_svd[x:x+self.size, y:y+self.size]
        magnetic_svd = np.where(np.isnan(magnetic_svd), np.nanmean(magnetic_svd), magnetic_svd)
        magnetic_svd = self.scaler_dict['magnetic_svd'].transform(magnetic_svd.reshape(-1, 1)).reshape(magnetic_svd.shape)

        magnetic_tlt = county.magnetic_tlt[x:x+self.size, y:y+self.size]
        magnetic_tlt = np.where(np.isnan(magnetic_tlt), np.nanmean(magnetic_tlt), magnetic_tlt)
        magnetic_tlt = self.scaler_dict['magnetic_tlt'].transform(magnetic_tlt.reshape(-1, 1)).reshape(magnetic_tlt.shape)

        magnetic_ana = county.magnetic_ana[x:x+self.size, y:y+self.size]
        magnetic_ana = np.where(np.isnan(magnetic_ana), np.nanmean(magnetic_ana), magnetic_ana)
        magnetic_ana = self.scaler_dict['magnetic_ana'].transform(magnetic_ana.reshape(-1, 1)).reshape(magnetic_ana.shape)

        magnetic = np.stack([magnetic, magnetic_fvd, magnetic_svd, magnetic_tlt, magnetic_ana], axis=0)
        magnetic = torch.from_numpy(magnetic).unsqueeze(0).repeat(top_rasters.shape[0], 1, 1, 1)

        ###--- Keep shallowest 3 and one random deeper formation ---###
        if county.btype == 'paleozoic':
            if len(top_rasters) > 3:
                rng = np.random.default_rng()
                rand_idx = rng.integers(3, len(top_rasters))
                sel = [0, 1, 2, rand_idx]

                top_rasters = top_rasters[sel]
                base_rasters = base_rasters[sel]
                elevation = elevation[sel]
                alphaearth = alphaearth[sel]
                magnetic = magnetic[sel]
                formation_info = formation_info[sel]

        ###--- Keep 4 random formations ---###
        elif county.btype == 'precambrian':
            if len(top_rasters) > 3:
                rng = np.random.default_rng()
                sel = rng.choice(top_rasters, 4, replace=False)

                top_rasters = top_rasters[sel]
                base_rasters = base_rasters[sel]
                elevation = elevation[sel]
                alphaearth = alphaearth[sel]
                magnetic = magnetic[sel]
                formation_info = formation_info[sel]

        max_f = 4
        f = top_rasters.shape[0]

        if f < max_f:
            pad = max_f - f

            top_rasters = torch.cat([top_rasters, top_rasters[-1:].expand(pad, -1, -1, -1)], dim=0)
            base_rasters = torch.cat([base_rasters, base_rasters[-1:].expand(pad, -1, -1, -1)], dim=0)
            elevation = torch.cat([elevation, elevation[-1:].expand(pad, -1, -1, -1)], dim=0)
            alphaearth = torch.cat([alphaearth, alphaearth[-1:].expand(pad, -1, -1, -1)], dim=0)
            magnetic = torch.cat([magnetic, magnetic[-1:].expand(pad, -1, -1, -1)], dim=0)
            formation_info = torch.cat([formation_info, formation_info[-1:].expand(pad, -1, -1, -1)], dim=0)

        return elevation, top_rasters, base_rasters, alphaearth, magnetic, formation_info

    def generate_indices(self):
        if self._county_patches is None:
            self._build_patch_lists()

        rng = np.random.default_rng()
        county_counts = rng.multinomial(self.count, self._county_weights).tolist()

        indices = []
        for idx, n in enumerate(county_counts):
            n_patches = len(self._county_patches[idx])
            if n_patches == 0 or n == 0:
                continue
            chosen = rng.choice(n_patches, size=n, replace=(n > n_patches))
            indices.extend((idx, int(p)) for p in chosen)

        rng.shuffle(indices)
        self.indices = indices[:self.count]

    def _build_patch_lists(self):
        print('Building patch lists')
    
        patch_lists = []

        for county in self.counties:
            print(f'    {county.path}')
            patches = county.get_patches(self.size)
            patch_lists.append(patches)

        self._county_patches = patch_lists
        self._build_weights_from_patches(patch_lists)

    def _build_weights_from_patches(self, patch_lists):
        print('Building weights')
    
        sizes = np.array([len(p) for p in patch_lists], dtype=float)
        sizes = np.maximum(sizes, 1.0)
        log_sizes = np.log(sizes) / self.temperature
        log_sizes -= log_sizes.max()
        weights = np.exp(log_sizes)
        self._county_weights = weights / weights.sum()

    def split_test(self, count, frac=0.05, seed=42):
        print('Splitting data')
    
        if self._county_patches is None:
            self._build_patch_lists()

        rng = np.random.default_rng(seed)

        test_patch_lists = []
        train_patch_lists = []

        for patches in self._county_patches:
            n_test = max(1, int(len(patches) * frac))
            chosen_idx = rng.choice(len(patches), size=n_test, replace=False)
            chosen_set = set(chosen_idx.tolist())

            test_patch_lists.append([patches[i] for i in chosen_idx])
            train_patch_lists.append([p for i, p in enumerate(patches) if i not in chosen_set])

        test_count = sum(len(p) for p in test_patch_lists)
        print(test_count)

        test_dataset = MultiCountyDataset(
            self.counties,
            self.scaler_dict,
            count=count,
            raster_size=self.size,
            temperature=self.temperature,
        )
        test_dataset._county_patches = test_patch_lists
        test_dataset._build_weights_from_patches(test_patch_lists)

        self._county_patches = train_patch_lists
        self._build_weights_from_patches(train_patch_lists)

        return test_dataset

    def select_boreholes(self, top_rasters, base_rasters, count, seed=None):
        rng = np.random.default_rng(seed)

        out = torch.zeros(4, count, 5, dtype=torch.float32)

        seen = set()
        sampled = 0

        while sampled < count:
            x = rng.integers(0, self.size)
            y = rng.integers(0, self.size)

            if (x, y) in seen:
                continue

            seen.add((x, y))

            for f in range(4):
                top = float(top_rasters[f, 0, x, y])
                base = float(base_rasters[f, 0, x, y])

                exists = 1.0 if (top - base) > 0.0 else 0.0
                out[f, sampled] = torch.tensor([top, base, exists, x, y])

            sampled += 1

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

    mask_snap = (rows % 10 == 0) & (cols % 10 == 0)

    return list(zip(rows[mask_snap].tolist(), cols[mask_snap].tolist()))

def to_sparse(arr):
    rows, cols = np.where(~np.isnan(arr))
    values = arr[rows, cols]

    return {'rows': rows, 'cols': cols, 'values': values, 'shape': arr.shape}

def patch_from_sparse(sparse, x, y, size):
    rows, cols, values = sparse['rows'], sparse['cols'], sparse['values']

    mask = (rows >= x) & (rows < x + size) & (cols >= y) & (cols < y + size)

    patch = np.full((size, size), np.nan, dtype=np.float32)
    patch[rows[mask] - x, cols[mask] - y] = values[mask]
    return patch

def load_rasters(path, order=None):
    if order is None:
        order = ['kwnd', 'dlp', 'dspl', 'omaq', 'odub', 'ogsv', 'ogpr', 'ogcm', 'odcr', 'opgw', 'ostp', 'opsh', 'opod',
                 'cp', 'cjdn', 'cstl', 'ctcg', 'cwoc', 'cecr', 'cmts']

    rasters = {}

    for idx in order:

        if not os.path.isfile(f'{path}/{idx}_top.npy'):
            continue

        rasters[f'{idx}_top']  = to_sparse(np.load(f'{path}/{idx}_top.npy'))
        rasters[f'{idx}_base'] = to_sparse(np.load(f'{path}/{idx}_base.npy'))

    context = {
        'elevation': np.load(f'{path}/elevation.npy'),
    }

    return rasters, context

def create_global_scaler_dict(counties):
    rng = np.random.default_rng(0)
    sampled_values = []

    scaler_dict = {}

    for county in counties:
        county_values = np.concatenate(
            [s['values'] for s in county.rasters.values()]
            + [county.elevation.reshape(-1)]  # elevation may have NaNs
        )
        county_values = county_values[~np.isnan(county_values)]
        n = max(1000, int(len(county_values) * 0.05))
        n = min(n, len(county_values))
        sampled_values.append(rng.choice(county_values, size=n, replace=False))

    all_values = np.concatenate(sampled_values).reshape(-1, 1)

    elevation_scaler = StandardScaler()
    elevation_scaler.fit(all_values)
    scaler_dict['elevation'] = elevation_scaler

    for county in counties:
        county_values = np.concatenate(
            [s['values'] for s in county.magnetic.values()]
            + [county.magnetic.reshape(-1)]
        )
        county_values = county_values[~np.isnan(county_values)]
        n = max(1000, int(len(county_values) * 0.25))
        n = min(n, len(county_values))
        sampled_values.append(rng.choice(county_values, size=n, replace=False))

    all_values = np.concatenate(sampled_values).reshape(-1, 1)

    magnetic_scaler = StandardScaler()
    magnetic_scaler.fit(all_values)
    scaler_dict['magnetic'] = magnetic_scaler

    for county in counties:
        county_values = np.concatenate(
            [s['values'] for s in county.magnetic_fvd.values()]
            + [county.magnetic_fvd.reshape(-1)]
        )
        county_values = county_values[~np.isnan(county_values)]
        n = max(1000, int(len(county_values) * 0.25))
        n = min(n, len(county_values))
        sampled_values.append(rng.choice(county_values, size=n, replace=False))

    all_values = np.concatenate(sampled_values).reshape(-1, 1)

    magnetic_fvd_scaler = StandardScaler()
    magnetic_fvd_scaler.fit(all_values)
    scaler_dict['magnetic_fvd'] = magnetic_fvd_scaler

    for county in counties:
        county_values = np.concatenate(
            [s['values'] for s in county.magnetic_svd.values()]
            + [county.magnetic_svd.reshape(-1)]
        )
        county_values = county_values[~np.isnan(county_values)]
        n = max(1000, int(len(county_values) * 0.25))
        n = min(n, len(county_values))
        sampled_values.append(rng.choice(county_values, size=n, replace=False))

    all_values = np.concatenate(sampled_values).reshape(-1, 1)

    magnetic_svd_scaler = StandardScaler()
    magnetic_svd_scaler.fit(all_values)
    scaler_dict['magnetic_svd'] = magnetic_svd_scaler

    for county in counties:
        county_values = np.concatenate(
            [s['values'] for s in county.magnetic_tlt.values()]
            + [county.magnetic_tlt.reshape(-1)]
        )
        county_values = county_values[~np.isnan(county_values)]
        n = max(1000, int(len(county_values) * 0.25))
        n = min(n, len(county_values))
        sampled_values.append(rng.choice(county_values, size=n, replace=False))

    all_values = np.concatenate(sampled_values).reshape(-1, 1)

    magnetic_tlt_scaler = StandardScaler()
    magnetic_tlt_scaler.fit(all_values)
    scaler_dict['magnetic_tlt'] = magnetic_tlt_scaler

    for county in counties:
        county_values = np.concatenate(
            [s['values'] for s in county.magnetic_ana.values()]
            + [county.magnetic_ana.reshape(-1)]
        )
        county_values = county_values[~np.isnan(county_values)]
        n = max(1000, int(len(county_values) * 0.25))
        n = min(n, len(county_values))
        sampled_values.append(rng.choice(county_values, size=n, replace=False))

    all_values = np.concatenate(sampled_values).reshape(-1, 1)

    magnetic_ana_scaler = StandardScaler()
    magnetic_ana_scaler.fit(all_values)
    scaler_dict['magnetic_ana'] = magnetic_ana_scaler

    return scaler_dict