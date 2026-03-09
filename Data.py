import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def load_rasters(path):
    path = Path(path).resolve()

    top_files = [f for f in path.glob(f'**/*_top.npy') if f.is_file()]
    base_files = [f for f in path.glob(f'**/*_base.npy') if f.is_file()]

    top_rasters = [np.load(f) for f in top_files]
    base_rasters = [np.load(f) for f in base_files]

    return top_rasters, base_rasters

def create_data(path, count=100):
    """
    Selects N random 200x200 pieces of land and compresses them into individual data pieces
    :param path: Data path
    :param count: Amount of pieces to generate
    :return: List of data chunks
    """
    top_rasters, base_rasters = load_rasters(path)
    elevation = np.load(f'{path}/elevation.npy')

    scaler = scaler_rasters(np.concatenate([top_rasters, base_rasters, [elevation]]))

    shape = top_rasters[0].shape

    top_rasters = [scaler.transform(idx.reshape(-1, 1)).reshape(shape)[200:-1000, 200:-200] for idx in top_rasters]
    base_rasters = [scaler.transform(idx.reshape(-1, 1)).reshape(shape)[200:-1000, 200:-200] for idx in base_rasters]

    elevation = scaler.transform(elevation.reshape(-1, 1)).reshape(shape)[200:-1000, 200:-200]

    data = []

    for _ in range(count):
        x = np.random.randint(low=0, high=top_rasters[0].shape[0] - 500)
        y = np.random.randint(low=0, high=top_rasters[0].shape[1] - 500)

        arr = np.full((500, 500, 2*len(top_rasters) + 1), np.nan)

        for idx in range(len(top_rasters)):

            arr[:, :, 2*idx] = top_rasters[idx][x:x+500, y:y+500]
            arr[:, :, 2*idx + 1] = base_rasters[idx][x:x+500, y:y+500]

        """Last channel of map should be elevation, possibly add geophysical data"""
        arr[:, :, arr.shape[2] - 1] = elevation[x:x+500, y:y+500]

        data.append(arr)

    return data

def scaler_rasters(rasters):
    scaler = StandardScaler()
    scaler.fit(rasters.reshape(-1, 1))

    return scaler
