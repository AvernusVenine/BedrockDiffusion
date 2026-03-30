import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class FormationProperties:
    age_ma: float                   # Age in millions of years — full geological scale
    lithologies: List[str]          # One or more lithologies
    dep_environment: str            # Depositional environment
    max_thickness: float


# ── Categorical encoding maps ─────────────────────────────────────────────────

LITHOLOGY_CLASSES = [
    'limestone',
    'dolomite',
    'sandstone',
    'shale',
    'siltstone',
    'mudstone',
    'conglomerate',
    'granite',
    'gneiss',
    'schist',
    'basalt',
    'quartzite',
    'mixed_carbonate',
    'undifferentiated',
]

DEP_ENV_CLASSES = [
    'shallow_marine_carbonate',
    'shallow_marine_clastic',
    'deep_marine',
    'fluvial',
    'glacial',
    'aeolian',
    'metamorphic',
    'igneous_intrusive',
    'igneous_extrusive',
    'undifferentiated',
]

# ── Age encoding ──────────────────────────────────────────────────────────────
# Log scale over full geological time 1 Ma → 4000 Ma
# log(1) = 0.0, log(4000) ≈ 8.29 — normalise to 0→1

import math
_LOG_AGE_MIN = math.log(1.0)
_LOG_AGE_MAX = math.log(4000.0)


def encode_age(age_ma: float) -> float:
    """Log-normalised age over full geological timescale."""
    return (math.log(max(age_ma, 1.0)) - _LOG_AGE_MIN) / (_LOG_AGE_MAX - _LOG_AGE_MIN)


# ── Thickness encoding ────────────────────────────────────────────────────────
# Also log-scaled since thickness ranges from metres to kilometres
_LOG_THICK_MIN = math.log(1.0)
_LOG_THICK_MAX = math.log(5000.0)


def encode_thickness(thickness_m: float) -> float:
    """Log-normalised thickness from 1m to 5000m."""
    return (math.log(max(thickness_m, 1.0)) - _LOG_THICK_MIN) / (_LOG_THICK_MAX - _LOG_THICK_MIN)


# ── Formation lookup table ────────────────────────────────────────────────────

FORMATION_PROPERTIES = {
    # ── Ordovician ────────────────────────────────────────────────────────────
    'omaq': FormationProperties(
        age_ma=446.0,
        lithologies=['limestone', 'dolomite', 'shale'],
        dep_environment='deep_marine',
        max_thickness=85.0,
    ),
    'odub': FormationProperties(
        age_ma=447.0,
        lithologies=['limestone', 'shale'],
        dep_environment='shallow_marine_carbonate',
        max_thickness=35.0,
    ),
    'ogsv': FormationProperties(
        age_ma=453.0,
        lithologies=['dolomite', 'limestone'],
        dep_environment='shallow_marine_carbonate',
        max_thickness=85.0,
    ),
    'ogpr': FormationProperties(
        age_ma=458.0,
        lithologies=['limestone', 'dolomite', 'shale'],
        dep_environment='shallow_marine_carbonate',
        max_thickness=62.0,
    ),
    'ogcm': FormationProperties(
        age_ma=460.0,
        lithologies=['limestone', 'shale'],
        dep_environment='shallow_marine_carbonate',
        max_thickness=65.0,
    ),
    'odcr': FormationProperties(
        age_ma=462.0,
        lithologies=['shale', 'limestone'],
        dep_environment='deep_marine',
        max_thickness=80.0,
    ),
    'opgw': FormationProperties(
        age_ma=465.0,
        lithologies=['shale', 'dolomite', 'limestone'],
        dep_environment='shallow_marine_carbonate',
        max_thickness=46.0,
    ),
    'ostp': FormationProperties(
        age_ma=468.0,
        lithologies=['sandstone', 'siltstone', 'shale'],
        dep_environment='shallow_marine_clastic',
        max_thickness=190.0,
    ),
    'opsh': FormationProperties(
        age_ma=470.0,
        lithologies=['limestone', 'dolomite'],
        dep_environment='shallow_marine_carbonate',
        max_thickness=240.0,
    ),
    'opod': FormationProperties(
        age_ma=473.0,
        lithologies=['dolomite', 'sandstone'],
        dep_environment='shallow_marine_carbonate',
        max_thickness=180.0,
    ),
    # ── Cambrian ──────────────────────────────────────────────────────────────
    'cjdn': FormationProperties(
        age_ma=490.0,
        lithologies=['sandstone'],
        dep_environment='fluvial',
        max_thickness=110.0,
    ),
    'cstl': FormationProperties(
        age_ma=494.0,
        lithologies=['siltstone', 'dolomite'],
        dep_environment='shallow_marine_clastic',
        max_thickness=130.0,
    ),
    'ctcg': FormationProperties(
        age_ma=497.0,
        lithologies=['sandstone'],
        dep_environment='shallow_marine_clastic',
        max_thickness=180.0,
    ),
    'cwoc': FormationProperties(
        age_ma=501.0,
        lithologies=['sandstone'],
        dep_environment='shallow_marine_clastic',
        max_thickness=100.0,
    ),
    'cecr': FormationProperties(
        age_ma=505.0,
        lithologies=['shale', 'sandstone', 'siltstone'],
        dep_environment='shallow_marine_clastic',
        max_thickness=250.0,
    ),
    'cmts': FormationProperties(
        age_ma=515.0,
        lithologies=['sandstone'],
        dep_environment='fluvial',
        max_thickness=375.0,
    ),
    'undiff': FormationProperties(
        age_ma=4000.0,
        lithologies=['undifferentiated'],
        dep_environment='undifferentiated',
        max_thickness=5000.0,
    ),
}


def encode_formation(key: str) -> torch.Tensor:
    props = FORMATION_PROPERTIES[key]

    # Continuous features
    continuous = [
        encode_age(props.age_ma),
        encode_thickness(props.max_thickness),
    ]

    # Multi-hot lithology — 1.0 for each lithology present
    lith_vec = [0.0] * len(LITHOLOGY_CLASSES)
    for lith in props.lithologies:
        lith_vec[LITHOLOGY_CLASSES.index(lith)] = 1.0

    # One-hot depositional environment
    dep_vec = [0.0] * len(DEP_ENV_CLASSES)
    dep_vec[DEP_ENV_CLASSES.index(props.dep_environment)] = 1.0

    vec = [*continuous, *lith_vec, *dep_vec]

    return torch.tensor(vec, dtype=torch.float32)


# Pre-compute all encodings at import time
FORMATION_ENCODINGS = {k: encode_formation(k) for k in FORMATION_PROPERTIES}

FORMATION_INFO_DIM = len(next(iter(FORMATION_ENCODINGS.values())))