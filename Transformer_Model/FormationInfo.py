import torch
from dataclasses import dataclass
from typing import List
import math


@dataclass
class FormationInfo:
    composition: List[str]

COMPOSITION_CLASSES = [
    #- Primary Lithology -#
    'limestone',
    'dolomite',
    'sandstone',
    'shale',
    'siltstone',
    'conglomerate',
    'undifferentiated',
]

FORMATION_PROPERTIES = {
    #- Undifferentiated -#
    'undiff': FormationInfo(
        composition=['undifferentiated']
    ),

    #- Cretaceous -#
    'kwnd': FormationInfo(
        composition=['sandstone', 'conglomerate', 'siltstone'],
    ),

    #- Paleozoic -#
    'dlp': FormationInfo(
        composition=['dolomite', 'limestone']
    ),

    'dspl': FormationInfo(
        composition=['dolomite']
    ),

    'omaq': FormationInfo(
        composition=['dolomite', 'sandstone']
    ),

    'odub': FormationInfo(
        composition=['limestone', 'dolomite', 'shale']
    ),

    'ogsv': FormationInfo(
        composition=['dolomite', 'limestone']
    ),

    'ogpr': FormationInfo(
        composition=['dolomite', 'limestone']
    ),

    'ogcm': FormationInfo(
        composition=['limestone', 'shale']
    ),

    'odcr': FormationInfo(
        composition=['shale', 'limestone']
    ),

    'opgw': FormationInfo(
        composition=['limestone', 'dolomite', 'shale']
    ),

    'ostp': FormationInfo(
        composition=['sandstone']
    ),

    'opsh': FormationInfo(
        composition=['dolomite', 'sandstone', 'shale']
    ),

    'opod': FormationInfo(
        composition=['dolomite', 'sandstone', 'shale']
    ),

    'cp': FormationInfo(
        composition=['sandstone', 'siltstone', 'shale']
    ),

    'cjdn': FormationInfo(
        composition=['sandstone']
    ),

    'cstl': FormationInfo(
        composition=['dolomite', 'siltstone', 'sandstone', 'shale']
    ),

    'ctcg': FormationInfo(
        composition=['sandstone', 'siltstone', 'shale']
    ),

    'cwoc': FormationInfo(
        composition=['sandstone']
    ),

    'cecr': FormationInfo(
        composition=['sandstone', 'siltstone', 'shale']
    ),

    'cmts': FormationInfo(
        composition=['sandstone']
    ),
}

def encode_formation(key):
    props = FORMATION_PROPERTIES[key]

    lith_vec = [0.0] * len(COMPOSITION_CLASSES)
    for lith in props.composition:
        lith_vec[COMPOSITION_CLASSES.index(lith)] = 1.0

    return torch.tensor(lith_vec, dtype=torch.float32)

FORMATION_ENCODINGS = {k: encode_formation(k) for k in FORMATION_PROPERTIES}
FORMATION_INFO_DIM = len(next(iter(FORMATION_ENCODINGS.values())))