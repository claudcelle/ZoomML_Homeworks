"""Missing-value filling on a copy of the model data."""

import numpy as np

from .data import ModelData


def fill_nan_values_of_model_data(
    D: ModelData, features_dict: dict = {}, target_dict: dict = {}
):
    import copy

    D_out = copy.deepcopy(D)
    for section, data in vars(D_out).items():
        data.features = data.features.fillna(features_dict)
        data.targets = data.targets.fillna(target_dict)
        data.X = data.features.to_numpy()
        data.y = data.targets.to_numpy()
        data.Xe = np.column_stack([np.ones(data.X.shape[0]), data.X])
    return D_out
