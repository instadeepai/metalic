# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

import numpy as np
from scipy.stats import spearmanr


def check_shapes(predictions: np.ndarray, targets: np.ndarray) -> None:
    assert (
        predictions.shape == targets.shape
    ), f"Predictions shape {predictions.shape} and targets shape {targets.shape} don't match"


def standard_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    target_mean = targets.mean(0)
    target_std = targets.std(0) if len(targets) > 1 else 1.0
    standard_predictions = (predictions - target_mean) / target_std
    standard_targets = (targets - target_mean) / target_std
    return mse(standard_predictions, standard_targets)


def norm_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    avg_target_l1 = np.abs(targets).mean(0)
    norm_predictions = predictions / avg_target_l1
    norm_targets = targets / avg_target_l1
    return mse(norm_predictions, norm_targets)


def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    check_shapes(predictions, targets)
    return ((targets - predictions) ** 2).mean(0)


def spearman(predictions: np.ndarray, targets: np.ndarray) -> float:
    check_shapes(predictions, targets)
    return spearmanr(targets, predictions)[0]


def get_metric_function(metric_name: str) -> Callable:
    if metric_name == "mse":
        return mse
    elif metric_name == "standardized_mse":
        return standard_mse
    elif metric_name == "normalized_mse":
        return norm_mse
    elif metric_name == "spearman":
        return spearman
    else:
        raise ValueError(f"Unknown metric_name {metric_name}")
