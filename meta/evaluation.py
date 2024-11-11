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

import logging
import time
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

from .models.base_metasurrogate import BaseMetaSurrogate
from .tasks.proteingym.tasks import ProteinGymMetaSLTask

log = logging.getLogger("rich")


def run_metasupervised_evaluation(
    task: ProteinGymMetaSLTask,
    surrogate: BaseMetaSurrogate,
    splits_to_evaluate: Tuple[str, ...] = (
        "validation",
    ),  # "train" is useful but takes a long time
) -> Tuple[Dict[str, Union[float, int, np.number]], pd.DataFrame]:
    """
    This function is primarily a wrapper around task.evaluate_surrogate.
    This function adds the time and creates a dataframe.
    """
    metrics = {}

    for split in splits_to_evaluate:
        t0 = time.time()
        # Evaluate the surrogate on the split. Note that the seed comes from the task
        split_metrics, preds, oracle_values, seqs = task.evaluate_surrogate(surrogate, split)
        split_metrics = dict(split_metrics)
        # Pad sequences to same length
        max_len = max([len(seq) for seq in seqs])
        padded_sequences = []
        for seq in seqs:
            padded_sequences.append(seq.ljust(max_len, "-"))
        # Data dict
        data = {
            "sequence": padded_sequences,
            "target": oracle_values,
        }
        if len(preds) > 0:
            data["prediction"] = preds
        # Convert to DataFrame
        predictions_df = pd.DataFrame(data)
        # Add evaluation time to metrics
        t1 = time.time()
        split_metrics["eval_time"] = t1 - t0
        # Add the metrics for this split to the overall metrics
        for key, value in split_metrics.items():
            metrics[f"{split}_{key}"] = value

    return metrics, predictions_df
