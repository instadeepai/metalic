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

import dataclasses
from typing import List, Optional

import numpy as np
import pandas as pd
from datasets import Dataset

from meta.constants import SUBS_ZERO_SHOT_COLS
from meta.dataclasses import Candidate, OraclePoints


def oracle_points_from_hf_dataset(
    dataset: Dataset,
    sequence_col: str = "sequence",
    value_col: str = "value",
    add_zero_shot: bool = False,
) -> OraclePoints:
    """Convert HF Dataset to OraclePoints."""
    candidate_points = [
        Candidate(sequence=seq, features={"mutant": mutant})
        for seq, mutant in zip(dataset[sequence_col], dataset["mutant"])
    ]
    # Add zero-shot features
    if add_zero_shot:
        zero_shot_cols = [
            f"standardized_{c}" for c in SUBS_ZERO_SHOT_COLS
        ] + SUBS_ZERO_SHOT_COLS
        valid_cols = [col for col in zero_shot_cols if col in dataset.column_names]
        if len(valid_cols) > 0:
            for col in valid_cols:
                zeroshot_data = dataset[
                    col
                ]  # Must limit to one dataset read per column for speed
                assert len(zeroshot_data) == len(
                    candidate_points
                ), f"Length mismatch: {len(zeroshot_data)} vs {len(candidate_points)}"
                for cp, data in zip(candidate_points, zeroshot_data):
                    cp.features[col] = data
    # Add oracle values
    oracle_values = np.asarray(dataset[value_col])
    return OraclePoints(candidate_points=candidate_points, oracle_values=oracle_values)


def oracle_points_from_dataframe(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
    value_col: str = "value",
    feature_cols: Optional[List[str]] = None,
) -> OraclePoints:
    """Convert HF Dataset to OraclePoints."""
    oracle_values = np.asarray(df[value_col])
    feature_cols = feature_cols or []
    candidate_points = []
    for feature_values in df[[sequence_col] + feature_cols].values:
        seq = feature_values[0]
        features = dict(zip(feature_cols, feature_values[1:]))
        candidate_points.append(Candidate(sequence=seq, features=features))
    return OraclePoints(candidate_points=candidate_points, oracle_values=oracle_values)


def oracle_points_to_dataframe(
    oracle_points: OraclePoints,
) -> pd.DataFrame:
    """Convert HF Dataset to OraclePoints."""
    rows = []
    for cand, value in zip(oracle_points.candidate_points, oracle_points.oracle_values):
        d = {"sequence": cand.sequence, "value": value}
        if cand.features is not None and isinstance(cand.features, dict):
            for k, v in cand.features.items():
                d[k] = v
        rows.append(d)

    return pd.DataFrame.from_records(rows)


def oracle_points_to_hf_dataset(oracle_points: OraclePoints) -> Dataset:
    """Conver OraclePoints to HF Dataset."""
    raise NotImplementedError()


@dataclasses.dataclass
class FitnessTaskMetadata:
    dms_name: str  # name of mutants dataset (DMS_id in ProteinGym)
    dms_category: Optional[str] = None
    wt_sequence: Optional[str] = None
    msa_file: Optional[str] = None
    msa_format: str = "a3m"
    pdb_file: Optional[str] = None
    # TODO: add search space?
