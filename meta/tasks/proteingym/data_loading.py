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
import os
from typing import Any, Dict, Optional, Set, Tuple

import datasets
import numpy as np
import pandas as pd

from meta.constants import (
    SUBS_ZERO_SHOT_COLS,
    SUBS_ZERO_SHOT_COLS_to_index,
)
from meta.tasks.utils import FitnessTaskMetadata

WT_VALUES = {
    "GFP_AEQVI_Sarkisyan_2016": 3.72,  # https://elifesciences.org/articles/75842.pdf
    "CAPSD_AAV2S_Sinai_substitutions_2021": -0.918194,  # see raw dataset from FLIP
}


logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]")
log = logging.getLogger("rich")


# This file is meta-data taken from protein gym
subs_meta = pd.read_csv(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "DMS_substitutions.csv")
).set_index("DMS_id")

def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()

def get_gym_metadata(dms_name: str) -> Dict[str, Any]:
    return subs_meta.loc[dms_name].to_dict()


def make_msa_filepath(msa_file: str) -> str:
    relative_path = "ProteinGym/MSA_files/{msa_file}"
    return relative_path


def make_pdb_filepath(pdb_file: str) -> str:
    raise NotImplementedError()


def make_gym_metadata(dms_name: str) -> FitnessTaskMetadata:
    metadata = get_gym_metadata(dms_name)
    # we'll assume these are relative paths to hub for now
    # but could equally well act as relative local filepaths.
    # TODO: figure out whether we need to download or can use fsppec
    msa_file = make_msa_filepath(metadata["MSA_filename"])
    assert os.path.isfile(msa_file), f"MSA file {msa_file} not found"
    # pdb_file = proteingym.make_pdb_filepath(metadata) TODO
    return FitnessTaskMetadata(
        dms_name=dms_name,
        wt_sequence=metadata["target_seq"],
        msa_file=msa_file,
        msa_format="gym",
        pdb_file=None,
    )


def get_from_harvard(dms_name: str, download_zero_shot: bool = False) -> str:
    # Check if the harvard file is already downloaded
    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    data_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_dir)))
    gym_dir = f"{data_base_dir}/data/ProteinGym/"
    harvard_dir = f"{gym_dir}/ProteinGym_v1.1"
    harvard_file = f"{gym_dir}/ProteinGym_v1.1.zip"
    os.makedirs(gym_dir, exist_ok=True)
    if not os.path.exists(harvard_file):
        # Download the file from harvard
        log.info("Downloading from harvard")
        url = "https://zenodo.org/records/13936340/files/ProteinGym_v1.1.zip?download=1"
        os.system(f"wget {url} -O {harvard_file}")
    # unzip if not unzipped
    zero_shot_file = os.path.join(harvard_dir, "zero_shot_substitutions_scores.zip")
    subs_file = os.path.join(harvard_dir, "DMS_ProteinGym_substitutions.zip")
    if not os.path.exists(zero_shot_file) or not os.path.exists(subs_file):
    	# unzip the file
        os.system(f"unzip {harvard_file} -d {gym_dir}")
    if download_zero_shot:
        # Check if zero-shot directory is already unzipped by checking for the existence of any subdir
        zero_shot_dir = os.path.join(harvard_dir, "zero_shot_substitutions_scores")
        os.makedirs(zero_shot_dir, exist_ok=True)
        has_subdirs = any(
            os.path.isdir(os.path.join(zero_shot_dir, subdir))
            for subdir in os.listdir(zero_shot_dir)
        )
        if not has_subdirs:
            # unzip the file
            os.system(f"unzip {zero_shot_file} -d {harvard_dir}")
        return zero_shot_dir
    # if not download_zero_shot, check for exact file:
    subs_dir = os.path.join(harvard_dir, "DMS_ProteinGym_substitutions")
    if not os.path.exists(f"{subs_dir}/{dms_name}.csv"):
        # Unzip the file
        os.system(f"unzip {subs_file} -d {harvard_dir}")
        assert os.path.exists(
            f"{subs_dir}/{dms_name}.csv"
        ), f"Failed to find {dms_name}.csv in {subs_dir} after unzipping"
    # Change data path to the harvard directory
    dms_data_path = f"{subs_dir}/{dms_name}.csv"
    return dms_data_path


def load_gym_dataset(  # noqa: CCR001
    dms_name: str, zero_shot_dms_names: Set[str] = set()  # noqa: B006
) -> datasets.Dataset:
    """Load Dataset instance from ProteinGym csv file

    Args:
        dms_name: name of dms dataset used internally by ProteinGym.
        e.g. IF1_ECOLI_Kelsic_2016. Equivalent to DMS_id column in ProteinGym
        reference file.
    """
    dms_data_path = get_from_harvard(dms_name)

    hf_data = datasets.load_dataset(
        "csv",
        data_files=dms_data_path,
    )["train"]

    if dms_name not in zero_shot_dms_names:
        return hf_data

    # Merge zero-shot predictions (required for NPT on validation data)
    zero_shot_path = get_from_harvard(dms_name, download_zero_shot=True)
    zero_shot_df = pd.DataFrame()
    for model in SUBS_ZERO_SHOT_COLS:
        if "_L" in model:
            csv_path = os.path.join(
                zero_shot_path, model.split("_L")[0], model, f"{dms_name}.csv"
            )
        else:
            csv_path = os.path.join(zero_shot_path, model, f"{dms_name}.csv")
        model_zero_shot_df = pd.read_csv(csv_path)
        index_str = SUBS_ZERO_SHOT_COLS_to_index[model]
        zero_shot_df[model] = model_zero_shot_df[index_str]

    # standardise zero shot predictions before split
    standardized_cols = [f"standardized_{c}" for c in SUBS_ZERO_SHOT_COLS]
    zero_shot_df[standardized_cols] = standardize(zero_shot_df[SUBS_ZERO_SHOT_COLS])
    zero_shot_df = zero_shot_df[SUBS_ZERO_SHOT_COLS + standardized_cols]
    if len(hf_data) > len(zero_shot_df):
        log.info(f"Lengths: {len(hf_data)} vs {len(zero_shot_df)}")
        raise ValueError(f"Not enough zero shot predictions for {dms_name}.")
    # Add zero_shot pandas dataframe to hf_data
    # Convert hf_data to a pandas DataFrame to combine it with zero_shot_df
    hf_data_df = hf_data.to_pandas()
    # Combine the two dataframes in pandas, but only keep row if the row is in hf_data
    zero_shot_df = zero_shot_df[zero_shot_df.index.isin(hf_data_df.index)]
    assert len(zero_shot_df) == len(hf_data_df), "Length mismatch"
    combined_df = pd.concat([hf_data_df, zero_shot_df], axis=1)
    # Convert the combined DataFrame back to a Hugging Face Dataset
    combined_dataset = datasets.Dataset.from_pandas(combined_df)

    return combined_dataset


def count_num_mutations(mutant_code: str) -> int:
    return len(mutant_code.split(":"))


def subsample_dataset(
    input_dataset: datasets.Dataset,
    n_rows: int,
    seed: Optional[int] = None,
    generator: Optional[np.random.Generator] = None,
) -> datasets.Dataset:
    """
    Sample a specified number of rows from a Hugging Face dataset.

    Args:
        input_dataset (datasets.Dataset): The original dataset to sample from.
        n_rows (int): The number of rows to sample.
        seed (int, optional): Random seed for reproducibility. Default is None.

    Returns:
        datasets.Dataset: A new dataset containing the sampled rows.
    """
    # Shuffle the input dataset
    shuffled_dataset = input_dataset.shuffle(seed=seed, generator=generator)

    # Select the first n_rows from the shuffled dataset
    sampled_dataset = shuffled_dataset.select(list(range(n_rows)))

    return sampled_dataset


def subsample_splits(
    train_dataset: datasets.Dataset,
    test_dataset: datasets.Dataset,
    num_train: int,
    num_test: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    # TODO handle num_train / num_test greater than dataset size
    assert num_train is not None
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = None
    train_dataset = subsample_dataset(train_dataset, num_train, generator=rng)
    if num_test is not None:
        test_dataset = subsample_dataset(test_dataset, num_test, generator=rng)
    return train_dataset, test_dataset


def split_dataset(
    dataset: datasets.Dataset,
    num_train: int,  # TODO: allow None for non-random splits?
    num_test: Optional[int] = None,
    split_type: str = "random",
    dms_name: Optional[str] = None,
    seed: Optional[int] = None,
    minimum_test_set_size: int = 32,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """Split dataset based on configured splits.

    Split types are based on split types used in the paper:
    `Benchmark tasks in fitness landscape inference for proteins` (FLIP)

    Currently supported:
        random: random train / test split
        low_vs_high: train on sequences with DMS_score < WT, test on higher
        one_vs_many: train on single mutants, test on multi-mutants

    Args:
        dataset: datasets.Dataset
        num_train: int number of training points
        num_test: number of test points. If None, set automatically to
            complement of num_train
        split_type: type of split (random, low_vs_high, one_vs_many)
        dms_name: name of dataset in ProteinGym. Only required for low_vs_high
            split_type. dms_name is value in DMS_id field of ProteinGym
            reference files.
        seed: random seed to use for splitting.
        minimum_test_set_size: if the training set size and the test set size are
            greater than the dataset size, we automatically adjust the test
            set size, raising an exception if there are fewer than this
            number of sequences in the test set (i.e. the complement of the
            training set)
    """
    if split_type == "random":
        if num_test is None:
            assert num_train is not None  # would lead to unexpected behaviour
        if num_test is not None and num_train + num_test > len(dataset):
            assert num_train < (
                len(dataset) - minimum_test_set_size
            ), f"num_train too large relative to dataset size ({num_train}) vs ({len(dataset)})"
            num_test = len(dataset) - num_train
            print(
                "Warning: train and test sizes combined are greater than dataset size: "
                f"automatically re-setting num_test to len(dataset)-num_train={num_test}"
            )
        splits = dataset.train_test_split(
            test_size=num_test, train_size=num_train, seed=seed
        )
        return splits["train"], splits["test"]
    elif split_type == "one_vs_many":
        condition = dataset["mutant"].map(count_num_mutations) == 1
        train_dataset = dataset.filter(condition)
        test_dataset = dataset.filter(condition)
        # TODO write test
        return subsample_splits(
            train_dataset,
            test_dataset,
            num_train=num_train,
            num_test=num_test,
            seed=seed,
        )
    else:
        assert (
            dms_name is not None and dms_name in WT_VALUES
        ), f"no wt value for dms_name {dms_name}"
        condition = dataset["DMS_score"] < WT_VALUES[dms_name]
        train_dataset = dataset.filter(condition)
        test_dataset = dataset.filter(~condition)
        return subsample_splits(
            train_dataset,
            test_dataset,
            num_train=num_train,
            num_test=num_test,
            seed=seed,
        )
