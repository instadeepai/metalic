import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from meta.constants import BASEDIR

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]")
log = logging.getLogger("rich")


NPT_DATA_DIR = os.path.join(
    BASEDIR, "data/ProteinNPT_data"
)  # TODO check consistency with scripts
NPT_REPO_DIR = os.path.join(BASEDIR, "baselines/proteinnpt")
SINGLES_DIR = os.path.join(NPT_DATA_DIR, "data/fitness/substitutions_singles")
MULTIPLES_DIR = os.path.join(NPT_DATA_DIR, "data/fitness/substitutions_multiples")
INDELS_DIR = os.path.join(NPT_DATA_DIR, "data/fitness/indels")
# TODO: what is the single_substitutions folder within the substitutions zero shot folder?
ZERO_SHOT_SUBSTITUTIONS_DIR = os.path.join(
    NPT_DATA_DIR, "data/zero_shot_fitness_predictions/substitutions"
)
ZERO_SHOT_INDELS_DIR = os.path.join(
    NPT_DATA_DIR, "data/zero_shot_fitness_predictions/indels"
)

SUBS_ZERO_SHOT_COLS = [
    "Tranception_L",
    "ESM1v",
    "MSA_Transformer_ensemble",
    "DeepSequence_ensemble",
    "TranceptEVE_L",
]


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


# TODO: check extensions
def make_msa_filepath(msa_file: str) -> str:
    return os.path.join(NPT_DATA_DIR, "data/MSA/MSA_files", msa_file)


def make_msa_weights_filepath(msa_weights_file: str) -> str:
    return os.path.join(NPT_DATA_DIR, "data/MSA/MSA_weights", msa_weights_file)

def load_mutants_df(
    dms_name: str,
    category: str = "substitutions_singles",
    merge_zero_shot_predictions: bool = False,
) -> pd.DataFrame:
    """Load mutants df.

    Based on proteinnpt utils.data_utils.get_train_val_test_data,
    but without support for multiple targets.
    """
    if category == "substitutions_singles":
        df = pd.read_csv(os.path.join(SINGLES_DIR, f"{dms_name}.csv"))
    elif category == "substitutions_multiples":
        df = pd.read_csv(os.path.join(MULTIPLES_DIR, f"{dms_name}.csv"))
    elif category == "indels":
        df = pd.read_csv(os.path.join(INDELS_DIR, f"{dms_name}.csv"))
    else:
        raise ValueError(f"Invalid category: {category}")
    if merge_zero_shot_predictions:
        log.info("Merging zero shot predictions")
        if category.startswith("substitutions"):
            log.info(
                "WARNING: not loading zero shot from singles directory but "
                "need to double check if we should"
            )
            zero_shot_df = pd.read_csv(
                os.path.join(ZERO_SHOT_SUBSTITUTIONS_DIR, f"{dms_name}.csv")
            )
            # standardise zero shot predictions before split: this is appropriate,
            # in the case where we know what sequences exist in the holdout set,
            # we just don't know their fitness values, as in the landscape tasks.
            # this is only used by zero shot covariate npt baselines
            standardized_cols = [f"standardized_{c}" for c in SUBS_ZERO_SHOT_COLS]
            zero_shot_df[standardized_cols] = standardize(
                zero_shot_df[SUBS_ZERO_SHOT_COLS]
            )
            zero_shot_df = zero_shot_df[
                ["mutant"] + SUBS_ZERO_SHOT_COLS + standardized_cols
            ]
        else:
            zero_shot_df = pd.read_csv(
                os.path.join(ZERO_SHOT_INDELS_DIR, f"{dms_name}.csv")
            )
            zero_shot_df["standardized_Tranception_L"] = standardize(
                zero_shot_df["Tranception_L"]
            )
            zero_shot_df = zero_shot_df[
                ["mutant", "Tranception_L", "standardized_Tranception_L"]
            ]

        df = pd.merge(df, zero_shot_df, how="inner", on="mutant")
    return df


def split_df(
    df: pd.DataFrame,
    num_train: int,  # TODO: allow None for non-random splits?
    num_test: Optional[int] = None,
    split_type: str = "random",
    # dms_name: Optional[str] = None,
    seed: Optional[int] = None,
    minimum_test_set_size: int = 32,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset based on configured splits.

    We use sklearn.train_test_split as this should have similar behaviour/args
    to datasets.Dataset.train_test_split in data/proteingym
    """
    if split_type == "random":
        if num_test is None:
            assert num_train is not None  # would lead to unexpected behaviour
        if num_test is not None and num_train + num_test > len(df):
            assert num_train < (
                len(df) - minimum_test_set_size
            ), f"num_train too large relative to dataset size ({num_train}) vs ({len(df)})"
            num_test = len(df) - num_train
            print(
                "Warning: train and test sizes combined are greater than dataset size: "
                f"automatically re-setting num_test to len(dataset)-num_train={num_test}"
            )
        if num_test is None:
            assert num_train is not None  # would lead to unexpected behaviour
        return train_test_split(
            df, test_size=num_test, train_size=num_train, random_state=seed
        )
    else:
        # TODO: support NPT-style splits or FLIP-style splits
        raise NotImplementedError()


def split_data_based_on_test_fold_index(
    dataframe: pd.DataFrame,
    test_fold_index: int,
    fold_variable_name: str = "fold_modulo_5",
    use_validation_set: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """From proteinnpt utils/data_utils.

    Args:
        test_fold_index: index of test fold: [0, 1, 2, 3, 4]
        fold_variable_name: name of column with fold information ['fold_modulo_5',
            'fold_random_5', 'fold_contiguous_5'
        ]
    """
    unique_folds = np.unique(dataframe[fold_variable_name])
    num_distinct_folds = len(unique_folds)
    if fold_variable_name == "fold_multiples":  # Q: what is this
        train = dataframe[dataframe[fold_variable_name] == 0]
        if use_validation_set:
            num_mutations_train = int(len(train) * 0.8)
            val = train[num_mutations_train + 1 :]  # noqa: E203
            train = train[: num_mutations_train + 1]
        else:
            val = None
        test = dataframe[dataframe[fold_variable_name] == 1]
    else:
        if use_validation_set:
            test = dataframe[dataframe[fold_variable_name] == test_fold_index]
            val_fold_index = (test_fold_index - 1) % num_distinct_folds
            val = dataframe[dataframe[fold_variable_name] == val_fold_index]
            train = dataframe[
                ~dataframe[fold_variable_name].isin([test_fold_index, val_fold_index])
            ]
        else:
            train = dataframe[dataframe[fold_variable_name] != test_fold_index]
            val = None
            test = dataframe[dataframe[fold_variable_name] == test_fold_index]
    del train[fold_variable_name]
    if use_validation_set:
        del val[fold_variable_name]
    del test[fold_variable_name]
    return train, val, test
