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
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Tuple, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf

from meta.dataclasses import OraclePoints
from meta.models.base_metasurrogate import BaseMetaSurrogate
from meta.tasks.metrics import get_metric_function
from meta.tasks.proteingym import data_loading
from meta.tasks.utils import oracle_points_from_hf_dataset

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]")
log = logging.getLogger("rich")


class ProteinGymMetaSLTask:  # Cannot implement SupervisedTask since the return types differ
    def __init__(
        self,
        task_name: str,
        dataset: DictConfig,
        metric_names: List[str],
        num_evals_for_chunked: int,  # Number of evaluations to perform per task dataset.
        mutant_type: str,  # Can be "single", multiple, or "both"
        allow_partial_query_set: bool,
        early_stop_data_type: str,
    ):
        self.name = task_name
        self.dataset_cfg = dataset or OmegaConf.create({})  # for typing
        self.metric_fns = {
            metric_name: get_metric_function(metric_name)
            for metric_name in metric_names
        }
        self._is_setup = False
        self._has_metadata = True
        self.data_splits: Dict[str, Dict[str, OraclePoints]] = {}
        self.metadata: Dict[str, Dict] = {}

        assert mutant_type in {"single", "multiple", "both"}
        self.mutant_type = mutant_type
        self.allow_partial_query_set = allow_partial_query_set
        self.num_evals_for_chunked = num_evals_for_chunked
        self.early_stop_data_type = early_stop_data_type

        self.val_dms_names = (
            None
            if self.dataset_cfg.val_dms_names is None
            else set(self.dataset_cfg.val_dms_names.split(","))
        )

        self.train_dms_names = (
            None
            if self.dataset_cfg.train_dms_names is None
            else set(self.dataset_cfg.train_dms_names.split(","))
        )

    @property
    def has_metadata(self) -> bool:
        return self._has_metadata

    def load_raw_dataset(self, load_zero_shot: bool) -> Any:  # noqa: CCR001
        dms_sets = data_loading.subs_meta  # get metadata
        self.metadata = dms_sets

        # Check that there are no overlapping wild types between single and multiple mutants
        single_mutants = dms_sets[~dms_sets["includes_multiple_mutants"]]
        multiple_mutants = dms_sets[dms_sets["includes_multiple_mutants"]]
        single_dms_names = set(single_mutants.index.tolist())
        multiple_dms_names = set(multiple_mutants.index.tolist())
        num_overlapping_wt = len(
            set(single_mutants["UniProt_ID"]) & set(multiple_mutants["UniProt_ID"])
        )
        assert (
            num_overlapping_wt == 0
        ), "Overlapping wild types between single and multiple mutants"

        # Limit to single or multiple mutants if specified
        if self.mutant_type == "single":
            dms_names = single_dms_names
        elif self.mutant_type == "multiple":
            dms_names = multiple_dms_names
        else:
            assert self.mutant_type == "both", "Invalid mutant type"
            dms_names = set(dms_sets.index.tolist())

        # Check that validation dms are in available dms names
        for val_dms_name in self.val_dms_names:  # type: ignore
            if val_dms_name not in dms_names:
                assert (
                    val_dms_name not in self.dataset_cfg.skip_dms_names
                ), "Validation dms cannot be in skip_dms_names."
                log.warning(
                    f"Found a validation dms, {val_dms_name}, not in available dms names"
                    " and not in skip_dms_names. Adding it manually. Note that this may"
                    " mean it has a different number of mutants from the training data."
                )
                dms_names.add(val_dms_name)

        # Load the datasets
        raw_dataset = {
            dms_name: data_loading.load_gym_dataset(
                dms_name,
                zero_shot_dms_names=(
                    dms_names if load_zero_shot else set()  # type: ignore
                ),  # type: ignore
            )
            for dms_name in dms_names
            if dms_name not in self.dataset_cfg.skip_dms_names
        }
        log.info(f"Loaded {len(raw_dataset)} datasets")

        # check for overlapping proteins between single and multiple mutants
        if self.mutant_type == "both":
            single_mutant_data = {
                k: v for k, v in raw_dataset.items() if k in set(single_mutants.index)
            }
            multiple_mutant_data = {
                k: v for k, v in raw_dataset.items() if k in set(multiple_mutants.index)
            }
            single_proteins = {
                s
                for dataset in single_mutant_data.values()
                for s in dataset["mutated_sequence"]
            }
            multiple_proteins = {
                s
                for dataset in multiple_mutant_data.values()
                for s in dataset["mutated_sequence"]
            }
            overlapping_proteins = single_proteins & multiple_proteins
            num_overlap = len(overlapping_proteins)
            if num_overlap > 0:
                print(
                    f"Warning: Found {num_overlap} overlapping "
                    f"proteins between single and multiple mutants."
                )
                multi_lanscapes_overlapping = [
                    dms_name
                    for dms_name, dataset in multiple_mutant_data.items()
                    if len(set(dataset["mutated_sequence"]) & overlapping_proteins) > 0
                ]
                print(
                    f"Multi-mutant datasets with overlap: {multi_lanscapes_overlapping}"
                )
                single_lanscapes_overlapping = [
                    dms_name
                    for dms_name, dataset in single_mutant_data.items()
                    if len(set(dataset["mutated_sequence"]) & overlapping_proteins) > 0
                ]
                print(
                    f"Single-mutant datasets with overlap: {single_lanscapes_overlapping}"
                )
            else:
                print("No overlapping proteins between single and multiple mutants.")

        # Remove datasets with proteins that are too long
        max_protein_lengths = [
            max(len(s) for s in d["mutated_sequence"]) for d in raw_dataset.values()
        ]
        log.info(f"Max protein lengths: {max_protein_lengths}")
        log.info(f"Max protein length overall: {max(max_protein_lengths)}")
        if self.dataset_cfg.max_protein_length is not None:
            for dms_name in list(raw_dataset.keys()):
                if (
                    max(len(s) for s in raw_dataset[dms_name]["mutated_sequence"])
                    > self.dataset_cfg.max_protein_length
                ):
                    raw_dataset.pop(dms_name)
        log.info(
            f"Max protein length after removing proteins: {self.dataset_cfg.max_protein_length}"
        )
        log.info(f"Loaded {len(raw_dataset)} datasets after removing long proteins")

        if self.dataset_cfg.restrict_multi is not None:
            # Restrict multi-mutant data using randomly using seed
            assert (
                self.mutant_type == "both"
            ), "Can only restrict multi-mutant data if using both types"
            multi_mutants_in_data = set(multiple_mutants.index) & set(
                raw_dataset.keys()
            )
            assert self.val_dms_names is not None, "Must specify val_dms_names"
            # Remove val_data_names from multi_dms_names
            multi_mutants_in_data = {
                n for n in multi_mutants_in_data if n not in self.val_dms_names
            }
            # Restrict to given number
            random_st = np.random.RandomState(seed=self.dataset_cfg.seed)
            restricted_multi_names = set(
                random_st.choice(
                    list(multi_mutants_in_data),
                    self.dataset_cfg.restrict_multi,
                    replace=False,
                )
            )
            assert len(restricted_multi_names) == self.dataset_cfg.restrict_multi
            for dms_name in list(raw_dataset.keys()):
                if (
                    dms_name not in restricted_multi_names
                    and dms_name not in self.val_dms_names
                    and dms_name not in single_dms_names
                ):
                    raw_dataset.pop(dms_name)
            log.info(f"Loaded {len(raw_dataset)} datasets after restricting multiples")

        # Sort raw dataset by name
        raw_dataset = dict(sorted(raw_dataset.items()))

        return raw_dataset

    def make_data_splits(
        self, load_zero_shot: bool
    ) -> Dict[str, Dict[str, OraclePoints]]:  # noqa: CCR001
        """Split dataset into train, and val."""

        datasets_dict = {}

        # Get the validation data
        data_remaining = self._raw_dataset
        random_st = np.random.RandomState(seed=self.dataset_cfg.seed)
        num_val = 0  # Set below
        if self.val_dms_names is None:
            # self.val_dms_names is not specified
            assert (
                self.dataset_cfg.num_val is not None
            ), "Must specify num_val if not using val_dms_names"
            assert (
                len(data_remaining) >= self.dataset_cfg.num_val
            ), "Not enough data for validation"
            # Select val_dms_names randomly
            self.val_dms_names = random_st.choice(  # type: ignore
                list(data_remaining.keys()), self.dataset_cfg.num_val, replace=False
            )
            num_val = self.dataset_cfg.num_val
        else:
            # self.val_dms_names is specified
            assert (
                self.dataset_cfg.num_val is None
            ), "Must not specify num_val if using val_dms_names"
            num_val = len(self.val_dms_names)
        # Define the validation datasets using the keys
        val_dataset = {k: data_remaining[k] for k in self.val_dms_names}  # type: ignore
        # Remove the selected validation datasets from the remaining data
        data_remaining = {
            k: v for k, v in data_remaining.items() if k not in self.val_dms_names  # type: ignore
        }

        # Check the total number of datasets available
        num_train = (
            len(self._raw_dataset) - num_val
            if self.dataset_cfg.num_train is None
            else self.dataset_cfg.num_train
        )
        num_requested_datasets = num_train + num_val
        assert len(self._raw_dataset) >= num_requested_datasets, (
            f"Only {len(self._raw_dataset)} datasets available, but"
            f" {num_requested_datasets} are requested."
        )
        log.info(
            f"Using {num_train} training datasets and {num_val} validation datasets"
        )

        # Get the training data
        if self.train_dms_names is None:
            self.train_dms_names = random_st.choice(
                list(data_remaining.keys()), num_train, replace=False
            )
        train_dataset = {k: data_remaining[k] for k in self.train_dms_names}
        # Remove the selected training datasets from the remaining data
        data_remaining = {
            k: v for k, v in data_remaining.items() if k not in self.train_dms_names
        }

        # Check to make sure the validation set is not in the training set
        assert (
            len(set(train_dataset.keys()) & set(val_dataset.keys())) == 0
        ), "Validation set overlaps with training set"

        # Convert the datasets to oracle points
        datasets_dict["train"] = {
            k: oracle_points_from_hf_dataset(
                v,
                sequence_col="mutated_sequence",
                value_col="DMS_score",
                add_zero_shot=load_zero_shot,
            )
            for k, v in sorted(train_dataset.items())
        }
        datasets_dict["validation"] = {
            k: oracle_points_from_hf_dataset(
                v,
                sequence_col="mutated_sequence",
                value_col="DMS_score",
                add_zero_shot=load_zero_shot,
            )
            for k, v in sorted(val_dataset.items())
        }

        return datasets_dict

    def data_summary(self) -> Mapping[str, Union[float, int]]:
        summary = {
            "num_train": len(self.data_splits["train"]),
            "train_mean": np.mean(
                [
                    np.mean(data.oracle_values)
                    for data in self.data_splits["train"].values()
                ]
            ),
        }
        if "validation" in self.data_splits:
            summary["num_val"] = len(self.data_splits["validation"])
            summary["val_mean"] = np.mean(
                [
                    np.mean(data.oracle_values)
                    for data in self.data_splits["validation"].values()
                ]
            )
        return summary

    def make_empty_dataset(self) -> Mapping[str, OraclePoints]:
        """Convenience method to return empty datasets in setup_datasets if no data available."""
        return {
            "dummy_dms": OraclePoints(candidate_points=[], oracle_values=np.empty((0,)))
        }

    def setup_datasets(self, load_zero_shot: bool) -> None:
        """Load and standardise datasets before exposing to surrogate."""
        self._raw_dataset = self.load_raw_dataset(load_zero_shot)
        self.data_splits = self.make_data_splits(load_zero_shot)
        assert (
            "train" in self.data_splits
        ), "Must set train dataset (though they can be empty)"

    def evaluate_surrogate_predictions(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> Mapping[str, Union[float, np.number]]:
        """Evaluate predictions of a surrogate model.

        Given target values associated with test datapoints.
        Returns a dictionary of metrics.

        Args:
            predictions: (n,) array of predictions
            targets: (n,) array of targets
        """
        return {
            metric_name: metric_fn(predictions, targets)
            for metric_name, metric_fn in self.metric_fns.items()
        }

    def evaluate_surrogate(  # noqa: CCR001
        self,
        surrogate: BaseMetaSurrogate,
        split_name: str,
    ) -> Tuple[
        Mapping[str, Union[float, np.number]], np.ndarray, np.ndarray, List[str]
    ]:
        """Evaluate surrogate model.

        Split the data into support sets (context) and query sets (targets).

        Returns metrics as well as predictions.
        """
        # Make a new random state based on the seed each time, so results stay the same
        random_st = np.random.RandomState(self.dataset_cfg.seed)
        all_oracle_values: List[float] = []
        all_predictions: List[float] = []
        all_seqs: List[str] = []
        metrics_samples = defaultdict(list)
        task_to_metric_to_samples = defaultdict(lambda: defaultdict(list))  # type: ignore
        support_size_for_chunked = (
            surrogate.train_config.default_support_sz
            if surrogate.train_config.support_size is None
            else surrogate.train_config.support_size
        )
        chunked_str = "chunked_" + str(support_size_for_chunked)
        generators_and_evals = [
            (chunked_str, self.num_evals_for_chunked),
        ]

        # Iterate over val data in different ways
        for gen_name, num_evals in generators_and_evals:

            # Iterate over numver of evaluations
            for _ in range(num_evals):

                # Store the metrics for each task in a list
                metrics_samples_by_task = defaultdict(list)
                # Iterate over the tasks
                for task_name, task_data in self.data_splits[split_name].items():

                    # Get the next batch of data from a generator
                    if gen_name == chunked_str:
                        # Get query size, and set default if randomized
                        query_size = (
                            surrogate.train_config.default_query_sz
                            if surrogate.train_config.query_size is None
                            else surrogate.train_config.query_size
                        )
                        # Generator chosen to match Hawkins-Hooker et al. splits
                        generator = surrogate.generate_support_chunked_query_splits(
                            {task_name: task_data},
                            support_size_for_chunked,
                            query_size,
                            128,
                            2000,
                            1,
                            random_st,
                            surrogate.train_config.landscape_normalization,
                            self.allow_partial_query_set,
                        )
                    else:
                        raise ValueError(f"Unknown generator name: {gen_name}")
                        # TODO implement chunked_512. Right now it requires too much memory.

                    # Get next support/query set from generator
                    support_set, query_set, task_name_from_gen, early_stop_set = next(
                        generator
                    )

                    # Make sure we are at the end of the generator
                    try:
                        next(generator)
                        raise ValueError("Generator did not end")
                    except StopIteration:
                        pass

                    # Makre sure chunked query set is not partial if not allowed
                    if not self.allow_partial_query_set and gen_name == chunked_str:
                        assert len(query_set) % query_size == 0, (
                            len(query_set.oracle_values),
                            query_size,
                        )

                    assert task_name_from_gen == task_name, "Task name mismatch"

                    # Determine early stopping data (overwrite if we are not using it)
                    early_stop_data_type = (
                        "none"
                        if early_stop_set is None  # happens if gen_name == "default"
                        else self.early_stop_data_type
                    )
                    original_support_set = support_set
                    if early_stop_data_type == "none":
                        early_stop_set = (
                            None  # Potentially overwrite if we are not using it
                        )
                    elif early_stop_data_type == "split":
                        # Split the support set into half for train and early stopping data
                        early_stop_set = original_support_set[
                            len(original_support_set) // 2 :
                        ]
                        support_set = original_support_set[
                            : len(original_support_set) // 2
                        ]
                    else:
                        assert (
                            early_stop_data_type == "additional"
                        ), f"Invalid early stop data: {early_stop_data_type}"

                    # Store predictions from surrogate and other baselines
                    modelname_and_preds = []

                    # Store all sequences and oracle values
                    all_seqs.extend(query_set.sequences)
                    all_oracle_values.extend(query_set.oracle_values)

                    # Predict on the query set
                    query_predictions = surrogate.predict(
                        (
                            support_set
                            if surrogate.should_finetune(original_support_set)
                            else original_support_set
                        ),  # do not split support if not finetuning
                        query_set.candidate_points,
                        task_name,
                        early_stop_set=early_stop_set,  # early_stop_set only used if finetuning
                    )
                    # # For debugging (replace query_predictions above with this):
                    # query_predictions = np.random.rand(
                    #     *query_set.oracle_values.shape
                    # )
                    # Store predictions
                    all_predictions.extend(query_predictions)

                    # Make sure the predictions are the right shape
                    assert len(query_predictions) == len(query_set.oracle_values), (
                        len(query_predictions),
                        len(query_set.oracle_values),
                    )
                    # Store the predictions
                    modelname_and_preds.append(("", query_predictions))

                    # Iterate over preds from surrogate(s)
                    for model_name, model_preds in modelname_and_preds:

                        # Get metrics for the predictions on just this evaluation on this landscape
                        task_metrics = self.evaluate_surrogate_predictions(
                            np.array(model_preds), np.array(query_set.oracle_values)
                        )

                        # Prepend generator and model name
                        task_metrics = {
                            f"{gen_name}{model_name}_{k}": v
                            for k, v in task_metrics.items()
                        }

                        # Remove NaNs, which can occur for spearman if all preds are the same
                        task_metrics = {
                            k: v for k, v in task_metrics.items() if not np.isnan(v)
                        }

                        # Save metrics for this task
                        for metric_name, metric in task_metrics.items():
                            metrics_samples_by_task[metric_name].append(metric)
                            task_to_metric_to_samples[task_name][metric_name].append(
                                metric
                            )

                # Save metric statistics over tasks
                for metric_name, metric_list in metrics_samples_by_task.items():
                    for statistic_name, statistic_fn in [
                        ("mean", np.mean),
                        ("std", np.std),
                        ("min", np.min),
                        ("max", np.max),
                    ]:
                        statistic_across_tasks = statistic_fn(np.array(metric_list))  # type: ignore
                        key_for_statistic = metric_name + "_task_" + statistic_name
                        metrics_samples[key_for_statistic].append(
                            statistic_across_tasks
                        )

        # Compute the mean over samples, each of which represents all tasks
        metrics = {
            metric_name: np.mean(sample_list)
            for metric_name, sample_list in metrics_samples.items()
        }

        # Compute the std over samples, each of which represents all tasks
        for metric_name, sample_list in metrics_samples.items():
            metrics[f"{metric_name}_std"] = (
                np.std(sample_list) if len(sample_list) > 1 else 0
            )

        # Save each sample for each metric
        for metric_name, sample_list in metrics_samples.items():
            for i, sample in enumerate(sample_list):
                metrics[f"{metric_name}_sample_{i}"] = sample

        # Save the mean for each landscape, over samples
        for task_name, metric_to_samples in task_to_metric_to_samples.items():
            for metric_name, metric_samples in metric_to_samples.items():
                metrics[f"{metric_name}_{task_name}"] = np.mean(metric_samples)

        return metrics, np.array(all_predictions), np.array(all_oracle_values), all_seqs
