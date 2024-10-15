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

import abc
import logging
from typing import (
    Callable,
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
import torch
from numpy import number

from meta.dataclasses import Candidate, OraclePoints
from meta.logger import Logger

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]")
log = logging.getLogger("rich")


class BaseMetaSurrogate(abc.ABC):
    def __init__(
        self,
        name: str,
        support_size: Optional[
            int
        ],  # Number of support set datapoints to pass as context (shot). None for any size.
        query_size: Optional[
            int
        ],  # Number of query set datapoints to pass as targets. None for any size.
        use_all_data: bool,  # If True, all data is in support or query set. Ignored if sizes given.
        max_context_sz: int,  # Maximum size of dataset for training. (Will subsample if larger.)
        num_outputs: int = 1,
    ) -> None:
        """
        Args:
            name: name of surrogate model
            num_outputs: number of values that are predicted by model for each candidate.
        """
        self.name = name
        self.num_outputs = num_outputs
        self.support_size = support_size
        self.query_size = query_size
        self.max_context_sz = max_context_sz
        self.use_all_data = use_all_data
        self.metadata: Dict[str, Dict] = {}
        self.save_dir: Optional[str] = None

    @abc.abstractmethod
    def cleanup(self) -> None:
        """Delete any temporary files, checkpoints etc that aren't being persisted."""
        pass

    def set_dir(self, save_dir: str) -> None:
        self.save_dir = save_dir

    @abc.abstractmethod
    def fit(
        self,
        train_data: Dict[str, OraclePoints],
        seed: int,
        logger: Logger,
        eval_func: Callable,
        num_steps: Optional[int] = None,
    ) -> None:
        """
        Train surrogate model on multiple datsets of labelled candidates.
        n.b. Pass train_data to generate_support_query_splits() to generate
        data for training.
        """
        pass

    def generate_support_chunked_query_splits(  # noqa: CCR001
        self,
        task_data: Dict[str, OraclePoints],
        support_size: int,
        query_size: int,
        early_stop_size: int,  # Size of eval set to use for early stopping
        eval_size: int,
        num_evals: int,
        random_st: np.random.RandomState,
        normalization: Optional[str],
        allow_partial_query_set: bool,
    ) -> Generator[Tuple[OraclePoints, OraclePoints, str, OraclePoints], None, None]:
        """Generate fixed splits of each task, as in PrefBO, for validation.
        Allows the query set to be chunked over multiple forwrd passes.
        Split the data into support sets (context) and query sets (targets).
        """
        for _ in range(num_evals):
            for task_name, oracle_points in task_data.items():
                oracle_points = oracle_points.normalize(normalization)
                effective_eval_size = eval_size
                expected_size = support_size + early_stop_size + eval_size
                min_size = support_size + early_stop_size
                if not allow_partial_query_set:
                    min_size += query_size
                if len(oracle_points) < min_size:
                    raise ValueError(
                        f"Expected at least {min_size} data points,"
                        f" but got {len(oracle_points)}"
                    )
                if len(oracle_points) < expected_size:
                    print(
                        f"Warning: Expected at least {expected_size} data points,"
                        f" but got {len(oracle_points)}"
                    )
                    # Eval on all remaining data if not enough for support and eval sets
                    effective_eval_size = (
                        len(oracle_points) - support_size - early_stop_size
                    )
                    if not allow_partial_query_set:
                        # Round eval_size to be divisible by query_size
                        effective_eval_size = (
                            effective_eval_size // query_size
                        ) * query_size
                        if effective_eval_size == 0:
                            raise ValueError(
                                "Expected enough data for at least one query set."
                            )
                        assert (
                            effective_eval_size % query_size == 0
                        ), "eval_size must be divisible by query_size"
                permuted_data = oracle_points.permutation(random_st)
                support_set = permuted_data[:support_size]
                early_stop_set = permuted_data[
                    support_size : support_size + early_stop_size
                ]
                eval_set = permuted_data[
                    support_size
                    + early_stop_size : support_size
                    + early_stop_size
                    + effective_eval_size
                ]
                yield support_set, eval_set, task_name, early_stop_set

    def generate_support_query_splits(  # noqa: CCR001
        self,
        task_data: Dict[str, OraclePoints],
        random_st: np.random.RandomState,
        num_evals: Optional[int],
        normalization: Optional[str],
    ) -> Generator[Tuple[OraclePoints, OraclePoints, str, None], None, None]:
        """Generate multiple splits of each tast for training and evaluation.

        Split the data into support sets (context) and query sets (targets).
        Also returns the task name from which the batch was sampled.
        """
        tasks = list(task_data.items())
        eval_num = 0
        while num_evals is None or eval_num < num_evals:
            perm = random_st.permutation(len(tasks))
            for i in perm:
                task_name, dataset = tasks[i]
                dataset = dataset.normalize(normalization)

                # Define the support and query sets
                cur_support_size = self.support_size  # Size can chance per iteration
                cur_query_size = self.query_size
                permuted_data = dataset.permutation(random_st)
                if len(permuted_data) > self.max_context_sz:
                    permuted_data = permuted_data[: self.max_context_sz]
                if self.support_size is None and self.query_size is None:
                    assert (
                        self.use_all_data
                    ), "Must specify support_size and/or query_size if use_all_data is False."
                    split = random_st.randint(
                        0,
                        len(permuted_data)
                        - 1,  # There can be nothing only in support set
                    )
                    # Split all data between support and query sets
                    support_set, query_set = (
                        permuted_data[:split],
                        permuted_data[split:],
                    )
                elif self.support_size is None:
                    # Select suport set size
                    max_support_size = len(permuted_data) - self.query_size  # type: ignore
                    cur_support_size = (
                        max_support_size
                        if self.use_all_data
                        else random_st.randint(
                            0, max_support_size  # 0 is possible support size
                        )
                    )
                elif self.query_size is None:
                    # Select query set size
                    max_query_size = len(permuted_data) - self.support_size
                    cur_query_size = (
                        max_query_size
                        if self.use_all_data
                        else random_st.randint(1, max_query_size)
                    )
                cur_support_size = cast(int, cur_support_size)
                cur_query_size = cast(int, cur_query_size)
                # Define the sets based on the selected sizes
                support_set, query_set = (
                    permuted_data[:cur_support_size],
                    permuted_data[cur_support_size : cur_support_size + cur_query_size],
                )

                yield support_set, query_set, task_name, None  # None for early stop set

            eval_num += 1

    @abc.abstractmethod
    def _predict(
        self,
        support_set: OraclePoints,
        query_set: List[Candidate],
        task_name: str,
        early_stop_set: Optional[OraclePoints] = None,
        return_params: bool = False,
    ) -> Union[np.ndarray, List[torch.nn.Parameter]]:
        pass

    def check_predictions_shape(
        self, candidate_points: List[Candidate], predictions: np.ndarray
    ) -> None:
        expected_shape = (
            (len(candidate_points),)
            if self.num_outputs == 1
            else (len(candidate_points), self.num_outputs)
        )
        assert predictions.shape == expected_shape, (
            f"Expected _predict to return array with shape ({expected_shape}), "
            f"got shape {predictions.shape}"
        )

    def set_metadata(self, task_metadata: Dict[str, Dict]) -> None:
        self.metadata = task_metadata

    def predict(
        self,
        support_set: OraclePoints,
        query_set: List[Candidate],
        task_name: str,
        early_stop_set: Optional[OraclePoints] = None,
    ) -> np.ndarray:
        """
        Compute predictions at the candidate points.
        """
        predictions = self._predict(
            support_set, query_set, task_name, early_stop_set=early_stop_set
        )
        self.check_predictions_shape(query_set, predictions)  # type: ignore
        return predictions  # type: ignore

    @abc.abstractmethod
    def get_training_summary_metrics(
        self,
    ) -> Mapping[str, Union[float, int, number]]:
        pass
