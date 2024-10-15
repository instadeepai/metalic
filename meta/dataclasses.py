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
# noqa: A005

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
from sklearn.model_selection import train_test_split


def create_all_single_mutants(
    candidate: Candidate,
    alphabet: str,
    mutation_range_start: Optional[int] = None,
    mutation_range_end: Optional[int] = None,
    include_null: bool = False,
) -> List[Candidate]:
    """Modified from Tranception:
    https://github.com/OATML-Markslab/Tranception/blob/main/design_app/Tranception_Design.ipynb
    """
    mutated_candidates = []
    if mutation_range_start is None:
        mutation_range_start = 1
    if mutation_range_end is None:
        mutation_range_end = len(candidate.sequence)
    for position, current_char in enumerate(
        candidate.sequence[mutation_range_start - 1 : mutation_range_end]  # noqa: E203
    ):
        for mutated_char in alphabet:
            if current_char != mutated_char:
                mutation = PointMutation(position, current_char, mutated_char)
                mutated_candidates.append(candidate.apply_mutation(mutation))
    if include_null:
        mutated_candidates.append(candidate)
    return mutated_candidates


@dataclass
class PointMutation:
    position: int
    from_char: str
    to_char: str

    def mutate(self, sequence: str) -> str:
        assert sequence[self.position] == self.from_char
        seq_list = list(sequence)
        seq_list[self.position] = self.to_char
        return "".join(seq_list)

    def __repr__(self) -> str:
        return f"{self.from_char}{self.position}{self.to_char}"

    @classmethod
    def from_str(cls, mutation_code: str) -> PointMutation:
        position = int(mutation_code[1:-1])
        from_char = mutation_code[0]
        to_char = mutation_code[-1]
        return cls(position, from_char, to_char)


@dataclass
class Candidate:
    def __init__(
        self,
        sequence: str,
        features: Any = None,
    ):
        self._sequence = sequence
        self._features = features

    def __repr__(self) -> str:
        return f"Candidate(sequence={self.sequence})"

    def __len__(self) -> int:
        return len(self.sequence)

    @property
    def sequence(self) -> str:
        return self._sequence

    @property
    def features(self) -> dict:
        return self._features

    def apply_mutation(self, mutation: PointMutation) -> Candidate:
        return Candidate(sequence=mutation.mutate(self._sequence))

    def apply_random_mutation(self, alphabet: str) -> Candidate:
        return self.apply_mutation(self.propose_random_mutation(alphabet))

    def propose_random_mutation(self, alphabet: str) -> PointMutation:
        position = np.random.choice(len(self.sequence))
        from_char = self.sequence[position]
        to_char = np.random.choice([c for c in alphabet if c != from_char])
        return PointMutation(position, from_char, to_char)

    def create_all_single_mutants(
        self,
        alphabet: str,
        mutation_range_start: Optional[int] = None,
        mutation_range_end: Optional[int] = None,
        include_null: bool = False,
    ) -> List[Candidate]:
        return create_all_single_mutants(
            self,
            alphabet,
            mutation_range_start=mutation_range_start,
            mutation_range_end=mutation_range_end,
            include_null=include_null,
        )


# N.B. much easier for typing not to have a separate poolsearchspace class...
# sure there's a way around it but haven't figured it out
@dataclass
class SearchSpace:

    """Dataclass for specifying search space."""

    alphabet: str
    length: Optional[int] = None
    candidate_pool: Optional[List[Candidate]] = None

    @property
    def pool_sequences(self) -> Optional[List[str]]:
        if self.candidate_pool is not None:
            return [cand.sequence for cand in self.candidate_pool]
        else:
            return None

    def check_is_valid(self, candidate: Candidate) -> None:
        if self.length is not None:
            assert (
                len(candidate) == self.length
            ), f"Candidates must be of length {self.length}, got {len(candidate)}"
        assert all(
            char in self.alphabet for char in candidate.sequence
        ), f"Not all letters in candidate {candidate} in alphabet {self.alphabet}"
        # TODO: make sure this isn't called unnecessarily
        if self.pool_sequences is not None:
            assert (
                candidate.sequence in self.pool_sequences
            ), f"Candidate {candidate} not in pool ({len(self.pool_sequences)})"

    def update(self, scored_candidates: OraclePoints) -> None:
        # n.b. training set of pool-based task must be in pool at init
        # to pass this check
        for cand in scored_candidates.candidate_points:
            self.check_is_valid(cand)
        if self.candidate_pool is not None:
            updated_pool_candidates = [
                candidate
                for candidate in self.candidate_pool
                if candidate.sequence not in scored_candidates.sequences
            ]
            self.candidate_pool = updated_pool_candidates


class OraclePoints:
    def __init__(
        self, candidate_points: List[Candidate], oracle_values: np.ndarray
    ) -> None:
        self.candidate_points = candidate_points
        self._oracle_values = oracle_values
        assert (
            len(candidate_points) == oracle_values.shape[0]
        ), "Candidates and oracles must be same length"

    @property
    def oracle_values(self) -> np.ndarray:
        return self._oracle_values

    @oracle_values.setter
    def oracle_values(self, new_oracle_values: np.ndarray) -> None:
        self._oracle_values = new_oracle_values

    @property
    def sequences(self) -> List[str]:
        return [cand.sequence for cand in self.candidate_points]

    def permutation(self, rand_st: np.random.RandomState) -> OraclePoints:
        perm = rand_st.permutation(len(self))
        return OraclePoints(
            [self.candidate_points[i] for i in perm],
            self.oracle_values[perm],
        )

    def normalize(self, norm_str: Optional[str]) -> OraclePoints:
        assert len(self) > 0, "Normalization requires at least one value"
        oracle_values = self.oracle_values
        if norm_str == "standardize":
            std = np.std(oracle_values) if len(self) > 1 else 1
            oracle_values = (oracle_values - oracle_values.mean()) / std
        elif norm_str == "normalize":
            oracle_values = oracle_values / np.mean(np.abs(oracle_values))
        else:
            assert norm_str is None, "Invalid normalization"
        return OraclePoints(self.candidate_points, oracle_values)

    def __len__(self) -> int:
        return len(self.candidate_points)

    def __getitem__(self, index: Union[int, slice]) -> OraclePoints:
        if isinstance(index, int):
            return OraclePoints(
                [self.candidate_points[index]],
                self.oracle_values[index : index + 1],
            )
        elif isinstance(index, slice):
            return OraclePoints(
                self.candidate_points[index],
                self.oracle_values[index],
            )
        else:
            raise TypeError("Index must be an integer or a slice")

    def append(
        self, candidate_points: List[Candidate], oracle_values: np.ndarray
    ) -> None:
        # TODO optionally return new instance (concat?)
        # TODO accept OraclePoints directly
        self.candidate_points.extend(candidate_points)
        self.oracle_values = np.concatenate((self.oracle_values, oracle_values), axis=0)


class AcquisitionPoints(NamedTuple):
    """
    Values returned by an acquisition function.
    """

    # Todo: May need to handle shape of ensemble, or GP mean/var.
    candidate_points: List[Candidate]
    acquisition_values: np.ndarray

    @property
    def sequences(self) -> List[str]:
        return [cand.sequence for cand in self.candidate_points]

    def __len__(self) -> int:
        return len(self.candidate_points)


@dataclass
class OptimizerState:
    def __init__(
        self,
        search_space: SearchSpace,
        datasets: Dict[str, OraclePoints],
        val_frac: float,
    ):
        self.search_space = search_space
        self.datasets = datasets
        self.val_frac = val_frac

    @property
    def train_dataset(self) -> OraclePoints:
        return self.datasets["train"]

    @property
    def test_dataset(self) -> OraclePoints:
        return self.datasets["test"]

    @property
    def validation_dataset(self) -> OraclePoints:
        return self.datasets["validation"]

    def add_to_training_dataset(self, scored_candidates: OraclePoints) -> None:
        # TODO check shapes (or will concatenate effectively check for us?)
        # N.B. this applies in the tell at init also
        if self.val_frac > 0:
            (
                new_train_candidates,
                new_valid_candidates,
                new_train_values,
                new_valid_values,
            ) = train_test_split(
                scored_candidates.candidate_points,
                scored_candidates.oracle_values,
                test_size=self.val_frac,
            )
        else:
            new_train_candidates = scored_candidates.candidate_points
            new_train_values = scored_candidates.oracle_values

        self.train_dataset.append(new_train_candidates, new_train_values)
        if self.val_frac > 0:
            self.validation_dataset.append(new_valid_candidates, new_valid_values)
        self.search_space.update(scored_candidates)  # e.g. update candidate pool

    def summary(self) -> Dict[str, Union[int, float]]:
        summary: Dict[str, Union[int, float]] = {
            "num_train": len(self.train_dataset),
            "num_test": len(self.test_dataset),
            "num_validation": len(self.validation_dataset),
        }
        if self.search_space.candidate_pool is not None:
            summary["candidate_pool_size"] = len(self.search_space.candidate_pool)
        return summary
