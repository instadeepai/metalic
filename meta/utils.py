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
import random
import re
from typing import Generator, List, Optional, TextIO, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from meta.constants import BASEDIR

log = logging.getLogger("rich")


def get_device(
    gpu_index: Optional[int] = None, force_cpu: bool = False
) -> torch.device:
    if torch.cuda.is_available() and not force_cpu:
        if gpu_index is not None:
            return torch.device(f"cuda:{gpu_index}")
        else:
            return torch.device("cuda")
    else:
        return torch.device("cpu")


def display_info(config: DictConfig) -> None:
    log.info(f"AIChor Path: {os.environ.get('AICHOR_INPUT_PATH')}")
    log.info("Working directory : {}".format(os.getcwd()))
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")


def _read_fasta_lines(  # noqa: CCR001
    lines: TextIO,
    keep_gaps: bool = True,
    keep_insertions: bool = True,
    to_upper: bool = False,
) -> Generator[Tuple[str, str], None, None]:
    """Modified from esm."""
    seq = desc = None

    def parse(s: str) -> str:
        if not keep_gaps:
            s = re.sub("-", "", s)
        if not keep_insertions:
            s = re.sub(r"[a-z\.]", "", s)
        return s.replace(".", "-").upper() if to_upper else s

    for line in lines:
        if len(line) > 0 and line[0] == ">":
            if seq is not None:
                yield desc, parse(seq)
            desc = line.strip()[1:]
            seq = ""
        else:
            assert isinstance(seq, str)
            seq += line.strip()

    assert isinstance(seq, str) and isinstance(desc, str)
    yield desc, parse(seq)


def fasta_generator(
    filepath: str,
    encoding: Optional[str] = None,
    keep_insertions: bool = True,
    keep_gaps: bool = True,
    to_upper: bool = False,
) -> Generator[Tuple[str, str], None, None]:
    with open(filepath, "r", encoding=encoding) as fin:
        yield from _read_fasta_lines(
            fin, keep_gaps=keep_gaps, keep_insertions=keep_insertions, to_upper=to_upper
        )


def read_fasta(
    filepath: str,
    encoding: Optional[str] = None,
    keep_insertions: bool = True,
    keep_gaps: bool = True,
    to_upper: bool = False,
) -> Tuple[List[str], List[str]]:
    gen = fasta_generator(
        filepath,
        encoding=encoding,
        keep_insertions=keep_insertions,
        keep_gaps=keep_gaps,
        to_upper=to_upper,
    )
    names, seqs = [], []
    for n, s in gen:
        names.append(n)
        seqs.append(s)
    return names, seqs


def read_msa(msa_file: str, msa_format: str) -> Tuple[List[str], List[str]]:
    if msa_format == "a3m":
        return read_fasta(msa_file, keep_insertions=False, to_upper=True)
    elif msa_format == "gym":
        return read_fasta(msa_file, keep_insertions=True, to_upper=True)
    else:
        raise NotImplementedError(f"MSA format {msa_format} not supported")


def get_current_git_commit_hash() -> str:
    # use case for AICHOR when VCS_SHA exists
    vcs_sha = os.getenv("VCS_SHA")
    if vcs_sha is not None:
        return vcs_sha
    # if it does not, assume we run locally and .git is available
    try:
        with open(os.path.join(BASEDIR, ".git/HEAD")) as f:
            ref = f.read().strip().split()[1]
        with open(os.path.join(BASEDIR, f".git/{ref}")) as f:
            commit_hash = f.read().strip()
    except (FileNotFoundError, IndexError) as e:
        log.info(f"Commit hash not found: {e}")
        commit_hash = "unknown commit hash"

    return commit_hash


def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    log.info(f"Random seed set as {seed}")
    log.info(np.random.randn(3))
