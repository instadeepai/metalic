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
from typing import Any, Mapping, Optional


class Logger(abc.ABC):
    """A logger has a `write` method."""

    @abc.abstractmethod
    def write(self, data: Mapping[str, Any], *args: Any, **kwargs: Any) -> None:
        """Writes `data` to destination (file, terminal, database, etc)."""

    @abc.abstractmethod
    def write_artifact(self, file_name: str) -> None:
        """Writes an artifact to destination."""

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the logger, not expecting any further write."""

    @abc.abstractmethod
    def get_checkpoint(self, file_path: str) -> Optional[str]:
        """Downloads the checkpoint from a run with a specific id"""

    def __enter__(self) -> "Logger":
        return self

    def __exit__(
        self, exc_type: Exception, exc_val: Exception, exc_tb: Exception
    ) -> None:
        self.close()
