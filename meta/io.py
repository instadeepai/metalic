import json
import os
import pickle
from typing import Any, Dict, List, Optional, TextIO

import numpy as np
import pandas as pd
import torch
import yaml
from s3fs.core import S3FileSystem


def load_yaml(yaml_file: str) -> Any:
    # * Make FullLoader safer by removing python/object/apply from the default FullLoader
    # https://github.com/yaml/pyyaml/pull/347
    # Move constructor for object/apply to UnsafeConstructor
    with open(yaml_file, "r") as yf:
        return yaml.load(yf, Loader=yaml.UnsafeLoader)


def load_json(jsonfile: str) -> Any:
    with open(jsonfile, "r") as jf:
        res = json.load(jf)
    return res


class FileHandler:
    """Wrapper that gets either an s3 endpoint or None. If an s3 endpoint is provided,
    all methods will read and save the file to the indicated s3 bucket,
    otherwise the local disk is used. The goal is to use the same functions when
    the script is run on Ichor or on a local machine.

    (Modified from ID notion)
    """

    def __init__(
        self, s3_endpoint: Optional[str] = None, bucket: Optional[str] = "input"
    ) -> None:
        self.s3_endpoint = s3_endpoint
        self.bucket = bucket
        # Assert that bucket is in {"input", "output"}. This could change along with
        # Ichor's infrastructure but is for now structured as an input and ouput
        # bucket.
        # WARNING: if "bucket" == "ouput", the files will be stored under a folder
        # named after the experiment whereas saving in the "input" bucket is more
        # straightforward as the path is directly used.

        # Q: is there not a local filesystem object in fsspec which can be
        # used directly in place of this handler?
        if s3_endpoint:
            self.s3 = S3FileSystem(client_kwargs={"endpoint_url": s3_endpoint})
            if bucket == "input":
                self.bucket_path = os.environ["AICHOR_INPUT_PATH"]
            elif bucket == "output":
                self.bucket_path = os.environ["AICHOR_OUTPUT_PATH"]
            else:
                raise ValueError("bucket should be in {input, output}")
        else:
            self.bucket_path = "./"

    def expand_path(self, path: str) -> str:
        return os.path.join(self.bucket_path, path)

    # Handling numpy
    def read_numpy(self, path: str) -> np.ndarray:
        """Wrapper around the numpy load method.

        Args:
            path (str): Path to where the array is to be read.

        Returns:
            np.ndarray: Array.
        """
        if self.s3_endpoint:
            with self.s3.open(
                os.path.join(self.bucket_path, path)  # type: ignore
            ) as f:
                array = np.load(f, allow_pickle=True)
        else:
            array = np.load(path, allow_pickle=True)
        return array

    def open(self, path: str, *args: Any, **kwargs: Any) -> TextIO:
        if self.s3_endpoint:
            return self.s3.open(os.path.join(self.bucket_path, path), *args, **kwargs)
        else:
            return open(os.path.join(self.bucket_path, path), *args, **kwargs)

    def save_numpy(self, path: str, array: np.ndarray) -> None:
        """Wrapper around the numpy save method.

        Args:
            path (str): Path where the array is to be saved.
            array (np.ndarray): Array to be saved.
        """
        if self.s3_endpoint:
            with self.s3.open(
                os.path.join(self.bucket_path, path), "wb"  # type: ignore
            ) as f:
                f.write(pickle.dumps(array))
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, array)

    # Handling text files
    def read_text(self, path: str) -> List[str]:
        """Wrapper around the python text read method.

        Args:
            path (str): Path to which the text is to be read.

        Returns:
            List[str]: lines read from the text file.
        """
        if self.s3_endpoint:
            with self.s3.open(
                os.path.join(self.bucket_path, path)  # type: ignore
            ) as f:
                lines: List[str] = f.readlines()
        else:
            with open(os.path.join(self.bucket_path, path)) as f:  # type: ignore
                lines = f.readlines()

        return lines

    def save_text(self, path: str, lines: List[str]) -> None:
        """Wrapper around the python text save method.

        Args:
            path (str): Path to which the text is to be saved.
            lines (np.ndarray): Lines to be written in the text file.
        """
        if self.s3_endpoint:
            with self.s3.open(
                os.path.join(self.bucket_path, path), "w"  # type: ignore
            ) as f:
                for line in lines:
                    f.write(line)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(os.path.join(self.bucket_path, path), "w") as f:  # type: ignore
                for line in lines:
                    f.write(line)

    # Handling json files
    def read_json(self, path: str) -> Dict:
        """Wrapper around the python json load method

        Args:
            path (str): Path to which the text is to be saved.

        Returns:
            Dict: Dictionary written in the json file;
        """
        if self.s3_endpoint:
            with self.s3.open(
                os.path.join(self.bucket_path, path)  # type: ignore
            ) as f:
                data: Dict = json.load(f)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(os.path.join(self.bucket_path, path)) as f:  # type: ignore
                data = json.load(f)

        return data

    def save_json(self, path: str, data: Dict) -> None:
        """Wrapper around the python json save method

        Args:
            path (str): Path to which the text is to be saved.
            data (Dict): Dictionary to be saved.
        """
        if self.s3_endpoint:
            with self.s3.open(
                os.path.join(self.bucket_path, path), "w"  # type: ignore
            ) as f:
                json.dump(data, f, indent=4)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(os.path.join(self.bucket_path, path), "w") as f:  # type: ignore
                json.dump(data, f, indent=4)

    def read_yaml(self, path: str) -> Dict:
        if self.s3_endpoint:
            with self.s3.open(os.path.join(self.bucket_path, path)) as f:
                return yaml.load(f, Loader=yaml.UnsafeLoader)
        else:
            with open(os.path.join(self.bucket_path, path)) as f:
                return yaml.load(f, Loader=yaml.UnsafeLoader)

    def save_yaml(self, path: str, data: Dict) -> None:
        if self.s3_endpoint:
            with self.s3.open(os.path.join(self.bucket_path, path), "w") as f:
                yaml.dump(data, f)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(os.path.join(self.bucket_path, path), "w") as f:
                yaml.dump(data, f)

    # Handling csv files
    def read_csv(
        self,
        path: str,
        header: str = "infer",
    ) -> pd.DataFrame:
        """Wrapper around the pd read_csv method

        Args:
            path (str): Path to which the text is to be saved.
            header (np.ndarray): Row to be used for the columns of the resulting
                DataFrame. defaults to None.

        Returns:
            pd.DataFrame: pandas Dataframe.
        """
        if self.s3_endpoint:
            assert "FSSPEC_S3_ENDPOINT_URL" in os.environ

        df = pd.read_csv(
            os.path.join(self.bucket_path, path),  # type: ignore
            header=header,
        )
        return df

    def save_csv(
        self, path: str, df: pd.DataFrame, header: bool = True, index: bool = False
    ) -> None:
        """Wrapper around the pd save_csv method

        Args:
            path (str): Path to which the text is to be saved.
            df (pd.DataFrame): DataFrame to be saved.
            header (bool, optional): Whether headers should be saved. Defaults to True.
            index (bool, optional): Whether index should be saved. Defaults to False.
        """
        if self.s3_endpoint:
            assert "FSSPEC_S3_ENDPOINT_URL" in os.environ
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(
            os.path.join(self.bucket_path, path),  # type: ignore
            index=index,
            header=header,
        )

    def save_surrogate(  # TODO: Check if this is working and where it is saving
        self, surrogate: torch.nn.Module, local_path: str, remote_path: str
    ) -> None:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        torch.save(surrogate.state_dict(), local_path)
        if self.s3_endpoint:
            assert "FSSPEC_S3_ENDPOINT_URL" in os.environ
            self.upload(local_path, remote_path)

    # Handling os operations
    def listdir(self, path: str) -> List[str]:
        """Wrapper around the listdir command.

        Args:
            path (str): Path to the folder that needs to be inspected

        Returns:
            List[str]: List of filenames in the folder.
        """
        if self.s3_endpoint:
            # Gets the list of paths from root for all files in the folder
            list_files = list(
                self.s3.ls(os.path.join(self.bucket_path, path))  # type: ignore
            )
            # Trim paths to get only the file names as in os.listdir
            list_files = [file.split(os.path.sep)[-1] for file in list_files]
            return list_files

        else:
            return os.listdir(path=path)

    def isfile(self, path: str) -> bool:
        if self.s3_endpoint:
            return self.s3.isfile(os.path.join(self.bucket_path, path))
        else:
            return os.path.isfile(path)

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        if not self.s3_endpoint:
            os.makedirs(path, exist_ok=exist_ok)

    def download(
        self, remote_path: str, local_path: str, recursive: bool = False, **kwargs: Any
    ) -> None:
        assert self.s3_endpoint
        self.s3.download(
            self.expand_path(remote_path), local_path, recursive=recursive, **kwargs
        )

    def upload(
        self, local_path: str, remote_path: str, recursive: bool = False, **kwargs: Any
    ) -> None:
        assert self.s3_endpoint
        self.s3.put(
            local_path, self.expand_path(remote_path), recursive=recursive, **kwargs
        )


input_handler = FileHandler(os.environ.get("S3_ENDPOINT"), bucket="input")
