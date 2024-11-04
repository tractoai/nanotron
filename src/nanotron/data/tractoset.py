import sys
import typing
import uuid
import tempfile
import warnings

import fsspec
import torch
import numpy as np
import yt.wrapper as yt
from torch import Tensor

from tractorun.backend.tractorch.dataset import YtDataset
from tractorun.backend.tractorch.serializer import TensorSerializer

from nanotron.data.nanoset import Nanoset


_T_co = typing.TypeVar("_T_co")


class YTTensorTransform:
    _serializer = TensorSerializer()

    def __call__(self, columns: list[str], row: dict) -> dict:
        return {
            name: self._serializer.desirialize(
                yt.yson.get_bytes(row[name])
            )
            for name in columns
        }


class TractoTableDataset(YtDataset[_T_co]):
    # the most optimal way to process datasets
    def __init__(
        self,
        yt_client: yt.YtClient,
        path: str,
        batch_size: int,
        sequence_length: int,
        start: int = 0,
        end: int | None = None,
        columns: list | None = None,
    ) -> None:
        self.yt_client = yt_client
        self.path = path
        self.start = start
        self.end = end
        self.columns = columns
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        super().__init__(
            yt_client=yt_client,
            path=path,
            start=start,
            end=end,
            columns=columns,
            transform=YTTensorTransform(),
        )

    def __iter__(self) -> typing.Iterator[_T_co]:
        for batch in super().__iter__():
            for sample in batch["input_ids"]:
                yield self._unfold_raw(sample)

    def __len__(self) -> int:
        return super().__len__() * self.batch_size

    def _unfold_raw(self, sample: Tensor) -> dict[str, Tensor]:
        result = {
            "input_ids": sample[:-1],
            "input_mask": torch.ones((self.sequence_length,), dtype=torch.bool),
            "label_ids": sample[1:],
            "label_mask": torch.ones((self.sequence_length,), dtype=torch.bool),
        }
        for value in result.values():
            assert value.shape == (self.sequence_length,)
        return result

    def to_dp(self, start: int, end: int) -> "TractoTableDataset":
        print(f"new start {start} and end {end}", file=sys.stderr)
        assert start % self.batch_size == 0

        start = start // self.batch_size
        end = end // self.batch_size

        return TractoTableDataset(
            yt_client=self.yt_client,
            path=self.path,
            batch_size=self.batch_size,
            columns=self.columns,
            sequence_length=self.sequence_length,
            end=end,
            start=start,
        )


class TractoFsFileDataset(torch.utils.data.Dataset):
    # just a wrapper for Nanoset
    # download file from YT and store it to the local fs
    def __init__(
        self,
        yt_client: yt.YtClient,
        yt_dataset_paths: str | list[str],
        sequence_length: int,
        token_size: int,
        train_split_num_samples: int,
        dataset_weights: float | None = None,
        random_seed: int = 1234,
    ) -> None:
        if isinstance(yt_dataset_paths, str):
            warnings.warn("dataset_folders should be of type List[str] but str was provided. Converting to List[str]")
            yt_dataset_paths = [yt_dataset_paths]
        dataset_dir = tempfile.mkdtemp()
        for yt_path in yt_dataset_paths:
            file_name = str(uuid.uuid4())
            stream = yt_client.read_file(yt_path)
            with open(file_name, "wb") as f:
                f.write(stream.read())
        self.dataset = Nanoset(
            dataset_folders=[dataset_dir],
            sequence_length=sequence_length,
            token_size=token_size,
            train_split_num_samples=train_split_num_samples,
            dataset_weights=dataset_weights,
            random_seed=random_seed,
        )

    def __len__(self) -> int:
        """
        Returns:
            int: The number of samples of the Nanoset
        """

        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self.dataset[idx]


class TractoMemFileDataset(torch.utils.data.Dataset):
    # download file from YT and map it to RAM
    def __init__(
        self,
        yt_client: yt.YtClient,
        yt_dataset_paths: str | list[str],
        sequence_length: int,
        token_size: int,
        train_split_num_samples: int,
        dataset_weights: float | None = None,
        random_seed: int = 1234,
    ) -> None:
        if isinstance(yt_dataset_paths, str):
            warnings.warn("dataset_folders should be of type List[str] but str was provided. Converting to List[str]")
            yt_dataset_paths = [yt_dataset_paths]

        fs = fsspec.filesystem("memory")
        dataset_dir = str(uuid.uuid4())
        for yt_path in yt_dataset_paths:
            file_name = str(uuid.uuid4())
            stream = yt_client.read_file(yt_path)
            with fs.open(f"mem://{dataset_dir}/{file_name}", "wb") as f:
                f.write(stream.read())
        self.dataset = Nanoset(
            dataset_folders=[dataset_dir],
            sequence_length=sequence_length,
            token_size=token_size,
            train_split_num_samples=train_split_num_samples,
            dataset_weights=dataset_weights,
            random_seed=random_seed,
        )

    def __len__(self) -> int:
        """
        Returns:
            int: The number of samples of the Nanoset
        """

        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self.dataset[idx]

