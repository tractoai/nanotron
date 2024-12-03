import sys
import typing
import uuid
import tempfile
import warnings

import fsspec
import torch
import yt.wrapper as yt
from torch import Tensor

from tractorun.backend.tractorch.dataset import YtDataset
from tractorun.backend.tractorch.serializer import TensorSerializer


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
