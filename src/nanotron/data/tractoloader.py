import dataclasses
from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader

import nanotron.distributed as dist
from nanotron import logging
from nanotron.data.tractoset import TractoTableDataset
from nanotron.dataloader import (
    EmptyInfiniteDataset,
    get_dataloader_worker_init,
)
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer

logger = logging.get_logger(__name__)


@dataclasses.dataclass
class TractosetDataCollatorForCLM:
    # we don't need a collator for iterable dataset
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Process the case when current rank doesn't require data. We return `TensorPointer` that points to ranks having the data.
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        if current_pp_rank not in [
            self.input_pp_rank,
            self.output_pp_rank,
        ]:
            assert all(len(example) == 0 for example in examples)
            return {
                "input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "input_mask": TensorPointer(group_rank=self.input_pp_rank),
                "label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "label_mask": TensorPointer(group_rank=self.output_pp_rank),
            }

        result: Dict[str, torch.Tensor] = {
            "input_ids": torch.vstack([example["input_ids"] for example in examples]),
            "input_mask": torch.vstack([example["input_mask"] for example in examples]),
            "label_ids": torch.vstack([example["label_ids"] for example in examples]),
            "label_mask": torch.vstack([example["label_mask"] for example in examples])
        }
        return result


def get_sampled_dataset(
    dl_ranks_size: int,
    dl_rank: int,
    train_dataset: TractoTableDataset,
    consumed_raws: int,
) -> TractoTableDataset:
    # here we just drop line that do not fit into the last chunk
    dp_chunk_size = len(train_dataset) // dl_ranks_size
    dp_chunk_size = dp_chunk_size - dp_chunk_size % train_dataset.batch_size
    start = dl_rank * dp_chunk_size
    end = start + dp_chunk_size

    if consumed_raws:
        start = start + consumed_raws

    return train_dataset.to_dp(start=start, end=end)


def build_tractoloader(
    dataset: TractoTableDataset,
    parallel_context: ParallelContext,
    input_pp_rank: int,
    output_pp_rank: int,
    micro_batch_size: int,
    dataloader_num_workers: int,
    consumed_train_samples: int = 0,
    dataloader_drop_last: bool = True,
    dataloader_pin_memory: bool = True,
) -> DataLoader:

    # Case of ranks not requiring data. We give them a dummy dataset, then the collator will do his job
    if dist.get_rank(parallel_context.pp_pg) not in [input_pp_rank, output_pp_rank]:
        dataset_length = len(dataset)
        dataset = EmptyInfiniteDataset(length=dataset_length)
        # No need to spawn a lot of workers, we can just use main
        dataloader_num_workers = 0

    # Compute size and rank of dataloader workers
    dp_ranks_size = parallel_context.dp_pg.size()
    dp_rank = parallel_context.dp_pg.rank()

    dataset = get_sampled_dataset(
        train_dataset=dataset,
        dl_ranks_size=dp_ranks_size,
        dl_rank=dp_rank,
        consumed_raws=(consumed_train_samples // dp_ranks_size),
    )

    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        drop_last=dataloader_drop_last,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dp_rank),
    )
