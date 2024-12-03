from pathlib import Path

import torch

from nanotron import distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.random import RandomStates
from nanotron.serialize.storage import Storage


def save_random_states(
    random_states: RandomStates,
    parallel_context: ParallelContext,
    storage: Storage,
):
    """All processes save their own random state"""
    storage.create_directory("random")
    path = "random/" + f"tp-{dist.get_rank(parallel_context.tp_pg)}-of-{parallel_context.tp_pg.size()}_dp-{dist.get_rank(parallel_context.dp_pg)}-of-{parallel_context.dp_pg.size()}_pp-{dist.get_rank(parallel_context.pp_pg)}-of-{parallel_context.pp_pg.size()}.pt"

    # TODO @thomasw21: That's annothing but this actually uses pickle, we might need to change that for something else
    storage.save(path, random_states)


def load_random_states(parallel_context: ParallelContext, storage: Storage) -> RandomStates:
    # TODO @thomasw21: This basically assumes that we have exactly the same topology as the one we used when saving.
    path = "random/" + f"tp-{dist.get_rank(parallel_context.tp_pg)}-of-{parallel_context.tp_pg.size()}_dp-{dist.get_rank(parallel_context.dp_pg)}-of-{parallel_context.dp_pg.size()}_pp-{dist.get_rank(parallel_context.pp_pg)}-of-{parallel_context.pp_pg.size()}.pt"

    # TODO @thomasw21: That's annothing but this actually uses pickle, we might need to change that for something else
    return storage.load(path)
