# nanotron

The objective of this repository is to provide easy distributed primitives in order to train a variety of models efficiently.

# Philosophy

- Make it fast. At least as fast as other open source versions.
- Make it minimal. We don't actually need to support all techniques and all versions of 3D parallelism. What matters is that we can efficiently use the "best" ones.
- Make everything explicit instead of transparent. As we move forward, making things transparent works well when it works well but is a horrible debugging experience if one doesn't understand the implications of techniques used. In order to mitigate this, we choose to be explicit in the way it does things

# Core Features

We support the following:
 - 3D parallelism, including one-forward-one-backward pipeline engine
 - ZeRO-1 optimizer
 - FP32 gradient accumulation
 - Parameter tying/sharding

# Examples

In the `/examples` directory, we provide a set of **self-sufficient** examples for different workloads, which you can use to quickly get started.
* [Train a GPT2 model](https://github.com/huggingface/nanotron/tree/main/examples/gpt2)
* [Train a GPT2 model with Multi-Headed Attention](https://github.com/huggingface/nanotron/tree/main/examples/gpt2_mqa)
* [Train a T5 model](https://github.com/huggingface/nanotron/tree/main/examples/t5)
* [Train a LLaMa model](https://github.com/huggingface/nanotron/tree/main/examples/llama)
* [RLHF Training (Includes SFT/Reward Modeling/DPO/PPO)](https://github.com/huggingface/nanotron/tree/main/examples/llama)
* [Benchmark throughput](./benchmarks/nanotron/README.md)

> Note: Most examples include a slow modeling (No dependencies, only Pytorch), and a fast modeling (Flash Attention, apex, ...). Make sure to install the dependencies if you want to run the fast modeling, then set the env `export USE_FAST=1`

# Installation

Requirements:
 - Python >= 3.10
 - PyTorch >= 2.0.0

To install:
```bash
git clone git@github.com:huggingface/nanotron.git
cd nanotron
pip install -e .
```

For the linting:
```bash
pre-commit install
pre-commit run --config .pre-commit-config.yaml --all-files
```

We also support a set of flavors that you can install using `pip install -e [$FLAVOR]`:
 - `dev`: Used is you are developping in `nanotron`. It installs in particular our linter mechanism. On top of that you have to run `pre-commit install` afterwards.
 - `test`: We use `pytest` in order to run out testing suite. In order to run tests in parallel, it will install `pytest-xdist`, which you can leverage by running `pytest -n 12 tests` (12 is the number of parallel test)
 - `s3fs`: Used if you want to save/load checkpoints directly from s3. It uses `s3fs` and consequently might be slow. We are working on an improve mechanism

Install also:
- Flash Attention: `pip install packaging; pip install flash-attn>=2.0  --no-build-isolation`
- Apex: see the [Github repo](https://github.com/NVIDIA/apex#linux) (if master errors out, try checking out release 23.08)
- Also good to have `s3fs` `transformers` `datasets` `python-etcd` `tensorboardX` `boto3`: `pip install s3fs transformers datasets python-etcd tensorboardX boto3`
- finally install `s5cmd`:
```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install s5cmd
```

# Doc on collective operations

This NVIDIA dic is nice on all collective operations (all_reduce, reduce_scatter, etc): https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html

# Usage

We showcase usage in the `examples` directory.

# Key concepts

Let's go through some key concepts.

## ParallelContext

`ParallelContext` is the base class that contains all information like process groups, ranks, world sizes when running 3D parallelism workloads. You can initialize it using the following:

```python
from nanotron.distributed import ParallelContext

parallel_context = ParallelContext.from_torch(
    tensor_parallel_size=2,
    data_parallel_size=2,
    pipeline_parallel_size=2
)
```

`ParallelMode` is an enum for accessing specific information like local rank, local world size, etc., about a specific parallelism dimension (TP, PP, DP).

For example, if you want to get the local rank of the current process in tensor parallelism, you can do:

```python
local_rank = parallel_context.get_local_rank(ParallelMode.TENSOR) # and the same for other parallelisms
```

In PyTorch, you can issue a collective communication (all-reduce, all-gather, ...) on a subgroup of all the ranks. It provides the granularity needed for 3D parallelism. You can access the process group of a parallelism dimension of a process as:

- `parallel_context.get_group(ParallelMode.GLOBAL)` or `parallel_context.get_group(ParallelMode.TENSOR)` # similar for other parallelisms
- `parallel_context.world_rank_matrix`: This allows one to compute the world rank knowing the 3D ranks of a given process, or inversely when using get_3d_ranks.
- `parallel_context.world_ranks_to_pg`: This is a more generic pattern that allows you to store a custom set of ProcessGroups, and query it via a list of world ranks.


## NanotronParameter

Given a specific computation workload, we can freely define how we distribute workloads. For example:

```python
from torch import nn
# Example: let's assume you want to run a Linear without bias
hidden_size = 8

# Single process way of running computation
module = nn.Linear(hidden_size, hidden_size) # Parameters: [H, H]
input = torch.randn(batch_size, hidden_size)
output = module(input)

# Sharded ways of running computation across `tp_pg` (`ProcessGroup`)
# Version 1
sharded_module = nn.Linear(hidden_size, hidden_size / tp_pg.size())
input = torch.randn(batch_size, hidden_size)
sharded_output = module(input)
torch.distributed.all_gather(output, sharded_output, group=tp_pg.size())

# Version 2
sharded_module = nn.Linear(hidden_size / tp_pg.size(), hidden_size)
sharded_input = torch.randn(batch_size, hidden_size / tp_pg.size())
sharded_output = module(sharded_input)
torch.distributed.all_reduce(output, sharded_output, group=tp_pg.size())

# Version 3
sharded_module = nn.Linear(hidden_size, hidden_size)
sharded_input = torch.randn(batch_size / tp_pg.size(), hidden_size)
torch.distributed.all_gather(input, sharded_input, group=tp_pg.size())
output = module(input) # Duplicate workload

# Version ....
```
Distributed workloads have the tendency to generate tradeoffs between duplicated computation and extra communication. There's multiple ways to run the same computation, what we can optimize is the amount of communication we do, as well as duplicated work. Sometime it's worth duplicating work in order to reduce communication significantly.

As seen in previous example, sometimes the parameters are sharded across multiple devices, and sometimes they are duplicated. In `nanotron`, we decided to add those additional metadatas to `nn.Parameter`. We call our new datastructure: `NanotronParameter`

## Sharded parameter

A sharded parameter has the following metadata attached:

```python
@dataclasses.dataclass
class SlicesPair:
    local_slices: Tuple[slice, ...]
    global_slices: Tuple[slice, ...]

@dataclasses.dataclass
class ShardedInfo:
    # All world ranks involved in the sharding.
    global_ranks: Tuple[int, ...]
    # Info of to what slice of the unsharded tensor (global_slices) the current sharded tensor corresponds (local_slices)
    local_global_slices_pairs: Tuple[SlicesPair, ...]
    # The shape of the unsharded tensor
    unsharded_shape: Tuple[int, ...]
```
Imaging we sharded a tensor t of shape [8, 64] across 2 ranks, 0 and 3, where rank 0 holds the first shard t[:, :32] and rank 3 holds the second shard t[:, 32:], then the sharded_info for them is:
```python
shard_info = ShardedInfo(global_ranks=(0,3), local_global_slices_pairs=(SlicesPair(local_slices=(slice(0,8), slice(0, 32),), global_slices=(slice(0,8), slice(0, 32)),),), unsharded_shape=(8, 64)) # world rank 0
shard_info = ShardedInfo(global_ranks=(0,3), local_global_slices_pairs=(SlicesPair(local_slices=(slice(0,8), slice(0, 32),), global_slices=(slice(0,8), slice(32, 64)),),), unsharded_shape=(8, 64)) # world rank 3
```

## Tied parameter

This signifies that multiple occurence of a given parameter are duplicated on multiple devices. Therefore we need a mechanism for them to be synced at all time. A typical example would be `lm_head` on top of transformers that's tied to the work embedding parameters. We attach the following metadata to the parameter:
```python
@dataclasses.dataclass
class TiedInfo:
    # We usually arbitrarily choose a name of a parameter, either `lm_head.weight` or `wte.weight` for example.
    name: str
    # This allows us to define the scope in which `name` is valid.
    root_module: nn.Module
    # All world ranks involved in the tying.
    global_ranks: Tuple[int, ...]
    # In order to keep parameter synced, we add a `reduce_op` value that defines what kind of reduce operation we apply to the gradient.
    # None signifies that we do not reduce
    reduce_op: Optional[dist.ReduceOp]
```

Most interesting in this dataclass is the `reduce_op` parameter. Sometimes duplicated workload can remove the need to sync gradients as by design gradient computation would have already computed the correct gradient. A typical example of this is classic TP implementation using `all-reduce`/`identity`.

Note: a parameter can be both sharded and tied. Both notion just have to involve different ranks. For example: lm_head and word embeddings can be sharded across TP, and tied between the first PP rank, and the last one.

## Tensor parallelism

Usually the go-to solution when models can't fit within a device. The basic idea is to figure out patterns where one can divide a single workload into multiple smaller workerloads that can run in parallel. We mimick tensor parallelism from Megatron-LM. Current supported modules:
 - ColumnLinear/RowLinear
 - ParallelVocabulary
 - Cross-Entropy over sharded logits
 - Distributed samplers for generation

[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) introduces that notion upon implementing one of the first large scale transformers:
![Tensor parallelism in transformer model](assets/tensor_parallelism_in_transformer.png)
(Source: [link](https://arxiv.org/abs/1909.08053))

## Pipeline parallelism

We can view the neural network as a sequence of operations. Instead of previous assumption where we split operations into smaller workloads that we can distribute. We take contiguous chunks and assign them to specific ranks. Instead of running parallel workloads, those are inherently sequential. In order to run them in parallel, we introduce fancy schedulers that process different batches in parallel:Rank 0 can be processing batch 1, while rank 1 is processing batch 0
 - Rank 0 starts to process batch 0
 - Rank 0 finishes to process batch 0
 - Rank 0 sends outputs to rank 1
 - Rank 1 starts to process batch 0
 - Rank 0 starts to process batch 1 (Rank 1 and Rank 0 are processing in parallel batches 1 and 0 respectively)
 - Rank 1 finishes to process batch 0
 - Rank 0 finishes to process batch 1

### PipelineBlock

The core component of our pipeline engine is a `PipelineBlock`.
It acts as the granularity for all our pipeline engines, we can define a specific workload that needs to happen on a specific device, ie rank.
Other ranks run a dummy `forward` where the forward pass returns `TensorPointer` which hold enough metadata in order to know where the output of the computation is.
```python
@dataclass
class TensorPointer:
    group_rank: int
```

Module defined within `PipelineBlock` can be directly be instantiated on the specific device.

In short, what does `PipelineBlock` does:
 - Receives either a set of `torch.Tensor`/`TensorPointer` as input
 - In case of `TensorPointer`, query the tensor from the specified rank we extract from it's state/context.
 - Run the defined computation if current rank is responsible for running computation
 - Return a dictionary `Dict[str, Union[torch.Tensor, TensorPointer]]`.
   `TensorPointer` as output are for ranks that didn't run computation and require to know where the output of the computation is.

```python
class PipelineBlock(nn.Module):
    def __init__(
       self,
       p2p, # point-to-point communication class
       module_builder, # module constructor in order to build module lazily
       module_kwargs, # module constructor arguments in order to build module lazily
       module_input_keys, # ranks that are not running compute to know the module input structure. Serves as a validation mechanism.
       module_output_keys, # metadata for ranks that are not running compute to know the module output structure.
    ):
        pass

# Example
# Lazy instantiation of a `nn.Linear`
model = PipelineBlock(
   p2p=p2p,
   module_builder=nn.Linear,
   module_kwargs={"in_features":3, "out_feature": 5},
   module_input_keys={"input"},
   module_output_keys={"output"}
)

model.build_and_set_rank(pp_rank) # Instantiate model parameters on `pp_rank` assigned device
```

In order to define which rank we use the `build_and_set_rank` method. It attaches the rank as a meta data, and builds the module on that specific rank.

Models have to be defined using a "surface" of `PipelineBlock`. Typically, above `PipelineBlock` it's all about defining the `PipelineBlock` computational direct acyclic graph, below is where device specific computation is defines.

As a non trivial example:
```python
class DummyModel(nn.Module):
    def __init__(
        self,
        p2p: P2P,
    ):
        super().__init__()
        self.dense1 = PipelineBlock(
            p2p=p2p,
            module_builder=nn.Linear,
            module_kwargs={"in_features": 10, "out_features": 10},
            module_input_keys={"input"},
            module_output_keys={"output"},
        )
        self.dense2 = PipelineBlock(
            p2p=p2p,
            module_builder=nn.Linear,
            module_kwargs={"in_features": 10, "out_features": 10},
            module_input_keys={"input"},
            module_output_keys={"output"},
        )
        # Doesn't hold any parameter, but we have to specify where the computation happens.
        self.loss = PipelineBlock(
            p2p=p2p,
            module_builder=lambda: lambda x: x.sum(),
            module_kwargs={},
            module_input_keys={"x"},
            module_output_keys={"output"},
        )

    def forward(self, x: Union[torch.Tensor, TensorPointer]):
        # x can be a `torch.Tensor` or a `TensorPointer` depending on the current rank, and where the pipeline blocks run their compute
        x = self.dense1(input=x)["output"]
        x = self.dense2(input=x)["output"]
        x = self.loss(x=x)["output"]
        return x
```


### Pipeline engine

We now support two kinds of engines: `AllForwardAllBackward`, `OneForwardOneBackward`

Pipeline engines are different schedules for the set of workloads. A great illustration for the different schedules we support for training can be found in [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM
](https://arxiv.org/abs/2104.04473). We support `All forward all backward` and `One forward one backward` currently (Figure 3 and top of figure 4).

![Pipeline engine](assets/pipeline_engine.png)
(Source: [link](https://arxiv.org/abs/2104.04473))

> **_IMPORTANT NOTE:_** When preparing your dataloader, make sure every tensor lives on a single rank, and other ranks must have `TensorPointer` to that rank. This is a requirement for the pipeline engine to work.

## ZeRO-1 optimizer

ZeRO stands for "Zero Redundancy Optimizer", also known as "FSDP" in Pytorch. The goal of such techniques is to shard tensors across multiple devices instead of duplicating them. Consequently it allows for significant memory gains at the cost of some communication overhead (with potential ability to overlap computation and communication). Sharding is done across data parallel dimension There are three stages:
 - `Stage 1`: The optimizer states are sharded.
 - `Stage 2`: The gradients are sharded
 - `Stage 3`: The model weight are sharded

As of now, we currently only support `stage 1`.

![ZeRO](assets/zero.png)
(Source: [link](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/))

# The awesome to have

## Recomputation utilities

Activation recomputation, also known as "activation checkpointing" is a memory saving technique. Pytorch automatically stores a set activation during the forward pass required for backward computation. However with large workloads, it might be worth recomputing specific activation in order to save memory. In `nanotron` we provide a decorator to implement this feature:

```python
class MyFancyModule(nn.Module):
    def __init__(self):
        ...
        self.do_checkpoint: bool = True

    @checkpoint_method(attr_name="do_checkpoint")
    def forward(self, x):
        ...
```

## On device initialization

Usual pytorch module constructor instantiate weights on cpu and then move them to gpus. This can blow up cpu memory as well as being overall quite slow.

```python
with init_on_device_and_dtype(device=torch.device("cuda"), dtype=torch.bfloat16):
    module = MyFancyModule() # This directly instantiate the model on your device

# If you want to bypass Pytorch weight initialization mechanism
with init_on_device_and_dtype(device=torch.device("meta"), dtype=torch.bfloat16):
    module = MyFancyModule()
module.to_empty(torch.device("cuda")) # bfloat 16 model loaded in gpu with weight not initialized (only the storage buffers are allocated)
```

## Unified API for logging

We provide a uniform API to logging, whether that's on tensorboard, on stdout or on Hugging Face hub:

```python
@dataclass
class LogItem:
    tag: str
    scalar_value: Union[float, int]
    log_format: Optional[str] = None
```

All logger need to implement a single method:
```python
class BaseLogger:
    @abstractmethod
    def add_scalars_from_list(self, log_entries: List[LogItem], iteration_step: int):
        ...
```

If you want to have tensorboard logger support: `pip install -e ".[tb-logger]"`.
If you want to have huggingface-hub tensorboard logger support: `pip install -e ".[hf-logger]"`.

## Random state handling primives

We currently have a mechanism to have an arbitrary number of `RandomState` in a `RandomStates`:
```python
class RandomState:
    random
    numpy
    torch
    torch_cuda

class RandomStates(MutableMapping[str, RandomState])
    pass
```

At all time we get get/set current random state in the current context
```python
def get_current_random_state():
   # This gets the current random_state from the current context
   pass

def set_random_state(random_state: RandomState):
   # This sets random state in the current context
   pass
```

In order to use specific `RandomState` for specific operations, typically when you want to synchronize `nn.Dropout` across multiple ranks for example, you can run `branch_random_state` context manager:
```python
def branch_random_state(random_states:RandomStates, key:str):
   # Context manager which sets the random state associated with `key` when entering
   # When exiting, we update the random state at `key` and restore previous random state.
   pass

# Usage
random_states = RandomStates({"my_own_random_state": get_current_random_state()})
with branch_random_state(random_states, "my_own_random_state"):
    output = nn.Dropout(0.1)(input)
```

Finally we provide a quick helper in order to get a synchronized random state across a process group.
```python
def get_synced_random_state(random_state: RandomState, pg: ProcessGroup):
   # This allows us to get a synchronized random state with other ranks within a single group

# Usage
random_states = RandomStates({"tp_synced_random_state": get_synced_random_state(random_state=get_current_random_state(), group=tp_pg)})
with branch_random_state(random_states, "tp_synced_random_state"):
    # Assuming that input is synced across TP, all ranks will apply the same random mask.
    output = nn.Dropout(0.1)(input)
```

# Distributed serialization mechanism

We rely on compute nodes having access to a single shared filesystem.

We use `safetensors` to store our checkpoints.

Current format:
```python
checkpoint_metadata.json # Stores version, topology, other metadata that would make the training resumable
optimizer
    optimizer_config.json # Stores enough information to reinstantiate which optimizer this runs.
    optimizer_tp-0-of-1_dp-0-of-1_pp-0-of-2.pt
    optimizer_tp-0-of-1_dp-0-of-1_pp-0-of-2.pt
lr_scheduler
    lr_scheduler_tp-0-of-1_dp-0-of-1_pp-0-of-2.pt
    lr_scheduler_tp-0-of-1_dp-0-of-1_pp-0-of-2.pt
random # Stores random states from each process in order to resume training from the point on.
    tp-0-of-1_dp-0-of-1_pp-0-of-2.pt
    tp-0-of-1_dp-0-of-1_pp-1-of-2.pt
model
    dense1
        model_weight.safetensors
        model_bias.safetensors
    dense2
        model_weight.safetensors
        model_bias.safetensors
```

Some observations:
 - checkpoints are NOT topology agnostic, this is due to both `random_states` and `sharded` tensors.
   Instead of trying to reconcile those and obtain a topology agnostic one, we want to support a `checkpoint_reshape` method.
   The motivations are the following:
   - When training, one spends a LOT more time `saving` checkpoints than loading. In doing so, having the fastest saving mechanism helps. Consequently not having any distributed communication/locking will help this.
   - Random states are not so easily reconcilable. Given random states for two seperate processes when we have TP=2, it's not obvious what should be the random state if we set to TP=1.
 - Optimizer states are aligned with parameters. It's usually the case where for each parameter you can define a optimizer state. But that's a limitation on the current serialization format.

# Development guidelines

If you plan on developping on `nanotron`, we suggest you install the `dev` flavor: `pip install -e ".[dev]"`

We use pre-commit to run a bunch of callbacks on each commit, mostly normalization code in order for the codebase to stay consistent. Please do run `pre-commit install`.

# Current restrictions:

- `nn.Module` inside PipelineBlocks have to return a `Dict[str,torch.Tensor]` or `torch.Tensor`.
- No conditional flow on top of pipeline, or at least making sure that all the process within a data parallel rank are performing the same sequence of operations:
  - First all but one process will be things on `TensorPointer` which would make input dependent control flow quite hard.
  - Second if you were to have input dependent control flow, causing two processes within a single data parallel rank to be different, then you might end up with weird communication issues.

# Credits

We would like to thank everyone working on LLMs, especially those sharing their work openly from which we took great inspiration: Nvidia for `Megatron-LM/apex`, Microsoft for `DeepSpeed`, HazyResearch for `flash-attn`
