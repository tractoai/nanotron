from typing import Dict, List, Tuple, Union

import torch
import torch.distributed as dist
from nanotron.models.base import NanotronModel
from nanotron.parallel import ParallelContext
from torch import nn
from torch.distributed import ReduceOp


def log_to_nn_state_dict(name, logs):
    from nanotron import constants

    if name not in constants.NN_STATES:
        constants.NN_STATES[name] = {}

    for key, value in logs.items():
        constants.NN_STATES[name][key] = value


def track_weight_and_grad_stats(name: str, module: nn.Module, parallel_context: ParallelContext):
    def compute_stats(tensors, metrics: List[str] = ["amax", "mean", "std", "var", "norm"]):
        from nanotron.fp8.utils import compute_stas

        # NAME_TO_FUNC = {
        #     "mean": lambda x: x.mean().item(),
        #     "std": lambda x: x.std().item(),
        #     "var": lambda x: x.var().item(),
        #     # "norm": lambda x: x.norm().item(),
        #     "l1_norm": lambda x: x.norm(p=1).item(),
        #     "l2_norm": lambda x: x.norm(p=2).item(),
        #     "min": lambda x: x.min().item(),
        #     "max": lambda x: x.max().item(),
        #     "amax": lambda x: x.abs().max().item(),
        #     "abs_mean": lambda x: x.abs().mean().item(),
        #     "kurtosis": lambda x: calculate_kurtosis(x),
        # }
        tensors = {"tensor": tensors} if not isinstance(tensors, dict) else tensors
        stats = {}

        for key, tensor in tensors.items():
            # if tensor.dtype == torch.long or tensor.dtype == torch.int or tensor.dtype == torch.bool:
            #     continue
            if tensor.is_floating_point() is False:
                continue

            # stats[key] = {}
            # for metric in metrics:
            #     stats[key][metric] = NAME_TO_FUNC[metric](tensor)
            stats[key] = compute_stas(tensor)

            # NOTE: now all reduce mean this across tp ranks
            tp_group = parallel_context.tp_pg
            for metric_name, metric_value in stats[key].items():
                stats[key][metric_name] = torch.tensor(metric_value, device=tensor.device, dtype=tensor.dtype)
                dist.all_reduce(stats[key][metric_name], op=ReduceOp.MAX, group=tp_group)

        return stats[list(stats.keys())[0]] if len(stats) == 1 else stats

    # logs: Dict[str, Dict[str, float]] = {}

    # if name not in logs:
    #     logs[name] = {}

    def _save_output_stats(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        param_names = [name for name, _ in module.named_parameters()]
        for param_name in param_names:
            if hasattr(module, param_name):
                param = getattr(module, param_name)
                stats = compute_stats(param.data)
                if stats is not None:
                    # logs[name][param_name] = stats
                    log_to_nn_state_dict(name, {param_name: stats})

        inputs = input if isinstance(input, tuple) else (input,)
        outputs = output if isinstance(output, tuple) else (output,)

        if len(inputs) > 1:
            for i, inp in enumerate(inputs):
                if inp.dtype == torch.long:
                    # NOTE: this is input ids in transformers
                    continue
                stats = compute_stats(inp)
                if stats is not None:
                    # logs[name][f"input:{i}"] = stats
                    log_to_nn_state_dict(name, {f"input:{i}": stats})

        elif len(inputs) == 1:
            stats = compute_stats(inputs[0])
            if stats is not None:
                # logs[name]["input"] = stats
                log_to_nn_state_dict(name, {"input": stats})

        if len(outputs) > 1:
            for i, out in enumerate(outputs):
                stats = compute_stats(out)
                if stats is not None:
                    # logs[name][f"output:{i}"] = stats
                    log_to_nn_state_dict(name, {f"output:{i}": stats})

        elif len(outputs) == 1:
            stats = compute_stats(outputs[0])
            if stats is not None:
                # logs[name]["output"] = stats
                log_to_nn_state_dict(name, {"output": stats})

    def _save_grad_stats(module: nn.Linear, grad_input, grad_output: torch.Tensor):
        if isinstance(grad_output, tuple):
            for i, grad in enumerate(grad_output):
                if grad is None:
                    continue

                stats = compute_stats(grad)
                if stats is not None:
                    # logs[name][f"grad_output:{i}"] = stats
                    log_to_nn_state_dict(name, {f"grad_output:{i}": stats})
        else:
            stats = compute_stats(grad_output)
            if stats is not None:
                # logs[name]["grad_output"] = stats
                log_to_nn_state_dict(name, {"grad_output": stats})

        if isinstance(grad_input, tuple):
            for i, grad in enumerate(grad_input):
                if grad is not None:
                    stats = compute_stats(grad)
                    if stats is not None:
                        # logs[name][f"grad_input:{i}"] = stats
                        log_to_nn_state_dict(name, {f"grad_input:{i}": stats})
        else:
            if grad_input is not None:
                stats = compute_stats(grad_input)
                if stats is not None:
                    # logs[name]["grad_input"] = stats
                    log_to_nn_state_dict(name, {"grad_input": stats})

    handles = []
    handles.append(module.register_forward_hook(_save_output_stats))
    # handles.append(module.register_backward_hook(_save_grad_stats))
    # return logs, handles
    return handles


def monitor_model(
    model: NanotronModel, parallel_context: ParallelContext
) -> Tuple[Dict[str, Union[torch.Tensor, float]], List]:
    handles = []
    leaf_modules = [(name, module) for name, module in model.named_modules()]

    for name, module in leaf_modules:
        # module_logs, module_handles = track_weight_and_grad_stats(name, module, parallel_context)
        module_handles = track_weight_and_grad_stats(name, module, parallel_context)
        # logs.update(module_logs)
        handles.extend(module_handles)

    # return logs, handles
    return handles


# def convert_logs_to_flat_logs(
#     logs: Dict[str, Dict[str, Dict[str, Union[torch.Tensor, float]]]]
# ) -> Dict[str, Union[torch.Tensor, float]]:
#     flat_logs = {}
#     for module_name, components in logs.items():
#         if isinstance(components, dict) is True:
#             for component_name, stats in components.items():
#                 for metric_name, metric_value in stats.items():
#                     flat_logs[f"{module_name}:{component_name}:{metric_name}"] = metric_value
#         else:
#             assert 1 == 1

#     return flat_logs


def convert_logs_to_flat_logs(
    logs, prefix: str = "", flat_logs: Dict[str, Union[torch.Tensor, float]] = None
) -> Dict[str, Union[torch.Tensor, float]]:
    if flat_logs is None:
        flat_logs = {}

    for key, value in logs.items():
        new_prefix = f"{prefix}:{key}" if prefix else key

        if isinstance(value, dict):
            convert_logs_to_flat_logs(value, new_prefix, flat_logs)
        else:
            flat_logs[new_prefix] = value

    return flat_logs