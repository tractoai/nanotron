"""
Converts a nanotron model to HF format
Command:
    torchrun --nproc_per_node=1 convert_nanotron_to_hf.py --checkpoint_path=nanotron-path --save_path=hf-path
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, Optional

import torch
from convert_weights import get_config_mapping, get_weight_mapping, load_nanotron_model
from nanotron.config import LlamaConfig as NanotronLlamaConfig
from nanotron.models import init_on_device_and_dtype
from nanotron.models.llama import LlamaForTraining
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import LlamaConfig as HFLlamaConfig

import os
import yt.wrapper as yt

from tractorun.run import prepare_and_get_toolbox
from tractorun.backend.tractorch import Tractorch


TEST_PROMPT = "What is the meaning of the word chutzpah?\nThe word chutzpah means"


def _handle_attention_block(
    qkv: torch.Tensor, part: Literal["q", "k", "v"], n_q_heads: int, n_kv_heads: int, d_qk: int
) -> torch.Tensor:
    # Huggingface Llama separates the q, k, v weights (as opposed to nanotron).
    # Furthermore, in the rotary embeddings in nanotron expects interleaved pairs of even
    # and odd dimensions GPT-J style, while the huggingface implementation expects
    # the whole 1st half and then the whole 2nd half GPT-NeoX style (for more information
    # see flash_attn.layers.rotary.RotaryEmbedding).
    # This function selects the proper chunk of the bundled qkv tensor and permutation
    # to ensure correct transformation to huggingface.

    def interleave(w: torch.Tensor):
        return w
        # w_new = []
        # for head_w in w.split(d_qk):
        #     head_w = head_w.view(d_qk // 2, 2, -1).transpose(0, 1).reshape(d_qk, -1)
        #     w_new.append(head_w)
        # return torch.cat(w_new)

    assert part in ["q", "k", "v"], "part must be one of [q, k, v]"

    index_end_q = n_q_heads * d_qk
    index_end_k = index_end_q + n_kv_heads * d_qk
    if part == "q":
        return interleave(qkv[:index_end_q])
    if part == "k":
        return interleave(qkv[index_end_q:index_end_k])
    return qkv[index_end_k:]


def _handle_gate_up_proj(gate_up_proj: torch.Tensor, gate: bool) -> torch.Tensor:
    # The gate and up projection are bundled in nanotron.
    # This function selects the proper chunk in the bundled weights to return
    # either the gate or the up projection only.
    weight_size = gate_up_proj.shape[0] // 2
    if gate:
        return gate_up_proj[:weight_size]
    else:
        return gate_up_proj[weight_size:]


def convert_nt_to_hf(nanotron_model: LlamaForTraining, hf_model: LlamaForCausalLM, model_config: NanotronLlamaConfig):
    """Converts the weights from the nanotron_model to hf_model, making modifications
    in-place."""

    nanotron_model_state_dict = nanotron_model.state_dict()

    hf_to_nt = get_weight_mapping(model_config, nt_to_hf=False)
    for module_name_hf, module_hf in hf_model.named_modules():
        for param_name_hf, param_hf in module_hf.named_parameters(recurse=False):
            # Get the Nanotron parameter
            nanotron_key = hf_to_nt[f"{module_name_hf}.{param_name_hf}"]
            param = nanotron_model_state_dict[nanotron_key]

            if "qkv_proj" in nanotron_key:
                proj_name = module_name_hf.split(".")[4][0]
                param = _handle_attention_block(
                    param,
                    proj_name,
                    model_config.num_attention_heads,
                    model_config.num_key_value_heads,
                    model_config.hidden_size // model_config.num_attention_heads,
                )

            elif "gate_up_proj" in nanotron_key:
                gate = "gate" in module_name_hf
                param = _handle_gate_up_proj(param, gate)

            with torch.no_grad():
                param_hf.copy_(param)


def get_hf_config(config: NanotronLlamaConfig) -> HFLlamaConfig:
    """Converts a nanotron configuration to huggingface configuration."""
    attrs = {key: getattr(config, value) for key, value in get_config_mapping(nt_to_hf=False).items()}
    return HFLlamaConfig(**attrs)


def convert_checkpoint_and_save(checkpoint_yt_path: str, save_yt_path: str, tokenizer_name: Optional[str] = None):
    """Loads the nanotron checkpoint in `checkpoint_path`, creates
    a new huggingface instance, copies the weights from the nanotron checkpoint
    and saves the transformed huggingface to `save_path`."""

    # Init nanotron model.

    raw_config = yt.read_file(checkpoint_yt_path + "/model_config.json").read().decode()
    attrs = json.loads(raw_config)
    model_config = NanotronLlamaConfig(**attrs)
    nanotron_model = load_nanotron_model(
        model_config=model_config,
        checkpoint_yt_path=checkpoint_yt_path,
    )
    # Init huggingface model.
    with init_on_device_and_dtype(torch.device("cuda"), torch.bfloat16):
        model_config_hf = get_hf_config(model_config)
        hf_model = LlamaForCausalLM._from_config(model_config_hf)

    # Copy weights, initialize tokenizer and save model.
    if tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.save_pretrained("/slot/sandbox/model")
    convert_nt_to_hf(nanotron_model, hf_model, model_config)
    hf_model.save_pretrained("/slot/sandbox/model")
    def dfs(path):
        local_path = "/slot/sandbox/model"
        yt_path = save_yt_path
        if path:
            local_path = f"{local_path}/{path}"
            yt_path = f"{yt_path}/{path}"
        if os.path.isdir(local_path):
            yt.create("map_node", yt_path, ignore_existing=True, recursive=True)
            for file in os.listdir(local_path):
                if path:
                    dfs(f"{path}/{file}")
                else:
                    dfs(file)
        else:
            with open(local_path, "rb") as f:
                data = f.read()
                yt.write_file(yt_path, data)
    dfs("")

def check_converted_model_generation(save_path: Path):
    """Loads a huggingface model and tokenizer from `save_path` and
    performs a dummy text generation."""

    tokenizer = AutoTokenizer.from_pretrained(save_path)
    input_ids = tokenizer(TEST_PROMPT, return_tensors="pt")["input_ids"].cuda()
    print("Inputs:", tokenizer.batch_decode(input_ids))

    model = LlamaForCausalLM.from_pretrained(save_path).cuda().bfloat16()
    out = model.generate(input_ids, max_new_tokens=100)
    print("Generation (converted): ", tokenizer.batch_decode(out))


if __name__ == "__main__":
    toolbox = prepare_and_get_toolbox(backend=Tractorch())
    os.environ["RANK"] = str(toolbox.coordinator.get_self_index())

    parser = ArgumentParser(description="Convert Nanotron weights to HF format")
    parser.add_argument("--checkpoint_yt_path", type=str, default="llama-7b", help="Path to the checkpoint")
    parser.add_argument("--save_yt_path", type=str, default="llama-7b-hf", help="Path to save the HF model")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    args = parser.parse_args()

    # Convert Nanotron model to HF format.
    convert_checkpoint_and_save(
        checkpoint_yt_path=args.checkpoint_yt_path, save_yt_path=args.save_yt_path, tokenizer_name=args.tokenizer_name
    )

    # Check if the conversion was successful by generating some text.
    # if args.tokenizer_name is not None:
    #    check_converted_model_generation(save_path=args.save_path)
