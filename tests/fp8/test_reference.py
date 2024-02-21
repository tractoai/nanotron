from copy import deepcopy

import torch
from msamp.common.dtype import Dtypes as MS_Dtypes
from msamp.nn import LinearReplacer
from msamp.optim import LBAdam
from torch import nn
from torch.optim import Adam

from nanotron.fp8.optim import FP8Adam
from utils import convert_linear_to_fp8


def test_optim():
    HIDDEN_SIZE = 16
    N_STEPS = 1
    LR = 1e-3

    ref_linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda")
    linear = deepcopy(ref_linear)
    linear = convert_linear_to_fp8(linear)
    msamp_linear = deepcopy(ref_linear)
    msamp_linear = LinearReplacer.replace(msamp_linear, MS_Dtypes.kfloat16)

    ref_optim = Adam(ref_linear.parameters(), lr=LR)
    msamp_optim = LBAdam(msamp_linear.parameters(), lr=LR)
    optim = FP8Adam(linear.parameters(), lr=LR)

    input = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda", requires_grad=False)

    for _ in range(N_STEPS):
        ref_output = ref_linear(input)
        ref_output.sum().backward()
        ref_optim.step()
        ref_optim.zero_grad()
        
        
        msamp_output = msamp_linear(input)
        msamp_output.sum().backward()
        # msamp_optim.all_reduce_grads(msamp_linear)
        msamp_optim.step()
        msamp_optim.zero_grad()
        
        output = linear(input)
        output.sum().backward()
        optim.step()
        optim.zero_grad()
    

    # NOTE: 3e-4 is from msamp
    torch.testing.assert_close(msamp_linear.weight.float(), ref_linear.weight, rtol=0, atol=3e-4)
    torch.testing.assert_close(msamp_linear.bias.float(), ref_linear.bias, rtol=0, atol=3e-4)

    # torch.testing.assert_close(linear.weight, ref_linear.weight, rtol=0.1, atol=3e-4)
    # torch.testing.assert_close(linear.bias, ref_linear.bias, rtol=0, atol=3e-4)


def test_fwd_and_bwd():
    HIDDEN_SIZE = 16
    ref_linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda")
    msamp_linear = deepcopy(ref_linear)
    msamp_linear = LinearReplacer.replace(msamp_linear, MS_Dtypes.kfloat16)

    linear = convert_linear_to_fp8(deepcopy(ref_linear))

    input = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda")

    ref_output = ref_linear(input)
    msamp_output = msamp_linear(input)
    output = linear(input)

    torch.testing.assert_close(msamp_output.float(), ref_output, rtol=0, atol=0.1)
    
    msamp_output.sum().backward()
    ref_output.sum().backward()
    output.sum().backward()

    torch.testing.assert_close(msamp_linear.weight.grad.float(), ref_linear.weight.grad, rtol=0.1, atol=0.1)
    torch.testing.assert_close(msamp_linear.bias.grad, ref_linear.bias.grad, rtol=0, atol=0.1)