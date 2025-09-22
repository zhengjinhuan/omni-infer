import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch.nn.parameter import Parameter, UninitializedParameter

from omni.adaptors.sglang.distributed import (
    get_local_world_rank,
    get_local_world_size,
    tensor_model_local_world_parallel_all_reduce,
)
from sglang.srt.distributed import (
    parallel_state,
)

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead as ParallelLMHeadGPU

class ParallelLMHead(ParallelLMHeadGPU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tp_rank = get_local_world_rank()
        self.tp_size = get_local_world_size()

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        with use_symmetric_memory(get_local_world_group()) as sm:
            output_parallel = self.quant_method.embedding(
                self, input_, self.weight, self.tp_rank, self.tp_size, bias
            )
            sm.tag(output_parallel)

        output = tensor_model_local_world_parallel_all_reduce(output_parallel)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias