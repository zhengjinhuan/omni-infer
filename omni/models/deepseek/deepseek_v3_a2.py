# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only DeepseekV3 model."""
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, Literal
import os
import torch, torch_npu
from torch import nn
from torch.nn.parameter import Parameter, UninitializedParameter
from transformers import PretrainedConfig
import torch.distributed as dist
import torchair._contrib.custom_torch_ops
import torchair as tng

from vllm.config import CacheConfig, QuantizationConfig, VllmConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.attention import Attention, AttentionMetadata
from vllm.distributed import (divide, 
                              get_pp_group,
                              get_ep_group,
                              get_dp_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather,
                              get_world_group)
from vllm.model_executor.layers.linear import (WEIGHT_LOADER_V2_SUPPORTED,
                                               ColumnParallelLinear,
                                               LinearBase, ReplicatedLinear,
                                               adjust_marlin_shard,
                                               adjust_scalar_to_fused_array,
                                               logger)
from vllm.config import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    PPMissingLayer, is_pp_missing_parameter, make_layers, make_empty_intermediate_tensors_factory)
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           PerTensorScaleParameter,
                                           RowvLLMParameter)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors

from omni.models.common.layers.activation import SiluAndMul
from omni.models.common.layers.layernorm import RMSNorm
from omni.models.common.layers.rotary_embedding import get_rope

from omni.models.common.layers.vocab_parallel_embedding import (
    ParallelLMHead, 
    VocabParallelEmbedding
)

from omni.models.common.layers.linear import (
    MergedReplicatedLinear,
    RowParallelLinearWithReduceScatter,
    RowParallelLinearCross
)
from omni.adaptors.vllm.distributed.communication_op import (
    reduce_scatter_two_stage, all_gather_two_stage, all_gather_local, reduce_scatter_local,
    reduce_scatter_world, all_gather_world, all_gather_cross,
    reduce_scatter_pipeline, all_gather_pipeline, prefill_reduce_scatter_pipeline,
    reduce_scatter_round_pipeline, all_gather_round_pipeline,
    all_to_all_local, reduce_scatter_cross)
from omni.adaptors.vllm.distributed.parallel_state import (
    get_npu_device_count,
    get_local_group_size, 
    get_local_group_rank,
    get_round_cross_group_from_list
)

from omni.models.common.layers.moe.fused_moe.layer import FusedMoE
from omni.models.common.config.model_config import model_extra_config


"""MLP 模块激活拆分长度，按64G显存拆分，需要根据序列长度以及性能确认最佳拆分长度"""
SEQ_SPLIT_LENGTH = 4096
SEQ_SPLIT_LENGTH_BEFORE_ALL_GATHER = 64 if model_extra_config.operator_opt_config.prefill_moe_all_to_all else 256
KVCACHE_NZ_DIM = 16
"""stream name"""
STREAM_TOPK_COMPUTE = 'topk_compute'
STREAM_SHARED_EXPERT = 'shared_expert'
STREAM_TOPK_COMM = 'topk_comm'
STREAM_INTERNODE_COMM_0 = 'internode_comm_0'
STREAM_INTERNODE_COMM_1 = 'internode_comm_1'
STREAM_INTERNODE_COMM_2 = 'internode_comm_2'

class DeepSeekMergedColumnParallelLinear(LinearBase):
    def __init__(self,
                 input_size: int,
                 output_sizes: List[int],
                 tp_size: int,
                 tp_rank: int,
                 bias: bool = True,
                 gather_output: bool = False,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        self.output_sizes = output_sizes
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        output_size = sum(output_sizes)
        super().__init__(input_size, output_size, skip_bias_add, params_dtype,
                         quant_config, prefix)

        self.gather_output = gather_output

        self.output_size_per_partition = divide(self.output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size)
                for output_size in self.output_sizes
            ]
        output_sizes = [output_size]

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2 if self.quant_method.__class__.__name__
                                         in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition,
                            dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        output = self.quant_method.apply(self, input_, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[int] = None):

        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.data[loaded_shard_id].copy_(loaded_weight)
            param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            return

        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)
            shard_size = loaded_weight.size(output_dim) // self.tp_size
            start_idx = self.tp_rank * shard_size

            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                 shard_size)

            param.shard_id.append(loaded_shard_id)
            param.shard_id_map[loaded_shard_id] = len(param.data_container)
            param.data_container.append(loaded_weight)
            if len(param.data_container) == 2:
                self.qweight = param.materialize_nested()
            return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        is_metadata = getattr(param, "is_metadata", False)
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0)

                param_data.copy_(loaded_weight)
                return
            current_shard_offset = 0
            shard_offsets: List[Tuple[int, int, int]] = []
            for i, output_size in enumerate(self.output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset)

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        if output_dim is not None:
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
            shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset)

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                            False)
            if use_bitsandbytes_4bit:
                shard_size = loaded_weight.shape[output_dim]
                shard_offset = loaded_weight.shape[output_dim] * \
                               loaded_shard_id

            param_data = param_data.narrow(output_dim, shard_offset,
                                           shard_size)
            start_idx = self.tp_rank * shard_size
            if not use_bitsandbytes_4bit:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     shard_size)
        elif is_metadata:
            shard_size = loaded_weight.shape[0]
            shard_offset = loaded_shard_id * shard_size
            param_data = param_data.narrow(0, shard_offset, shard_size)

        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id)

        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions.")

        param_data.copy_(loaded_weight)

    def _load_fused_module_from_checkpoint(self, param: BasevLLMParameter,
                                           loaded_weight: torch.Tensor):
        """
        Handle special case for models where MLP layers are already
        fused on disk. In this case, we have no shard id. This function
        determmines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """

        current_shard_offset = 0
        shard_offsets: List[Tuple[int, int, int]] = []
        for i, output_size in enumerate(self.output_sizes):
            shard_offsets.append((i, current_shard_offset, output_size))
            current_shard_offset += output_size

        for shard_id, shard_offset, shard_size in shard_offsets:
            if isinstance(param, (PackedColumnParameter, PackedvLLMParameter
                                  )) and param.packed_dim == param.output_dim:
                shard_size, shard_offset = \
                    param.adjust_shard_indexes_for_packing(
                        shard_size=shard_size, shard_offset=shard_offset)

            loaded_weight_shard = loaded_weight.narrow(param.output_dim,
                                                       shard_offset,
                                                       shard_size)
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def weight_loader_v2(self,
                         param: BasevLLMParameter,
                         loaded_weight: torch.Tensor,
                         loaded_shard_id: Optional[int] = None):
        if loaded_shard_id is None:
            if isinstance(param, PerTensorScaleParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight,
                                                shard_id=0)
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight)
                return
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size

        param.load_merged_column_weight(loaded_weight=loaded_weight,
                                        shard_id=loaded_shard_id,
                                        shard_offset=shard_offset,
                                        shard_size=shard_size)


class DeepSeekRowParallelLinear(LinearBase):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 tp_size: int,
                 tp_rank: int,
                 bias: bool = True,
                 input_is_parallel: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 reduce_results: bool = True,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__(input_size, output_size, skip_bias_add, params_dtype,
                         quant_config, prefix)

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.input_size_per_partition = divide(input_size, self.tp_size)

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=[self.output_size],
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2 if self.quant_method.__class__.__name__
                                         in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the "
                             "results can lead to incorrect results")

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        input_dim = getattr(param, "input_dim", None)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        if is_gguf_weight and isinstance(param, UninitializedParameter):
            weight_shape = list(loaded_weight.shape)
            if input_dim:
                weight_shape[input_dim] = weight_shape[input_dim] // self.tp_size
            param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)

        param_data = param.data
        if input_dim is not None and not use_bitsandbytes_4bit:
            shard_size = param_data.shape[input_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx,
                                                 shard_size)

        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        param_data.copy_(loaded_weight)

    def weight_loader_v2(self, param: BasevLLMParameter,
                         loaded_weight: torch.Tensor):
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        param.load_row_parallel_weight(loaded_weight=loaded_weight)

    def forward(self, input_):
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output = self.quant_method.apply(self, input_, bias=bias_)
        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias

    def extra_repr(self) -> str:
        s = f"input_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", reduce_results={self.reduce_results}"
        return s


class ParallelDeepseekMLP(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            tp_parallel: Literal["global", "local", "no_tp"] = "no_tp",
            quant_config: Optional[QuantizationConfig] = None,
            reduce_results: bool = True,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_parallel = tp_parallel

        if tp_parallel == "local":
            self.tp_size = get_local_group_size()
            self.tp_rank = get_local_group_rank()
        elif tp_parallel == "global":
            self.tp_size = get_ep_group().world_size
            self.tp_rank = get_ep_group().rank_in_group
        elif tp_parallel == "no_tp":
            self.tp_size = 1
            self.tp_rank = 0

        self.gate_up_proj = DeepSeekMergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        if os.environ["ROLE"] == "decode":
            self.gate_up_proj.throw_dequant = True
        self.down_proj = DeepSeekRowParallelLinear(intermediate_size,
                                                   hidden_size,
                                                   tp_size=self.tp_size,
                                                   tp_rank=self.tp_rank,
                                                   bias=False,
                                                   quant_config=quant_config,
                                                   reduce_results=False,
                                                   prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn_obj = SiluAndMul()
        self.quant_symbol = True if quant_config else False
        self.device_count = get_npu_device_count()
        self.node_rank = get_world_group().rank_in_group // get_npu_device_count()
        self.which_half = get_world_group().rank_in_group // (get_world_group().world_size // 2)

    def act_fn(self, x, quant_symbol):
        if quant_symbol and isinstance(x, tuple):
            x = dict(zip(['x_int8', 'pertoken_scale'], x))
            x['out_scale'] = self.gate_up_proj.weight_scale
        return self.act_fn_obj(x, quant_symbol)

    def forward(self, x, residual, attn_metadata, pertoken_scale=None, no_communication=False):
        if self.tp_parallel == "no_tp" or no_communication:
            return self.forward_no_tp(x, residual, attn_metadata, pertoken_scale)

        if self.tp_parallel == "local":
            return self.forward_local_tp(x, residual, attn_metadata)

        if self.tp_parallel == "global":
            return self.forward_global_tp(x, residual, attn_metadata)

    def forward_no_tp(self, x, residual, attn_metadata, pertoken_scale=None):
        if pertoken_scale is None:
            x, pertoken_scale = torch_npu.npu_dynamic_quant(x)

        x = {'x_int8': x,
             'pertoken_scale': pertoken_scale}
        gate_up, _ = self.gate_up_proj.forward(x)
        x = self.act_fn(gate_up, self.quant_symbol)
        x, _ = self.down_proj.forward(x)

        return x, residual

    def forward_local_tp(self, x, residual, attn_metadata):
        pad_size = 0
        is_prefill = (attn_metadata is None or attn_metadata.prefill)
        if is_prefill and model_extra_config.parall_config.dp_size > 1:
            local_length = x.shape[0]
            reduce_length = torch.tensor(x.shape[0], dtype=torch.int64, device="npu")
            dist.all_reduce(reduce_length, op=dist.ReduceOp.MAX, async_op=False)
            global_max_length = reduce_length.item()
            pad_size = global_max_length - x.shape[0]

            x = torch.nn.functional.pad(
                x, (0, 0, 0, pad_size)
            )

        x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        x = all_gather_local(x, idx=0, dim=0)
        pertoken_scale = all_gather_local(pertoken_scale, idx=1, dim=0)

        x = {'x_int8': x,
             'pertoken_scale': pertoken_scale}
        gate_up, _ = self.gate_up_proj.forward(x)
        x = self.act_fn(gate_up, self.quant_symbol)
        x, _ = self.down_proj.forward(x)

        x = reduce_scatter_local(x, idx=0)

        if is_prefill and pad_size > 0:
            x = x[:local_length, :]
        return x, residual

    def forward_global_tp(self, x, residual, attn_metadata):
        is_prefill = (attn_metadata is None or attn_metadata.prefill)
        if not is_prefill:
            x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
            global_pertoken_scale = all_gather_two_stage(pertoken_scale, idx=1, dim=0, reverse=True)
            if model_extra_config.operator_opt_config.enable_round_pipeline_comm:
                x = all_gather_round_pipeline(x, idx=0, node_rank=self.node_rank, dim=0)
            elif model_extra_config.operator_opt_config.enable_pipeline_comm:
                x = all_gather_pipeline(x, idx=0, which_half=self.which_half, dim=0)
                global_pertoken_scale = global_pertoken_scale.view(2, -1, self.device_count, pertoken_scale.shape[0]) \
                    .transpose(1, 2).reshape(-1)
            else:
                x = all_gather_two_stage(x, idx=0, dim=0)
                global_pertoken_scale = global_pertoken_scale.view(-1, self.device_count, pertoken_scale.shape[0]) \
                    .transpose(0, 1).reshape(-1)
        else:
            pad_size = 0
            if model_extra_config.parall_config.dp_size > 1:
                local_length = x.shape[0]
                reduce_length = torch.tensor(x.shape[0], dtype=torch.int64, device="npu")
                dist.all_reduce(reduce_length, op=dist.ReduceOp.MAX, async_op=False)
                global_max_length = reduce_length.item()
                pad_size = global_max_length - x.shape[0]

                x = torch.nn.functional.pad(
                    x, (0, 0, 0, pad_size)
                )

            x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
            x = all_gather_two_stage(x, idx=0, dim=0)
            global_pertoken_scale = all_gather_two_stage(pertoken_scale, idx=1, dim=0)

        x = {'x_int8': x,
             'pertoken_scale': global_pertoken_scale}
        gate_up, _ = self.gate_up_proj.forward(x)
        x = self.act_fn(gate_up, self.quant_symbol)
        x, _ = self.down_proj.forward(x)

        if not is_prefill:
            if model_extra_config.operator_opt_config.enable_round_pipeline_comm:
                x = reduce_scatter_round_pipeline(x, idx=0, node_rank=self.node_rank)
            elif model_extra_config.operator_opt_config.enable_pipeline_comm:
                x = reduce_scatter_pipeline(x, idx=0, which_half=self.which_half)
            else:
                x = reduce_scatter_two_stage(x, idx=0)
        else:
            x = reduce_scatter_two_stage(x, idx=0)

        if is_prefill and pad_size > 0:
            x = x[:local_length, :]
        return x, residual


class DeepseekMoE(nn.Module):

    def __init__(
            self,
            config: PretrainedConfig,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ):
        super().__init__()
        self.ep_size = get_ep_group().world_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.n_total_experts = config.n_routed_experts
        self.device_count = get_npu_device_count()
        self.node_rank = get_world_group().rank_in_group // get_npu_device_count()
        self.which_half = get_world_group().rank_in_group // (get_world_group().world_size // 2)
        self.routed_scaling_factor = config.routed_scaling_factor
        if self.ep_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.ep_size} is greater than "
                f"the number of experts {config.n_routed_experts}.")

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")

        self.gate_dtype = torch.bfloat16
        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.n_routed_experts,
                                     bias=False,
                                     quant_config=None,
                                     params_dtype=self.gate_dtype,
                                     prefix=f"{prefix}.gate")
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float), requires_grad=False)
        else:
            self.gate.e_score_correction_bias = None

        self.experts = FusedMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
        )
        self.warm_up = True
        self.shared_experts_tp = None
        if config.n_shared_experts is not None:
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)

            self.shared_experts_tp = "no_tp"

            self.shared_experts = ParallelDeepseekMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                tp_parallel=self.shared_experts_tp,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts"
            )

        self.ep_shift_tensor = None

        self.attn_prefetch = None

    def forward_decode(self, hidden_states: torch.Tensor, residual: torch.Tensor,
                       attn_metadata: AttentionMetadata, layer_id: int, kv_prefetch: torch.Tensor = None) -> torch.Tensor:
        MAX_PREFETCH_SIZE = 90000000
        LARGE_BATCH, MEDIUM_BATCH, SMALL_BATCH = False, False, False
        if hidden_states.shape[0] >= 120:
            LARGE_BATCH = True
        elif hidden_states.shape[0] >= 60:
            MEDIUM_BATCH = True
        else:
            SMALL_BATCH = True

        hidden_states_int8, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
        with tng.scope.npu_stream_switch(STREAM_TOPK_COMPUTE):
            router_logits, _ = self.gate.forward(hidden_states)
            topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                                self.experts.top_k, self.experts.use_grouped_topk,
                                                                self.experts.renormalize,
                                                                self.experts.topk_group, self.experts.num_expert_group,
                                                                self.experts.custom_routing_function,
                                                                self.experts.scoring_func,
                                                                self.experts.e_score_correction_bias,
                                                                self.routed_scaling_factor,
                                                                layer=self.experts)
            topk_ids = self.experts.apply_expert_load_balance(
                topk_ids=topk_ids, 
                best_topk_ids=attn_metadata.decode.best_topk if attn_metadata is not None and attn_metadata.decode is not None else None
            )
            if attn_metadata is not None and attn_metadata.decode is not None:
                actual_batch_mask = attn_metadata.decode.mc2_mask \
                                                        .to(torch.int32).view(-1, 1) \
                                                        .repeat(1, self.experts.top_k)
                topk_ids = actual_batch_mask * topk_ids + (1 - actual_batch_mask) * self.n_total_experts

            topk_cat = torch.cat((topk_weights, topk_ids.to(torch.float), pertoken_scale.unsqueeze(-1)), dim=-1)

        with tng.scope.npu_stream_switch(STREAM_SHARED_EXPERT):
            if LARGE_BATCH or MEDIUM_BATCH:
                shared_output, _ = self.shared_experts(hidden_states_int8, residual=None, attn_metadata=attn_metadata,
                                                    pertoken_scale=pertoken_scale)
                if model_extra_config.operator_opt_config.enable_prefetch:
                    torch_npu.npu_prefetch(self.experts.w13_weight, shared_output, MAX_PREFETCH_SIZE)
        
        with tng.scope.npu_stream_switch(STREAM_TOPK_COMM):
            topk_local_all = all_gather_local(topk_cat, idx=1, dim=0)

        input_ag = all_gather_local(hidden_states_int8, idx=0, dim=0)
        with tng.scope.npu_stream_switch(STREAM_INTERNODE_COMM_0):
            round0_swp = tng.scope.npu_wait_tensor(hidden_states_int8, hidden_states_int8)
            round0_swp = get_round_cross_group_from_list(round=0).swap(round0_swp, method="all2allv")
        with tng.scope.npu_stream_switch(STREAM_INTERNODE_COMM_1):
            round1_swp = tng.scope.npu_wait_tensor(hidden_states_int8, input_ag)
            round1_swp = get_round_cross_group_from_list(round=1).swap(round1_swp, method="all2allv")
        with tng.scope.npu_stream_switch(STREAM_INTERNODE_COMM_2):
            round2_swp = tng.scope.npu_wait_tensor(hidden_states_int8, round1_swp)
            round2_swp = get_round_cross_group_from_list(round=2).swap(round2_swp, method="all2allv")
        
        with tng.scope.npu_stream_switch(STREAM_TOPK_COMM):
            topk_local_all_wait = tng.scope.npu_wait_tensor(topk_local_all, round0_swp)
            topk_all = all_gather_cross(topk_local_all_wait, idx=1, dim=0)
        round0_swp = tng.scope.npu_wait_tensor(round0_swp, input_ag)
        round0_ag = all_gather_local(round0_swp, idx=0, dim=0)
        round1_swp = tng.scope.npu_wait_tensor(round1_swp, round0_ag)
        round1_ag = all_gather_local(round1_swp, idx=0, dim=0)
        round2_swp = tng.scope.npu_wait_tensor(round2_swp, round1_ag)
        round2_ag = all_gather_local(round2_swp, idx=0, dim=0)


        with tng.scope.npu_stream_switch(STREAM_SHARED_EXPERT):
            if SMALL_BATCH:
                hidden_states_int8 = tng.scope.npu_wait_tensor(hidden_states_int8, input_ag)
                shared_output, _ = self.shared_experts(hidden_states_int8, residual=None, attn_metadata=attn_metadata,
                                                    pertoken_scale=pertoken_scale)
                if model_extra_config.operator_opt_config.enable_prefetch:
                    torch_npu.npu_prefetch(self.experts.w13_weight, input_ag, MAX_PREFETCH_SIZE)


        with tng.scope.npu_stream_switch(STREAM_TOPK_COMPUTE):
            topk_weights, topk_ids, global_pertoken_scale = torch.split(topk_all,
                                                                        [topk_weights.shape[-1], topk_ids.shape[-1], 1],
                                                                        dim=-1)
            topk_ids = torch.round(topk_ids).to(torch.int32)
            global_pertoken_scale = global_pertoken_scale.squeeze(-1)

        if self.node_rank == 0:
            global_hidden_states = torch.cat([input_ag, round0_ag, round1_ag, round2_ag], dim=0)
        elif self.node_rank == 1:
            global_hidden_states = torch.cat([round0_ag, input_ag, round2_ag, round1_ag], dim=0)
        elif self.node_rank == 2:
            global_hidden_states = torch.cat([round1_ag, round2_ag, input_ag, round0_ag], dim=0)
        elif self.node_rank == 3:
            global_hidden_states = torch.cat([round2_ag, round1_ag, round0_ag, input_ag], dim=0)

        final_hidden_states = self.experts(
            hidden_states=global_hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            pertoken_scale=global_pertoken_scale,
            attn_metadata=attn_metadata
        )

        if self.node_rank == 0:
            input_self, round0, round1, round2 = torch.split(final_hidden_states, final_hidden_states.shape[0] // 4, dim=0)
        elif self.node_rank == 1:
            round0, input_self, round2, round1 = torch.split(final_hidden_states, final_hidden_states.shape[0] // 4, dim=0)
        elif self.node_rank == 2:
            round1, round2, input_self, round0 = torch.split(final_hidden_states, final_hidden_states.shape[0] // 4, dim=0)
        elif self.node_rank == 3:
            round2, round1, round0, input_self = torch.split(final_hidden_states, final_hidden_states.shape[0] // 4, dim=0)

        round2 = round2.to(torch.bfloat16)
        with tng.scope.npu_stream_switch(STREAM_TOPK_COMPUTE):
            round1 = round1.to(torch.bfloat16)
            round0 = round0.to(torch.bfloat16)
            input_self = input_self.to(torch.bfloat16)

            if self.attn_prefetch is not None:
                torch_npu.npu_prefetch(self.attn_prefetch.q_a_proj.weight, input_self, MAX_PREFETCH_SIZE)
                torch_npu.npu_prefetch(self.attn_prefetch.kv_a_proj_with_mqa.weight, input_self, MAX_PREFETCH_SIZE)
                torch_npu.npu_prefetch(self.attn_prefetch.q_b_proj.weight, input_self, MAX_PREFETCH_SIZE)
                torch_npu.npu_prefetch(self.attn_prefetch.W_UK, input_self, MAX_PREFETCH_SIZE)
            if kv_prefetch is not None and isinstance(kv_prefetch, Tuple) and kv_prefetch[0].numel():
                torch_npu.npu_prefetch(kv_prefetch[0], input_self, MAX_PREFETCH_SIZE)
        
        round2_rs = reduce_scatter_local(round2, idx=0)
        round1 = tng.scope.npu_wait_tensor(round1, round2_rs)
        round1_rs = reduce_scatter_local(round1, idx=0)
        round0 = tng.scope.npu_wait_tensor(round0, round1_rs)
        round0_rs = reduce_scatter_local(round0, idx=0)
        input_self = tng.scope.npu_wait_tensor(input_self, round0_rs)
        input_rs = reduce_scatter_local(input_self, idx=0)
        with tng.scope.npu_stream_switch(STREAM_INTERNODE_COMM_2):
            round2_swp = get_round_cross_group_from_list(round=2).swap(round2_rs, method="all2allv")
        with tng.scope.npu_stream_switch(STREAM_INTERNODE_COMM_1):
            round1_swp = get_round_cross_group_from_list(round=1).swap(round1_rs, method="all2allv")
        with tng.scope.npu_stream_switch(STREAM_INTERNODE_COMM_0):
            round0_swp = get_round_cross_group_from_list(round=0).swap(round0_rs, method="all2allv")

        final_hidden_states = input_rs + round0_swp + round1_swp + round2_swp + shared_output

        return final_hidden_states, residual

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor, attn_metadata: AttentionMetadata,
                layer_id: int, kv_prefetch: torch.Tensor = None) -> torch.Tensor:
        is_prefill = (attn_metadata is None or attn_metadata.prefill is not None)
        if is_prefill and model_extra_config.operator_opt_config.pd_seperate_prefill:
            return self.forward_prefill_pd_seperate(hidden_states, residual, attn_metadata, layer_id)
        elif not is_prefill and model_extra_config.operator_opt_config.enable_round_pipeline_comm:
            return self.forward_decode(hidden_states, residual, attn_metadata, layer_id, kv_prefetch)
        else:
            return self.forward_normal(hidden_states, residual, attn_metadata, layer_id)

    def forward_prefill_pd_seperate(self, hidden_states: torch.Tensor, residual: torch.Tensor,
                                    attn_metadata: AttentionMetadata, layer_id: int) -> torch.Tensor:
        MULTISTREAM_THRESHOLD = 1200
        GMM_CHUNK_SIZE = MULTISTREAM_THRESHOLD * get_ep_group().world_size
        enable_prefill_moe_multi_stream = model_extra_config.operator_opt_config.prefill_moe_multi_stream and hidden_states.shape[0] <= MULTISTREAM_THRESHOLD
        enable_prefill_pipeline_comm = model_extra_config.operator_opt_config.prefill_enable_pipeline_comm and hidden_states.shape[0] <= MULTISTREAM_THRESHOLD
        hidden_states_int8, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)

        if enable_prefill_moe_multi_stream:
            shared_stream = torch.npu.Stream()
            curr_stream = torch.npu.current_stream()
            shared_stream.wait_stream(curr_stream)
            with torch.npu.stream(shared_stream):
                global_hidden_states = all_gather_world(hidden_states_int8, idx=0, dim=0)
        else:
            global_hidden_states = all_gather_world(hidden_states_int8, idx=0, dim=0)

        if self.warm_up:
            self.warm_up = False

        if self.n_shared_experts is not None:
            shared_output, _ = self.shared_experts(hidden_states_int8, None, attn_metadata, pertoken_scale)

        router_logits, _ = self.gate.forward(hidden_states.to(self.gate_dtype))
        topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                            self.experts.top_k, self.experts.use_grouped_topk,
                                                            self.experts.renormalize,
                                                            self.experts.topk_group, self.experts.num_expert_group,
                                                            self.experts.custom_routing_function,
                                                            self.experts.scoring_func,
                                                            self.experts.e_score_correction_bias,
                                                            self.routed_scaling_factor,
                                                            layer=self.experts
                                                            )
        topk_ids = self.experts.apply_expert_load_balance(
            topk_ids=topk_ids, 
            best_topk_ids=attn_metadata.decode.best_topk if attn_metadata is not None and attn_metadata.decode is not None else None
        )
        
        topk_cat = torch.cat((topk_weights, topk_ids.to(torch.float), pertoken_scale.unsqueeze(-1)), dim=-1)
        topk_all = all_gather_world(topk_cat, idx=1, dim=0)
        topk_weights, topk_ids, global_pertoken_scale = torch.split(topk_all, [topk_weights.shape[-1], topk_ids.shape[-1],1], dim=-1)
        topk_ids = torch.round(topk_ids).to(torch.int32)

        global_pertoken_scale = global_pertoken_scale.squeeze(-1)

        if enable_prefill_moe_multi_stream:
            torch.npu.current_stream().wait_stream(shared_stream)
            shared_stream.wait_stream(torch.npu.current_stream())

        
        final_hidden_states = self.chunked_gmm(
            hidden_states=global_hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            pertoken_scale=global_pertoken_scale,
            attn_metadata=attn_metadata,
            chunk_size=GMM_CHUNK_SIZE
        )

        if enable_prefill_pipeline_comm:
            final_hidden_states = prefill_reduce_scatter_pipeline(final_hidden_states, idx=1, which_half=self.which_half)
        else:
            final_hidden_states = reduce_scatter_world(final_hidden_states, idx=0)

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states, residual

    def chunked_gmm(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                    pertoken_scale: torch.Tensor, attn_metadata: AttentionMetadata, chunk_size: int):

        if hidden_states.shape[0] > chunk_size:
            out = []
            hidden_states_list = torch.split(hidden_states, chunk_size)
            topk_weights_list = torch.split(topk_weights, chunk_size)
            topk_ids_list = torch.split(topk_ids, chunk_size)
            pertoken_scale_list = torch.split(pertoken_scale, chunk_size)
            for hid_states, topk_w, topk_id, scale in zip(hidden_states_list, topk_weights_list, topk_ids_list, pertoken_scale_list):
                out.append(self.experts(hidden_states=hid_states,
                                        topk_weights=topk_w,
                                        topk_ids=topk_id,
                                        pertoken_scale=scale,
                                        attn_metadata=attn_metadata))
            return torch.cat(out)

        return self.experts(hidden_states=hidden_states,
                            topk_weights=topk_weights,
                            topk_ids=topk_ids,
                            pertoken_scale=pertoken_scale,
                            attn_metadata=attn_metadata)


    def forward_normal(self, hidden_states: torch.Tensor, residual: torch.Tensor, attn_metadata: AttentionMetadata,
                       layer_id: int) -> torch.Tensor:
        is_prefill = (attn_metadata is None or attn_metadata.prefill is not None)

        hidden_states_int8, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)

        if not is_prefill and model_extra_config.operator_opt_config.enable_round_pipeline_comm:
            global_hidden_states = all_gather_round_pipeline(hidden_states_int8, idx=0, node_rank=self.node_rank, dim=0)
        elif not is_prefill and model_extra_config.operator_opt_config.enable_pipeline_comm:
            global_hidden_states = all_gather_pipeline(hidden_states_int8, idx=0, which_half=self.which_half, dim=0)
        else:
            global_hidden_states = all_gather_two_stage(hidden_states_int8, idx=0, dim=0)

        if self.warm_up:
            self.warm_up = False
        # decode profile_run
        if is_prefill or not model_extra_config.operator_opt_config.enable_kv_rmsnorm_rope_cache:
            if self.n_shared_experts is not None:
                shared_output, _ = self.shared_experts(hidden_states, None, attn_metadata)

            router_logits, _ = self.gate.forward(hidden_states.to(self.gate_dtype))
            topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                                self.experts.top_k, self.experts.use_grouped_topk,
                                                                self.experts.renormalize,
                                                                self.experts.topk_group, self.experts.num_expert_group,
                                                                self.experts.custom_routing_function,
                                                                self.experts.scoring_func,
                                                                self.experts.e_score_correction_bias,
                                                                self.routed_scaling_factor,
                                                                layer=self.experts)
            topk_ids = self.experts.apply_expert_load_balance(
                topk_ids=topk_ids, 
                best_topk_ids=attn_metadata.decode.best_topk if attn_metadata is not None and attn_metadata.decode is not None else None
            )

            topk_cat = torch.cat((topk_weights, topk_ids.to(torch.float), pertoken_scale.unsqueeze(-1)), dim=-1)
            topk_all = all_gather_two_stage(topk_cat, idx=1, dim=0)
            topk_weights, topk_ids, global_pertoken_scale = torch.split(
                topk_all, [topk_weights.shape[-1], topk_ids.shape[-1], 1], dim=-1)
            topk_ids = torch.round(topk_ids).to(torch.int32)
            global_pertoken_scale = global_pertoken_scale.squeeze(-1)
        else:
            if model_extra_config.operator_opt_config.moe_multi_stream_tune:
                with tng.scope.npu_stream_switch('22'):
                    hidden_states = tng.scope.npu_wait_tensor(hidden_states, hidden_states)
                    router_logits, _ = self.gate.forward(hidden_states)
                    topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                                        self.experts.top_k, self.experts.use_grouped_topk,
                                                                        self.experts.renormalize,
                                                                        self.experts.topk_group,
                                                                        self.experts.num_expert_group,
                                                                        self.experts.custom_routing_function,
                                                                        self.experts.scoring_func,
                                                                        self.experts.e_score_correction_bias,
                                                                        self.routed_scaling_factor,
                                                                        layer=self.experts,
                                                                    )
                    topk_ids = self.experts.apply_expert_load_balance(
                        topk_ids=topk_ids, 
                        best_topk_ids=attn_metadata.decode.best_topk if attn_metadata is not None and attn_metadata.decode is not None else None
                    )
                    if attn_metadata is not None and attn_metadata.decode is not None:
                        actual_batch_mask = attn_metadata.decode.mc2_mask \
                                                                .to(torch.int32).view(-1, 1) \
                                                                .repeat(1, self.experts.top_k)
                        topk_ids = actual_batch_mask * topk_ids + (1 - actual_batch_mask) * self.n_total_experts
                    
                    if not model_extra_config.operator_opt_config.decode_moe_dispatch_combine:
                        pertoken_scale = tng.scope.npu_wait_tensor(pertoken_scale, pertoken_scale)
                        topk_cat = torch.cat((topk_weights, topk_ids.to(torch.float), pertoken_scale.unsqueeze(-1)), dim=-1)

                if self.n_shared_experts is not None:
                    with tng.scope.npu_stream_switch('21'):
                        hidden_states = tng.scope.npu_wait_tensor(hidden_states, hidden_states)
                        shared_output, _ = self.shared_experts(hidden_states, None, attn_metadata)

                with tng.scope.npu_stream_switch('23'):
                    if not model_extra_config.operator_opt_config.decode_moe_dispatch_combine:
                        topk_cat = tng.scope.npu_wait_tensor(topk_cat, topk_cat)
                        topk_all = all_gather_two_stage(topk_cat, idx=1, dim=0, reverse=True)

                with tng.scope.npu_stream_switch('22'):
                    if not model_extra_config.operator_opt_config.decode_moe_dispatch_combine:
                        topk_all = tng.scope.npu_wait_tensor(topk_all, topk_all)
                        if model_extra_config.operator_opt_config.enable_round_pipeline_comm:
                            pass
                        elif model_extra_config.operator_opt_config.enable_pipeline_comm:
                            topk_all = topk_all.view(2, -1, self.device_count, topk_weights.shape[0], topk_all.shape[-1]) \
                                .transpose(1, 2).reshape(-1, topk_all.shape[-1])
                        else:
                            topk_all = topk_all.view(-1, self.device_count, topk_weights.shape[0], topk_all.shape[-1]) \
                                .transpose(0, 1).reshape(-1, topk_all.shape[-1])

                        topk_weights, topk_ids, global_pertoken_scale = torch.split(topk_all, [topk_weights.shape[-1],
                                                                                            topk_ids.shape[-1], 1],
                                                                                    dim=-1)
                        topk_ids = torch.round(topk_ids).to(torch.int32)
                        global_pertoken_scale = global_pertoken_scale.squeeze(-1)
            else:
                router_logits, _ = self.gate.forward(hidden_states.to(self.gate_dtype))
                topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                                    self.experts.top_k, self.experts.use_grouped_topk,
                                                                    self.experts.renormalize,
                                                                    self.experts.topk_group,
                                                                    self.experts.num_expert_group,
                                                                    self.experts.custom_routing_function,
                                                                    self.experts.scoring_func,
                                                                    self.experts.e_score_correction_bias,
                                                                    self.routed_scaling_factor,
                                                                    layer=self.experts,
                                                                	)
                topk_ids = self.experts.apply_expert_load_balance(
                    topk_ids=topk_ids, 
                    best_topk_ids=attn_metadata.decode.best_topk if attn_metadata is not None and attn_metadata.decode is not None else None
                )

                if self.n_shared_experts is not None:
                    shared_output, _ = self.shared_experts(hidden_states, None, attn_metadata)

                topk_cat = torch.cat((topk_weights, topk_ids.to(torch.float), pertoken_scale.unsqueeze(-1)), dim=-1)
                topk_all = all_gather_two_stage(topk_cat, idx=1, dim=0, reverse=True)

                # if model_extra_config.operator_opt_config.enable_round_pipeline_comm:
                #     pass
                # elif model_extra_config.operator_opt_config.enable_pipeline_comm:
                #     topk_all = topk_all.view(2, -1, self.device_count, topk_weights.shape[0], topk_all.shape[-1]) \
                #         .transpose(1, 2).reshape(-1, topk_all.shape[-1])
                # else:
                #     topk_all = topk_all.view(-1, self.device_count, topk_weights.shape[0], topk_all.shape[-1]) \
                #         .transpose(0, 1).reshape(-1, topk_all.shape[-1])

                topk_weights, topk_ids, global_pertoken_scale = torch.split(topk_all, [topk_weights.shape[-1],
                                                                                        topk_ids.shape[-1], 1],
                                                                            dim=-1)
                topk_ids = torch.round(topk_ids).to(torch.int32)
                global_pertoken_scale = global_pertoken_scale.squeeze(-1)

        final_hidden_states = self.experts(
            hidden_states=global_hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            pertoken_scale=global_pertoken_scale,
            attn_metadata=attn_metadata
        )
        
        if not is_prefill and model_extra_config.operator_opt_config.enable_round_pipeline_comm:
            final_hidden_states = reduce_scatter_round_pipeline(final_hidden_states, idx=0,
                                                                node_rank=self.node_rank, dtype=torch.float32)
        elif not is_prefill and model_extra_config.operator_opt_config.enable_pipeline_comm:
            final_hidden_states = reduce_scatter_pipeline(final_hidden_states, idx=0, which_half=self.which_half,
                                                          dtype=torch.float32)
        else:
            final_hidden_states = reduce_scatter_two_stage(final_hidden_states, idx=0)

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states, residual


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class AscendDeepseekAttention_MLA(nn.Module):

    def __init__(
            self,
            config: PretrainedConfig,
            hidden_size: int,
            num_heads: int,
            qk_nope_head_dim: int,
            qk_rope_head_dim: int,
            v_head_dim: int,
            q_lora_rank: int,
            kv_lora_rank: int,
            rope_theta: float = 10000,
            rope_scaling: Optional[Dict[str, Any]] = None,
            max_position_embeddings: int = 8192,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_local_heads = num_heads // self.tp_size
        self.scaling = self.qk_head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.merge_qkv = True if model_extra_config.operator_opt_config.merge_qkv else False
        if self.q_lora_rank is not None:
            if self.merge_qkv:
                self.qkv_a_proj = MergedReplicatedLinear(self.hidden_size,
                                                         [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                                                         bias=False,
                                                         quant_config=quant_config,
                                                         prefix=f"{prefix}.qkv_a_proj")
            else:
                self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                                 self.q_lora_rank,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_a_proj")
                self.kv_a_proj_with_mqa = ReplicatedLinear(
                    self.hidden_size,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.kv_a_proj_with_mqa")
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)
            self.norm_res = {}
            if not model_extra_config.operator_opt_config.pd_seperate_prefill:
                for batch_size in model_extra_config.operator_opt_config.decode_gear_list:
                    self.norm_res[batch_size] = torch.zeros([batch_size * self.tp_size, self.q_lora_rank], dtype=torch.bfloat16, device="npu")
            self.q_b_proj = ColumnParallelLinear(self.q_lora_rank,
                                                 self.num_heads *
                                                 self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_b_proj")
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads *
                                               self.qk_head_dim,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.q_proj")
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa")

        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.kv_b_proj")

        if model_extra_config.operator_opt_config.prefill_enable_mla_alltoall_local:
            self.o_proj = RowParallelLinearCross(self.num_heads * self.v_head_dim,
                                                self.hidden_size,
                                                bias=False,
                                                quant_config=quant_config,
                                                prefix=f"{prefix}.o_proj")
        else:
            self.o_proj = RowParallelLinearWithReduceScatter(self.num_heads * self.v_head_dim,
                                                             self.hidden_size,
                                                             bias=False,
                                                             quant_config=quant_config,
                                                             prefix=f"{prefix}.o_proj")
        rope_scaling["rope_type"] = 'deepseek_yarn'
        self.rotary_emb = get_rope(qk_rope_head_dim,
                                   rotary_dim=qk_rope_head_dim,
                                   max_position=max_position_embeddings,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling,
                                   is_neox_style=False)

        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.attn_prefill = Attention(
            num_heads=self.num_local_heads,
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.scaling,
            use_mla=True,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            rotary_emb=self.rotary_emb,
            q_proj=self.q_proj if self.q_lora_rank is None else self.q_b_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa if hasattr(self, 'kv_a_proj_with_mqa') else None,
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
            qkv_a_proj=self.qkv_a_proj if hasattr(self, 'qkv_a_proj') else None,
            q_a_layernorm=self.q_a_layernorm if hasattr(self, 'q_a_layernorm') else None,
            q_b_proj=self.q_b_proj if hasattr(self, 'q_b_proj') else None,
            q_a_proj=self.q_a_proj if hasattr(self, 'q_a_proj') else None
        )

        kv_b_proj_weight = self.kv_b_proj.weight.T
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        self.W_UK, self.W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        self.W_UK = self.W_UK.permute(1, 2, 0)
        self.W_UV = self.W_UV.transpose(0, 1)
        self.is_init = False
        self.is_quant = False if quant_config is None else True
        self.eps = config.rms_norm_eps

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            rotary_emb_cos: Optional[torch.Tensor] = None,
            rotary_emb_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.is_init and not model_extra_config.operator_opt_config.pd_seperate_prefill:
            self.W_UK = torch.nn.Parameter(self.W_UK.contiguous(), requires_grad=False)
            self.W_UV = torch.nn.Parameter(self.W_UV.contiguous(), requires_grad=False)
            self.empty_out = torch.empty(1, dtype=torch.bfloat16)
            self.is_init = True

        is_prefill = (attn_metadata is None or attn_metadata.prefill is not None)
        
        if is_prefill:
            if model_extra_config.operator_opt_config.pd_seperate_prefill:
                return self.forward_prefill_pd_seperate(positions, hidden_states, kv_cache, attn_metadata, rotary_emb_cos, rotary_emb_sin)
            else:
                return self.forward_normal(positions, hidden_states, kv_cache, attn_metadata)
        elif model_extra_config.operator_opt_config.enable_kv_rmsnorm_rope_cache:
            return self.forward_absorb_kv_rmsnorm_rope_cache(positions, hidden_states, kv_cache, attn_metadata)
        else:
            return self.forward_absorb(positions, hidden_states, kv_cache, attn_metadata)

    def forward_prefill_pd_seperate(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            rotary_emb_cos: Optional[torch.Tensor] = None,
            rotary_emb_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if rotary_emb_cos is None or rotary_emb_sin is None:
            cos, sin = self.rotary_emb.get_cos_sin(positions)
        else:
            cos, sin = rotary_emb_cos, rotary_emb_sin

        MULTISTREAM_THRESHOLD = 1200
        enable_prefill_mla_multi_stream = model_extra_config.operator_opt_config.prefill_mla_multi_stream and hidden_states.shape[0] <= MULTISTREAM_THRESHOLD

        if enable_prefill_mla_multi_stream:
            kv_stream = torch.npu.Stream()
            kv_stream2 = torch.npu.Stream()
            curr_stream = torch.npu.current_stream()

        if self.q_lora_rank is not None:
            if self.merge_qkv:
                qkv = self.qkv_a_proj(hidden_states)[0]
                qkv = all_gather_world(qkv, idx=0, dim=0)
                q, latent_cache = torch.split(qkv, [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
                q = self.q_a_layernorm(q)
            else:
                h_quant, h_scale = torch_npu.npu_dynamic_quant(hidden_states)
                hidden_states = {'x_int8': h_quant,
                                 'pertoken_scale': h_scale}
                latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

                if enable_prefill_mla_multi_stream:
                    kv_stream2.wait_stream(curr_stream)
                    with torch.npu.stream(kv_stream2):
                        latent_cache = all_gather_world(latent_cache, idx=0, dim=0)
                else:
                    latent_cache = all_gather_world(latent_cache, idx=0, dim=0)

                q = self.q_a_proj(hidden_states)[0]
                q = self.q_a_layernorm(q)
                q_quant, q_scale = torch_npu.npu_dynamic_quant(q)
                q_scale = all_gather_world(q_scale, idx=1, dim=0)
                q_quant = all_gather_world(q_quant, idx=1, dim=0)
                q = {'x_int8': q_quant,
                     'pertoken_scale': q_scale}

            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = all_gather_world(q, idx=0, dim=0)
            latent_cache = all_gather_world(latent_cache, idx=0, dim=0)

        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = q_pe.unsqueeze(2)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin)
        q_pe = q_pe.squeeze(2)
        q = torch.cat([q_nope, q_pe], dim=-1)

        if attn_metadata is not None:
            if enable_prefill_mla_multi_stream:
                kv_stream.wait_stream(kv_stream2)
                with torch.npu.stream(kv_stream):
                    if isinstance(kv_cache, Dict):
                        kv_cache = kv_cache.get("kv_cache")
                    if kv_cache is not None and isinstance(kv_cache, Tuple) and kv_cache[0].numel() > 0: 
                        _, _, k_pe, kv_a = torch_npu.npu_kv_rmsnorm_rope_cache(
                            latent_cache.view(-1, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim),
                            self.kv_a_layernorm.weight,
                            cos.view(-1, 1, 1, self.qk_rope_head_dim),
                            sin.view(-1, 1, 1, self.qk_rope_head_dim),
                            attn_metadata.slot_mapping,
                            kv_cache[1],
                            kv_cache[0],
                            epsilon=self.kv_a_layernorm.variance_epsilon,
                            cache_mode="PA_NZ",
                            is_output_kv=True)
                    else:
                        latent_cache = latent_cache.view(-1, latent_cache.size(-1))
                        kv_a, k_pe = torch.split(latent_cache, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                        kv_a = self.kv_a_layernorm(kv_a)
                        k_pe = k_pe.unsqueeze(1)
                        k_pe = k_pe.unsqueeze(2)
                        k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
                        k_pe = k_pe.squeeze(2)
            else:
                if isinstance(kv_cache, Dict):
                    kv_cache = kv_cache.get("kv_cache")
                if kv_cache is not None and isinstance(kv_cache, Tuple) and kv_cache[0].numel() > 0: 
                    _, _, k_pe, kv_a = torch_npu.npu_kv_rmsnorm_rope_cache(
                        latent_cache.view(-1, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim),
                        self.kv_a_layernorm.weight,
                        cos.view(-1, 1, 1, self.qk_rope_head_dim),
                        sin.view(-1, 1, 1, self.qk_rope_head_dim),
                        attn_metadata.slot_mapping,
                        kv_cache[1],
                        kv_cache[0],
                        epsilon=self.kv_a_layernorm.variance_epsilon,
                        cache_mode="PA_NZ",
                        is_output_kv=True)
                else:
                    latent_cache = latent_cache.view(-1, latent_cache.size(-1))
                    kv_a, k_pe = torch.split(latent_cache, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                    kv_a = self.kv_a_layernorm(kv_a)
                    k_pe = k_pe.unsqueeze(1)
                    k_pe = k_pe.unsqueeze(2)
                    k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
                    k_pe = k_pe.squeeze(2)
            
            prefill_metadata = attn_metadata.prefill
            if len(prefill_metadata.seq_qlen_group) == 1:
                # normally execute
                actual_seq_qlen = prefill_metadata.seq_qlen_group[0] if prefill_metadata is not None else [q.shape[0]]
                actual_seq_kvlen = prefill_metadata.seq_kvlen_group[0] if prefill_metadata is not None else [q.shape[0]]

                if enable_prefill_mla_multi_stream:
                    with torch.npu.stream(kv_stream):
                        kv = self.kv_b_proj.forward(kv_a)[0]
                        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
                        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                        k = torch.cat([k_nope, k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_local_heads, 1)], dim=-1)
                else:
                    kv = self.kv_b_proj.forward(kv_a)[0]
                    kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
                    k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                    k = torch.cat([k_nope, k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_local_heads, 1)], dim=-1)

                if enable_prefill_mla_multi_stream:
                    torch.npu.current_stream().wait_stream(kv_stream)
                    kv_stream.wait_stream(torch.npu.current_stream())
                    torch.npu.current_stream().wait_stream(kv_stream2)
                    kv_stream2.wait_stream(torch.npu.current_stream())

                if prefill_metadata.max_query_len > 1:
                    attn_mask = ~torch.tril(
                    torch.ones((2048, 2048), dtype=torch.bool, device="npu")
                ) #(self.attn_prefill.impl.SHARE_MASK_TRIL_SPARSE)
                else:
                    attn_mask = None
        
                if q.shape[0] != actual_seq_qlen[-1]:
                    actual_seq_qlen.append(q.shape[0])
                if k.shape[0] != actual_seq_kvlen[-1]:
                    actual_seq_kvlen.append(k.shape[0])

                attn_output = torch_npu.npu_fused_infer_attention_score(
                    q, k, v,
                    num_heads=self.num_local_heads,
                    input_layout="TND",
                    scale=self.scaling,
                    sparse_mode=3,
                    atten_mask=attn_mask,
                    actual_seq_lengths=actual_seq_qlen,
                    actual_seq_lengths_kv=actual_seq_kvlen)[0].view(-1, self.num_local_heads, self.v_head_dim)

                q, k, v = None, None, None
                kv, k_nope = None, None
            else:
                attn_output = torch.empty(q.shape[0],
                                        self.num_local_heads,
                                        self.v_head_dim,
                                        device=q_nope.device,
                                        dtype=q_nope.dtype)
                computed_tokens = 0
                for iter, (actual_seq_qlen, actual_seq_kvlen) in enumerate(zip(
                        prefill_metadata.seq_qlen_group,
                        prefill_metadata.seq_kvlen_group)
                ):
                    prefill_q = q[computed_tokens:computed_tokens + actual_seq_qlen[-1]]
                    if prefill_metadata.kv_index_list and kv_cache is not None and isinstance(kv_cache, Tuple) and \
                            kv_cache[0].numel() > 0:

                        block_num, block_size, head_size, _ = kv_cache[0].shape
                        kv_cache_a = (kv_cache[0]
                                    .view(block_num, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM))
                        kv_cache_pe = (kv_cache[1]
                                    .view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size,
                                            KVCACHE_NZ_DIM))
                        kv_cache_a = kv_cache_a.transpose(1, 3)
                        kv_cache_pe = kv_cache_pe.transpose(1, 3)

                        kv_a = kv_cache_a.reshape(-1, kv_cache[0].shape[-1]) \
                            .index_select(0, prefill_metadata.kv_index_list[iter]).contiguous()
                        k_pe = kv_cache_pe.reshape(-1, kv_cache[1].shape[-1]) \
                            .index_select(0, prefill_metadata.kv_index_list[iter]).contiguous()
                    prefill_kv_a = kv_a[:actual_seq_kvlen[-1]]
                    prefill_k_pe = k_pe[:actual_seq_kvlen[-1]]

                    kv = self.kv_b_proj.forward(prefill_kv_a)[0]
                    kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
                    k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                    prefill_k = torch.cat(
                        [k_nope, prefill_k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_local_heads, 1)],
                        dim=-1)

                    if prefill_metadata.max_query_len > 1:
                        attn_mask = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device="npu"))
                        #attn_mask = (self.attn_prefill.impl.SHARE_MASK_TRIL_SPARSE)
                    else:
                        attn_mask = None

                    prefill_v = v
                    attn_output[computed_tokens:computed_tokens + actual_seq_qlen[-1]] = \
                        torch_npu.npu_fused_infer_attention_score(
                            prefill_q,
                            prefill_k,
                            prefill_v,
                            num_heads=self.num_local_heads,
                            input_layout="TND",
                            scale=self.scaling,
                            sparse_mode=3,
                            atten_mask=attn_mask,
                            actual_seq_lengths=actual_seq_qlen,
                            actual_seq_lengths_kv=actual_seq_kvlen)[0].view(-1, self.num_local_heads, self.v_head_dim)

                    computed_tokens += actual_seq_qlen[-1]
                    prefill_q, prefill_k, prefill_v = None, None, None
                    kv, k_nope = None, None,
                    q_nope, q_pe = None, None

            if model_extra_config.operator_opt_config.prefill_enable_mla_alltoall_local:
                attn_output = attn_output.reshape(attn_output.shape[0], -1)
                attn_output = attn_output.reshape(self.tp_size // get_npu_device_count(), get_npu_device_count(),
                                                attn_output.shape[0] // self.tp_size, -1) \
                                        .transpose(0, 1).reshape(attn_output.shape[0], -1)
                attn_output = all_to_all_local(attn_output, idx=0)
                output, _ = self.o_proj.forward(attn_output)
                output = reduce_scatter_cross(output, idx=0)
            else:
                attn_output = attn_output.view(-1, self.num_local_heads * self.v_head_dim)
                output = self.o_proj.forward(attn_output)[0]
        else:
            attn_output = torch.zeros(q.shape[0],
                                      self.num_local_heads,
                                      self.v_head_dim,
                                      device=q_nope.device,
                                      dtype=q_nope.dtype)
            if model_extra_config.operator_opt_config.prefill_enable_mla_alltoall_local:
                attn_output = attn_output.reshape(attn_output.shape[0], -1)
                attn_output = attn_output.reshape(self.tp_size // get_npu_device_count(), get_npu_device_count(),
                                                attn_output.shape[0] // self.tp_size, -1) \
                                        .transpose(0, 1).reshape(attn_output.shape[0], -1)
                attn_output = all_to_all_local(attn_output, idx=0)
                output, _ = self.o_proj.forward(attn_output)
                output = reduce_scatter_cross(output, idx=0)
            else:
                attn_output = attn_output.view(-1, self.num_local_heads * self.v_head_dim)
                output = self.o_proj.forward(attn_output)[0]

        attn_output = None
        return output

    def forward_normal(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            if self.merge_qkv:
                qkv = self.qkv_a_proj(hidden_states)[0]
                qkv = tensor_model_parallel_all_gather(qkv, dim=0)
                q, latent_cache = torch.split(qkv, [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
            else:
                q = self.q_a_proj(hidden_states)[0]
                latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
                q = tensor_model_parallel_all_gather(q, dim=0)
                latent_cache = tensor_model_parallel_all_gather(latent_cache, dim=0)
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = tensor_model_parallel_all_gather(q, dim=0)
            latent_cache = tensor_model_parallel_all_gather(latent_cache, dim=0)

        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q = None
        q_pe = q_pe.unsqueeze(2)
        cos, sin = self.rotary_emb.get_cos_sin(positions)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin)
        q_pe = q_pe.squeeze(2)

        if isinstance(kv_cache, Dict):
            kv_cache = kv_cache.get("kv_cache")
        if kv_cache is not None and isinstance(kv_cache, Tuple) and kv_cache[0].numel() > 0:
            _, _, k_pe, kv_a = torch_npu.npu_kv_rmsnorm_rope_cache(
                latent_cache.view(-1, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim),
                self.kv_a_layernorm.weight,
                cos.view(-1, 1, 1, self.qk_rope_head_dim),
                sin.view(-1, 1, 1, self.qk_rope_head_dim),
                attn_metadata.slot_mapping,
                kv_cache[1],
                kv_cache[0],
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA_NZ",
                is_output_kv=True)
        else:
            latent_cache = latent_cache.view(-1, latent_cache.size(-1))
            kv_a, _ = torch.split(latent_cache, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            latent_cache = latent_cache.unsqueeze(1)
            kv_a = self.kv_a_layernorm(kv_a)
            k_pe = latent_cache[:, :, self.kv_lora_rank:]
            k_pe = k_pe.unsqueeze(2)
            k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
            k_pe = k_pe.squeeze(2)

        prefill_metadata = attn_metadata.prefill if attn_metadata is not None else None
        attn_output = torch.empty(q_nope.shape[0],
                                  self.num_local_heads,
                                  self.v_head_dim,
                                  device=q_nope.device,
                                  dtype=q_nope.dtype)
        
        if prefill_metadata is not None:
            computed_tokens = 0
            for iter, (actual_seq_qlen, actual_seq_kvlen) in enumerate(zip(
                    prefill_metadata.seq_qlen_group,
                    prefill_metadata.seq_kvlen_group)
            ):
                prefill_q = torch.concat(
                    (q_nope[computed_tokens:computed_tokens + actual_seq_qlen[-1]],
                    q_pe[computed_tokens:computed_tokens + actual_seq_qlen[-1]]), dim=-1)
                if prefill_metadata.kv_index_list and kv_cache is not None and isinstance(kv_cache, Tuple) and \
                        kv_cache[0].numel() > 0:
                    block_num, block_size, head_size, _ = kv_cache[0].shape
                    kv_cache_a = (kv_cache[0]
                                .view(block_num, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM))
                    kv_cache_pe = (kv_cache[1]
                                .view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM))
                    kv_cache_a = kv_cache_a.transpose(1, 3)
                    kv_cache_pe = kv_cache_pe.transpose(1, 3)
                    kv_a = kv_cache_a.reshape(-1, kv_cache[0].shape[-1]) \
                        .index_select(0, prefill_metadata.kv_index_list[iter]).contiguous()
                    k_pe = kv_cache_pe.reshape(-1, kv_cache[1].shape[-1]) \
                        .index_select(0, prefill_metadata.kv_index_list[iter]).contiguous()
                prefill_kv_a = kv_a[:actual_seq_kvlen[-1]]
                prefill_k_pe = k_pe[:actual_seq_kvlen[-1]]
                is_prefill = (attn_metadata is None or attn_metadata.prefill is not None)
                if not is_prefill:
                    self.kv_b_proj.weight = torch.nn.Parameter(torch.cat((self.W_UK.permute(2,0,1), self.W_UV.transpose(0,1)), dim=-1) \
                                                                    .view(self.kv_lora_rank,-1).T, requires_grad=False)
                    kv = self.kv_b_proj.forward(prefill_kv_a)[0]
                    self.kv_b_proj.weight = None
                else:
                    kv = self.kv_b_proj.forward(prefill_kv_a)[0]
                kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
                k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                kv = None
                prefill_k = torch.cat(
                    [k_nope, prefill_k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_local_heads, 1)], dim=-1)
                k_nope, prefill_k_pe = None, None

                if prefill_metadata.max_query_len > 1:
                    attn_mask = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device="npu"))
                    #attn_mask = (self.attn_prefill.impl.SHARE_MASK_TRIL_SPARSE)
                else:
                    attn_mask = None
                v = torch.nn.functional.pad(v, [0, self.qk_rope_head_dim], value=0)
                attn_output[computed_tokens:computed_tokens + actual_seq_qlen[-1]] = \
                    torch_npu.npu_fusion_attention(
                        prefill_q,
                        prefill_k,
                        v,
                        head_num=self.num_local_heads,
                        input_layout="TND",
                        scale=self.scaling,
                        sparse_mode=3,
                        atten_mask=attn_mask,
                        actual_seq_qlen=actual_seq_qlen,
                        actual_seq_kvlen=actual_seq_kvlen)[0][..., :self.v_head_dim]
                computed_tokens += actual_seq_qlen[-1]
                prefill_q, prefill_k, v = None, None, None

            q_nope, q_pe = None, None
        else:
            attn_output.fill_(0)

        attn_output = attn_output.view(-1, self.num_local_heads * self.v_head_dim)
        output = self.o_proj.forward(attn_output)[0]

        attn_output = None
        return output

    def forward_absorb_kv_rmsnorm_rope_cache(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)

        key_cache, value_cache = kv_cache
        output_dtype = key_cache.dtype

        if self.q_lora_rank is not None:
            q_lowrank = self.q_a_proj(hidden_states)[0]
        else:
            q_lowrank = self.q_proj(hidden_states)[0]

        with tng.scope.npu_stream_switch('11'):
            kv = hidden_states
            kv = self.kv_a_proj_with_mqa(kv)[0]

        tng.scope.npu_wait_tensor(q_lowrank, q_lowrank)


        if self.q_lora_rank is not None:
            q = self.q_a_layernorm(q_lowrank)
            q = self.q_b_proj(q)[0]
        else:
            q = q_lowrank
        bsz, _ = q.shape
        q_len = 1
        q = q.view(bsz, self.num_local_heads, 1, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim).transpose(0, 1)
        q_nope = (
            torch.matmul(q_nope, self.W_UK)
            .transpose(1, 0)
            .view(bsz, q_len, self.num_local_heads, -1)
        )
        with tng.scope.npu_stream_switch('11'):
            kv = kv.unsqueeze(1).unsqueeze(1)
            cos, sin = attn_metadata.decode.cos, attn_metadata.decode.sin
            tmp_slot_mapping = attn_metadata.slot_mapping
            block_num, block_size, head_size, _ = key_cache.shape
            k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                kv, self.kv_a_layernorm.weight,
                cos, sin, tmp_slot_mapping,
                value_cache, key_cache,
                epsilon=self.kv_a_layernorm.variance_epsilon, cache_mode="PA_NZ")

            k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
            k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)

            tng.scope.npu_wait_tensor(q_pe, k_nope)

            q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin)
            q_nope = q_nope.view(bsz, 1, self.num_local_heads, self.kv_lora_rank)
            q_pe = q_pe.view(bsz, 1, self.num_local_heads, -1)

        bsz, q_len, _, q_dim = q_nope.size()
        
        attn_output, _ = tng.ops.npu_fused_infer_attention_score(
            q_nope, k_nope, k_nope, query_rope=q_pe, key_rope=k_rope,
            num_heads=self.num_local_heads,
            num_key_value_heads=1, input_layout="BSND",
            scale=self.scaling,
            antiquant_mode=0, antiquant_scale=None,
            block_table=attn_metadata.decode.block_table,
            block_size=128,
            actual_seq_lengths_kv=attn_metadata.decode.seq_lens,
        )
        attn_output = attn_output.squeeze(1).transpose(0, 1)
        attn_output = (
            torch.matmul(attn_output, self.W_UV)
            .transpose(1, 0)
            .reshape(bsz, q_len, -1)
        )
        attn_output = attn_output.view(
            -1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj.forward(attn_output)
        return output

    def forward_absorb(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
        key_cache, value_cache = kv_cache
        output_dtype = key_cache.dtype

        if self.q_lora_rank is not None:
            q_lowrank = self.q_a_proj(hidden_states)[0]
        else:
            q_lowrank = self.q_proj(hidden_states)[0]

        kv = hidden_states
        kv = self.kv_a_proj_with_mqa(kv)[0]

        if self.q_lora_rank is not None:
            q = self.q_a_layernorm(q_lowrank)
            q = self.q_b_proj(q)[0]
        else:
            q = q_lowrank
        bsz, _ = q.shape
        q_len = 1
        q = q.view(bsz, self.num_local_heads, 1, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim).transpose(0, 1)
        q_nope = (
            torch.matmul(q_nope, self.W_UK)
            .transpose(1, 0)
            .view(bsz, q_len, self.num_local_heads, -1)
        )

        kv = kv.unsqueeze(1).unsqueeze(1)
        cos, sin = attn_metadata.decode.cos, attn_metadata.decode.sin
        tmp_slot_mapping = attn_metadata.slot_mapping
        block_num, block_size, head_size, _ = key_cache.shape
        k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
            kv, self.kv_a_layernorm.weight,
            cos, sin, tmp_slot_mapping,
            value_cache, key_cache,
            epsilon=self.kv_a_layernorm.variance_epsilon, cache_mode="PA_NZ")

        k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
        k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)

        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin)
        q_nope = q_nope.view(bsz, 1, self.num_local_heads, self.kv_lora_rank)
        q_pe = q_pe.view(bsz, 1, self.num_local_heads, -1)

        bsz, q_len, _, q_dim = q_nope.size()
        attn_output, _ = tng.ops.npu_fused_infer_attention_score(
            q_nope, k_nope, k_nope, query_rope=q_pe, key_rope=k_rope,
            num_heads=self.num_local_heads,
            num_key_value_heads=1, input_layout="BSND",
            scale=self.scaling,
            antiquant_mode=0, antiquant_scale=None,
            block_table=attn_metadata.block_table,
            block_size=128,
            actual_seq_lengths_kv=attn_metadata.seq_lens,
        )

        attn_output = attn_output.squeeze(1).transpose(0, 1)
        attn_output = (
            torch.matmul(attn_output, self.W_UV)
            .transpose(1, 0)
            .reshape(bsz, q_len, -1)
        )
        attn_output = attn_output.view(
            -1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj.forward(attn_output)
        return output


class DeepseekDecoderLayer(nn.Module):

    def __init__(
            self,
            config: PretrainedConfig,
            prefix: str,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.layer_name = f"{prefix}.self_attn.attn"
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)

        layer_idx = int(prefix.split(sep='.')[-1])
        self.self_attn = AscendDeepseekAttention_MLA(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank
            if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0):
            self.mlp = DeepseekMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe = True
        else:
            dense_tp_parallel = "no_tp"
            if model_extra_config.operator_opt_config.enable_dense_local_tp == 1:
                dense_tp_parallel = "local"
            elif model_extra_config.operator_opt_config.enable_dense_local_tp == 0:
                dense_tp_parallel = "global"

            self.mlp = ParallelDeepseekMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                tp_parallel=dense_tp_parallel,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe = False
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

        self.enable_dp_attention = get_dp_group().world_size > 1
        if self.enable_dp_attention:
            self.dp_rank = get_dp_group().rank_in_group
            self.dp_size = get_dp_group().world_size
            self.dp_group = get_dp_group().device_group
        else:
            self.dp_rank = None
            self.dp_size = None
            self.dp_group = None

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            residual: Optional[torch.Tensor],
            layer_id: Optional[int] = None,
            rotary_emb_cos: Optional[torch.Tensor] = None,
            rotary_emb_sin: Optional[torch.Tensor] = None,
            kv_prefetch: torch.Tensor = None
    ) -> torch.Tensor:
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.layer_name]
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # hidden_states, residual = self.input_layernorm(
            #     hidden_states, residual, quant_symbol=True)
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual, quant_symbol=True)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            rotary_emb_cos=rotary_emb_cos,
            rotary_emb_sin=rotary_emb_sin
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        is_prefill = (attn_metadata is None or attn_metadata.prefill is not None)
        if (model_extra_config.operator_opt_config.prefill_moe_all_to_all or model_extra_config.parall_config.dp_size > 1) and is_prefill:
            reduce_length = torch.tensor(hidden_states.shape[0], dtype=torch.int64, device="npu")
            local_length = hidden_states.shape[0]
            dist.all_reduce(reduce_length, op=dist.ReduceOp.MAX, async_op=False)
            global_max_length = reduce_length.item()
            pad_size = global_max_length - hidden_states.shape[0]

            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, pad_size)
            )
            residual = torch.nn.functional.pad(
                residual, (0, 0, 0, pad_size)
            )
            hidden_states_list = hidden_states.split(SEQ_SPLIT_LENGTH_BEFORE_ALL_GATHER)
            residual_list = residual.split(SEQ_SPLIT_LENGTH_BEFORE_ALL_GATHER)
            hidden_state_out = []
            residual_out = []
            for i in range(len(hidden_states_list)):
                if self.is_moe == True:
                    hidden_states, residual = self.mlp(hidden_states_list[i], residual_list[i], attn_metadata, layer_id)
                else:
                    hidden_states, residual = self.mlp(hidden_states_list[i], residual_list[i], attn_metadata)
                hidden_state_out.append(hidden_states)
                residual_out.append(residual)
            hidden_states = torch.cat(hidden_state_out)[:local_length]
            residual = torch.cat(residual_out)[:local_length]
        else:
            if self.is_moe == True:
                hidden_states, residual = self.mlp(hidden_states, residual, attn_metadata, layer_id, kv_prefetch)
            else:
                hidden_states, residual = self.mlp(hidden_states, residual, attn_metadata)

        return hidden_states, residual

    CACHED_GATHERED_BUFFER = None

    def get_cached_gathered_buffer(self, token_nums, dtype):
        if DeepseekDecoderLayer.CACHED_GATHERED_BUFFER is None \
                or DeepseekDecoderLayer.CACHED_GATHERED_BUFFER.shape[0] != token_nums \
                or DeepseekDecoderLayer.CACHED_GATHERED_BUFFER.dtype != dtype:
            DeepseekDecoderLayer.CACHED_GATHERED_BUFFER = torch.zeros((token_nums, self.hidden_size), dtype=dtype,
                                                                      device='npu')
        return DeepseekDecoderLayer.CACHED_GATHERED_BUFFER


class DeepseekV3Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        qk_rope_head_dim = config.qk_rope_head_dim
        rope_scaling["rope_type"] = 'deepseek_yarn'
        self.rotary_emb = get_rope(qk_rope_head_dim,
                                   rotary_dim=qk_rope_head_dim,
                                   max_position=max_position_embeddings,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling,
                                   is_neox_style=False)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                # parallel_lmhead=(model_extra_config.parall_config.dp_size > 1),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekDecoderLayer(
                config,
                prefix,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers")

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        self.enable_dp_attention = get_dp_group().world_size > 1
        if self.enable_dp_attention:
            self.dp_rank = get_dp_group().rank_in_group
            self.dp_size = get_dp_group().world_size
            self.dp_group = get_dp_group().device_group

        self.is_init = False
        self.first_k_dense_replace = config.first_k_dense_replace
        self.num_hidden_layers = config.num_hidden_layers

    CACHED_LOCAL_NUM_TOKENS = None
    CACHED_GLOBAL_NUM_TOKENS = None

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids, reduce=1)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors],
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        if model_extra_config.operator_opt_config.enable_prefetch and not self.is_init:
            prefetch_start_layer = self.start_layer if self.start_layer > self.first_k_dense_replace else self.first_k_dense_replace
            prefetch_end_layer = self.end_layer if self.end_layer < self.num_hidden_layers - 1 else self.num_hidden_layers - 1
            for layer_id in range(prefetch_start_layer, prefetch_end_layer):
                self.layers[layer_id].mlp.attn_prefetch = self.layers[layer_id + 1].self_attn
            self.is_init = True

        rotary_emb_cos, rotary_emb_sin = self.rotary_emb.get_cos_sin(positions)

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            layer_id = i - self.first_k_dense_replace
            if model_extra_config.operator_opt_config.enable_prefetch and i < self.end_layer - 1 and kv_caches is not None:
                kv_prefetch = kv_caches[i + 1 - self.start_layer]
            else:
                kv_prefetch = None
            hidden_states, residual = layer(positions, hidden_states,
                                            kv_caches[i - self.start_layer] if kv_caches is not None else None,
                                            attn_metadata, residual, layer_id,
                                            rotary_emb_cos, rotary_emb_sin,
                                            kv_prefetch)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)

        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
        
        return hidden_states


@support_torch_compile
class DeepseekV3ForCausalLM(nn.Module):
    
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DeepseekV3Model(vllm_config=vllm_config, prefix="model")
        self.lm_head = ParallelLMHead(self.config.vocab_size,
                                      self.config.hidden_size,
                                      quant_config=self.quant_config,
                                      parallel_lmhead=(model_extra_config.parall_config.dp_size > 1))
        self.logits_processor = LogitsProcessor(self.config.vocab_size,
                                                logits_as_input=True)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        self.return_hidden_states = True

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor] = None,
            attn_metadata: AttentionMetadata = None,
            selected_indices: Optional[torch.Tensor] = None,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds = None,
            **kwargs
    ) -> Optional[torch.Tensor]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors)
        
        if attn_metadata is None:
            logits = self.compute_lmhead(hidden_states[-1:, ...], None)
        else:
            logits = self.compute_lmhead(hidden_states, selected_indices)

        if self.return_hidden_states:
            return hidden_states, logits
        else:
            return logits

    def compute_lmhead(
            self,
            hidden_states: torch.Tensor,
            selected_indices: Optional[torch.Tensor] = None,
            embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if selected_indices is not None:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            if hidden_states.shape[0] != selected_indices.shape[0]:
                hidden_states = hidden_states.index_select(0, selected_indices)

        logits = self.lm_head(hidden_states, embedding_bias)

        return logits

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        # actual_batch = sampling_metadata.selected_token_indices.shape[0]
        # if logits is not None and logits.shape[0] != actual_batch:
        #     logits = logits[:actual_batch, :]
        return logits

    def sample(
            self,
            logits: Optional[torch.Tensor],
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
                torch.zeros((batch_size, self.config.hidden_size),
                            dtype=dtype,
                            device=device),
            "residual":
                torch.zeros((batch_size, self.config.hidden_size),
                            dtype=dtype,
                            device=device),
        })

    def load_weights(self, weights: Iterable[Tuple[str,
    torch.Tensor]]) -> Set[str]:
        if model_extra_config.operator_opt_config.merge_qkv:
            stacked_params_mapping = [
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
                ("qkv_a_proj", "q_a_proj", 0),
                ("qkv_a_proj", "kv_a_proj_with_mqa", 1),
            ]
        else:
            stacked_params_mapping = [
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if self.config.architectures[0] == 'DeepseekV3ForCausalLM' and self.config.num_nextn_predict_layers > 0:
                layer_idx = self.config.num_hidden_layers
                if name.startswith(f"model.layers.{layer_idx}"):
                    continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue
                    
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue
                    
                    if name not in params_dict:
                        continue
                
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def should_use_eager_mode(self, *args, **kwargs):
        attn_metadata = kwargs.get('attn_metadata', None)
        
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.model.layers[self.model.start_layer].layer_name]

        if attn_metadata is None:
            return True

        if attn_metadata.prefill:
            return True

        return False
