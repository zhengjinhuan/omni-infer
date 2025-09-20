# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from enum import IntEnum, auto
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch_npu
from sglang.srt.distributed import (get_moe_expert_parallel_world_size,
                                    get_tensor_model_parallel_world_size,
                                    parallel_state)
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.linear import (MergedColumnParallelLinear,
                                      RowParallelLinear)
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import add_prefix
from torch import nn
from transformers import PretrainedConfig

from omni.adaptors.sglang.layers.moe.ep_moe.layer import NpuDeepEPMoE
from omni.adaptors.sglang.layers.moe.token_dispatcher.deepep import \
    NpuDeepEPDispatcher


class DeepseekMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        forward_batch=None,
        use_reduce_scatter: bool = False,
        **kwargs,
    ):
        if (self.tp_size == 1) and x.shape[0] == 0:
            return x

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x, skip_all_reduce=use_reduce_scatter
        )
        return x


class ReplicatedDeepseekMLP(DeepseekMLP):
    pass


class MoEGate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )

        self.e_score_correction_bias = None
        if config.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((config.n_routed_experts), dtype=torch.float32)
            )

    def forward(self, hidden_states):
        import torch.nn.functional as F
        return F.linear(hidden_states, self.weight, None)


class DeepseekMoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__()

        self.config = config
        self.layer_id = layer_id
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_size = get_moe_expert_parallel_world_size()

        self.num_fused_shared_experts = 0
        if not global_server_args_dict["disable_shared_experts_fusion"]:
            self.num_fused_shared_experts = config.n_shared_experts

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = MoEGate(config=config)

        self.topk = TopK(
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            num_fused_shared_experts=self.num_fused_shared_experts,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=config.routed_scaling_factor,
        )

        deepep_mode = global_server_args_dict["deepep_mode"]
        ep_num_redundant_experts = global_server_args_dict["ep_num_redundant_experts"]

        self.experts = NpuDeepEPMoE(
            num_experts=config.n_routed_experts + self.num_fused_shared_experts + ep_num_redundant_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_id,
            quant_config=quant_config,
            routed_scaling_factor=config.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
            deepep_mode=deepep_mode, # moe_a2a_backend
        )

        self.shared_experts = None

        if (config.n_shared_experts is not None and self.num_fused_shared_experts == 0):

            self.shared_experts = ReplicatedDeepseekMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                tp_size=1, # disable tp for shared experts when enable deepep moe
                tp_rank=0,
            )

            # shared_experts_is_fp8
            if self.shared_experts.gate_up_proj.weight.dtype == torch.float8_e4m3fn:
                if (hasattr(self.shared_experts.gate_up_proj.quant_method, "quant_config")
                    and self.shared_experts.gate_up_proj.quant_method.quant_config.get_name() == "moe_wna16"):
                    assert (
                        self.shared_experts.gate_up_proj.quant_method.quant_config.weight_block_size
                        == self.shared_experts.down_proj.quant_method.quant_config.weight_block_size
                    )

        self.top_k = config.num_experts_per_tok
        self.deepep_dispatcher = None

        if global_server_args_dict["moe_a2a_backend"].is_deepep():
            # TODO: we will support tp < ep in the future

            self.deepep_dispatcher = NpuDeepEPDispatcher(
                group=parallel_state.get_tp_group().device_group,
                router_topk=self.top_k,
                permute_fusion=True,
                num_experts=config.n_routed_experts + ep_num_redundant_experts,
                num_local_experts=config.n_routed_experts // self.tp_size,
                hidden_size=config.hidden_size,
                params_dtype=config.torch_dtype,
                deepep_mode=deepep_mode,
                async_finish=True,
                return_recv_hook=True,
                n_shared_experts=config.n_shared_experts,
                n_routed_experts=config.n_routed_experts,
                num_experts_per_tok=config.num_experts_per_tok,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        shared_output = None

        if hidden_states.shape[0] > 0 and not forward_batch.is_prefill_idle:
            router_logits = self.gate(hidden_states) # router_logits: (num_tokens, n_experts)

            if self.shared_experts is not None:
                shared_output = self.shared_experts(hidden_states)

            topk_weights, topk_idx, _ = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_idx = torch.randperm(256)[:hidden_states.size(0) * self.top_k].reshape(
                hidden_states.size(0), self.top_k
                ).npu()

            topk_weights = torch.empty(
                (hidden_states.size(0), self.top_k),
                dtype=torch.float32,
                device=hidden_states.device,
            )

        expert_tokens = None
        dynamic_scale = None

        if self.deepep_dispatcher is not None:
            topk_ids = topk_idx

            (
                hidden_states,
                dynamic_scale,
                topk_idx,
                expert_tokens,
                ep_recv_counts,
                tp_recv_counts,
                expanded_x,
                expanded_row_idx,
            ) = self.deepep_dispatcher.dispatch(
                hidden_states=hidden_states,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                forward_batch=forward_batch,
            )

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            expert_tokens=expert_tokens,
            dynamic_scale=dynamic_scale,
            can_run_graph=forward_batch.can_run_graph,
        )

        if self.deepep_dispatcher is not None:
            final_hidden_states = self.deepep_dispatcher.combine(
                hidden_states=final_hidden_states,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                forward_batch=forward_batch,
                topk_ids=topk_ids,
                ep_send_counts=ep_recv_counts,
                tp_send_counts=tp_recv_counts,
                shared_output=shared_output,
                expanded_x=expanded_x,
                expanded_row_idx=expanded_row_idx,
            )

        return final_hidden_states

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]
