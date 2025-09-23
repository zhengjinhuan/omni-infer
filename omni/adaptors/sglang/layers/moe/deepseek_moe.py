# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from enum import IntEnum, auto
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
import torch.distributed as dist
import torchair as tng
import torch_npu
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_world_group,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.utils import add_prefix

from omni.adaptors.sglang.layers.moe.ep_moe.layer import FusedMoE


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
        
        from sglang.srt.layers.activation import SiluAndMul
        self.act_fn = SiluAndMul()

    def forward(
            self,
            x,
            forward_batch=None,
            use_reduce_scatter: bool = False,
            **kwargs):
        if (self.tp_size == 1) and x.shape[0] == 0:
            return x

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x, skip_all_reduce=use_reduce_scatter)

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
        self.world_size = get_world_group().world_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_size = get_moe_expert_parallel_world_size()
        self.top_k = config.num_experts_per_tok

        self.quant_symbol = True if quant_config else False

        self.num_fused_shared_experts = 0
        if not global_server_args_dict["disable_shared_experts_fusion"]:
            self.num_fused_shared_experts = config.n_shared_experts

        assert global_server_args_dict["moe_a2a_backend"].is_deepep()

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

        # ========================== init gate ==============================

        self.gate = MoEGate(config=config)

        # ====================== init FusedMoE ======================

        deepep_mode = global_server_args_dict["deepep_mode"]
        ep_num_redundant_experts = global_server_args_dict["ep_num_redundant_experts"]

        self.experts = FusedMoE(
            num_experts=config.n_routed_experts + self.num_fused_shared_experts + ep_num_redundant_experts,
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            num_fused_shared_experts=self.num_fused_shared_experts,
            layer_id=self.layer_id,
            quant_config=quant_config,
            routed_scaling_factor=config.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
            deepep_mode=deepep_mode, # moe_a2a_backend
        )

        # ====================== init for self._forward_decode ======================
        
        if self.quant_symbol:
            self.local_expert_num = self.experts.w13_weight.shape[0]
            self.experts.weight_num_bits = 8 # W8A8 as default
            self.in_scale_2 = torch.ones(
                (self.local_expert_num, config.moe_intermediate_size),
                dtype=torch.float32,
                device="npu")
            torch._dynamo.mark_static(self.in_scale_2) # call the mark_static to reduce memory usage

        self.tuning_config = None # not using gmm_nz

        # ====================== init shared_experts ======================
        
        self.shared_experts = None
        if (config.n_shared_experts is not None
            and self.num_fused_shared_experts == 0):

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

        # ====================== misc ======================
        
        self.num_experts = config.n_routed_experts + ep_num_redundant_experts
        
        self.group = get_world_group().device_group
        self.global_rank = get_world_group().rank_in_group
        self.group_name = self.group._get_backend(torch.device("npu")).get_hccl_comm_name(self.global_rank)

        moe_rs_group = get_pp_group().device_group
        moe_rs_group_rank = get_pp_group().rank_in_group
        self.moe_rs_group_name = moe_rs_group._get_backend(torch.device("npu")).get_hccl_comm_name(moe_rs_group_rank)

        self.experts_tp_size = 1
        self.shared_expert_rank_num = 0 # route_share_on_same_card

        epsilon = 1e-2
        self.smooth_scale = torch.nn.Parameter(
            torch.ones(size=(self.num_experts, config.hidden_size), dtype=torch.float32) * (1 - epsilon) + epsilon)  # smooth scale, now dpsk use smooth_scale == 1

    def forward(
            self,
            hidden_states: torch.Tensor,
            forward_batch: Optional[ForwardBatch] = None,
            use_reduce_scatter: bool = False,
            **kwargs) -> torch.Tensor:

        if forward_batch.is_extend_in_batch:
            return self._forward_prefill(hidden_states, forward_batch)
        else:
            return self._forward_decode(hidden_states, forward_batch)

    def _forward_common(self, hidden_states, forward_batch):

        shared_output = None

        if hidden_states.shape[0] > 0 and not forward_batch.is_prefill_idle:

            # ========================== gate ==============================

            router_logits = self.gate(hidden_states) # router_logits: (num_tokens, n_experts)

            # ========================== shared_experts ==============================

            if self.shared_experts is not None:
                shared_output = self.shared_experts(hidden_states)

            # ========================== select_experts ==============================

            topk_weights, topk_idx, _ = FusedMoE.select_experts(
                router_logits=router_logits,
                top_k=self.experts.top_k,
                use_grouped_topk=True,
                renormalize=self.config.norm_topk_prob,
                topk_group=self.config.topk_group,
                num_expert_group=self.config.n_group,
                e_score_correction_bias=self.gate.e_score_correction_bias,
                routed_scaling_factor=self.config.routed_scaling_factor)

        else:
            topk_idx = torch.randperm(256)[:hidden_states.size(0) * self.top_k].reshape(
                hidden_states.size(0), self.top_k
                ).npu()

            topk_weights = torch.empty(
                (hidden_states.size(0), self.top_k),
                dtype=torch.float32,
                device="npu")

        return (hidden_states, topk_idx, forward_batch, topk_weights, shared_output)

    def _forward_prefill(self, hidden_states, forward_batch) -> torch.Tensor:

        # ====================== common ======================

        (
            hidden_states,
            topk_idx,
            forward_batch,
            topk_weights,
            shared_output
        ) = self._forward_common(hidden_states, forward_batch)

        # ======================FusedMoE.forward======================

        final_hidden_states_list = self.experts(
            hidden_states=hidden_states,
            topk_ids=topk_idx,
            forward_batch=forward_batch,
            comm_group=self.group,
            dynamic_scale=self.smooth_scale,
        )
        if len(final_hidden_states_list) != 3:
            raise RuntimeError("len(final_hidden_states_list) != 3")
        hidden_states = final_hidden_states_list[0]
        gathered_tokens = final_hidden_states_list[1]
        expanded_row_idx = final_hidden_states_list[2]

        # finalize-routing
        hidden_states = torch_npu.npu_moe_finalize_routing(
            gathered_tokens,
            skip1=shared_output,
            skip2=None,
            bias=None,
            scales=topk_weights.to(gathered_tokens.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None,
            drop_pad_mode=2,
        )

        return hidden_states

    def _forward_decode(self, hidden_states, forward_batch) -> torch.Tensor:
        
        act_dtype = hidden_states.dtype

        # ====================== common ======================

        (
            hidden_states,
            topk_idx,
            forward_batch,
            topk_weights,
            shared_output
        ) = self._forward_common(hidden_states, forward_batch)

        # ====================== dispatch ======================

        expert_quant_mode = 2 # 0: non-quant; 1: static quant(not supported now); 2: dynamic quant
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        topk_ids = topk_idx.to(torch.int)

        (
            hidden_states,
            dynamic_scale,
            topk_idx,
            expert_tokens,
            ep_recv_counts,
            tp_recv_counts,
        ) = torch_npu.npu_moe_distribute_dispatch_v2(
            x=hidden_states,
            expert_ids=topk_ids,
            expert_shard_type=0,
            shared_expert_rank_num=self.shared_expert_rank_num,
            moe_expert_num=self.num_experts,
            global_bs=0,
            scales=self.smooth_scale,
            quant_mode=expert_quant_mode,
            group_ep=self.group_name,
            ep_world_size=self.world_size,
            ep_rank_id=self.global_rank,
            group_tp=self.moe_rs_group_name,
            tp_world_size=self.experts_tp_size
            )[:6]

        group_list = expert_tokens.to(torch.int64)

        # ====================== FusedMoE.forward ======================

        expand_x = hidden_states
        weight1_3 = self.experts.w13_weight
        weight2 = self.experts.w2_weight

        if self.quant_symbol:

            if self.experts.weight_num_bits == 8:
                weight_scale1_3 = self.experts.w13_weight_scale.squeeze(-1) # adapt shape
                weight_scale2 = self.experts.w2_weight_scale.squeeze(-1).to(torch.bfloat16) # adapt shape and dtype
            elif self.experts.weight_num_bits == 4:
                weight_scale1_3 = self.experts.w13_weight_int4_scale
                weight_scale2 = self.experts.w2_weight_int4_scale
                weight_bias1_3 = self.experts.w13_weight_bias
                weight_bias2 = self.experts.w2_weight_bias
            else:
                raise NotImplementedError(f"Unsupported compress tensor type. num bits: {self.experts.weight_num_bits}")

            if expert_quant_mode != 0:
                pertoken_scale = dynamic_scale
            else:
                expand_x, pertoken_scale = torch_npu.npu_dynamic_quant(expand_x)

        if self.quant_symbol:
            if self.experts.weight_num_bits == 8: # w8a8

                gate_up_proj = torch_npu.npu_grouped_matmul(
                    [expand_x], 
                    [weight1_3], 
                    bias=None,
                    group_list=group_list,
                    split_item=3,
                    output_dtype=torch.int32, 
                    group_type=0,
                    group_list_type=1)[0]

                gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                    gate_up_proj, 
                    weight_scale=weight_scale1_3, 
                    activation_scale=pertoken_scale, 
                    bias=None, 
                    quant_scale=self.in_scale_2, 
                    quant_offset=None,
                    group_index=group_list, 
                    activate_left=True, 
                    quant_mode=1) # 1: dynamic quant; 0: static quant(not supported now)

                hidden_states = torch_npu.npu_grouped_matmul(
                    [gate_up_proj], 
                    [weight2], 
                    scale=[weight_scale2],
                    per_token_scale=[pertoken_scale],
                    bias=None,
                    group_list=group_list,
                    split_item=3, 
                    output_dtype=act_dtype,
                    group_type=0,
                    group_list_type=1)[0]

            elif self.experts.weight_num_bits == 4:
                gate_up_proj = torch_npu.npu_grouped_matmul(
                    [expand_x],
                    [weight1_3],
                    bias=[weight_bias1_3],
                    scale=[weight_scale1_3],
                    offset=None, 
                    antiquant_scale=None,
                    antiquant_offset=None,
                    per_token_scale=[pertoken_scale],
                    group_list=group_list,
                    activation_input=None, 
                    activation_quant_scale=None,
                    activation_quant_offset=None,
                    split_item=3, 
                    group_type=0,
                    group_list_type=1, 
                    act_type=0,
                    tuning_config=self.tuning_config,
                    output_dtype=torch.bfloat16)[0]

                fake_scale = torch.ones(weight_bias1_3.shape, dtype=torch.float32, device="npu").view(-1,weight_bias1_3.shape[1])
                pertoken_scale = torch.ones(pertoken_scale.shape, dtype=torch.float32, device="npu")

                gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                    gate_up_proj,
                    weight_scale=fake_scale,
                    activation_scale=pertoken_scale,
                    bias=None,
                    quant_scale=None,
                    quant_offset=None,
                    group_index=group_list,
                    activate_left=True,
                    quant_mode=1) # 1: dynamic quant; 0: static quant(not supported now)

                hidden_states = torch_npu.npu_grouped_matmul(
                    [gate_up_proj],
                    [weight2],
                    scale=[weight_scale2],
                    per_token_scale=[pertoken_scale],
                    bias=[weight_bias2],
                    group_list=group_list,
                    split_item=3,
                    output_dtype=act_dtype,
                    group_type=0,
                    group_list_type=1,
                    tuning_config=self.tuning_config)[0]
            else:
                raise NotImplementedError(f"Unsupported compress tensor type. num bits: {self.experts.weight_num_bits}")
        else: # bf16
            gate_up_proj = torch_npu.npu_grouped_matmul(
                [expand_x], 
                [weight1_3], 
                bias=None, 
                group_list=group_list,
                split_item=3, 
                group_type=0,
                group_list_type=1)[0]
        
            gate_up_proj = torch_npu.npu_swiglu(gate_up_proj)

            hidden_states = torch_npu.npu_grouped_matmul(
                [gate_up_proj], 
                [weight2],
                bias=None,
                group_list=group_list,
                split_item=3,
                output_dtype=act_dtype,
                group_type=0,
                group_list_type=1)[0]

        # ====================== combine ======================

        hidden_states = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=hidden_states,
            expert_ids=topk_ids,
            assist_info_for_combine=topk_idx,
            expert_scales=topk_weights.to(torch.float32),
            expert_shard_type=0,
            shared_expert_x=shared_output,
            shared_expert_rank_num=self.shared_expert_rank_num,
            moe_expert_num=self.num_experts,
            global_bs=0,
            ep_send_counts=ep_recv_counts,
            group_ep=self.group_name,
            ep_world_size=self.world_size,
            ep_rank_id=self.global_rank,
            tp_send_counts=tp_recv_counts,
            group_tp=self.moe_rs_group_name,
            tp_world_size=self.experts_tp_size)

        return hidden_states

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]
