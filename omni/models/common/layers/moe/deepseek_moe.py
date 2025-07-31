# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

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
from typing import Dict, Optional
import torch, torch_npu
from torch import nn
from transformers import PretrainedConfig
import torchair as tng
torch._logging.set_logs(recompiles=True)
# vllm adaptor
from vllm.platforms import current_platform
from vllm.config import QuantizationConfig
from vllm.attention import AttentionMetadata
from vllm.distributed import (
    get_ep_group,
    get_dp_group,
    get_world_group,
)
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from omni.models.common.layers.linear import (
    MergedReplicatedLinear,
)
from omni.models.common.layers.activation import SiluAndMul
from omni.adaptors.vllm.distributed.communication_op import (
    all_gather_two_stage,
    reduce_scatter_two_stage
)
from omni.models.common.layers.moe.fused_moe.layer import FusedMoE
from omni.models.common.config.model_config import model_extra_config
from omni.models.common.layers.moe.fused_moe.fused_moe import fused_experts_w8a8_moe_dispatch_combine

if model_extra_config.operator_opt_config.use_omni_placement:
    from omni_planner import OmniPlanner

"""NPU Stream Switch Names"""
STREAM_SHARED_EXPERT = 'stream_shared_expert'
SEQ_SPLIT_LENGTH = 4096


class ReplicatedDeepseekMLP(nn.Module):
    """Replicates the inputs and weights across multiple GPUs. No memory saving."""
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            quant_config: Optional[QuantizationConfig] = None,
            reduce_results: bool = True,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedReplicatedLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.gate_up_proj.throw_dequant = True
        self.down_proj = ReplicatedLinear(intermediate_size,
                                          hidden_size,
                                          bias=False,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn_obj = SiluAndMul()
        self.quant_symbol = True if quant_config else False


    def act_fn(self, x, quant_symbol):
        if quant_symbol and isinstance(x, tuple):
            x = dict(zip(['x_int8', 'pertoken_scale'], x))
            x['out_scale'] = self.gate_up_proj.weight_scale
        return self.act_fn_obj(x, quant_symbol)

    def forward(self, x):
        if isinstance(x, Dict):
            token_num = x.get('x_int8').shape[0]
        else:
            token_num = x.shape[0]
        if token_num > SEQ_SPLIT_LENGTH:  # Split seq to reduce memory usage
            x_list = x.split(SEQ_SPLIT_LENGTH)
            out = []
            for i in range(len(x_list)):
                x = x_list[i]
                gate_up, _ = self.gate_up_proj.forward(x)
                x = self.act_fn(gate_up, self.quant_symbol)
                x, _ = self.down_proj.forward(x)
                out.append(x)
            return torch.concat(out)
        gate_up, _ = self.gate_up_proj.forward(x)
        x = self.act_fn(gate_up, self.quant_symbol)
        x, _ = self.down_proj.forward(x)
        return x


class DeepseekMoE(nn.Module):

    def __init__(
            self,
            config: PretrainedConfig,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix
        self.ep_size = get_ep_group().world_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.device_count = torch.npu.device_count()

        self.redundancy_shared_expert_num = model_extra_config.parall_config.redundancy_shared_expert_num
        self.quant_symbol = quant_config is not None
        self.is_init_gate = False
        if self.ep_size > (config.n_routed_experts + self.redundancy_shared_expert_num):
            raise ValueError(
                f"Tensor parallel size {self.ep_size} is greater than "
                f"the number of experts {config.n_routed_experts} + {self.redundancy_shared_expert_num}.")

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.n_routed_experts,
                                     bias=False,
                                     quant_config=None,
                                     params_dtype=torch.float32,
                                     prefix=f"{prefix}.gate")
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float), requires_grad=False)
        else:
            self.gate.e_score_correction_bias = None

        self.shared_experts = None
        self.experts = None
        self.global_rank = get_world_group().rank_in_group
        self.planner = None
        self.moe_layer_idx = None
        self.expert_mapping = None
        
        if self.global_rank >= self.redundancy_shared_expert_num:
            moe_prefix = f"{prefix}.experts"
            # omni placement for redundancy route experts
            if model_extra_config.operator_opt_config.use_omni_placement:
                self.planner = OmniPlanner(config_file= model_extra_config.operator_opt_config.omni_placement_config_path, device="npu",
                                           rank=get_world_group().rank_in_group,
                                           world_size=get_world_group().world_size,
                                           num_experts=config.n_routed_experts,
                                           num_redundancy_shared_expert_rank=self.redundancy_shared_expert_num)
                self.moe_layer_idx = OmniPlanner.get_deepseek_v3_moe_layer_idx(moe_prefix)
                self.expert_mapping = self.planner.expert_mapping_on_current_layer(self.moe_layer_idx)
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
                prefix=moe_prefix,
                scoring_func=config.scoring_func,
                e_score_correction_bias=self.gate.e_score_correction_bias,
                planner=self.planner,
                moe_layer_idx=self.moe_layer_idx,
                expert_mapping=self.expert_mapping,
				first_k_dense_replace=config.first_k_dense_replace
            )
        if config.n_shared_experts is not None and \
            (self.redundancy_shared_expert_num == 0 or self.global_rank < self.redundancy_shared_expert_num):
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            # omni placement for redundancy shared experts
            if self.redundancy_shared_expert_num > 0 and OmniPlanner is not None:
                # The order that first initializing OmniPlanner, then ReplicatedDeepseekMLP, should correspond to the router expert rank initialization order in the layer.py file.
                self.planner = OmniPlanner(config_file=model_extra_config.operator_opt_config.omni_placement_config_path, device="npu",
                                           rank=self.global_rank, world_size=self.ep_size,
                                           num_experts=config.n_routed_experts,
                                           num_redundancy_shared_expert_rank=self.redundancy_shared_expert_num)
                self.moe_layer_idx = OmniPlanner.get_deepseek_v3_moe_layer_idx(f"{prefix}.share_experts", first_k_dense_replace=config.first_k_dense_replace)
                self.expert_mapping = self.planner.expert_mapping_on_current_layer(self.moe_layer_idx, is_prefill=False)

            self.shared_experts = ReplicatedDeepseekMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        if self.experts is not None:
            self.w13_prefetch_size = model_extra_config.operator_opt_config.expert_gate_up_prefetch * 1024 * 1024
            self.w2_prefetch_size = 0
            self.local_expert_num = self.experts.w13_weight.shape[0]
            if self.quant_symbol:
                self.in_scale_2 = torch.ones((self.local_expert_num, self.experts.w13_weight_scale.shape[-1] // 2), dtype=torch.float32, device=current_platform.device_type)
                torch._dynamo.mark_static(self.in_scale_2)
                if self.ep_size > 64:
                    self.w2_prefetch_size = model_extra_config.operator_opt_config.expert_down_prefetch * 1024 * 1024

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor, attn_metadata: AttentionMetadata, layer_id: int, next_attention_weights: Optional[dict]=None) -> torch.Tensor:
        if self.redundancy_shared_expert_num > 0:
            if attn_metadata is None or attn_metadata.prefill is not None:
                return self.forward_separate_expert_prefill(hidden_states, residual, attn_metadata)
            else:
                return self.forward_separate_expert_decode(hidden_states, residual, attn_metadata)
        else:
            if not self.is_init_gate:
                self.gate.weight.data = torch_npu.npu_format_cast(self.gate.weight.data, 2)
                self.is_init_gate = True
            if attn_metadata is None or attn_metadata.prefill is not None:
                return self._forward_prefill_norm(hidden_states, residual, attn_metadata)
            else:
                return self._forward_decode_norm(hidden_states, residual, attn_metadata, layer_id, next_attention_weights)

    def _forward_prefill_norm(self, hidden_states: torch.Tensor, residual: torch.Tensor, attn_metadata: AttentionMetadata) -> torch.Tensor:
        shared_output = self.shared_experts(hidden_states)

        if not model_extra_config.operator_opt_config.prefill_dispatch_combine:
            hidden_states_int8, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
            global_hidden_states = get_world_group().all_gather(hidden_states_int8, dim=0)
        else:
            global_hidden_states = hidden_states
            global_pertoken_scale = None

        router_logits, _ = self.gate.forward(hidden_states.float())
        topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                                    self.experts.top_k, self.experts.use_grouped_topk, self.experts.renormalize,
                                                                    self.experts.topk_group, self.experts.num_expert_group, self.experts.custom_routing_function,
                                                                    self.experts.scoring_func, self.experts.e_score_correction_bias, self.routed_scaling_factor,
                                                                    layer=self.experts  # ENABLE_OMNI_PLANNER
                                                                    )
        topk_ids = self.experts.apply_expert_load_balance(topk_ids=topk_ids)
            # skip when use dispatch&combine
        if not model_extra_config.operator_opt_config.prefill_dispatch_combine:
            topk_cat = torch.cat((topk_weights, topk_ids.to(torch.float), pertoken_scale.unsqueeze(-1)), dim=-1)
            topk_all = get_world_group().all_gather(topk_cat, dim=0)
            topk_weights, topk_ids, global_pertoken_scale = torch.split(
                topk_all, [topk_weights.shape[-1], topk_ids.shape[-1], 1], dim=-1)
            topk_ids = torch.round(topk_ids).to(torch.int32)
            global_pertoken_scale = global_pertoken_scale.squeeze(-1)

        final_hidden_states_list = self.experts(
            hidden_states=global_hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            pertoken_scale=global_pertoken_scale,
            attn_metadata=attn_metadata
        )

        if model_extra_config.operator_opt_config.prefill_dispatch_combine:
            if len(final_hidden_states_list) != 4:
                raise RuntimeError("len(final_hidden_states_list) != 4")
            final_hidden_states = final_hidden_states_list[0]
            gathered_tokens = final_hidden_states_list[1]
            expanded_row_idx = final_hidden_states_list[3]
        else:
            final_hidden_states = final_hidden_states_list

        # skip when use dispatch&combine
        if not model_extra_config.operator_opt_config.prefill_dispatch_combine:
            final_hidden_states = get_world_group().reduce_scatter(final_hidden_states)

        if model_extra_config.operator_opt_config.prefill_dispatch_combine:
            final_hidden_states = torch_npu.npu_moe_finalize_routing(
                gathered_tokens,
                skip1=shared_output,
                skip2=None,
                bias=None,
                scales=topk_weights.to(gathered_tokens.dtype),
                expanded_src_to_dst_row=expanded_row_idx,
                export_for_source_row=None,
                drop_pad_mode=2
            )
        else:
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states, residual

    def _forward_decode_norm(self, hidden_states: torch.Tensor, residual: torch.Tensor, attn_metadata: AttentionMetadata, layer_id: int, next_attention_weights: Optional[dict]=None) -> torch.Tensor:
        if model_extra_config.operator_opt_config.moe_multi_stream_tune and \
            model_extra_config.operator_opt_config.moe_dispatch_combine:
            if model_extra_config.operator_opt_config.use_super_kernel:
                with tng.scope.super_kernel(self.prefix, 'stream-fusion=1'):
                    return self._forward_decode_dispatch_combine(hidden_states, residual, attn_metadata, layer_id, next_attention_weights)
            else:
                return self._forward_decode_dispatch_combine(hidden_states, residual, attn_metadata, layer_id, next_attention_weights)
        if model_extra_config.operator_opt_config.moe_multi_stream_tune:
            with tng.scope.npu_stream_switch('21'):
                hidden_states = tng.scope.npu_wait_tensor(hidden_states, hidden_states)
                shared_output = self.shared_experts(hidden_states)
        else:
            shared_output = self.shared_experts(hidden_states)

        if not model_extra_config.operator_opt_config.moe_dispatch_combine:
            hidden_states_int8, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
            global_hidden_states = get_world_group().all_gather(hidden_states_int8, dim=0)
        else:
            global_hidden_states = hidden_states
            global_pertoken_scale = None

        router_logits, _ = self.gate.forward(hidden_states.float())
        # Here, we do a 2d-3d conversion and then convert back to 2d to trigger the fusion rule, fusing add rms and cast into AddRmsNormCast.
        hidden_states_3d = hidden_states.unsqueeze(1)
        hidden_states = hidden_states_3d.squeeze(1)
        topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                            self.experts.top_k, self.experts.use_grouped_topk,
                                                            self.experts.renormalize,
                                                            self.experts.topk_group, self.experts.num_expert_group,
                                                            self.experts.custom_routing_function,
                                                            self.experts.scoring_func,
                                                            self.experts.e_score_correction_bias,
                                                            self.routed_scaling_factor,
                                                            layer=self.experts  # ENABLE_OMNI_PLANNER
                                                            )
        topk_ids = self.experts.apply_expert_load_balance(topk_ids=topk_ids, best_topk_ids=attn_metadata.decode.best_topk)
        if not model_extra_config.operator_opt_config.moe_dispatch_combine:
            topk_cat = torch.cat((topk_weights, topk_ids.to(torch.float), pertoken_scale.unsqueeze(-1)), dim=-1)
            topk_all = get_world_group().all_gather(topk_cat, dim=0)

            topk_all = topk_all.view(-1, self.device_count, topk_weights.shape[0], topk_all.shape[-1]) \
                                .transpose(0, 1) \
                                .reshape(-1, topk_all.shape[-1])
            topk_weights, topk_ids, global_pertoken_scale = torch.split(topk_all, [topk_weights.shape[-1], topk_ids.shape[-1], 1], dim=-1)
            topk_ids = torch.round(topk_ids).to(torch.int32)
            global_pertoken_scale = global_pertoken_scale.squeeze(-1)

        final_hidden_states = self.experts(
            hidden_states=global_hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            pertoken_scale=global_pertoken_scale,
            attn_metadata=attn_metadata
        )

        if not model_extra_config.operator_opt_config.moe_dispatch_combine:
            final_hidden_states = get_world_group().reduce_scatter(final_hidden_states)

        final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states, residual

    def _forward_decode_dispatch_combine(self, hidden_states: torch.Tensor, residual: torch.Tensor, attn_metadata: AttentionMetadata, layer_id: int, next_attention_weights: Optional[dict]=None) -> torch.Tensor:
        is_prefill = (attn_metadata is None or attn_metadata.prefill is not None)
        router_logits, _ = self.gate.forward(hidden_states.float())
        # Here, we do a 2D to 3D conversion, and then convert back to 2D to trigger the fusion rule, fusing add rms and cast into AddRmsNormCast.
        hidden_states_3d = hidden_states.unsqueeze(1)
        hidden_states = hidden_states_3d.squeeze(1)

        with tng.scope.npu_stream_switch(STREAM_SHARED_EXPERT):
            hidden_states = tng.scope.npu_wait_tensor(hidden_states, router_logits)
            # shared_experts w13
            gate_up_share, _ = self.shared_experts.gate_up_proj.forward(hidden_states)
        wait_gate = gate_up_share if isinstance(gate_up_share, torch.Tensor) else gate_up_share[0]
        
        # expert weight prefetch
        if self.w13_prefetch_size > 0:
            torch_npu.npu_prefetch(self.experts.w13_weight, wait_gate, self.w13_prefetch_size)
        if self.w2_prefetch_size > 0:
            torch_npu.npu_prefetch(self.experts.w2_weight, wait_gate, self.w2_prefetch_size)

        topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                                self.experts.top_k, self.experts.use_grouped_topk,
                                                                self.experts.renormalize,
                                                                self.experts.topk_group, self.experts.num_expert_group,
                                                                self.experts.custom_routing_function,
                                                                self.experts.scoring_func,
                                                                self.experts.e_score_correction_bias,
                                                                self.routed_scaling_factor,
                                                                layer=self.experts  # ENABLE_OMNI_PLANNER
                                                                )
        topk_ids = self.experts.apply_expert_load_balance(topk_ids=topk_ids, best_topk_ids=attn_metadata.decode.best_topk)

        mc2_mask = attn_metadata.decode.mc2_mask if attn_metadata is not None and attn_metadata.decode is not None else None
        layer = self.experts
        
        max_num_deployed_expert = self.local_expert_num * get_dp_group().world_size
        act_dtype = hidden_states.dtype
        shared_expert_rank_num = 0
        kwargs = {
            "x": hidden_states,
            "expert_ids": topk_ids,  # [n*topk]
            "expert_shard_type": 0,  # Set it to 0 for now
            "shared_expert_rank_num": shared_expert_rank_num,  # 32
            "moe_expert_num": max_num_deployed_expert, #ENABLE_OMNI_PLANNER, 0 redundancy 256, 1 redundancy expert 320
            "global_bs": 0,  # 0 Default (all); all tokens can be set
        }

        experts_tp_size = layer.tp_size
        world_size = get_world_group().world_size
        # In fact, what we get is the die number, and the ep group is not adapted by default.
        # The default ep group is experts_num/die_num.
        global_rank = get_world_group().rank_in_group
        all_to_all_group_size = world_size // experts_tp_size

        kwargs.update({
            "scales": None,  # Quantization coefficient
            "quant_mode": layer.quant_mode,  # 0: Non-quantization; 1: Static quantization; 2: Dynamic quantization
            "group_ep": layer.moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
            "ep_world_size": all_to_all_group_size,
            "ep_rank_id": global_rank // experts_tp_size,
            "group_tp": layer.moe_rs_group_name,
            "tp_world_size": experts_tp_size,
            "tp_rank_id": global_rank % experts_tp_size,
            "x_active_mask": mc2_mask,
        })

        if model_extra_config.operator_opt_config.enable_mc2_v2:
            output = torch_npu.npu_moe_distribute_dispatch_v2(**kwargs)
        else:
            output = torch_npu.npu_moe_distribute_dispatch(**kwargs)
        expand_x, dynamic_scale, expand_idx, expert_token_nums, ep_recv_counts = output[0:5]

        group_list = expert_token_nums.to(torch.int64)
        if model_extra_config.operator_opt_config.use_omni_placement and layer.planner.enable_dump and self.experts.moe_layer_idx < 58:
            layer.planner.record_activation(layer.moe_layer_idx, group_list, is_prefill)

        # cal experts
        weight1_3 = self.experts.w13_weight
        weight2 = self.experts.w2_weight
        if self.quant_symbol:
            weight_scale1_3 = self.experts.w13_weight_scale
            weight_scale2 = self.experts.w2_weight_scale

            if self.experts.quant_mode:  # 0: no quant 1: static quant 2: dynamic quant
                pertoken_scale = dynamic_scale
            else:
                expand_x, pertoken_scale = torch_npu.npu_dynamic_quant(expand_x)

        with tng.scope.npu_stream_switch(STREAM_SHARED_EXPERT):
            wait_gate = gate_up_share if isinstance(gate_up_share, torch.Tensor) else gate_up_share[0]
            wait_gate = tng.scope.npu_wait_tensor(wait_gate, expand_x)
            if not isinstance(gate_up_share, torch.Tensor):
                gate_up_share = (wait_gate, gate_up_share[1])
            intermediate_hiddenstates_share = self.shared_experts.act_fn(gate_up_share, self.shared_experts.quant_symbol)
        if self.quant_symbol:
            # w8a8
            gate_up_proj = torch_npu.npu_grouped_matmul([expand_x], [weight1_3], bias=None, group_list=group_list,
                                                        split_item=3, output_dtype=torch.int32, group_type=0,
                                                        group_list_type=1)[0]
            
            gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                gate_up_proj, weight_scale=weight_scale1_3, activation_scale=pertoken_scale, bias=None, quant_scale=self.in_scale_2, quant_offset=None,
                group_index=group_list, activate_left=True, quant_mode=1)

            hidden_states_experts = torch_npu.npu_grouped_matmul([gate_up_proj], [weight2], scale=[weight_scale2],
                                            per_token_scale=[pertoken_scale],bias=None,
                                            group_list=group_list, split_item=3, output_dtype=act_dtype,
                                            group_type=0,
                                            group_list_type=1)[0]
        else:
            # bf16
            gate_up_proj = torch_npu.npu_grouped_matmul([expand_x], [weight1_3], bias=None, group_list=group_list,
                                                    split_item=3, group_type=0, group_list_type=1)[0]
        
            gate_up_proj = torch_npu.npu_swiglu(gate_up_proj)

            hidden_states_experts = torch_npu.npu_grouped_matmul([gate_up_proj], [weight2],bias=None,
                                            group_list=group_list, split_item=3, output_dtype=act_dtype,
                                            group_type=0, group_list_type=1)[0]

        # moeCombine
        kwargs = {
            "expand_x": hidden_states_experts,
            "expert_ids": topk_ids,  # [n*topk]
            "expand_idx": expand_idx,
            "expert_scales": topk_weights.to(torch.float32),  # weight [n*topk]
            "expert_shard_type": 0,
            "shared_expert_rank_num": shared_expert_rank_num,
            "moe_expert_num":  max_num_deployed_expert, #ENABLE_OMNI_PLANNER, 0 redundancy 256, 1 redundancy expert 320
            "global_bs": 0,  # 0 Default (all); all tokens can be set
        }
        tp_recv_counts = output[5]
        stage3_kwargs = {
            "ep_send_counts": ep_recv_counts,  # dispatch's send_counts
            "group_ep": layer.moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
            "ep_world_size": all_to_all_group_size,
            "ep_rank_id": global_rank // experts_tp_size,
            "tp_send_counts": tp_recv_counts,
            "group_tp": layer.moe_rs_group_name,
            "tp_world_size": experts_tp_size,
            "tp_rank_id": global_rank % experts_tp_size,
            "x_active_mask": mc2_mask,
        }
        kwargs.update(stage3_kwargs)

        with tng.scope.npu_stream_switch(STREAM_SHARED_EXPERT):
            if isinstance(intermediate_hiddenstates_share, dict):
                intermediate_hiddenstates_share['x_int8'] = tng.scope.npu_wait_tensor(intermediate_hiddenstates_share.get('x_int8'), hidden_states_experts)
            else:
                intermediate_hiddenstates_share = tng.scope.npu_wait_tensor(intermediate_hiddenstates_share, hidden_states_experts)
            shared_output, _ = self.shared_experts.down_proj.forward(intermediate_hiddenstates_share)

        # prefetch weights for attention next layer
        if next_attention_weights is not None and next_attention_weights['q_a_proj_weight'] is not None:
                attn_prefetch_size = model_extra_config.operator_opt_config.attn_prefetch * 1024 * 1024
                attn_prefetch_flag = shared_output
                torch_npu.npu_prefetch(next_attention_weights['q_a_proj_weight'], attn_prefetch_flag, attn_prefetch_size)
                if self.quant_symbol:
                    torch_npu.npu_prefetch(next_attention_weights['kv_a_proj_with_mqa_weight'], attn_prefetch_flag, attn_prefetch_size)
                torch_npu.npu_prefetch(next_attention_weights['q_b_proj_weight'], attn_prefetch_flag, attn_prefetch_size)
                torch_npu.npu_prefetch(next_attention_weights['W_UK'], attn_prefetch_flag, attn_prefetch_size)

        if model_extra_config.operator_opt_config.enable_mc2_v2:
            expand_idx = kwargs.pop('expand_idx', None)
            kwargs['assist_info_for_combine'] = expand_idx
            hidden_states_route = torch_npu.npu_moe_distribute_combine_v2(**kwargs)
        else:
            hidden_states_route = torch_npu.npu_moe_distribute_combine(**kwargs)

        if shared_output is not None:
            final_hidden_states = (hidden_states_route, shared_output)

        return final_hidden_states, residual

    def forward_separate_expert_decode(self,
                                       hidden_states: torch.Tensor,
                                       residual: torch.Tensor,
                                       attn_metadata: AttentionMetadata) -> torch.Tensor:
        router_logits, _ = self.gate.forward(hidden_states.float())
        
        # Here, we do a 2D to 3D conversion, and then convert back to 2D to trigger the fusion rule, fusing add rms and cast into AddRmsNormCast.
        hidden_states_3d = hidden_states.unsqueeze(1)
        hidden_states = hidden_states_3d.squeeze(1)

        topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                            self.top_k, self.use_grouped_topk,
                                                            self.renormalize,
                                                            self.topk_group, self.num_expert_group,
                                                            self.custom_routing_function,
                                                            self.scoring_func,
                                                            self.gate.e_score_correction_bias,
                                                            self.routed_scaling_factor,
                                                            layer=self.experts)
        max_num_deployed_expert=self.n_routed_experts
        if model_extra_config.operator_opt_config.use_omni_placement:
            if self.shared_experts is not None and self.planner.is_moe_layer(self.moe_layer_idx):
                hidden_states, topk_ids, topk_weights = self.planner.plan(layer_idx_moe=self.moe_layer_idx,
                                                                          tokens=hidden_states,
                                                                          token_expert_ids=topk_ids,
                                                                          token_expert_scores=topk_weights,
                                                                          top_k=self.top_k,
                                                                          expert_mapping=self.expert_mapping,
                                                                          is_prefill=False)
                max_num_deployed_expert_per_rank = self.planner.get_max_num_deployed_expert_per_rank()
                max_num_deployed_expert = max_num_deployed_expert_per_rank * (self.ep_size - self.redundancy_shared_expert_num)
            elif self.experts is not None and self.experts.planner.is_moe_layer(self.experts.moe_layer_idx):
                max_num_deployed_expert_per_rank = self.experts.planner.get_max_num_deployed_expert_per_rank()
                max_num_deployed_expert = max_num_deployed_expert_per_rank * (self.ep_size - self.redundancy_shared_expert_num)
        if model_extra_config.operator_opt_config.best_ep and attn_metadata.decode.best_topk is not None:
            fake_topk_ids = attn_metadata.decode.best_topk
            topk_ids = tng.scope.npu_wait_tensor(fake_topk_ids, topk_ids)
        hidden_states = fused_experts_w8a8_moe_dispatch_combine(self.shared_experts or self.experts,
                                                                hidden_states,
                                                                topk_weights,
                                                                topk_ids,
                                                                max_num_deployed_expert=max_num_deployed_expert,
                                                                is_prefill=False,
                                                                is_route_expert=self.experts is not None)
        return hidden_states, residual

    def forward_separate_expert_prefill(self, hidden_states: torch.Tensor, residual: torch.Tensor, attn_metadata: AttentionMetadata) -> torch.Tensor:
        global_hidden_states = all_gather_two_stage(hidden_states, idx=0, dim=0)
        if self.shared_experts:
            avg_tokens_per_shared_experts = global_hidden_states.shape[0] // self.redundancy_shared_expert_num
            shared_experts_mask = torch.zeros(global_hidden_states.shape[0], 1, dtype=torch.int32, device="npu")
            shared_experts_mask[self.global_rank * avg_tokens_per_shared_experts : (self.global_rank + 1) * avg_tokens_per_shared_experts] = 1
            shared_experts_hidden_states = global_hidden_states * shared_experts_mask
            shared_output = self.shared_experts(shared_experts_hidden_states, attn_metadata)
        else:
            shared_output = torch.zeros_like(global_hidden_states)
        shared_output = reduce_scatter_two_stage(shared_output, idx=0)

        if self.experts:
            router_logits, _ = self.gate.forward(global_hidden_states)
            topk_weights, topk_ids, _ = FusedMoE.select_experts(global_hidden_states, router_logits,
                                                                self.experts.top_k, self.experts.use_grouped_topk,
                                                                self.experts.renormalize,
                                                                self.experts.topk_group, self.experts.num_expert_group,
                                                                self.experts.custom_routing_function,
                                                                self.experts.scoring_func,
                                                                self.experts.e_score_correction_bias,
                                                                self.routed_scaling_factor,
                                                                layer=self.experts)
            global_hidden_states, global_pertoken_scale = torch_npu.npu_dynamic_quant(global_hidden_states)
            output = self.experts(
                hidden_states=global_hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                pertoken_scale=global_pertoken_scale,
                attn_metadata=attn_metadata
            )
        else:
            output = torch.zeros_like(global_hidden_states)
        final_hidden_states = reduce_scatter_two_stage(output, idx=0)
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        return final_hidden_states, residual
