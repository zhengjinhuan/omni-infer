from transformers import PretrainedConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.layers.rotary_embedding import get_rope, get_rope_wrapper
from sglang.srt.managers.schedule_batch import global_server_args_dict
from torch import nn
import torch

from typing import Any, Dict, Iterable, Optional, Tuple
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    is_fp8_fnuz,
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.utils import (
    BumpAllocator,
    LazyValue,
    add_prefix,
    bind_or_assign,
    get_bool_env_var,
    get_device_sm,
    get_int_env_var,
    is_flashinfer_available,
    is_non_idle_and_non_empty,
    log_info_on_rank0,
    use_intel_amx_backend,
)

from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
import torch_npu


class AttentionPreparer(nn.Module):
    def __init__(
        self,
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
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",

        fused_qkv_a_proj_with_mqa = None,
        q_a_layernorm = None,
        q_b_proj = None,
        q_proj = None,
        kv_a_proj_with_mqa = None,
        kv_b_proj = None,
        kv_a_layernorm = None,
        rotary_emb = None,
        attn_mha = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.num_heads = num_heads
        assert num_heads % attn_tp_size == 0
        self.num_local_heads = num_heads // attn_tp_size

        self.w_kc = None
        self.w_scale_k = None
        self.use_deep_gemm_bmm = False

        self.fused_qkv_a_proj_with_mqa = fused_qkv_a_proj_with_mqa
        self.q_a_layernorm = q_a_layernorm
        self.q_b_proj = q_b_proj
        self.q_proj = q_proj
        self.kv_a_proj_with_mqa = kv_a_proj_with_mqa
        self.kv_b_proj = kv_b_proj
        self.kv_a_layernorm = kv_a_layernorm
        self.rotary_emb = rotary_emb
        self.attn_mha = attn_mha


    def _process_q_and_latent_cache(self, hidden_states):
        if self.q_lora_rank is not None:
            fused_qkv_a_proj_out = self.fused_qkv_a_proj_with_mqa(hidden_states)[0]
            q, latent_cache = fused_qkv_a_proj_out.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        return q, latent_cache


    def _process_absorb_k_nope(self, latent_cache):
        k_nope = latent_cache[..., : self.kv_lora_rank]
        k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)
        return k_nope


    def _process_split_qk(self, q, latent_cache):
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[:, :, self.kv_lora_rank:]
        return q_nope, q_pe, k_pe


    def _process_normal_get_kv(self, latent_cache):
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a)
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim:]
        return  k_nope, v, kv_a, latent_cache


    def _process_normal_rope_qk(self, positions, latent_cache, q, q_pe, k_nope):
        k_pe = latent_cache[:, :, self.kv_lora_rank:]
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim:] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe
        return q, k, k_pe


    def _process_normal_update_forward_batch(self, latent_cache, kv_a, k_pe, forward_batch):
        latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
        latent_cache[:, :, self.kv_lora_rank:] = k_pe
        # Save latent cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
        )
        return forward_batch


    def init_weights(self, w_kc, w_scale_k):
        self.w_kc = w_kc
        self.w_scale_k = w_scale_k


    def _process_absorb_q_nope(self, q_nope):
        if self.use_deep_gemm_bmm:
            q_nope_val, q_nope_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(q_nope.transpose(0, 1))
            )
            q_nope_out = q_nope.new_empty(
                (self.num_local_heads, aligned_m, self.kv_lora_rank)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (self.w_kc, self.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)
        return q_nope_out



    def forward_normal_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        q, latent_cache = self._process_q_and_latent_cache(hidden_states)
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_nope, v, kv_a, latent_cache = self._process_normal_get_kv(latent_cache)
        q, k, k_pe = self._process_normal_rope_qk(positions, latent_cache, q, q_pe, k_nope)
        forward_batch = self._process_normal_update_forward_batch(latent_cache, kv_a, k_pe, forward_batch)
        return q, k, v, forward_batch


    def forward_absorb_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        q, latent_cache = self._process_q_and_latent_cache(hidden_states)
        k_nope = self._process_absorb_k_nope(latent_cache)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank:].unsqueeze(1)
        q_nope_out = self._process_absorb_q_nope(q_nope)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        return q_pe, k_pe, q_nope_out, k_nope, forward_batch, zero_allocator