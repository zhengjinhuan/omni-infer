# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import os

def patch_vllm_distributed():
    from vllm import distributed
    from omni.adaptors.vllm.distributed.parallel_state import (
        initialize_model_parallel,
        GroupCoordinator
    )

    distributed.parallel_state.GroupCoordinator = GroupCoordinator
    distributed.initialize_model_parallel = initialize_model_parallel
    distributed.parallel_state.initialize_model_parallel = initialize_model_parallel
    print("++++++++++++++++++++++++patch_vllm_distributed++++++++++++++++++++++++++")
 
def patch_rope():
    from vllm.model_executor.layers import rotary_embedding
 
    from omni.models.common.layers.rotary_embedding import get_rope
    rotary_embedding.get_rope = get_rope
    print("+++++++++++++++++++++++patch_rope+++++++++++++++++++++++++++")
 
def patch_embedding():
    from vllm.model_executor.layers import vocab_parallel_embedding
    from omni.models.common.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
    vocab_parallel_embedding.VocabParallelEmbedding = VocabParallelEmbedding
    vocab_parallel_embedding.ParallelLMHead = ParallelLMHead
    vocab_parallel_embedding.VocabParallelEmbedding.forward = VocabParallelEmbedding.forward_vocab

def patch_sampler():
    from omni.models.common.layers.sampler import AscendSampler
    from vllm.model_executor.layers import sampler
    sampler.Sampler = AscendSampler
    from vllm.model_executor.layers import rejection_sampler
    from omni.models.common.layers.sampler import RejectionSampler, _multinomial
    rejection_sampler.RejectionSampler = RejectionSampler
    rejection_sampler._multinomial = _multinomial
    print("++++++++++++++++++++++patch_sampler++++++++++++++++++++++++++++")

def patch_compilation():
    from omni.adaptors.vllm.compilation.decorators import _support_torch_compile
    from vllm.compilation import decorators
    decorators._support_torch_compile = _support_torch_compile
    print("+++++++++++++++++++++++patch_compilation+++++++++++++++++++++++++++")

def get_attr_by_names(src_config, attrs, default_value):
        for attr in attrs:
            value = getattr(src_config, attr, 0)
            if value > 0:
                return value
        return default_value
        
def patch_pangu():
    from vllm.config import ModelConfig
    

    @property
    def is_deepseek_mla(self) -> bool:
        kv_lora_dim_names = ['attention_kv_lora_dim', 'kv_lora_rank']
        kv_lora_dim = get_attr_by_names(self.hf_text_config, kv_lora_dim_names, None)
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in \
            ('deepseek_v2', 'deepseek_v3', 'deepseek_mtp', 'pangu_ultra_moe'):
            return kv_lora_dim is not None
        elif self.hf_text_config.model_type == 'eagle':
            # if the model is an EAGLE module, check for the
            # underlying architecture
            return self.hf_text_config.model.model_type in \
                    ('deepseek_v2', 'deepseek_v3', 'pangu_ultra_moe') \
                and kv_lora_dim is not None
        return False

    def _verify_with_expert_parallelism(self) -> None:
        num_expert_names = [
            "moe_num_experts",  # Dbrx
            "num_experts",  # Jamba
            "n_routed_experts",  # DeepSeek
            "num_local_experts",  # Mixtral
            "num_routed_experts", # Pangu
        ]
        num_experts = 0
        for name in num_expert_names:
            num_experts = getattr(self.hf_text_config, name, 0)
            if num_experts > 0:
                break
        if num_experts < 1:
            raise ValueError(
                "Number of experts in the model must be greater than 0 "
                "when expert parallelism is enabled.")
    
    def get_head_size(self) -> int:
        if self.is_deepseek_mla:
            qk_rope_dim_names = ['attention_qk_rope_dim', 'qk_rope_head_dim']
            kv_lora_dim_names = ['attention_kv_lora_dim', 'kv_lora_rank']
            qk_rope_dim = get_attr_by_names(self.hf_text_config, qk_rope_dim_names, 0)
            kv_lora_dim = get_attr_by_names(self.hf_text_config, kv_lora_dim_names, 0)
            if self.use_mla:
                return kv_lora_dim + qk_rope_dim
            else:
                qk_dim_names = ['attention_qk_dim', 'qk_nope_head_dim']
                qk_dim = get_attr_by_names(self.hf_text_config, qk_dim_names, 0)
                if qk_rope_dim and qk_dim:
                    return qk_rope_dim + qk_dim

    ModelConfig.is_deepseek_mla = is_deepseek_mla
    ModelConfig._verify_with_expert_parallelism = _verify_with_expert_parallelism
    ModelConfig.get_head_size = get_head_size
    print("++++++++++++++++++++++patch_pangu++++++++++++++++++++++++++++")

_patch_done = False

def patch_all():
    global _patch_done
    if _patch_done:
        return
    patch_vllm_distributed()
    patch_rope()
    patch_embedding()
    patch_sampler()
    patch_compilation()
    patch_pangu()
    _patch_done = True

patch_all() 
