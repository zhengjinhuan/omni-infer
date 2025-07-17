# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from omni.adaptors.vllm.patches import model_patch 
from vllm import ModelRegistry
import os

import os
if os.getenv("PROFILING_NAMELIST", None):
    print("<<<Profiler patch environmental variable is enabled, applying profiler patches.")
    from omni.adaptors.vllm.patches.profiler_patches import apply_profiler_patches


def register_model():

    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM",
        "omni.models.deepseek.deepseek_v2:CustomDeepseekV2ForCausalLM")

    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM",
        "omni.models.deepseek.deepseek_v3:DeepseekV3ForCausalLM")
 
    ModelRegistry.register_model(
        "DeepSeekMTPModel",
        "omni.models.deepseek.deepseek_mtp:DeepseekV3MTP")
    
    ModelRegistry.register_model(
        "DeepSeekMTPModelDuo",
        "omni.models.deepseek.deepseek_mtp:DeepseekV3MTPDuo")
 
    ModelRegistry.register_model(
        "DeepSeekMTPModelTres",
        "omni.models.deepseek.deepseek_mtp:DeepseekV3MTPTres")
 
    if int(os.getenv("RANDOM_MODE", default='0')) or int(os.getenv("CAPTURE_MODE", default='0')) or int(os.getenv("REPLAY_MODE", default='0')):
        from omni.models.mock.mock import mock_model_class_factory
 
        from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
        ModelRegistry.register_model(
            "Qwen2ForCausalLM",
            mock_model_class_factory(Qwen2ForCausalLM))
            
        from omni.models.deepseek.deepseek_v3 import DeepseekV3ForCausalLM
        ModelRegistry.register_model(
            "DeepseekV3ForCausalLM",
            mock_model_class_factory(DeepseekV3ForCausalLM))