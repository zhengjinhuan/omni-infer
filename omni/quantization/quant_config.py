#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#
# By using quantization case, this file is called before worker patch achieve,

from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, cast
from pydantic import BaseModel

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.fused_moe import (FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               RowParallelLinear,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.parameter import PerTensorScaleParameter
from vllm.model_executor.utils import set_weight_attrs

from omni.models.common.layers.fused_moe.layer import (FusedMoE, UnquantizedFusedMoEMethod)
from omni.adaptors.vllm.utils import ASCEND_QUATIZATION_METHOD

from .quantizer import AscendQuantizer

from vllm.attention.backends.abstract import AttentionMetadata
from .compressed_tensors.compressed_tensors_linear import AscendCompressedTensorsW8A8Int8LinearMethod
from .compressed_tensors.compressed_tensors_moe import AscendCompressedTensorsW8A8Int8MoEMethod, \
    AscendCompressedTensorsW4A8Int8MoEMethod
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target, is_activation_quantization_format,
    should_ignore_layer)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsLinearMethod, \
    CompressedTensorsKVCacheMethod
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig

@register_quantization_config(ASCEND_QUATIZATION_METHOD)
class AscendQuantConfig(CompressedTensorsConfig):
    """Config class for Ascend
    
    This class is a general class that parse quantization configs
    that are supported on ascend hardware.
    """

    def __init__(self, quant_config: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.quant_description = quant_config

    def __repr__(self) -> str:
        return "AscendQuantConfig:\n" + super().__repr__()

    @classmethod
    def get_name(cls) -> str:
        return ASCEND_QUATIZATION_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "Ascend hardware dose not support \"get_min_capability\" feature.")

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quant_model_description.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CompressedTensorsConfig":
        cls.quant_description = config
        target_scheme_map: Dict[str, Any] = dict()
        ignore = cast(List[str], config.get("ignore"))
        quant_format = cast(str, config.get("format"))

        # The quant_config has multiple config_groups, each containing
        # an input_activations key with details about how the activations are
        # quantized, a weights key indicating how the weights are quantized,
        # and a list of targets under the `targets` key, dictating which
        # layers are impacted by the quantization details. The quantization
        # details follow the structure defined by the QuantizationArgs
        # pydantic model, which is used to verify the structure of the
        # quant_config and also store the details for later use.
        for _, quant_config in config["config_groups"].items():
            targets = quant_config.get("targets")
            for target in targets:
                target_scheme_map[target] = {}
                # adapt: do not validate parameters
                module_num_bits = quant_config.get("weights").get("num_bits")
                quant_config["weights"]["num_bits"] = 0
                target_scheme_map[target][
                    "weights"] = QuantizationArgs.parse_obj(quant_config.get("weights"))
                quant_config["weights"]["num_bits"] = module_num_bits
                target_scheme_map[target]["weights"].num_bits = module_num_bits
                try:
                    target_scheme_map[target][
                        "input_activations"] = QuantizationArgs.parse_obj(
                        quant_config.get("input_activations"))
                except Exception:
                    target_scheme_map[target]["input_activations"] = None

        return cls(quant_config=config,
                   target_scheme_map=target_scheme_map,
                   ignore=ignore,
                   quant_format=quant_format,
                   kv_cache_scheme=config.get("kv_cache_scheme"),
                   sparsity_scheme_map=None,
                   sparsity_ignore_list=None)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        if torch.npu.is_available():
            return ASCEND_QUATIZATION_METHOD
        return None

    def _get_weight_num_bits(self,
                             layer_name: str,
                             weight_quant: BaseModel) -> bool:
        if isinstance(weight_quant.num_bits, dict):
            for module, module_num_bits in weight_quant.num_bits.items():
                if module in layer_name:
                    return module_num_bits
            raise ValueError(
                f"weight name mismatch, please check weights num_bits in config.json and model weight name. layer_name={layer_name}")

        else:
            return weight_quant.num_bits

    def _is_dynamic_token_w8a8(self,
                               weight_quant: BaseModel,
                               input_quant: BaseModel,
                               weight_num_bits: int) -> bool:
        is_8_bits = weight_num_bits == input_quant.num_bits == 8
        weight_strategy = (
                weight_quant.strategy == QuantizationStrategy.TENSOR.value
                or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
                or weight_quant.strategy == QuantizationStrategy.GROUP.value)
        is_token = (weight_strategy and input_quant.strategy
                    == QuantizationStrategy.TOKEN.value)
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_token and weight_quant.symmetric and is_dynamic

    def _is_dynamic_token_w4a8(self,
                               weight_quant: BaseModel,
                               input_quant: BaseModel,
                               weight_num_bits: int) -> bool:
        is_w4a8_bits = (weight_num_bits == 4) and (input_quant.num_bits == 8)
        weight_strategy = (
                weight_quant.strategy == QuantizationStrategy.TENSOR.value
                or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
                or weight_quant.strategy == QuantizationStrategy.GROUP.value)
        is_token = (weight_strategy and input_quant.strategy
                    == QuantizationStrategy.TOKEN.value)
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_w4a8_bits and is_token and weight_quant.symmetric and is_dynamic

    def get_moe_method(self, prefix):
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        weight_quant = self.target_scheme_map["Linear"].get("weights")
        input_quant = self.target_scheme_map["Linear"].get(
            "input_activations")

        weight_num_bits = self._get_weight_num_bits("mlp.experts", weight_quant)
        if self._is_dynamic_token_w8a8(weight_quant, input_quant, weight_num_bits):
            return AscendFusedMoEMethod(self, prefix, self.packed_modules_mapping)
            # return AscendCompressedTensorsW8A8Int8MoEMethod()
        elif self._is_dynamic_token_w4a8(weight_quant, input_quant, weight_num_bits):
            return AscendCompressedTensorsW4A8Int8MoEMethod(self)
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}")

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention
        if isinstance(layer, LinearBase):
            return AscendLinearMethod(self, prefix,
                                      self.packed_modules_mapping)
        elif isinstance(layer, Attention) and \
            'fa_quant_type' in self.quant_description.keys() and \
            self.quant_description['fa_quant_type'] is not None:
            return AscendKVCacheMethod(self, prefix)
        elif isinstance(layer, FusedMoE):
            moe_method = self.get_moe_method(prefix)
            if isinstance(moe_method, AscendFusedMoEMethod):
                layer.num_bits = 8
            elif isinstance(moe_method, AscendCompressedTensorsW4A8Int8MoEMethod):
                layer.num_bits = 4
            else:
                layer.num_bits = 0
            return moe_method
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class AscendLinearMethod(LinearMethodBase):
    """Linear method for Ascend quantization.

    This class calls AscendQuantizer to search a specific quantization
    implementations supported on ascend hardware for linear methods.

    Args:
        quant_config: The Ascend quantization config.
    """

    def __init__(self, quant_config: AscendQuantConfig, prefix: str,
                 packed_modules_mapping: Dict[str, Any]) -> None:
        self.quantizer = AscendQuantizer.get_quantizer(
            quant_config.quant_description, prefix, packed_modules_mapping)
        self.quant_method = self.quantizer.get_linear_method()
        self.weight_quant_strategy = quant_config.quant_description.get('config_groups', {}).get('group_0', {}).get('weights').get('strategy')

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight_dict = self.quant_method.get_weight(input_size_per_partition,
                                                   output_size_per_partition,
                                                   params_dtype)
        for weight_name, weight_param in weight_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})
            layer.register_parameter(weight_name, param)
            set_weight_attrs(param, extra_weight_attrs)

        pertensor_dict = self.quant_method.get_pertensor_param(params_dtype, output_size_per_partition, self.weight_quant_strategy)
        for pertensor_name, pertensor_param in pertensor_dict.items():
            param = PerTensorScaleParameter(data=pertensor_param,
                                            weight_loader=weight_loader)
            # disable warning
            param.ignore_warning = True
            layer.register_parameter(pertensor_name, param)

        perchannel_dict = self.quant_method.get_perchannel_param(
            output_size_per_partition, params_dtype, self.weight_quant_strategy)
        for perchannel_name, perchannel_param in perchannel_dict.items():
            param = torch.nn.Parameter(perchannel_param, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(perchannel_name, param)
            set_weight_attrs(param, extra_weight_attrs)
        setattr(layer, "init_state", 0)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        inner_gather:bool = False,
    ) -> torch.Tensor:
        return self.quant_method.apply(layer, x, bias, inner_gather)


class AscendKVCacheMethod(BaseKVCacheMethod):
    """KVCache method for Ascend quantization.

    This class calls AscendQuantizer to search a specific quantization
    implementations supported on ascend hardware for kvcache methods.

    Args:
        quant_config: The Ascend quantization config.
    """

    def __init__(self, quant_config: AscendQuantConfig, prefix: str) -> None:
        self.quantizer = AscendQuantizer.get_quantizer(
            quant_config.quant_description, prefix)
        self.quant_method = self.quantizer.get_attention_method()

    def create_weights(self, layer: torch.nn.Module) -> None:
        # Different from linear method, there are no weight processing/slicing
        # steps for attention in vllm. So the whole process of create weights
        # is hidden into the specific quant method.
        self.quant_method.create_weights(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def apply(self,
              layer: torch.nn.Module,
              query: torch.Tensor,
              key: torch.Tensor,
              value: torch.Tensor,
              k_cache: List[torch.Tensor],
              v_cache: List[torch.Tensor],
              scale: torch.Tensor,
              block_tables: torch.Tensor,
              isPrefill: bool,
              attn_metadata,
              output,
              seq_lens_tensor_cpu: Optional[int] = None) -> torch.Tensor:
        return self.quant_method.apply(layer,
                                       query,
                                       key,
                                       value,
                                       k_cache,
                                       v_cache,
                                       scale,
                                       block_tables,
                                       isPrefill,
                                       attn_metadata.attn_mask,
                                       attn_metadata.slot_mapping,
                                       output,
                                       seq_lens_tensor_cpu=seq_lens_tensor_cpu)


class AscendFusedMoEMethod(FusedMoEMethodBase):
    """FusedMoE method for Ascend quantization.

    This class calls AscendQuantizer to search a specific quantization
    implementations supported on ascend hardware for kvcache methods.

    Args:
        quant_config: The Ascend quantization config.
    """

    def __init__(self, quant_config: AscendQuantConfig, prefix: str,
                 packed_modules_mapping: Dict[str, Any]):
        self.quantizer = AscendQuantizer.get_quantizer(
            quant_config.quant_description, prefix, packed_modules_mapping)
        self.quant_method = self.quantizer.get_moe_method()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_param = self.quant_method.get_weight(
            num_experts, intermediate_size_per_partition, hidden_size,
            params_dtype)
        for param_key, param_value in weight_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        dynamic_quant_param = self.quant_method.get_dynamic_quant_param(
            num_experts, intermediate_size_per_partition, hidden_size,
            params_dtype)
        for param_key, param_value in dynamic_quant_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        pertoken_scale: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        return self.quant_method.apply(
            layer, x, topk_weights, topk_ids, pertoken_scale, attn_metadata)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)
