# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass, field
import json
import threading
from vllm.logger import logger
import omni.adaptors.vllm.envs as envs

@dataclass
class ModelParallelConfig:
    dp_size: int = 1
    o_proj_tp_size: int = 1
    
    redundancy_shared_expert_num: int = 0
    
@dataclass    
class ModelProfilingConfig:
    pass
 
@dataclass
class ModelPrecisionDiffConfig:
    pass
 
@dataclass
class ModelOperatorOptConfig:
    enable_kv_rmsnorm_rope_cache: bool = True
    prefill_dispatch_combine: bool = True
    prefill_enable_mla_alltoall: bool = False
    enable_node_mlp: bool = False
    moe_multi_stream_tune: bool = False
    best_ep: bool = False
    enable_pd_separated: bool = False
    merge_qkv: bool = False
    two_stage_comm: bool = False
    use_chunked_prefill: bool = False
    use_w8a8_dynamic_quant: bool = True
    gmm_nz: bool = False
    unquant_bmm_nz: bool = False
    moe_dispatch_combine: bool = True
    use_omni_placement: bool = False
    omni_placement_config_path:str = None
    enable_moe_expert_parallel: bool = True
    use_a3_high_performance_cann: bool = True
    use_super_kernel: bool = False
    enable_prefill_micro_batch: bool = False
    use_mlaprolog: bool = False
    opt_w2_scale_cast: bool = False
    enable_mc2_v2: bool = False
    decode_gear_list: list[int] = field(default_factory=lambda: [1])
    enable_combine_addrmsnorm_fusion: bool = False
    control_accept_rate: float = -1 # <0 or >1 不控制, >=0 and <=1 控制MTP开启时接受率为该值，几乎必然导致输出结果异常，仅保证只投机1个token时满足这一数值

    enable_round_pipeline_comm: bool = False
    enable_pipeline_comm: bool = False
    pd_seperate_prefill: bool = False
    prefill_enable_long_seq: bool = False
    enable_prefetch: bool = False
    prefill_moe_multi_stream: bool = True
    prefill_enable_mla_alltoall_local: bool = True
    prefill_enable_pipeline_comm: bool = True
    prefill_mla_multi_stream: bool = True
    enable_dense_local_tp: int = 1
    
    def __post_init__(self):
        # Check the dependencies of use_omni_placement and omni_placement_config_path
        if self.use_omni_placement and not self.omni_placement_config_path:
            raise ValueError(
                "When use_omni_placement=True, omni_placement_config_path must be provided!"
            )
@dataclass      
class ModelExtraConfig:
    parall_config: ModelParallelConfig = field(default_factory=ModelParallelConfig)
    profiling_config: ModelProfilingConfig = field(default_factory=ModelProfilingConfig)
    precision_diff_config: ModelPrecisionDiffConfig = field(default_factory=ModelPrecisionDiffConfig)
    operator_opt_config: ModelOperatorOptConfig = field(default_factory=ModelOperatorOptConfig)
    model_extra_cfg_path: str = ""
    

def init_model_extra_config() -> ModelExtraConfig:
    model_config = ModelExtraConfig()
    model_extra_cfg_path = envs.MODEL_EXTRA_CFG_PATH

    try:
        with open(model_extra_cfg_path, 'r') as f:
            config_data = json.load(f)
        # Recursively create nested objects
        parall_config = ModelParallelConfig(**config_data['model_parallel_config'])
        operator_opt_config = ModelOperatorOptConfig(**config_data['operator_optimizition_config'])
        model_config = ModelExtraConfig(
                parall_config=parall_config,
                operator_opt_config=operator_opt_config,
                model_extra_cfg_path=model_extra_cfg_path)
    except FileNotFoundError:
        logger.warning(f"[WARNING] Config file not found: {model_extra_cfg_path}, using default configuration.")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"[ERROR] Invalid JSON format in config file: {e}")
    except KeyError as e:
        raise RuntimeError(f"[ERROR] Missing required key in config data: {e}")
    except TypeError as e:
        raise RuntimeError(f"[ERROR] Config structure mismatch or incorrect field types: {e}")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Unexpected error while loading model extra config: {e}")

    return model_config

_model_extra_config = None
_model_extra_config_lock = threading.Lock()

def get_model_extra_config():
    global _model_extra_config
    if _model_extra_config is None:
        with _model_extra_config_lock:
            if _model_extra_config is None:
                _model_extra_config = init_model_extra_config()
    return _model_extra_config

def update_model_extra_config(**kwargs):
    global model_extra_config
    model_extra_config = get_model_extra_config()
    operator_opt_config = getattr(model_extra_config, 'operator_opt_config', None)
    if operator_opt_config is not None and kwargs:
        for key, value in kwargs.items():
            if hasattr(operator_opt_config, key):
                setattr(operator_opt_config, key, value)
                logger.info(f"{key} loads from additional config: {value}")
            else:
                logger.warning(f"[WARNING] operator_opt_config has no attribute: {key}")

model_extra_config = get_model_extra_config()

