# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in compliance with the License,
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
from typing import Dict, List, Optional

import ray
import torch
import torch.nn as nn
import torch_npu
from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment, set_custom_all_reduce
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.logger import logger
from vllm.model_executor import set_random_seed
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, init_cached_hf_modules
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.worker_base import WorkerBase

from omni.adaptors.vllm.distributed.parallel_state import init_ascend_model_parallel
from omni.adaptors.vllm.platform import NPUPlatform
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class ProfilerConfig:
    """Manages profiler initialization for NPUWorker."""

    @staticmethod
    def initialize(torch_profiler_trace_dir: Optional[str]) -> Optional[torch_npu.profiler.profile]:
        """Initialize the NPU profiler if enabled.

        Args:
            torch_profiler_trace_dir: Directory to save profiler traces, if enabled.

        Returns:
            Optional[torch_npu.profiler.profile]: The profiler instance, or None if disabled.
        """
        if not torch_profiler_trace_dir:
            return None

        logger.info("Profiling enabled. Traces will be saved to: %s", torch_profiler_trace_dir)
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=False,
            record_op_args=False,
            gc_detect_threshold=None,
        )
        return torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            with_stack=True,
            profile_memory=True,
            with_modules=True,
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_profiler_trace_dir)
        )


class DistributedEnvironmentSetup:
    """Handles distributed environment initialization for NPUWorker."""

    def __init__(self, worker: 'NPUWorker'):
        self.worker = worker
        self.vllm_config = worker.vllm_config
        self.parallel_config = self.vllm_config.parallel_config

    def initialize(self) -> None:
        """Initialize the distributed environment and parallel groups."""
        set_custom_all_reduce(not self.parallel_config.disable_custom_all_reduce)
        init_distributed_environment(
            world_size=self.parallel_config.world_size,
            rank=self.worker.rank,
            distributed_init_method=self.worker.distributed_init_method,
            local_rank=self.worker.local_rank,
            backend="hccl"
        )
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=self.parallel_config.tensor_parallel_size,
            pipeline_model_parallel_size=self.parallel_config.pipeline_parallel_size
        )
        expert_tensor_parallel_size = self._get_expert_tensor_parallel_size()
        init_ascend_model_parallel(
            tensor_model_parallel_size=self.parallel_config.tensor_parallel_size,
            pipeline_model_parallel_size=self.parallel_config.pipeline_parallel_size,
            expert_tensor_parallel_size=expert_tensor_parallel_size
        )
        ensure_kv_transfer_initialized(self.vllm_config)

    def _get_expert_tensor_parallel_size(self) -> int:
        """Get the expert tensor parallel size from additional configuration.

        Returns:
            int: The expert tensor parallel size, defaulting to 1 if not specified.
        """
        additional_config = self.vllm_config.additional_config
        return int(additional_config.get("expert_tensor_parallel_size", 1)) if additional_config else 1

__origin_get_device_properties__ = torch.npu.get_device_properties
class NPUDeviceProperties:
    def __init__(self, device):
        self.properties = __origin_get_device_properties__(device)
        self.multi_processor_count = properties.multi_processor_count \
            if hasattr(self.properties, 'multi_processor_count') else 0


def get_device_properties(device):
    return NPUDeviceProperties(device)

class NPUWorker(WorkerBase):
    """Worker for executing models on NPU devices in vLLM-Ascend."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        **kwargs
    ) -> None:
        """Initialize the NPU worker.

        Args:
            vllm_config: The vLLM configuration.
            local_rank: The local rank of the worker.
            rank: The global rank of the worker.
            distributed_init_method: The method for distributed initialization.
            is_driver_worker: Whether this is the driver worker.
            **kwargs: Additional parameters for compatibility.

        Raises:
            ValueError: If multiprocessing configuration is inconsistent.
        """
        from omni import ops
        ops.register_dummy_fusion_op()

        adjusted_local_rank = self._adjust_local_rank(vllm_config, local_rank)

        torch.npu.get_device_properties = get_device_properties
        torch.cuda = torch.npu
        vllm_config.model_config.disable_cascade_attn = True

        super().__init__(
            vllm_config=vllm_config,
            local_rank=adjusted_local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker
        )

        self.cache_dtype = self._get_cache_dtype()
        if vllm_config.model_config.trust_remote_code:
            init_cached_hf_modules()

        self.profiler = ProfilerConfig.initialize(envs.VLLM_TORCH_PROFILER_DIR)
        self.model_runner: Optional[GPUModelRunner] = None
        self.init_npu_memory: Optional[int] = None

    def _adjust_local_rank(self, vllm_config: VllmConfig, local_rank: int) -> int:
        """Adjust local rank for multiprocessing configurations.

        Args:
            vllm_config: The vLLM configuration.
            local_rank: The original local rank.

        Returns:
            int: The adjusted local rank.
        """
        if not envs.VLLM_ENABLE_V1_MULTIPROCESSING or not envs.VLLM_USE_RAY_SPMD_WORKER:
            return local_rank

        dp_size = vllm_config.parallel_config.data_parallel_size
        node_count = len(ray.nodes())
        dp_rank_size = dp_size // node_count
        if dp_rank_size < 1:
            raise ValueError(f"Data parallel size {dp_size} is too small for {node_count} nodes")
        local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        adjusted_dp_rank = local_dp_rank % dp_rank_size
        return local_rank + adjusted_dp_rank * vllm_config.parallel_config.tensor_parallel_size

    def _get_cache_dtype(self) -> torch.dtype:
        """Determine the cache data type.

        Returns:
            torch.dtype: The data type for the KV cache.
        """
        return (self.model_config.dtype if self.cache_config.cache_dtype == "auto"
                else STR_DTYPE_TO_TORCH_DTYPE[self.cache_config.cache_dtype])

    def init_device(self) -> None:
        """Initialize the NPU device and distributed environment."""
        if self.device_config.device.type != "npu":
            raise RuntimeError(f"Unsupported device type: {self.device_config.device}")
        self.device = torch.device(f"npu:{self.local_rank}")
        NPUPlatform.set_device(self.device)
        NPUPlatform.empty_cache()
        self.init_npu_memory = NPUPlatform.mem_get_info()[0]
        DistributedEnvironmentSetup(self).initialize()
        set_random_seed(self.model_config.seed)
        
        self.model_runner = GPUModelRunner(self.vllm_config, self.device)

    def determine_available_memory(self) -> int:
        """Determine the available memory for KV cache.

        Returns:
            int: The number of bytes available for KV cache.

        Raises:
            RuntimeError: If memory profiling fails.
        """
        kv_caches: Dict[str, torch.Tensor] = {}
        for layer_name, layer_spec in self.model_runner.get_kv_cache_spec().items():
            if isinstance(layer_spec, FullAttentionSpec):
                kv_caches[layer_name] = (
                    torch.tensor([], dtype=layer_spec.dtype, device=self.device),
                    torch.tensor([], dtype=layer_spec.dtype, device=self.device)
                )
            else:
                raise NotImplementedError(f"Unsupported KV cache spec: {type(layer_spec)}")

        runner_kv_caches: List[torch.Tensor] = []
        bind_kv_cache(kv_caches, self.vllm_config.compilation_config.static_forward_context, runner_kv_caches)
        NPUPlatform.empty_cache()
        self.model_runner.profile_run()

        free_npu_memory, total_npu_memory = NPUPlatform.mem_get_info()
        peak_memory = self.init_npu_memory - free_npu_memory
        if peak_memory <= 0:
            raise RuntimeError(
                f"Memory profiling error: Initial free memory {self.init_npu_memory}, "
                f"current free memory {free_npu_memory}. Ensure NPU memory is cleaned up."
            )

        gc.collect()
        NPUPlatform.empty_cache()
        usable_memory = max(total_npu_memory * self.cache_config.gpu_memory_utilization - peak_memory, 0)
        logger.info(f"Available memory: {usable_memory}, total memory: {total_npu_memory}")
        return int(usable_memory)

    def execute_model(self, scheduler_output: SchedulerOutput) -> Optional[ModelRunnerOutput]:
        """Execute the model with the given scheduler output.

        Args:
            scheduler_output: The scheduler output containing request data.

        Returns:
            Optional[ModelRunnerOutput]: The model output, or None if not rank 0.
        """
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.rank == 0 else None

    def load_model(self) -> None:
        """Load the model weights onto the NPU."""
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> None:
        """Compile or warm up the model for different input sizes."""
        warmup_sizes = self._filter_warmup_sizes()
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Warming up model for size %d", size)
            self.model_runner._dummy_run(size)
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()
        set_random_seed(self.model_config.seed)

    def _filter_warmup_sizes(self) -> List[int]:
        """Filter warmup sizes based on compilation configuration.

        Returns:
            List[int]: The filtered list of warmup sizes.
        """
        warmup_sizes = self.vllm_config.compilation_config.compile_sizes.copy()
        if not self.model_config.enforce_eager:
            warmup_sizes = [
                size for size in warmup_sizes
                if size not in self.vllm_config.compilation_config.cudagraph_capture_sizes
            ]
        return warmup_sizes

    def get_model(self) -> nn.Module:
        """Get the model instance.

        Returns:
            nn.Module: The model instance.
        """
        return self.model_runner.get_model()

    def get_kv_cache_spec(self) -> Dict[str, KVCacheSpec]:
        """Get the KV cache specification.

        Returns:
            Dict[str, KVCacheSpec]: The KV cache specification.
        """
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Initialize the KV cache from the given configuration.

        Args:
            kv_cache_config: The KV cache configuration.
        """
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def initialize_cache(self, kv_cache_configs: List[KVCacheConfig]) -> None:
        """Initialize the KV cache for the worker's rank.

        Args:
            kv_cache_configs: List of KV cache configurations.
        """
        self.model_runner.initialize_kv_cache(kv_cache_configs[self.rank])

    def profile(self, is_start: bool = True) -> None:
        """Start or stop the profiler.

        Args:
            is_start: Whether to start (True) or stop (False) the profiler.

        Raises:
            RuntimeError: If the profiler is not enabled.
        """
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()

    def execute_dummy_batch(self) -> None:
        """Execute a dummy batch for testing or warmup."""
        self.model_runner._dummy_run(1)

    def sleep(self, level: int = 1) -> None:
        """Put the worker into sleep mode (not supported).

        Args:
            level: The sleep level.

        Raises:
            RuntimeError: Always, as sleep mode is not supported.
        """
        raise RuntimeError("Sleep mode is only supported on v0")

    def wake_up(self, tags: Optional[List[str]] = None) -> None:
        """Wake up the worker from sleep mode (not supported).

        Args:
            tags: Optional tags for wakeup.

        Raises:
            RuntimeError: Always, as sleep mode is not supported.
        """
        raise RuntimeError("Sleep mode is only supported on v0")
