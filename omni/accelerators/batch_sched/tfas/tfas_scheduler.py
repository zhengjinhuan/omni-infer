from vllm.v1.core.sched.scheduler import Scheduler
import time
from collections import deque
from vllm.v1.request import Request, RequestStatus
import math
from vllm.v1.structured_output import StructuredOutputManager
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.logger import logger
from omni.models.common.config.model_config import model_extra_config

class TFASScheduler(Scheduler):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config, 
            kv_cache_config, 
            structured_output_manager, 
            mm_registry, 
            include_finished_set, 
            log_stats)
        if (self.vllm_config.kv_transfer_config is not None and 
            self.vllm_config.kv_transfer_config.is_kv_consumer):
            raise ValueError("TFASScheduler does not support KV consumer mode.")
        tfas_config = getattr(model_extra_config, "tfas_scheduler_config", None)
        if tfas_config is None:
            raise ValueError(
                "Missing tfas_scheduler_config in model_extra_config."
            )

        required_fields = ["intercept", 
                           "slope", 
                           "waiting_time_out", 
                           "token_budget"]
        for field in required_fields:
            if not hasattr(tfas_config, field):
                raise ValueError(
                    f"Missing required field '{field}' in tfas_scheduler_config."
                )

        self.tfas_intercept = tfas_config.intercept
        self.tfas_slope = tfas_config.slope
        self.tfas_waiting_time_out = tfas_config.waiting_time_out
        self.tfas_token_budget = tfas_config.token_budget

        logger.info(
            "TFASScheduler enabled"
            "(intercept=%s, slope=%s, timeout=%s, token_budget=%s)",
            self.tfas_intercept,
            self.tfas_slope,
            self.tfas_waiting_time_out,
            self.tfas_token_budget
        )

            
    def schedule(self):
        now_time = time.time()
        self.waiting = deque(
            sorted(self.waiting, key=lambda req: self._length_sort_time_decay(
                now_time, req))
        )
        upper_bound = self._compute_upper_bound(self.waiting)
        upper_bound = self._accumulate_until_bound(
            self.waiting, upper_bound)
        self.max_num_scheduled_tokens = min(
            self.scheduler_config.max_num_batched_tokens, upper_bound)
        scheduler_output = super().schedule()
        return scheduler_output
            
    def _accumulate_until_bound(self, queue: deque[Request], bound):
        total_request_len = 0
        for request in queue:
            if (request.status == RequestStatus.WAITING_FOR_REMOTE_KVS and 
                    request.request_id not in self.finished_recving_kv_req_ids):
                continue
            if (request.status == RequestStatus.WAITING_FOR_FSM):
                structured_output_req = request.structured_output_request
                if (not structured_output_req or 
                    not structured_output_req.grammar):
                    continue
            total_request_len += request.num_tokens_with_spec
            if total_request_len > bound:
                return bound + request.num_tokens_with_spec
        return bound
    
    def _compute_upper_bound(self, waiting_queue: list[Request]) -> int:
        """
        Compute the token budget upper bound based on the waiting queue.
        """
        tokens_in_waiting_queue = sum(
            req.num_prompt_tokens for req in waiting_queue)
        req_in_waiting_queue = tokens_in_waiting_queue / 1024

        bound1 = self.tfas_token_budget
        bound2 = int(
            math.sqrt(
                req_in_waiting_queue * self.tfas_intercept / self.tfas_slope
            ) * 1024
        )
        return max(bound1, bound2)

    def _length_sort_time_decay(self, now_time: float, request: Request) -> int:
        """
        Sort key function: apply time decay.
        Requests waiting longer than the timeout get the lowest priority (key=0).
        """
        if now_time - request.arrival_time > self.tfas_waiting_time_out:
            return 0
        else:
            return request.num_prompt_tokens

class TFASProfilerScheduler(Scheduler):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config, 
            kv_cache_config, 
            structured_output_manager, 
            mm_registry, 
            include_finished_set, 
            log_stats)
        if (self.vllm_config.kv_transfer_config is not None and 
            self.vllm_config.kv_transfer_config.is_kv_consumer):
            raise ValueError(
                "TFASProfilerScheduler does not support KV consumer mode.")
        tfas_config = getattr(model_extra_config, "tfas_scheduler_config", None)
        if tfas_config is None:
            raise ValueError(
                "Missing tfas_scheduler_config in model_extra_config."
            )
        if not hasattr(tfas_config, "profiler_grow_frequency"):
            raise ValueError(
                "Missing required field 'profiler_grow_frequency'"
                "in tfas_scheduler_config."
            )

        self.trigger_num = 0
        self.grow_frequency = tfas_config.profiler_grow_frequency
        logger.info("TFASProfilerScheduler enabled"
                    " (grow frequency={self.grow_frequency})")

    def schedule(self):
        self.trigger_num += 1
        self.max_num_running_reqs = min(
            self.trigger_num // self.grow_frequency + 1, 
            self.scheduler_config.max_num_seqs)
        logger.info("[TFASProfilerScheduler] tfas_max_num_seqs"
                    f"set to {self.max_num_running_reqs}")
        scheduler_output = super().schedule()
        return scheduler_output
