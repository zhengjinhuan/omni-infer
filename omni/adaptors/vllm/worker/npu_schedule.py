#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
from collections import deque
from dataclasses import dataclass, fields

from typing import Iterable, Optional, Type, Union

from vllm.config import VllmConfig, SchedulerConfig
from vllm.logger import logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.utils import cdiv
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

@dataclass
class HybridSchedulerConfig(SchedulerConfig):
    enable_chunked_prefill: bool = False
    policy: str = "fcfs"
    num_scheduler_steps: int = 1
    scheduler_cls: Union[str, Type[object]] = (
        "omni.adaptors.vllm.worker.npu_schedule.NpuHybridScheduler")

    @classmethod
    def initialize_from_config(
        cls,
        vllm_scheduler_config: SchedulerConfig,
        ascend_scheduler_config: Optional[dict] = None,
    ):
        scheduler_config = {
            field.name: getattr(vllm_scheduler_config, field.name)
            for field in fields(vllm_scheduler_config) if field.init
        }
        # Override default values into original SchedulerConfig
        scheduler_config["enable_chunked_prefill"] = False
        scheduler_config["policy"] = "fcfs"
        scheduler_config["num_scheduler_steps"] = 1
        scheduler_config["scheduler_cls"] = (
            "omni.adaptors.vllm.worker.npu_schedule.NpuHybridScheduler")
        # Override params in original SchedulerConfig with params in additional_config.ascend_scheduler_config
        if ascend_scheduler_config:
            for k, v in ascend_scheduler_config.items():
                scheduler_config[k] = v
        return cls(**scheduler_config)

    def __post_init__(self) -> None:
        self.max_num_encoder_input_tokens = self.max_num_batched_tokens
        self.encoder_cache_size = self.max_num_batched_tokens
        self.chunked_prefill_enabled = self.enable_chunked_prefill
        if self.policy != "fcfs":
            raise NotImplementedError(
                f"currently AscendScheduler only supports fcfs policy, got {self.policy}"
            )
        if self.is_multimodal_model:
            raise NotImplementedError(
                "currently AscendScheduler only supports LLM models.")
        if self.num_scheduler_steps > 1:
            raise NotImplementedError(
                "currently AscendScheduler doesn't support multi-step.")
        if self.send_delta_data:
            raise NotImplementedError(
                "currently AscendScheduler doesn't support send_delta_data.")
        if self.delay_factor > 0:
            raise NotImplementedError(
                "currently AscendScheduler doesn't support scheduler_delay_factor."
            )

class NpuHybridScheduler(Scheduler):
    """This Scheduler extends vllm's original v1 scheduler
    with prefill-first scheduling strategy."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(vllm_config, kv_cache_config,
                         structured_output_manager, mm_registry,
                         include_finished_set, log_stats)
        self.scheduled_req_ids: set[str] = set()
        self.running: list[Request] = []

        if self.vllm_config.kv_transfer_config is not None and \
            self.vllm_config.kv_transfer_config.is_kv_consumer:
            raise ValueError(
                "AscendScheduler cannot be used for decode nodes. ")

    def schedule(self) -> SchedulerOutput:
        if self.scheduler_config.chunked_prefill_enabled:
            return super().schedule()
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_block_ids: dict[str, list[list[int]]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens

        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # Record scheduled LoRA requests.
        scheduled_loras: set[int] = set()

        # Use a temporary deque to collect requests that need to be skipped
        # and put back at the head of the waiting queue later
        skipped_waiting_requests: deque[Request] = deque()

        # Schedule prefill requests first.
        while self.waiting and token_budget > 0:
            if len(self.running) == self.max_num_running_reqs:
                break

            request = self.waiting[0]
            def skip_cur_request():
                self.waiting.popleft()
                skipped_waiting_requests.appendleft(request)

            # Check that adding the request still respects the max_loras
            # constraint.
            if (self.lora_config and request.lora_request and
                (len(scheduled_loras) == self.lora_config.max_loras
                 and request.lora_request.lora_int_id not in scheduled_loras)):
                # Scheduling would exceed max_loras, skip.
                skip_cur_request()
                continue

            prompt_limit = self._get_prompt_limit(request)
            # Get already-cached tokens.
            computed_blocks, num_computed_tokens = (
                self.kv_cache_manager.get_computed_blocks(request))
            num_new_tokens = request.num_tokens - num_computed_tokens
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            max_tokens_in_kvcache = (self.kv_cache_config.num_blocks *
                                     self.block_size)
            prompt_limit = min(prompt_limit, max_tokens_in_kvcache)

            # Finish request that exceeds prompt_limit or kv cache size.
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d",
                    num_new_tokens,
                    prompt_limit,
                )
                request.status = RequestStatus.FINISHED_IGNORED
                self.finished_req_ids.add(request.request_id)  # type: ignore
                self.waiting.popleft()
                continue
            if num_new_tokens > token_budget:
                # Scheduling would exceed token_budget, skip.
                skip_cur_request()
                continue
            assert num_new_tokens > 0
            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens, num_computed_tokens, computed_blocks)
            if new_blocks is None:
                # The request cannot be scheduled.
                break

            self.waiting.popleft()
            self.running.append(request)
            self.scheduled_req_ids.add(request.request_id)
            # Check request status.
            if request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            elif request.status == RequestStatus.PREEMPTED:
                scheduled_resumed_reqs.append(request)
            else:
                raise RuntimeError(f"Invalid request status: {request.status}")

            if self.lora_config and request.lora_request:
                scheduled_loras.add(request.lora_request.lora_int_id)

            all_block_ids = computed_blocks.get_block_ids()[0] + new_blocks.get_block_ids()[0]    
            req_to_new_block_ids[request.request_id] = [all_block_ids]

            # Update request info.
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.extendleft(skipped_waiting_requests)

        # If no prefill requests are scheduled,
        # Schedule decode requests next.
        if len(self.scheduled_req_ids) == 0:
            req_index = 0
            while req_index < len(self.running) and token_budget > 0:
                request = self.running[req_index]
                if request.request_id in self.scheduled_req_ids:
                    # This request has already been scheduled.
                    req_index += 1
                    continue

                num_new_tokens = (request.num_tokens_with_spec -
                                  request.num_computed_tokens)
                if (0 < self.scheduler_config.long_prefill_token_threshold <
                        num_new_tokens):
                    num_new_tokens = (
                        self.scheduler_config.long_prefill_token_threshold)
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens == 1

                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request, num_new_tokens)
                    if new_blocks is None:
                        # The request cannot be scheduled.
                        # Preempt the lowest-priority request.
                        preempted_req = self.running.pop()
                        self.kv_cache_manager.free(preempted_req)
                        preempted_req.status = RequestStatus.PREEMPTED
                        preempted_req.num_computed_tokens = 0
                        self.waiting.appendleft(preempted_req)
                        preempted_reqs.append(preempted_req)
                        if preempted_req == request:
                            # No more request to preempt.
                            can_schedule = False
                            break
                    else:
                        # The request can be scheduled.
                        can_schedule = True
                        break
                if not can_schedule:
                    break
                assert new_blocks is not None

                # Schedule the request.
                scheduled_running_reqs.append(request)
                self.scheduled_req_ids.add(request.request_id)
                req_to_new_block_ids[request.request_id] = new_blocks.get_block_ids()

                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                req_index += 1

                # Speculative decode related.
                if request.spec_token_ids:
                    num_scheduled_spec_tokens = (num_new_tokens +
                                                 request.num_computed_tokens -
                                                 request.num_tokens)
                    if num_scheduled_spec_tokens > 0:
                        # Trim spec_token_ids list to num_scheduled_spec_tokens.
                        del request.spec_token_ids[num_scheduled_spec_tokens:]
                        scheduled_spec_decode_tokens[request.request_id] = (
                            request.spec_token_ids)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = 0
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=True,
            ) for req in scheduled_resumed_reqs
        ]
        running_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=False,
            ) for req in scheduled_running_reqs
        ]
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=resumed_reqs_data + running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,  # type: ignore
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids={},
            grammar_bitmask=None,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += num_scheduled_token

        self.finished_req_ids = set()  # type: ignore
        return scheduler_output

    def _get_prompt_limit(self, request: Request) -> int:
        if (self.scheduler_config.chunked_prefill_enabled
                and not self.scheduler_config.is_multi_step):
            prompt_limit = self.scheduler_config.max_model_len
        else:
            prompt_limit = min(
                self.scheduler_config.max_model_len,
                self.scheduler_config.max_num_batched_tokens,
            )

        # Model is fine tuned with long context. Return the fine tuned max_len.
        if request.lora_request and request.lora_request.long_lora_max_len:
            assert prompt_limit <= request.lora_request.long_lora_max_len
            return request.lora_request.long_lora_max_len
        else:
            return prompt_limit

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.
 
        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            if request.status == RequestStatus.RUNNING:
                self.running.remove(request)
                self.scheduled_req_ids.discard(request.request_id)
            else:
                self.waiting.remove(request)
            request.status = finished_status
            self._free_request(request)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
        num_steps=None,
    ) -> EngineCoreOutputs:

        if len(model_runner_output.sampled_token_ids) == 0:
            return EngineCoreOutputs(
                outputs=[],
                scheduler_stats=None,
            )
        sampled_token_ids = model_runner_output.sampled_token_ids
        spec_token_ids = model_runner_output.spec_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        logger.warning(
            f" { sampled_token_ids = } { spec_token_ids = } { logprobs = } { prompt_logprobs_dict = }"
        )
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        new_running: list[Request] = []
        outputs: list[EngineCoreOutput] = []
        spec_decoding_stats: Optional[SpecDecodingStats] = None

        # NOTE(woosuk): As len(self.running) can be up to 1K or more, the below
        # loop can be a performance bottleneck. We should do our best to avoid
        # expensive operations inside the loop.
        for request in self.running:
            req_id = request.request_id
            num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_tokens_scheduled == 0:
                # The request was not scheduled in this step.
                new_running.append(request)
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index]

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if scheduled_spec_token_ids:
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens, where is given by:
                # len(scheduled_spec_token_ids) + 1 - len(generated_token_ids).
                num_tokens_rejected = (len(scheduled_spec_token_ids) + 1 -
                                       len(generated_token_ids))
                request.num_computed_tokens -= num_tokens_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=len(scheduled_spec_token_ids),
                    num_accepted_tokens=len(generated_token_ids) - 1)

            cached_encoder_input_ids = (
                self.encoder_cache_manager.get_cached_input_ids(request))
            # OPTIMIZATION: Avoid list(set) if the set is empty.
            if cached_encoder_input_ids:
                for input_id in list(cached_encoder_input_ids):
                    mm_positions = request.mm_positions[input_id]
                    start_pos = mm_positions.offset
                    num_tokens = mm_positions.length
                    if start_pos + num_tokens <= request.num_computed_tokens:
                        # The encoder output is already processed and stored
                        # in the decoder's KV cache.
                        self.encoder_cache_manager.free_encoder_input(
                            request, input_id)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids

            # Append generated tokens and check for stop. Note that if
            # a request is still being prefilled, we expect the model runner
            # to return empty token ids for the request.
            for num_new, output_token_id in enumerate(new_token_ids, 1):
                request.append_output_token_ids(output_token_id)

                # Check for stop and update request state.
                # This must be called before we make the EngineCoreOutput.
                stopped = check_stop(request, self.max_model_len)

                if stopped:
                    self._free_request(request)
                    del new_token_ids[num_new:]  # Trim new tokens if needed.
                    break

            # Extract sample logprobs if needed.
            if request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and request.use_structured_output:
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # check above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            # Add newly generated spec token ids to the request.
            if spec_token_ids is not None:
                if request.use_structured_output:
                    metadata = request.structured_output_request
                    assert metadata is not None and metadata.grammar is not None
                    # Needs to happen after new_token_ids are accepted.
                    request.spec_token_ids = metadata.grammar.validate_tokens(
                        spec_token_ids[req_index])
                else:
                    request.spec_token_ids = spec_token_ids[req_index]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids:
                # Add EngineCoreOutput for this Request.
                outputs.append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        stop_reason=request.stop_reason,
                        events=request.take_events()))
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

            self.scheduled_req_ids.remove(req_id)
            if not stopped:
                new_running.append(request)

        # Return the cached request data to the queue so they can be reused.
        for req_data in scheduler_output.scheduled_cached_reqs:
            # NOTE(rob): since we free stopped reqs above, adding stopped reqs
            # to _cached_reqs_data will cause a memory leak.
            if req_data.req_id not in self.finished_req_ids:
                self._cached_reqs_data[req_data.req_id].append(req_data)

        self.running = new_running
        engine_core_outputs = EngineCoreOutputs(
            outputs=outputs,
            scheduler_stats=self.make_stats(spec_decoding_stats),
        )
        if self.include_finished_set:
            # TODO currently sending duplicates here, improve this
            engine_core_outputs.finished_requests = (
                scheduler_output.finished_req_ids | self.finished_req_ids)

        return engine_core_outputs
