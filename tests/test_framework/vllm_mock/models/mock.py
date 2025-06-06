from vllm import ModelRegistry
import inspect

from typing import Any, Dict, Optional, Union, List
from filelock import FileLock
import json
import io
import base64
import os
import numpy
import time
import ast

import torch
from torch import nn
from vllm.attention import AttentionMetadata
from vllm.distributed import (
    get_dp_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.sequence import IntermediateTensors
from vllm.forward_context import ForwardContext, get_forward_context

from vllm.logger import logger


def access_variable(variable_name, stack_level=1):
    """
    Accesses a variable from a specific level in the call stack.

    Args:
        variable_name: The name of the variable to access (string).
        stack_level: The level in the call stack to look for the variable
                     (1 for caller's frame, 2 for caller's caller, etc.).

    Returns:
        The value of the variable, or None if not found.
    """
    try:
        frame_info = inspect.stack()[stack_level]
        frame = frame_info.frame
        if variable_name in frame.f_locals:
            return frame.f_locals[variable_name]
        elif variable_name in frame.f_globals:
            return frame.f_globals[variable_name]
        else:
            return None
    except IndexError:
        return None


class MockModel:
    pass


def mock_model_class_factory(base_class: type) -> type:

    def __init__(self, *, vllm_config, prefix: str = ""):
        self.no_npu = os.getenv("NO_NPU_MOCK", default=False)
        if self.no_npu:
            nn.Module.__init__(self)

            config = vllm_config.model_config.hf_config
            self.config = config

            self.vocab_size = config.vocab_size

            from vllm.model_executor.models.utils import (
                make_empty_intermediate_tensors_factory,
            )

            self.make_empty_intermediate_tensors = (
                make_empty_intermediate_tensors_factory(
                    ["hidden_states", "residual"], config.hidden_size
                )
            )
        else:
            base_class.__init__(self, vllm_config=vllm_config, prefix=prefix)

        self.seed = 42
        self.block_size = vllm_config.cache_config.block_size
        self.torch_compile_mode = (
            vllm_config.additional_config
            and "enable_graph_mode" in vllm_config.additional_config
            and vllm_config.additional_config["enable_graph_mode"]
        )
        self.torch_compile_mode = self.torch_compile_mode or os.getenv(
            "TORCH_COMPILE_MODE_MOCK", default=False
        )
        self.prefill_process = os.getenv("PREFILL_PROCESS", default=False)
        self.kv_cache_mode = os.getenv("KV_CACHE_MODE", default=False)
        self.capture_mode = os.getenv("CAPTURE_MODE", default=False)
        self.replay_mode = os.getenv("REPLAY_MODE", default=False)
        self.mock_capture_dir = os.getenv(
            "MOCK_CAPTURE_DIR", default="/home/kc/capture"
        )
        self.mock_capture_file_lock = os.getenv(
            "MOCK_CAPTURE_FILE_LOCK", default=".lock"
        )
        self.mock_capture_file = os.getenv(
            "MOCK_CAPTURE_FILE", default="mock_cache_20250519"
        ) + ("p" if self.prefill_process else "")
        self.simulate_elapsed_time = os.getenv("SIMULATE_ELAPSED_TIME", default=False)
        self.random_mode = os.getenv("RANDOM_MODE", default=False)
        self.forward_time = int(os.getenv("FORWARD_TIME", "0"))  # ms
        self.mock_compute_logits = os.getenv(
            "MOCK_COMPUTE_LOGITS", default=self.random_mode
        )

        if not os.path.exists(self.mock_capture_dir):
            logger.debug(f">>>Creating {self.mock_capture_dir}.")
            import pathlib

            pathlib.Path(self.mock_capture_dir).mkdir(parents=True, exist_ok=True)
        else:
            logger.debug(f">>>{self.mock_capture_dir} already exists.")

        if self.random_mode:
            logger.debug(f">>>Running forward in random mode. ")
            logger.debug(f">>>Forward time set to {self.forward_time} ms")

        self.input_id_cache = {}
        self.req_id_to_prompt = {}
        self.dummy_run = False

        if self.replay_mode:
            logger.debug(
                f">>>Replay mode is on. Loading mock_cache from "
                + f"{os.path.join(self.mock_capture_dir, self.mock_capture_file)}"
            )
            initialize_mock_cache(self)

    def initialize_mock_cache(self):
        self.mock_cache_forward = {}
        self.mock_cache_compute_logits = {}
        self.mock_cache_sample = {}
        with open(
            os.path.join(self.mock_capture_dir, self.mock_capture_file), "r"
        ) as f:
            for l in f:
                line = json.loads(l)
                if line["method"] == "forward":
                    input_str = line["input_str"]
                    output_str = line["output_str"]
                    elapsed_time = line["elapsed_time"]
                    if self.prefill_process:
                        kv_cache_str = line["kv_cache_str"]
                        kv_cache_shape = line["kv_cache_shape"]
                        self.mock_cache_forward[input_str] = (
                            output_str,
                            elapsed_time,
                            kv_cache_str,
                            kv_cache_shape,
                        )
                    else:
                        self.mock_cache_forward[input_str] = (output_str, elapsed_time)
                elif line["method"] == "compute_logits":
                    input_str = line["input_str"]
                    output_str = line["output_str"]
                    self.mock_cache_compute_logits[input_str] = output_str
                elif line["method"] == "sample":
                    input_str = line["input_str"]
                    output_str = line["output_str"]
                    self.mock_cache_sample[input_str] = output_str

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        attn_metadata = (
            get_forward_context().attn_metadata if not attn_metadata else attn_metadata
        )
        self.dummy_run = (
            attn_metadata is None
        )  # Dummy run info for compute_logits and other methods

        # Fast random mock output / dummy run except for capture mode and
        # kv_cache_mode does not need to call model forward
        if self.random_mode:
            return generate_random_output(self, input_ids, positions, attn_metadata)

        if attn_metadata is None and not self.capture_mode and not self.kv_cache_mode:
            return generate_random_output(self, input_ids, positions, attn_metadata)

        # Dummy run in kv cache mode or capture mode needs to call model forward,
        # because other workers are waiting
        if attn_metadata is None:
            return base_class.forward(
                self,
                input_ids,
                positions,
                kv_caches,
                attn_metadata,
                intermediate_tensors,
                inputs_embeds,
            )

        block_table, seq_lens, query_lens, new_reqs = extract_attn_metadata(
            attn_metadata
        )

        # For decode node in PD separation: get the original prompt ids from call stack (model runner)
        req_ids = get_req_ids_for_prefill(self, positions)

        # In KV cache mode for decode nodes, the capture-replay cache input key is
        # the entirety of the KV cache, needs computation by the model.
        # The prefill process also runs KV cache computation for capturing,
        # but does not use it as a key but instead as the output to record
        concatenated_kv_caches, output, elapsed_time = get_kv_caches(
            self,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            block_table,
            seq_lens,
            kv_caches,
            attn_metadata,
        )

        # In non-KV cache mode as well as for prefill node capture / replay,
        # we use the cached prompt input ids as the capture-replay cache key, which needs to be cached
        maintain_forward_cache(self, input_ids, block_table, query_lens, new_reqs)

        # Capture the input and model output (or KV cache for prefill nodes), or alternatively replay
        if not self.replay_mode:
            output = run_and_maybe_capture_model(
                self,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                block_table,
                query_lens,
                req_ids,
                concatenated_kv_caches,
                output,
                elapsed_time,
                kv_caches,
                attn_metadata,
            )
            return output
        else:
            output = replay_output_and_kv_caches(
                self,
                input_ids,
                positions,
                block_table,
                seq_lens,
                query_lens,
                req_ids,
            )
            return output

    def replay_output_and_kv_caches(
        self,
        input_ids,
        positions,
        block_table,
        seq_lens,
        query_lens,
        req_ids,
    ):
        if not self.torch_compile_mode:
            start_time = time.time()

        output, saved_kv_caches, total_captured_elapsed_time = replay_mock_cache(
            self,
            input_ids,
            block_table,
            query_lens,
            req_ids,
            positions,
        )
        set_kv_caches(self, block_table, seq_lens, saved_kv_caches)

        if not self.torch_compile_mode:
            end_time = time.time()
            elapsed_time = end_time - start_time
            if (
                self.simulate_elapsed_time
                and total_captured_elapsed_time > elapsed_time
            ):
                time.sleep(total_captured_elapsed_time - elapsed_time)

        return output

    def run_and_maybe_capture_model(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
        block_table,
        query_lens,
        req_ids,
        concatenated_kv_caches,
        output,
        elapsed_time,
        kv_caches,
        attn_metadata,
    ):
        if output is None:  # Run model if it did not run yet
            output, elapsed_time = base_forward(
                self,
                input_ids,
                positions,
                kv_caches,
                attn_metadata,
                intermediate_tensors,
                inputs_embeds,
            )

        if self.capture_mode:
            capture_mock_cache(
                self,
                output,
                block_table,
                query_lens,
                elapsed_time,
                concatenated_kv_caches,
                req_ids,
                positions,
            )

        return output

    def maintain_forward_cache(self, input_ids, block_table, query_lens, new_reqs):
        if not self.kv_cache_mode:
            maintain_input_token_cache(
                self, input_ids, block_table, query_lens, new_reqs
            )

        if self.prefill_process and (self.capture_mode or self.replay_mode):
            maintain_input_token_cache(
                self, input_ids, block_table, query_lens, new_reqs
            )

    def get_kv_caches(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
        block_table,
        seq_lens,
        kv_caches,
        attn_metadata,
    ):
        if self.kv_cache_mode and (not self.prefill_process or self.capture_mode):
            output, elapsed_time = base_forward(
                self,
                input_ids,
                positions,
                kv_caches,
                attn_metadata,
                intermediate_tensors,
                inputs_embeds,
            )
            concatenated_kv_caches = collect_kv_caches(self, block_table, seq_lens)
            return concatenated_kv_caches, output, elapsed_time
        else:
            concatenated_kv_caches = [None] * len(block_table)
            return concatenated_kv_caches, None, None

    def get_req_ids_for_prefill(self, positions):
        if not self.prefill_process and self.kv_cache_mode:
            scheduler_output, input_batch = find_scheduled_data_in_call_stack()
            for req in (
                scheduler_output.scheduled_new_reqs
                if scheduler_output.scheduled_new_reqs
                else []
            ):
                self.req_id_to_prompt[req.req_id] = req.prompt_token_ids
            req_ids = input_batch.req_ids
        else:
            req_ids = [None] * len(positions)
        return req_ids

    def find_scheduled_data_in_call_stack():
        for i in [6] + list(range(15)):
            scheduler_output = access_variable("scheduler_output", i)
            input_batch = (
                access_variable("self", i).input_batch
                if hasattr(access_variable("self", i), "input_batch")
                else None
            )
            if scheduler_output is not None and input_batch is not None:
                break
        return scheduler_output, input_batch

    def base_forward(
        self,
        input_ids,
        positions,
        kv_caches,
        attn_metadata,
        intermediate_tensors,
        inputs_embeds,
    ):
        if not self.torch_compile_mode:
            start_time = time.time()

        output = base_class.forward(
            self,
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors,
            inputs_embeds,
        )

        if not self.torch_compile_mode:
            end_time = time.time()
            elapsed_time = end_time - start_time
        else:
            elapsed_time = 0

        return output, elapsed_time

    def set_kv_caches(self, block_table, seq_lens, saved_kv_caches):
        if self.prefill_process:
            for i_layer, layer in enumerate(self.model.layers):
                attn_obj = (
                    layer.self_attn.attn
                    if hasattr(layer.self_attn, "attn")
                    else layer.self_attn.mla_attn
                )
                for i, seq_len in enumerate(seq_lens):
                    for block_seq_num, block_id in enumerate(block_table[i]):
                        set_kv_cache(
                            self,
                            saved_kv_caches,
                            i_layer,
                            layer,
                            attn_obj,
                            i,
                            seq_len,
                            block_seq_num,
                            block_id,
                        )

    def set_kv_cache(
        self,
        saved_kv_caches,
        i_layer,
        layer,
        attn_obj,
        i,
        seq_len,
        block_seq_num,
        block_id,
    ):
        if block_seq_num <= seq_len // self.block_size:
            if hasattr(layer.self_attn, "attn"):
                attn_obj.kv_cache[get_forward_context().virtual_engine][
                    :,
                    block_id,
                    : seq_len - block_seq_num * self.block_size,
                    0,
                    :,
                ] = saved_kv_caches[i][i_layer][
                    :,
                    block_seq_num
                    * self.block_size : min(
                        seq_len,
                        block_seq_num * self.block_size + self.block_size,
                    ),
                ]
            else:
                attn_obj.kv_cache[get_forward_context().virtual_engine][
                    block_id,
                    : seq_len - block_seq_num * self.block_size,
                    0,
                    :,
                ] = saved_kv_caches[i][i_layer][
                    block_seq_num
                    * self.block_size : min(
                        seq_len,
                        block_seq_num * self.block_size + self.block_size,
                    )
                ]

    def replay_mock_cache(
        self,
        input_ids,
        block_table,
        query_lens,
        req_ids,
        positions,
    ):
        outputs = []
        saved_kv_caches = []
        total_captured_elapsed_time = 0.0
        for block_row, query_len, req_id, position in zip(
            block_table, query_lens, req_ids, positions
        ):
            input_id_cache_key = get_unique_req_identifier(block_row)
            input_str = self.cache_repr_from_inputs(
                (
                    self.input_id_cache[input_id_cache_key]
                    if input_id_cache_key in self.input_id_cache
                    else None
                ),
                req_id,
                position,
            )

            if input_str not in self.mock_cache_forward:
                raise KeyError(
                    f"The input to this model has not been captured before, or temp is not zero:"
                    + f"\n{input_str[:min(len(input_str), 100)]}"
                )

            # Load outputs from replay cache key
            if self.prefill_process:
                (output_str, captured_elapsed_time, kv_cache_str, kv_cache_shape) = (
                    self.mock_cache_forward[input_str]
                )
                tensor_bytes = base64.b64decode(kv_cache_str)
                saved_kv_cache = (
                    torch.Tensor(numpy.frombuffer(tensor_bytes, dtype=numpy.float32))
                    .to(input_ids.device)
                    .type(torch.bfloat16)
                    .view(ast.literal_eval(kv_cache_shape))
                )
                saved_kv_caches.append(saved_kv_cache)
            else:
                (output_str, captured_elapsed_time) = self.mock_cache_forward[input_str]

            tensor_bytes = base64.b64decode(output_str)
            output = (
                torch.Tensor(numpy.frombuffer(tensor_bytes, dtype=numpy.float32))
                .to(input_ids.device)
                .type(torch.bfloat16)
                .view(query_len, -1)
            )
            outputs.append(output)
            total_captured_elapsed_time += captured_elapsed_time
        return (
            torch.concat(outputs, dim=0),
            saved_kv_caches,
            total_captured_elapsed_time,
        )

    def get_unique_req_identifier(block_row):
        return block_row[0].item()  # The first block table entry is unique for each request

    def capture_mock_cache(
        self,
        output,
        block_table,
        query_lens,
        elapsed_time,
        concatenated_kv_caches,
        req_ids,
        positions,
    ):
        if get_pp_group().is_last_rank and (
            get_tp_group().is_last_rank or self.kv_cache_mode
        ):
            curr_idx = 0
            for block_row, query_len, concatenated_kv_cache, req_id, position in zip(
                block_table,
                query_lens,
                (
                    concatenated_kv_caches
                    if self.kv_cache_mode
                    else [None] * len(block_table)
                ),
                req_ids,
                positions,
            ):
                input_id_cache_key = get_unique_req_identifier(block_row)
                input_str = self.cache_repr_from_inputs(
                    (
                        self.input_id_cache[input_id_cache_key]
                        if input_id_cache_key in self.input_id_cache
                        else None
                    ),
                    req_id,
                    position,
                )
                if self.prefill_process:
                    kv_cache_str = base64.b64encode(
                        concatenated_kv_cache.type(torch.float32)
                        .cpu()
                        .numpy()
                        .tobytes()
                    ).decode("utf-8")
                    kv_cache_shape = str(tuple(concatenated_kv_cache.shape))
                output_str = base64.b64encode(
                    output[curr_idx : curr_idx + query_len]
                    .type(torch.float32)
                    .cpu()
                    .numpy()
                    .tobytes()
                ).decode("utf-8")
                curr_idx += query_len

                lock = FileLock(
                    os.path.join(self.mock_capture_dir, self.mock_capture_file_lock)
                )
                with lock:
                    with open(
                        os.path.join(self.mock_capture_dir, self.mock_capture_file), "a"
                    ) as f:
                        logger.debug(f">>>Dump to {self.mock_capture_file}.")
                        f.write(
                            json.dumps(
                                {
                                    "method": "forward",
                                    "input_str": input_str,
                                    "output_str": output_str,
                                    "elapsed_time": elapsed_time,
                                    "kv_cache_str": (
                                        kv_cache_str if self.prefill_process else ""
                                    ),
                                    "kv_cache_shape": (
                                        kv_cache_shape if self.prefill_process else ""
                                    ),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

    def maintain_input_token_cache(self, input_ids, block_table, query_lens, new_reqs):
        input_id_cache_keys = [
            get_unique_req_identifier(block_row) for block_row in block_table
        ]

        # Make the cache from allocated block (req identifier) to all past token ids
        curr_idx = 0
        for input_id_cache_key, new_req, query_len in zip(
            input_id_cache_keys, new_reqs, query_lens
        ):
            if new_req:
                self.input_id_cache[input_id_cache_key] = (
                    torch.Tensor([]).type(torch.int32).to(input_ids.device)
                )

            if input_id_cache_key not in self.input_id_cache:
                raise KeyError(
                    f"The previously allocated block shifted and cannot be found, "
                    + f"or KV cache mode is required for prefilled decode nodes:\n{input_id_cache_key}"
                )

            self.input_id_cache[input_id_cache_key] = torch.cat(
                [
                    self.input_id_cache[input_id_cache_key],
                    input_ids[curr_idx : curr_idx + query_len],
                ]
            )
            curr_idx += query_len

    def collect_kv_caches(self, block_table, seq_lens):
        if hasattr(self.model.layers[0].self_attn, "attn"):
            return collect_kv_caches_attn(self, block_table, seq_lens)
        else:
            return collect_kv_caches_mla_attn(self, block_table, seq_lens)

    def collect_kv_caches_attn(self, block_table, seq_lens):
        kv_caches = [[] for _ in range(len(seq_lens))]
        for layer in self.model.layers:
            attn_obj = layer.self_attn.attn
            for i, seq_len in enumerate(seq_lens):
                kv_cache_entries = torch.ones(
                    2,
                    seq_len,
                    attn_obj.kv_cache[get_forward_context().virtual_engine].shape[-1],
                )
                for block_seq_num, block_id in enumerate(block_table[i]):
                    if block_seq_num <= seq_len // self.block_size:
                        kv_cache_entries[
                            :,
                            block_seq_num
                            * self.block_size : min(
                                seq_len,
                                block_seq_num * self.block_size + self.block_size,
                            ),
                        ] = attn_obj.kv_cache[get_forward_context().virtual_engine][
                            :,
                            block_id,
                            : seq_len - block_seq_num * self.block_size,
                            0,
                            :,
                        ]
                kv_caches[i].append(kv_cache_entries)
        concatenated_kv_caches = [torch.stack(kvs) for kvs in kv_caches]
        return concatenated_kv_caches

    def collect_kv_caches_mla_attn(self, block_table, seq_lens):
        kv_caches = [[] for _ in range(len(seq_lens))]
        for layer in self.model.layers:
            attn_obj = layer.self_attn.mla_attn
            for i, seq_len in enumerate(seq_lens):
                kv_cache_entries = torch.ones(
                    seq_len,
                    attn_obj.kv_cache[get_forward_context().virtual_engine].shape[-1],
                )
                for block_seq_num, block_id in enumerate(block_table[i]):
                    if block_seq_num <= seq_len // self.block_size:
                        kv_cache_entries[
                            block_seq_num
                            * self.block_size : min(
                                seq_len,
                                block_seq_num * self.block_size + self.block_size,
                            )
                        ] = attn_obj.kv_cache[get_forward_context().virtual_engine][
                            block_id,
                            : seq_len - block_seq_num * self.block_size,
                            0,
                            :,
                        ]
                kv_caches[i].append(kv_cache_entries)
        concatenated_kv_caches = [torch.stack(kvs) for kvs in kv_caches]
        return concatenated_kv_caches

    def extract_attn_metadata(attn_metadata):
        # Extract the correct metadata depending on vllm / vllm_ascend versions.
        if not hasattr(attn_metadata, "decode"):
            block_table = attn_metadata.block_tables
            seq_lens = attn_metadata.seq_lens
            query_lens = attn_metadata.query_lens
        elif attn_metadata.decode and attn_metadata.prefill:
            block_table = torch.cat(
                (attn_metadata.decode.block_table, attn_metadata.prefill.block_table),
                dim=0,
            )
            seq_lens = (
                attn_metadata.prefill.seq_lens
            )  # On reaching new versions of vllm_ascend, this may be incorrect and need cat like block table
            query_lens = torch.cat(
                (
                    torch.ones_like(attn_metadata.decode.seq_lens).type(torch.int32),
                    attn_metadata.prefill.query_lens,
                ),
                dim=0,
            )
        elif attn_metadata.prefill:
            block_table = attn_metadata.prefill.block_table
            seq_lens = attn_metadata.prefill.seq_lens
            query_lens = attn_metadata.prefill.query_lens
        elif attn_metadata.decode:
            block_table = attn_metadata.decode.block_table
            seq_lens = attn_metadata.decode.seq_lens
            query_lens = torch.ones_like(attn_metadata.decode.seq_lens).type(
                torch.int32
            )
        new_reqs = seq_lens == query_lens
        return block_table, seq_lens, query_lens, new_reqs

    def generate_random_output(self, input_ids, positions, attn_metadata):
        if not self.torch_compile_mode:
            st = time.time()

        output = torch.randn(
            (len(positions), self.config.hidden_size),
            dtype=torch.bfloat16,
            device=input_ids.device,
        )
        if self.prefill_process:
            # Fast random mock output for KV cache on prefill nodes,
            # except in dummy run when kv_cache is not allocated yet
            if attn_metadata is None:
                pass
            else:
                generate_random_kv_cache(self)

        if not self.torch_compile_mode:
            et = time.time()
            duration = et - st
            if self.forward_time != 0:
                resid_time = self.forward_time - (duration * 1000)
                if resid_time > 0:
                    time.sleep(resid_time / 1000)  # sleep in ms

        return output

    def generate_random_kv_cache(self):
        torch.manual_seed(self.seed)
        self.seed += 1
        for layer in self.model.layers:
            attn_obj = (
                layer.self_attn.attn
                if hasattr(layer.self_attn, "attn")
                else layer.self_attn.mla_attn
            )
            for virt_kv_cache in attn_obj.kv_cache:
                virt_kv_cache[...] = torch.randn(
                    virt_kv_cache.shape,
                    dtype=torch.bfloat16,
                    device=virt_kv_cache.device,
                )

    # Define the cache key
    def cache_repr_from_inputs(self, prompt_token_ids, req_id, position):
        if not self.prefill_process and self.kv_cache_mode:
            return str(self.req_id_to_prompt[req_id]) + "p" + str(position.item())
        else:
            return base64.b64encode(prompt_token_ids.cpu().numpy().tobytes()).decode(
                "utf-8"
            )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> Optional[torch.Tensor]:

        # Can turn off mocking compute_logits
        if not self.mock_compute_logits:
            return base_class.compute_logits(self, hidden_states, sampling_metadata)

        # Fast random mock output / dummy run does not need to compute anything
        if self.random_mode or self.dummy_run:
            return torch.randn(
                (len(hidden_states), self.config.vocab_size),
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )

        # Prefill process needs to compute everything, unless random mode / dummy run
        if self.prefill_process:
            return base_class.compute_logits(self, hidden_states, sampling_metadata)

        # Capture the input and model output, or alternatively replay
        if not self.replay_mode:
            logger.debug(f">>>Running compute_logits in capture mode.")
            output = run_and_maybe_capture_logits(
                self, hidden_states, sampling_metadata
            )
            return output
        else:
            logger.debug(f">>>Running compute_logits in replay mode.")
            outputs = replay_logits(self, hidden_states)
            return outputs

    def replay_logits(self, hidden_states):
        outputs = []
        for hidden_state in hidden_states:
            input_str = base64.b64encode(
                hidden_state.type(torch.float32).cpu().numpy().tobytes()
            ).decode("utf-8")
            output_str = self.mock_cache_compute_logits[input_str]
            tensor_bytes = base64.b64decode(output_str)
            output = (
                torch.Tensor(numpy.frombuffer(tensor_bytes, dtype=numpy.float32))
                .to(hidden_states.device)
                .type(torch.bfloat16)
                .view(-1, self.config.vocab_size)
            )
            outputs.append(output)
        outputs = torch.concat(outputs, dim=0)
        return outputs

    def run_and_maybe_capture_logits(self, hidden_states, sampling_metadata):
        output = base_class.compute_logits(self, hidden_states, sampling_metadata)

        # Save captured input-output pair
        if self.capture_mode:
            for hidden_state, o in zip(hidden_states, output):
                input_str = base64.b64encode(
                    hidden_state.type(torch.float32).cpu().numpy().tobytes()
                ).decode("utf-8")
                output_str = base64.b64encode(
                    o.type(torch.float32).cpu().numpy().tobytes()
                ).decode("utf-8")

                lock = FileLock(
                    os.path.join(self.mock_capture_dir, self.mock_capture_file_lock)
                )
                with lock:
                    with open(
                        os.path.join(self.mock_capture_dir, self.mock_capture_file),
                        "a",
                    ) as f:
                        logger.debug(f">>>Dump to {self.mock_capture_file}.")
                        f.write(
                            json.dumps(
                                {
                                    "method": "compute_logits",
                                    "input_str": input_str,
                                    "output_str": output_str,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

        return output

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata,
    ):
        return base_class.sample(
            self, logits, sampling_metadata
        )  # not needed to capture

    def load_weights(self, weights):
        if self.no_npu:
            return set()
        else:
            # load full model. If want to save memory, either use No-NPU version,
            # or add code from model here and ignore unused parameters
            return base_class.load_weights(self, weights)

    mock_model_class = type(
        base_class.__name__ + "MockModel",
        (base_class, MockModel),
        {
            "__init__": __init__,
            "forward": forward,
            "load_weights": load_weights,
            "compute_logits": compute_logits,
            "sample": sample,
            "cache_repr_from_inputs": cache_repr_from_inputs,
        },
    )
    return mock_model_class
