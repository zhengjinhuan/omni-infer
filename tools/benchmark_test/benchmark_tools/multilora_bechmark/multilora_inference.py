import os
os.environ["INFER_MODE"] = 'default'
os.environ["USE_MASKFREE_ATTN"] = '1'
os.environ["MASKFREE_ATTN_THRESHOLD"] = '16384'

from typing import List, Optional, Tuple
import pandas as pd
import random
import math
import torch_npu

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest

import json
from pathlib import Path

def create_test_prompts(
        lora_id: int,
        prompt: str,
        lora_path: str,
        num_tokens: int
) -> Tuple[str, SamplingParams, Optional[LoRARequest]]:
    """Create a list of test prompts with their sampling parameters.
    """
    return (  
            prompt,       
            # We set both min and max token equal to num_tokens to generate a fixed number of output tokens
            SamplingParams(temperature=0.0,
                           min_tokens=num_tokens,
                           max_tokens=num_tokens),
            LoRARequest(f"lora-{lora_id}", lora_id, lora_path))

def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]],
                     num_tokens: int):
    """Continuously process a list of prompts and handle the outputs."""
    # We add all the requests before calling the first engine.step()
    request_id = 1
    while test_prompts:
        prompt, sampling_params, lora_request = test_prompts.pop(0)
        engine.add_request(str(request_id),
                            prompt,
                            sampling_params,
                            lora_request=lora_request)
        request_id += 1

    outputs = []
    start_event = torch_npu.npu.Event(enable_timing=True)
    # We put each engine.step() between two timing events to 
    # capture the time taken for that step on the kernel-side.
    start_event.record()
    events = [start_event]
    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        step_end_event = torch_npu.npu.Event(enable_timing=True)
        step_end_event.record()
        events.append(step_end_event)

        for request_output in request_outputs:
            if request_output.finished:
                outputs.append(request_output)

    # We need to synchronize to make sure all the asynchronous kernel calls are finished
    torch_npu.npu.synchronize()

    num_steps = len(events) - 1
    step_times_ms = []
    for i in range(num_steps):
        step_time_ms = events[i].elapsed_time(events[i + 1])
        step_times_ms.append(step_time_ms)

    num_decode_steps = num_tokens - 1
    num_prefill_steps = num_steps - num_decode_steps

    # vLLM would batch the prefill phase as well, so we need to 
    # wait for all the prefill steps to finish befor starting the decode phase.
    prefill_latency_ms = sum(step_times_ms[:num_prefill_steps])
    # Each step would generate a new token for each active request (i.e., batch size at the beginning).
    decode_latency_ms = pd.Series(step_times_ms[num_prefill_steps:]).mean()

    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    e2e_latency_ms = events[0].elapsed_time(events[-1])
    return outputs, prefill_latency_ms, decode_latency_ms, e2e_latency_ms, step_times_ms

def warm_up(engine, adapters, prompts, num_tokens, lora_adapters_path, identical_lora_id, batch_size):
    """Process one warm up request for building graph"""
    run_tests(engine, adapters, prompts, num_tokens, lora_adapters_path, identical_lora_id, batch_size)

def initialize_engine(common_config) -> LLMEngine:
    """Initialize the LLMEngine."""
    print(common_config)
    engine_args = EngineArgs(model=common_config["base_model_path"],
                                dtype=common_config["dtype"],
                                enable_lora=common_config["enable_lora"],
                                max_loras=common_config["max_loras"],
                                max_lora_rank=common_config["max_lora_rank"],
                                max_cpu_loras=common_config["max_cpu_loras"],
                                enforce_eager=common_config["enforce_eager"],
                                max_model_len=common_config["max_model_len"],
                                gpu_memory_utilization=common_config["gpu_memory_utilization"],
                                tensor_parallel_size=common_config["tensor_parallel_size"],
                                disable_async_output_proc=common_config["disable_async_output_proc"],
                                max_num_seqs=common_config["max_num_seqs"])
    return LLMEngine.from_engine_args(engine_args)

def get_per_request_report(outputs):

    output_data = []
    for output in outputs:
        num_generated_tokens = len(output.outputs[0].token_ids)
        output_dict = {
            '#tokens': num_generated_tokens,
            'first_token_time': output.metrics.first_token_time - output.metrics.first_scheduled_time,
            'completion_time': output.metrics.last_token_time - output.metrics.first_token_time,
            'prompt_num_tokens': len(output.prompt_token_ids),
            'output_num_tokens': num_generated_tokens,
            'finish_reason': output.outputs[0].finish_reason,
        }

        output_dict['tokens_per_second'] = num_generated_tokens / output_dict['completion_time']
        output_dict['latency'] = output_dict['completion_time'] / num_generated_tokens
        output_data.append(output_dict)
    return output_data

def get_test_prompts(adapters, prompts, batch_size, num_tokens, lora_adapters_path, workload_type, identical_lora_id):
    if (workload_type == "distinct"):
        sample = list(range(batch_size))
    elif (workload_type == "uniform"):
        upper_limit = math.ceil(math.sqrt(batch_size))
        number_range = list(range(upper_limit))
        sample = [random.choice(number_range) for _ in range(batch_size)]
    else:
        sample = [identical_lora_id] * batch_size

    return [create_test_prompts(index + 1, adapters[index].get("prompt", prompts[index]),
                                f'{lora_adapters_path}/{adapters[index]["name"]}',
                                num_tokens) for index in sample]

def run_tests(engine, adapters, prompts, num_tokens, lora_adapters_path, identical_lora_id, batch_size, workload_type="distinct"):
    test_prompts = get_test_prompts(adapters, prompts, batch_size, num_tokens, lora_adapters_path, workload_type, identical_lora_id)
    outputs, prefill_latency_ms, decode_latency_ms, e2e_latency_ms, step_times_ms = process_requests(engine, test_prompts, num_tokens)
    step_latency_df = pd.DataFrame({'step_latency_ms': step_times_ms})

    process_data = get_per_request_report(outputs)
    process_data = pd.DataFrame(process_data)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(process_data)
    print("prefill_latency_ms:", prefill_latency_ms, "decode_latency_ms:",
          decode_latency_ms, "e2e_latency_ms:", e2e_latency_ms)

    return process_data, {"batch_size": batch_size, "prefill_latency_ms": prefill_latency_ms,
                          "decode_latency_ms": decode_latency_ms, "e2e_latency_ms": e2e_latency_ms}, step_latency_df

def get_adapters(lora_adapters_config_path, rank):
    with open(lora_adapters_config_path, 'r') as file:
        adapters_json = json.load(file)

    for adapters_same_rank in adapters_json:
        if (rank == int(adapters_same_rank["rank"])):
            return adapters_same_rank["adapters"]
    
    return []

def init():
    with open('config/common.json', 'r') as file:
        common_config = json.load(file)

    random.seed(common_config["random_seed"])
    with open(common_config["no_lora_prompts_path"], 'r') as file:
        prompts = json.load(file)

    adapters = get_adapters(lora_adapters_config_path=common_config["lora_adapters_config_path"], rank=common_config["max_lora_rank"])

    # To store report files
    Path(common_config["report_dir"]).mkdir(parents=True, exist_ok=True)

    engine = initialize_engine(common_config)
    return engine, adapters, prompts, common_config

def main():
    engine, adapters, prompts, common_config = init()

    latencies = []

    # Control the batch sizes here
    batch_sizes = [1, 2, 4, 8, 16, 32]
    for batch_size in batch_sizes:
        # Warm-up for building computational graph
        warm_up(engine, adapters, prompts, common_config["warmup_num_tokens"], common_config["lora_adapters_path"], common_config["identical_lora_id"], batch_size)

        process_data, per_batch_latency, step_latency_df = run_tests(engine, adapters, prompts, common_config["num_tokens"],
                                                    common_config["lora_adapters_path"], common_config["identical_lora_id"],
                                                    batch_size, common_config["workload_type"])
        
        process_data.to_csv(f'{common_config["report_dir"]}/per_request_latency-{batch_size}.csv', index=False)
        step_latency_df.to_csv(f'{common_config["report_dir"]}/step_latency-{batch_size}.csv', index=False)
        latencies.append(per_batch_latency)

    latencies = pd.DataFrame(latencies, columns=["batch_size", "prefill_latency_ms",
                                                 "decode_latency_ms", "e2e_latency_ms"], index=None)
    latencies.to_csv(f'{common_config["report_dir"]}/per_batch_latency.csv', index=False)
    
if __name__ == '__main__':
    main()
