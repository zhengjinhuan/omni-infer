import os
os.environ["INFER_MODE"] = 'default'
os.environ["USE_MASKFREE_ATTN"] = '1'
os.environ["MASKFREE_ATTN_THRESHOLD"] = '16384'

from typing import List, Tuple
import pandas as pd
import random
import torch_npu

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams

import json
from pathlib import Path

def create_test_prompts(
        prompt: str,
        seq_len: int
) -> Tuple[str, SamplingParams]:
    """Create a list of test prompts with their sampling parameters.
    """
    return (  
            prompt,       
            # We set both min and max token equal to seq_len to generate a fixed number of output tokens
            SamplingParams(temperature=0.0,
                           min_tokens=seq_len,
                           max_tokens=seq_len))

def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]],
                     seq_len: int):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 1
    # We add all the requests before calling the first engine.step()
    while test_prompts:
        prompt, sampling_params = test_prompts.pop(0)
        engine.add_request(str(request_id),
                            prompt,
                            sampling_params)
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

    num_decode_steps = seq_len - 1
    num_prefill_steps = num_steps - num_decode_steps

    # vLLM would batch the prefill phase as well, so we need to 
    # wait for all the prefill steps to finish befor starting the decode phase.
    prefill_latency_ms = sum(step_times_ms[:num_prefill_steps])
    # Each step would generate a new token for each active request (i.e., batch size at the beginning).
    decode_latency_ms = pd.Series(step_times_ms[num_prefill_steps:]).mean()

    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    e2e_latency_ms = events[0].elapsed_time(events[-1])
    return outputs, prefill_latency_ms, decode_latency_ms, e2e_latency_ms, step_times_ms

def warm_up(engine: LLMEngine, prompts, batch_size: int, seq_len: int):
    run_tests(engine, prompts, batch_size, seq_len)


def initialize_engine(common_config) -> LLMEngine:
    """Initialize the LLMEngine."""
    engine_args = EngineArgs(model=common_config["base_model_path"],
                                dtype=common_config["dtype"],
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

def get_test_prompts(prompts, batch_size, seq_len):
    test_prompts = [create_test_prompts(prompts[index]["prompt"], seq_len) for index in range(batch_size)]
    return test_prompts

def run_tests(engine, prompts, batch_size, seq_len):
    test_prompts = get_test_prompts(prompts, batch_size, seq_len)
    outputs, prefill_latency_ms, decode_latency_ms, e2e_latency_ms, step_times_ms = process_requests(engine, test_prompts, seq_len)
    step_latency_df = pd.DataFrame({'step_latency_ms': step_times_ms})

    process_data = get_per_request_report(outputs)
    process_data = pd.DataFrame(process_data)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(process_data)
    print("prefill_latency_ms:", prefill_latency_ms, "decode_latency_ms:",
          decode_latency_ms, "e2e_latency_ms:", e2e_latency_ms)

    return process_data, {"batch_size": batch_size, "prefill_latency_ms": prefill_latency_ms,
                          "decode_latency_ms": decode_latency_ms, "e2e_latency_ms": e2e_latency_ms}, step_latency_df

def init():
    with open('config/common.json', 'r') as file:
        common_config = json.load(file)

    random.seed(common_config["random_seed"])
    with open(common_config["no_lora_prompts_path"], 'r') as file:
        prompts = json.load(file)

    # To store report files
    Path(common_config["report_dir"]).mkdir(parents=True, exist_ok=True)

    engine = initialize_engine(common_config)
    return engine, prompts, common_config

def main():
    engine, prompts, common_config = init()

    latencies = []

    # Control the batch sizes here
    batch_sizes = [1, 2, 4, 8, 16, 32]
    for batch_size in batch_sizes:
        # Warm-up for building computational graph
        warm_up(engine, prompts, batch_size, common_config["warmup_num_tokens"])

        process_data, per_batch_latency, step_latency_df = run_tests(engine, prompts, batch_size, common_config["num_tokens"])
        process_data.to_csv(f'{common_config["report_dir"]}/per_request_latency-{batch_size}.csv', index=False)
        step_latency_df.to_csv(f'{common_config["report_dir"]}/step_latency-{batch_size}.csv', index=False)
        latencies.append(per_batch_latency)

    latencies = pd.DataFrame(latencies, columns=["batch_size", "prefill_latency_ms",
                                                 "decode_latency_ms", "e2e_latency_ms"], index=None)
    latencies.to_csv(f'{common_config["report_dir"]}/per_batch_latency.csv', index=False)
    
if __name__ == '__main__':
    main()
