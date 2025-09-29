import argparse
import json
import os
import random
import threading
import time
from queue import Queue
from typing import Tuple, List, Iterable

import numpy as np
import requests
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer
import pandas as pd

REQUEST_QUEUE = Queue()
REQUEST_LATENCY = Queue()


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output

def sample_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_len: int,
):
    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    # Filter out the conversations with less than 2 turns.
    dataset = [
        data["prompt"]
        for data in dataset
    ]
    # Tokenize the prompts.
    prompts = [prompt for prompt in dataset if not pd.isnull(prompt)]
    prompt_token_ids = tokenizer(prompts).input_ids
    tokenized_dataset = []
    for i in range(len(prompts)):
        prompt_len = len(prompt_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(tokenized_dataset, num_requests)
    prompt_lens = [prompt_len for _, prompt_len, _ in sampled_requests]
    output_lens = [output_len for _, _, output_len in sampled_requests]
    # Put requests to queue
    for request_item in sampled_requests:
        REQUEST_QUEUE.put(request_item)


def send_request(
        backend: str,
        api_url: str,
        args
) -> None:
    while not REQUEST_QUEUE.empty():
        prompt, prompt_len, output_len = REQUEST_QUEUE.get()
        headers = {"User-Agent": "Benchmark Client"}

        if backend == "vllm":
            pload = {
                "prompt": prompt,
                "n": 1,
                "best_of": 1,
                "use_beam_search": False,
                "temperature": 0,
                "top_k": -1,
                "top_p": 1,
                "max_tokens": output_len,
                "ignore_eos": True,
                "stream": True,
            }
        else:
            raise ValueError(f"Unknown backend: {backend}")

        prefill_time = None

        request_start_time = time.perf_counter()
        response = requests.post(api_url, timeout=120, headers=headers, json=pload, stream=True, verify=False)

        for tmp in get_streaming_response(response):
            if prefill_time is None:
                prefill_time = time.perf_counter()

        request_end_time = time.perf_counter()
        request_latency = request_end_time - request_start_time
        prefill_latency = prefill_time - request_start_time
        if output_len > 1:
            increment_latency = request_end_time - prefill_time
        else:
            increment_latency = 0

        REQUEST_LATENCY.put(
            (prompt_len, output_len, request_latency, prefill_latency, increment_latency))

        REQUEST_QUEUE.task_done()


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/generate"
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    sample_requests(args.dataset, args.num_prompts, tokenizer, args.output_len)
    benchmark_start_time = time.perf_counter()
    threads = []
    for _ in range(args.process_num):
        thread = threading.Thread(target=send_request, args=(args.backend, api_url,args))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {args.num_prompts / benchmark_time:.2f} requests/s")
    total_output = np.sum([output_len for _, output_len, _, _, _ in REQUEST_LATENCY.queue])
    print("QPS:", f"{total_output / benchmark_time :.2f} tokens/s")

    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency, _, _ in REQUEST_LATENCY.queue])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency, _, _ in REQUEST_LATENCY.queue
    ])
    print(f"Average latency per token: {avg_per_token_latency * 1000:.2f} ms")
    avg_per_output_token_latency = np.mean([
        latency / output_len
        for _, output_len, latency, _, _ in REQUEST_LATENCY.queue
    ])
    print("Average latency per output token: "
          f"{avg_per_output_token_latency * 1000:.2f} ms")

    avg_prefill_latency = np.mean([
        prefill_latency
        for _, _, _, prefill_latency, _ in REQUEST_LATENCY.queue
    ])
    print("Average prefill latency per token: "
          f"{avg_prefill_latency * 1000:.2f} ms")

    avg_increment_latency = np.mean([
        increment_latency / (output_len - 1 + 1e-10)
        for _, output_len, _, _, increment_latency in REQUEST_LATENCY.queue
    ])
    print("Average increment latency per token: "
          f"{avg_increment_latency * 1000:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--dataset", type=str, default="/home/whk/906ascend_vllm0625/llm_tools/llm_evaluation/benchmark_eval/opencompass/data/humaneval/human-eval-v2-20210705.jsonl",
                        help="Path to the dataset.")
    parser.add_argument("--tokenizer", type=str, default="/data/nfs/benchmark/tokenizer/llama-13b/",
                        help="Name or path of the tokenizer.")
    parser.add_argument("--best-of", type=int, default=1,
                        help="Generates `best_of` sequences per prompt and "
                             "returns the best one.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=80,
                        help="Number of prompts to process.")
    parser.add_argument("--output_len", type=int, default=1024,
                        help="The output length")
    parser.add_argument("--process-num", type=int, default=8,
                        help="Number of requests per second. If this is inf, "
                             "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process to synthesize "
                             "the request arrival times.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code', action='store_true',
                        help='trust remote code from huggingface')
    run_args = parser.parse_args()
    main(run_args)

