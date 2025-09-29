import aiohttp
import argparse
import asyncio
import bisect
import csv
import json
import logging
import numpy as np
import os
import random
import time
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from transformers import PreTrainedTokenizerBase
from typing import AsyncGenerator, List, Tuple, Union

from benchmark_utils import (get_tokenizer, get_api_url, get_request_data, do_request, save_to_csv, get_csv_path,
                             operate_profile)

logging.basicConfig(level=logging.DEBUG,
                    filename='serving_debug.log',
                    filemode='w',
                    format=
                    '%(asctime)s-%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_WITHOUT_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def alpaca_prompt_format(prompt):
    """
    alpaca数据集格式
    """
    if 'input' in prompt:
        prompt_text = PROMPT_WITH_INPUT.format_map(prompt)
    else:
        prompt_text = PROMPT_WITHOUT_INPUT.format_map(prompt)
    return prompt_text


def get_alpaca_dataset(dataset):
    """
    获取alpaca数据集
    """
    dataset = [
        (data["input"], data["output"])
        for data in dataset
    ]

    return dataset


def get_sharegpt_dataset(dataset):
    """
    获取sharegpt数据集
    """
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]

    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    return dataset


def get_custom_dataset(dataset):
    """
    获取custom数据集
    """
    dataset = [
        (data["input"], data["output"])
        for data in dataset
    ]

    return dataset


def sample_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        dataset_type: str,
        max_tokens: int,
        max_prompt_tokens: int,
) -> List[Tuple[str, int, int]]:
    """
    加载数据集采样请求
    """
    with open(dataset_path) as f:
        dataset = json.load(f)

    if dataset_type == "alpaca":
        dataset = get_alpaca_dataset(dataset)
    elif dataset_type == "sharegpt":
        dataset = get_sharegpt_dataset(dataset)
    else:
        dataset = get_custom_dataset(dataset)

    prompts = [prompt for prompt, _ in dataset]
    completions = [completion for _, completion in dataset]

    prompt_token_ids = tokenizer(prompts).input_ids
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > max_prompt_tokens:
            # Prune too long sequences.
            continue
        if prompt_len + output_len > max_tokens:
            output_len = max_tokens - prompt_len
            if output_len <= 0:
                continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


async def send_request(
        request_latency_record: List,
        backend: str,
        api_url: str,
        prompt: str,
        prompt_len: int,
        output_len: int,
        best_of: int,
        use_beam_search: bool,
        app_code: str = None,
        model: str = None,
        served_model_name: str = None,
        num_scheduler_steps: int = 1,
        temperature: float = 0,
        top_k: int = -1,
        top_p: float = 1,
) -> None:
    """
    发送一次客户端请求
    """
    headers, pload, confirm_error_output = get_request_data(backend,
                                                            prompt,
                                                            prompt_len,
                                                            output_len,
                                                            best_of,
                                                            use_beam_search,
                                                            app_code,
                                                            model,
                                                            served_model_name,
                                                            temperature=temperature,
                                                            top_k=top_k,
                                                            top_p=top_p)

    time_record, _ = await do_request(api_url, headers, pload, confirm_error_output, output_len, num_scheduler_steps,
                                      backend)

    output_tokens = len(time_record) - 1

    if output_tokens < output_len:
        logger.error(f"output_tokens: {output_tokens}, output_len: {output_len}")
        logger.info("Maybe: The --num-scheduler-steps parameter is not set."
                    "If --num-scheduler-steps and --multi-step-stream-outputs=false are set on the server, --num-scheduler-steps must be set on the benchmark.")

    request_latency_record.append((prompt_len, output_len, time_record))


async def get_request(
        input_requests: List[Tuple[str, int, int]],
        request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    """
    读取测试客户端请求
    """
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def benchmark(
        request_latency_record: List,
        backend: str,
        api_url: str,
        input_requests: List[Tuple[str, int, int]],
        best_of: int,
        use_beam_search: bool,
        request_rate: float,
        app_code: str = None,
        model: str = None,
        served_model_name: str = None,
        num_scheduler_steps: int = 1,
        concurrent_num: int = 1,
        is_evenly_rate: bool = False,
        temperature: float = 0,
        top_k: int = -1,
        top_p: float = 1,
) -> None:
    """
    benchmark test
    """
    batch_num = 0
    tasks: List[asyncio.Task] = []
    pbar = tqdm(total=len(input_requests), desc="request")
    for request in input_requests:
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(send_request(request_latency_record,
                                                backend, api_url, prompt,
                                                prompt_len, output_len,
                                                best_of, use_beam_search,
                                                app_code,
                                                model,
                                                served_model_name,
                                                num_scheduler_steps,
                                                temperature,
                                                top_k,
                                                top_p))
        tasks.append(task)
        pbar.update()

        batch_num += 1
        if batch_num >= concurrent_num:
            batch_num = 0
            interval = 1.0 / request_rate if is_evenly_rate else np.random.exponential(1.0 / request_rate)
            await asyncio.sleep(interval)
    pbar.close()
    await tqdm_asyncio.gather(*tasks, desc='finish')


def main(args: argparse.Namespace):
    """
    main entry
    """
    logger.info(args)
    if len({len(args.num_prompts), len(args.request_rate), len(args.concurrent_num)}) != 1:
        logger.error(f"The array length of num_prompts, request_rate and concurrent_num is different!")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.benchmark_csv)), exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = get_api_url(args.backend, args.host, args.port, args.url)
    tokenizer = get_tokenizer(args.tokenizer)

    logger.info(f"Warmup ...")
    warmup_request_rate = 100
    warmup_input_samples = sample_requests(args.dataset, 256, tokenizer, args.dataset_type, args.max_tokens,
                                           args.max_prompt_tokens)
    asyncio.run(
        benchmark(
            [],
            args.backend, api_url,
            warmup_input_samples,
            args.best_of,
            args.use_beam_search,
            warmup_request_rate,
            args.app_code,
            args.tokenizer,
            args.served_model_name,
            args.num_scheduler_steps
        )
    )

    logger.info(f"Sample requests ...")
    request_time = args.request_time  # seconds
    num_samples = max(args.num_prompts)

    input_samples = sample_requests(args.dataset, num_samples, tokenizer, args.dataset_type, args.max_tokens,
                                    args.max_prompt_tokens)

    # profile start.
    operate_profile(args.profile, "start", app_code=args.app_code, hosts=args.host, ports=args.port,
                    level=args.service_profile_level)

    all_latency_record = []
    benchmark_head = ["DATASET", "AVG_Input_Length", "AVG_Output_Length", "REQ_RATE", "QPS", "AVG_REQ_LATENCY",
                      "AVG_PER_REQ_OUTPUT_TOKEN_THROUGHPUT", "AVG_LATENCY_PER_TOKEN_FOR_A_SINGLE_REQ", "AVG_TTFT",
                      "AVG_OUTPUT_TOKEN_THROUGHPUT"]
    benchmark_csv = get_csv_path(args.benchmark_csv)
    for i, request_rate in enumerate(args.request_rate):
        request_num = args.num_prompts[i]
        concurrent_num = args.concurrent_num[i]
        logger.info(
            f"Benchmark running with request_rate: {request_rate}, request_num: {request_num}, concurrent_num: {concurrent_num}")
        requests = input_samples[:request_num]
        latency_record = do_benchmark(api_url, args, request_rate, requests, args.num_scheduler_steps, concurrent_num)
        all_latency_record.append(latency_record)
        save_to_csv(benchmark_head, [latency_record], benchmark_csv)

    # profile stop.
    operate_profile(args.profile, "stop", app_code=args.app_code, hosts=args.host, ports=args.port,
                    level=args.service_profile_level)

    all_latency_record.sort(key=lambda element: element[3])

    os.remove(benchmark_csv)
    save_to_csv(benchmark_head, all_latency_record, benchmark_csv)

    logger.info(benchmark_head)
    for latency_record in all_latency_record:
        logger.info(latency_record)
    logger.info(f"Benchmark serving with {args.dataset_type} dataset finished")


def do_benchmark(api_url, args, request_rate, requests, num_scheduler_steps, concurrent_num):
    # (prompt len, output len, time_record)
    request_latency_record: List[Tuple[int, int, List]] = []
    asyncio.run(
        benchmark(
            request_latency_record,
            args.backend, api_url,
            requests,
            args.best_of,
            args.use_beam_search,
            request_rate,
            args.app_code,
            args.tokenizer,
            args.served_model_name,
            num_scheduler_steps,
            concurrent_num,
            args.is_evenly_rate,
            args.temperature,
            args.top_k,
            args.top_p
        )
    )

    benchmark_start_time = np.min([time_record[0] for _, _, time_record in request_latency_record])
    benchmark_end_time = np.max([time_record[-1] for _, _, time_record in request_latency_record])
    benchmark_time = benchmark_end_time - benchmark_start_time
    logger.info(f"所有请求耗时: {benchmark_time:.2f} s")

    request_num = len(requests)
    benchmark_requests = request_num / benchmark_time
    logger.info(f"请求吞吐: {benchmark_requests:.3f} requests/s")
    # Compute the latency statistics.
    avg_latency_list = [
        time_record[-1] - time_record[0]
        for _, _, time_record in request_latency_record]
    avg_latency = np.mean(avg_latency_list)
    logger.info(f"平均请求时延: {avg_latency:.3f} s")

    avg_latency_list = [round(item, 2) for item in avg_latency_list]
    logging.debug(
        "request_rate %s, avg: %s, max: %s, 请求时延 : %s",
        request_rate, avg_latency, np.max(avg_latency_list), avg_latency_list)

    avg_per_token_latency = np.mean([
        (time_record[-1] - time_record[0]) / (prompt_len + output_len)
        for prompt_len, output_len, time_record in request_latency_record
    ]) * 1000

    logger.info(f"平均每token(输入+输出)时延: {avg_per_token_latency:.2f} ms")
    avg_per_output_token_latency = np.mean([
        (time_record[-1] - time_record[1]) / output_len
        for _, output_len, time_record in request_latency_record
    ]) * 1000

    logger.info("平均每输出token时延: "
                f"{avg_per_output_token_latency:.2f} ms")

    avg_per_output_tokens = np.mean([
        output_len / (time_record[-1] - time_record[1])
        for _, output_len, time_record in request_latency_record
    ])
    logger.info("平均输出tokens吞吐: "
                f"{avg_per_output_tokens:.2f} tokens/s")
    avg_prefill_latency = np.mean([
        time_record[1] - time_record[0]
        for _, _, time_record in request_latency_record
    ]) * 1000
    logger.info("平均首tokens时延: "
                f"{avg_prefill_latency:.2f} ms")
    avg_prompt_len = np.mean([
        prompt_len
        for prompt_len, _, _ in request_latency_record
    ])
    logger.info(f"输入平均长度: {avg_prompt_len:.1f} tokens")
    avg_output_len = np.mean([
        output_len
        for _, output_len, _ in request_latency_record
    ])
    logger.info(f"输出平均长度: {avg_output_len:.1f} tokens")
    total_output_tokens = np.sum([
        output_len
        for _, output_len, _ in request_latency_record
    ])
    logger.info("输出总tokens: "
                f"{total_output_tokens} tokens")
    total_output_tokens_th = total_output_tokens / benchmark_time
    logger.info("输出tokens总吞吐: "
                f"{total_output_tokens_th:.3f} tokens/s")
    time.sleep(60)

    return (args.dataset_type, avg_prompt_len, avg_output_len,
            request_rate, benchmark_requests, avg_latency,
            avg_per_output_tokens, avg_per_output_token_latency, avg_prefill_latency,
            total_output_tokens_th)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--backend", type=str, default="openai-chat",
                        choices=["vllm", "mindspore", "base", "tgi", "openai", "trt", "openai-chat"])
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="9288")
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--app-code", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--dataset-type", type=str, default="sharegpt",
                        choices=["alpaca", "sharegpt", "custom"])
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--best-of", type=int, default=1,
                        help="Generates `best_of` sequences per prompt and "
                             "returns the best one.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", nargs='+', type=int, default=[1000, 1000, 1000],
                        help="Number of prompts to process.")
    parser.add_argument("--request-rate", nargs='+', type=float, default=[1, 4, 8],
                        help="Number of requests per second. If this is inf, "
                             "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process to synthesize "
                             "the request arrival times.")
    parser.add_argument("--request-time", type=int, default=300,
                        help="requests time in seconds.")
    parser.add_argument("--max-requests", type=int, default=3000,
                        help="max requests.")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max tokens to process.")
    parser.add_argument("--max-prompt-tokens", type=int, default=900,
                        help="Max tokens to process.")
    parser.add_argument("--benchmark-csv", type=str, default="benchmark_serving.csv",
                        help="Path to the csv.")
    parser.add_argument("--served-model-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-scheduler-steps", type=int, default=1)
    parser.add_argument("--is-evenly-rate", action="store_true")
    parser.add_argument("--concurrent-num", nargs='+', type=int, default=[1, 1, 1])
    parser.add_argument("--temperature", type=float, default=0,
                        help="Float that controls the randomness of the sampling. Zero means greedy sampling.")
    parser.add_argument("--top-k", type=int, default=-1,
                        help="Integer that controls the number of top tokens to consider. "
                             "Set to -1 to consider all tokens.")
    parser.add_argument("--top-p", type=float, default=1,
                        help="Float that controls the cumulative probability of the top tokens to consider. "
                             "Must be in (0, 1]. Set to 1 to consider all tokens.")
    parser.add_argument("--profile", action="store_true",
                        help="Use Torch Profiler or Service Profiler."
                             "The vLLM service must be started with env VLLM_SERVICE_PROFILER_DIR set.")
    parser.add_argument("--service-profile-level", type=str, required=False,
                        choices=["Level_none", "Level0", "Level1", "Level2"],
                        help="Set Service Profiler level. Support Level_none, Level0, Level1, Level2."
                             "The default value is Level0.")
    args_global = parser.parse_args()
    main(args_global)
