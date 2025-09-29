import argparse
import asyncio
import json
import os
import random
import logging
from typing import AsyncGenerator, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from benchmark_utils import (generate_str, get_tokenizer, get_api_url, get_request_data, do_request,
                             save_to_csv, statistics_and_print_performance_data, get_csv_path,
                             generate_hello_str, statistics_and_print_embedding_performance_data,
                             operate_profile)
from benchmark_climbing import run_climbing

import uvloop

import gc
gc.disable()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DATASET = None
EMBEDDING_OUTPUT_TOKENS = 1
WARM_UP_PARALLEL_NUM = 4
WARM_UP_EPOCHS = 1
WARM_UP_PROMPT_TOKENS = 128
WARM_UP_OUTPUT_TOKENS = 128


def sample_fixed_requests(dataset_path, prompt_tokens: int, output_tokens: int, requests_num: int,
                          tokenizer: PreTrainedTokenizerBase = None):
    if prompt_tokens is None or requests_num is None or requests_num < 0:
        raise ValueError
    fixed_dataset_path = os.path.join(dataset_path, str(prompt_tokens) + ".json")
    if not os.path.exists(fixed_dataset_path):
        raise ValueError(f"Fixed dataset path {fixed_dataset_path} is not existed.")
    with open(fixed_dataset_path, "r", encoding="utf-8") as f:
        test_data = [item["input"] for item in json.load(f)]
        filtered_dataset = []
        for _ in range(requests_num):
            prompt = random.choice(test_data)
            prompt_len = len(tokenizer.encode(prompt)) if tokenizer is not None else prompt_tokens
            filtered_dataset.append((prompt, prompt_len, output_tokens))
        return filtered_dataset


def sample_random_requests(
        tokenizer: PreTrainedTokenizerBase,
        prompt_tokens: int,
        output_tokens: int,
        prefix_caching_num: int = 0,
        parallel_num: int = 1,
) -> List[Tuple[str, int, int]]:
    requests_list = []
    for _ in range(parallel_num):
        caching_prompt = (generate_hello_str(tokenizer, prefix_caching_num) +
                          generate_str(tokenizer, prompt_tokens - prefix_caching_num))
        requests_list.append((caching_prompt, prompt_tokens, output_tokens))
    return requests_list


# just copy from vllm.
def sample_sharegpt_requests(
        dataset_path,
        requests_num,
        tokenizer: PreTrainedTokenizerBase,
        modified_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    if modified_output_len is not None and modified_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    global CACHE_DATASET
    if CACHE_DATASET is None:
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        # Only keep the first two turns of each conversation.
        dataset = [subset for subset in dataset if len(subset["conversations"]) >= 2]
        dataset = [(subset["conversations"][0]["value"],
                    subset["conversations"][1]["value"]) for subset in dataset]

        # Shuffle the dataset.
        random.shuffle(dataset)

        CACHE_DATASET = dataset
    else:
        dataset = CACHE_DATASET

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == requests_num:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        output = dataset[i][1]
        output_token_ids = tokenizer(output).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(output_token_ids
                         ) if modified_output_len is None else modified_output_len
        if prompt_len < 4 or output_len < 4:
            # Filter sequences that are too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Filter sequences that are too long.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


def sample_human_eval_requests(
        dataset_path: str,
        requests_num: int,
        tokenizer: PreTrainedTokenizerBase,
        modified_output_len: Optional[int] = None,
):
    # Load the dataset.
    global CACHE_DATASET
    if CACHE_DATASET is None:
        with open(dataset_path, encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
        # Filter out the conversations with less than 2 turns.
        dataset = [(data["prompt"], data["canonical_solution"]) for data in dataset]

        # Shuffle the dataset.

        random.shuffle(dataset)
        CACHE_DATASET = dataset
    else:
        dataset = CACHE_DATASET

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == requests_num:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        output = dataset[i][1]
        output_token_ids = tokenizer(output).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(output_token_ids
                         ) if modified_output_len is None else modified_output_len
        if prompt_len == 0 or output_len == 0:
            continue

        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


async def get_request(
        input_requests,
        req_rate,
        use_pd_separate
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for req in input_requests:
        yield req

        if req_rate == float("inf"):
            # If the request rate is infinite, then we don't need to wait.
            continue
        if use_pd_separate:
            # Sample the request interval from the even distribution.
            interval = 1.0 / req_rate
        else:
            # Sample the request interval from the exponential distribution.
            interval = np.random.exponential(1.0 / req_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


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
        use_spec_decode: bool = False,
        temperature: float = 0,
        top_k: int = -1,
        top_p: float = 1,
) -> None:
    headers, pload, confirm_error_output = get_request_data(backend,
                                                            prompt,
                                                            prompt_len,
                                                            output_len,
                                                            best_of,
                                                            use_beam_search,
                                                            app_code,
                                                            model,
                                                            served_model_name,
                                                            use_spec_decode,
                                                            temperature,
                                                            top_k,
                                                            top_p)

    time_record, chunk_record = await do_request(api_url, headers, pload, confirm_error_output,
                                                 output_len, num_scheduler_steps, backend, use_spec_decode)

    output_tokens = len(time_record) - 1

    # output_tokens will smaller than output_len when use spec decode, fix it in update_spec_output_tokens function.
    if not use_spec_decode and output_tokens < output_len:
        logger.error("output_tokens: %d < output_len: %d", output_tokens, output_len)
        logger.info("Maybe: The --num-scheduler-steps or --use-spec-decode parameter is not set. "
                    "1.If --num-scheduler-steps and --multi-step-stream-outputs=false are set on the server, --num-scheduler-steps must be set on the benchmark. "
                    "2.If --use-spec-decode is set, --num-speculative-tokens also needs to be set for the benchmark, which is the same as that on the server. ")

    request_latency_record.append((prompt_len, output_len, time_record, chunk_record))


async def benchmark(
        request_latency_record: List,
        backend: str,
        api_urls: List,
        input_requests: List[Tuple[str, int, int]],
        best_of: int,
        use_beam_search: bool,
        request_rate: float,
        parallel_num: int,
        epochs: int,
        app_code: str = None,
        model: str = None,
        served_model_name: str = None,
        num_scheduler_steps: int = 1,
        use_spec_decode: bool = False,
        use_pd_separate: bool = False,
        temperature: float = 0,
        top_k: int = -1,
        top_p: float = 1,
) -> None:
    num_url = len(api_urls)
    input_index = 0
    for ep in tqdm(range(epochs), desc="epoch"):
        input_parallel = []
        for _ in range(parallel_num):
            input_parallel.append(input_requests[input_index])
            input_index += 1
            if input_index >= len(input_requests):
                input_index = 0

        tasks: List[asyncio.Task] = []
        url_index = 0
        async for request in get_request(input_parallel, request_rate, use_pd_separate):
            prompt, prompt_len, output_len = request
            api_url = api_urls[url_index % num_url]
            task = asyncio.create_task(send_request(request_latency_record,
                                                    backend, api_url, prompt,
                                                    prompt_len, output_len,
                                                    best_of, use_beam_search,
                                                    app_code,
                                                    model,
                                                    served_model_name,
                                                    num_scheduler_steps,
                                                    use_spec_decode,
                                                    temperature,
                                                    top_k,
                                                    top_p))
            tasks.append(task)
            url_index += 1
        await asyncio.gather(*tasks)


def group_get_output_tokens_per_step(chunk_list):
    output_list = []
    cur_output_token = 0
    for chunk in chunk_list:
        if chunk.startswith('data:'):
            output = chunk[5:].strip()
        else:
            output = chunk.strip().rstrip("\0")

        if output != '[DONE]':
            output = json.loads(output)
            if usage := output.get("usage"):
                output_list.append(usage['completion_tokens'] - cur_output_token)
                cur_output_token = usage['completion_tokens']
    return output_list


def update_spec_output_tokens(request_latency_record, backend):
    if backend not in ["openai", "openai-chat"]:
        raise ValueError(
            "Backend %s is not supported in spec decode benchmark, it might return the incorrect results.", backend)

    request_output_lens_list = [group_get_output_tokens_per_step(chunk_list) for _, _, _, chunk_list in
                                request_latency_record]

    for req_index in range(len(request_latency_record)):
        (prompt_len, output_len, time_record, chunk_record) = request_latency_record[req_index]
        output_lens_per_step = request_output_lens_list[req_index]
        new_time_record = [time_record[0]]
        for o_len, timestamp in zip(output_lens_per_step, time_record[1:]):
            new_time_record.extend([timestamp] * o_len)

        output_tokens = len(new_time_record) - 1
        if len(new_time_record) - 1 < output_len:
            logger.warning("output_tokens: %d < output_len: %d, maybe caused by the difference between the number of "
                           "tokens re-encoded by the tokenizer and the number of original tokens.", output_tokens,
                           output_len)

        # replace chunk_record to output_step for acceptance rate of speculative decode.
        request_latency_record[req_index] = (prompt_len, output_len, new_time_record, len(output_lens_per_step))


def get_dataset_requests(args, tokenizer, prompt_tokens, output_tokens, parallel_num):
    if args.dataset_type == "random":
        logger.info(
            "Benchmark running with parallel_num: %d, "
            "prompt_tokens: %d, output_tokens: %d",
            parallel_num, prompt_tokens, output_tokens
        )
        return sample_random_requests(tokenizer, prompt_tokens, output_tokens, args.prefix_caching_num, parallel_num)
    else:
        if not os.path.exists(args.dataset_path):
            raise ValueError(f"Dataset path {args.dataset_path} is not existed.")

        logger.info(
            "Benchmark running with parallel_num: %d, "
            "dataset: %s, output_tokens: %s",
            parallel_num,
            args.dataset_type, "real" if args.use_real_dataset_output_tokens else str(output_tokens)
        )

        if args.dataset_type == "sharegpt":
            return sample_sharegpt_requests(args.dataset_path, parallel_num * args.epochs, tokenizer,
                                            None if args.use_real_dataset_output_tokens else output_tokens)
        elif args.dataset_type == "human-eval":
            return sample_human_eval_requests(args.dataset_path, parallel_num * args.epochs,
                                              tokenizer,
                                              None if args.use_real_dataset_output_tokens else output_tokens)
        elif args.dataset_type == "fixed":
            return sample_fixed_requests(args.dataset_path, prompt_tokens, output_tokens,
                                         parallel_num * args.epochs, tokenizer)
        else:
            raise ValueError("Unsupported dataset.")


def check_parameters(args, api_url, tokenizer, output_tokens_list):
    if args.prefix_caching_num > min(args.prompt_tokens):
        raise ValueError("prefix_caching_num must be less than prompt_token.")
    return


def check_validity_and_process_num_prompts(num_prompts, parallel_nums):
    if len(num_prompts) != len(parallel_nums):
        raise ValueError("num_prompts length should equal parallel_num length.")
    div_result = None
    for index, _ in enumerate(num_prompts):
        if parallel_nums[index] == 0:
            raise ValueError("parallel_num parameter has 0.")
        if num_prompts[index] % parallel_nums[index] != 0:
            raise ValueError("num_prompts cannot be divided by parallel_num.")
        current_div_result = num_prompts[index] // parallel_nums[index]
        if div_result is None:
            div_result = current_div_result
        elif div_result != current_div_result:
            raise ValueError("Inconsistent divisive results.")
    return div_result


def get_benchmark_head(args, all_latency_record):
    benchmark_head = ["Input_Length", "Output_Length", "Concurrency",
                      "Total_Token_Throughput(tokens/s)", "Output_Token_Throughput(tokens/s)",
                      "TP75_TTFT(s)", "TP90_TTFT(s)", "TP95_TTFT(s)", "TP99_TTFT(s)", "MAX_TTFT(s)", "AVG_TTFT(s)",
                      "TP75_TPOT(s)", "TP90_TPOT(s)", "TP95_TPOT(s)", "TP99_TPOT(s)", "MAX_TPOT(s)", "AVG_TPOT(s)",
                      "TP90_TPOT_SEC(s)", "TP95_TPOT_SEC(s)", "TP99_TPOT_SEC(s)", "MAX_TPOT_SEC(s)", "AVG_TPOT_SEC(s)",
                      "TP90_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s)", "TP95_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s)",
                      "TP99_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s)", "MIN_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s)",
                      "MAX_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s)", "AVG_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s)",
                      "TP75_E2E(s)", "TP90_E2E(s)", "TP95_E2E(s)", "TP99_E2E(s)", "MAX_E2E(s)", "AVG_E2E(s)",
                      "Total_Time(s)", "QPS", "Fail_Rate", "Backend", "Temperature", "Top_k", "Top_p"]

    if args.use_spec_decode and args.num_speculative_tokens >= 0 and len(all_latency_record[-1]) != len(benchmark_head):
        accept_head = ["TP90_SPEC_ACCEPT_RATE", "TP99_SPEC_ACCEPT_RATE", "MAX_SPEC_ACCEPT_RATE",
                       "MIN_SPEC_ACCEPT_RATE", "AVG_SPEC_ACCEPT_RATE"]
        benchmark_head = benchmark_head + accept_head

    if args.backend == "embedding":
        benchmark_head = ["Input_Length", "Concurrency", "QPS", "INPUT_TOKEN_THROUGHPUT",
                          "TP90_REQ_LATENCY", "TP99_REQ_LATENCY", "MAX_REQ_LATENCY", "AVG_REQ_LATENCY"]

    return benchmark_head


def get_urls(args):
    hosts = args.host.split(",")
    ports = args.port.split(",")
    if len(hosts) == 1:
        api_urls = [get_api_url(args.backend, hosts[0], port, args.url) for port in ports]
    else:
        if len(hosts) != len(ports):
            raise "the number of hosts must be equal to the number of ports"
        api_urls = [get_api_url(args.backend, host, port, args.url) for host, port in zip(hosts, ports)]
    return api_urls


def run_parallel(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_urls = get_urls(args)
    tokenizer = get_tokenizer(args.tokenizer)

    output_tokens_list = args.output_tokens
    if args.backend == "embedding":
        output_tokens_list = [EMBEDDING_OUTPUT_TOKENS] * len(args.prompt_tokens)

    for api_url in api_urls:
        check_parameters(args, api_url, tokenizer, output_tokens_list)
    logger.info("api_urls: %s", api_urls)

    if args.num_prompts:
        args.epochs = check_validity_and_process_num_prompts(args.num_prompts, args.parallel_num)

    logger.info("Warmup ...")
    warm_up_prompt_tokens = WARM_UP_PROMPT_TOKENS if (not args.prefix_caching_num and
                                                      args.dataset_type != "fixed") else args.prompt_tokens[0]
    input_requests = get_dataset_requests(args, tokenizer, warm_up_prompt_tokens, WARM_UP_OUTPUT_TOKENS,
                                          WARM_UP_PARALLEL_NUM)
    asyncio.run(
        benchmark([], args.backend, api_urls, input_requests, args.best_of,
                  args.use_beam_search, args.request_rate, WARM_UP_PARALLEL_NUM, WARM_UP_EPOCHS, args.app_code,
                  args.tokenizer, args.served_model_name, args.num_scheduler_steps, args.use_spec_decode,
                  args.use_pd_separate, args.temperature, args.top_k, args.top_p)
    )

    if args.dataset_type != "random" or args.dataset_type != "fixed":
        logger.info("When use sharegpt or human-eval dataset, the number of the --input-tokens will be ignored.")
    if args.use_real_dataset_output_tokens:
        logger.info("When use --use-real-dataset-output-tokens, the number of the --output-tokens will be ignored.")
    if args.use_spec_decode and args.num_speculative_tokens >= 0:
        logger.info("When enable --use-spec-decode and --num-speculative-tokens >= 0, the acceptance rate of "
                    "speculative inference will be collected. Ensure that --num-speculative-tokens in the benchmark is "
                    "equal to --num-speculative-tokens in the vllm service.")

    # profile start.
    operate_profile(args.profile, "start", app_code=args.app_code, hosts=args.host, ports=args.port,
                    level=args.service_profile_level)

    all_latency_record = []
    benchmark_csv = get_csv_path(args.benchmark_csv)
    for parallel_num in args.parallel_num:
        for i, prompt_tokens in enumerate(args.prompt_tokens):
            output_tokens = output_tokens_list[i]

            input_requests = get_dataset_requests(args, tokenizer, prompt_tokens, output_tokens, parallel_num)

            request_latency_record: List[Tuple[int, int, List]] = []
            asyncio.run(benchmark(request_latency_record, args.backend, api_urls, input_requests, args.best_of,
                                  args.use_beam_search, args.request_rate, parallel_num,
                                  args.epochs, args.app_code, args.tokenizer, args.served_model_name,
                                  args.num_scheduler_steps, args.use_spec_decode, args.use_pd_separate,
                                  args.temperature, args.top_k, args.top_p))
            if args.use_spec_decode:
                update_spec_output_tokens(request_latency_record, args.backend)

            if args.backend == "embedding":
                latency_record = statistics_and_print_embedding_performance_data(args, parallel_num,
                                                                                 request_latency_record,
                                                                                 all_latency_record)
            else:
                latency_record = statistics_and_print_performance_data(args, prompt_tokens, output_tokens, parallel_num,
                                                                       request_latency_record, all_latency_record)
            benchmark_head = get_benchmark_head(args, all_latency_record)
            save_to_csv(benchmark_head, [latency_record], benchmark_csv)

    # profile stop.
    operate_profile(args.profile, "stop", app_code=args.app_code, hosts=args.host, ports=args.port,
                    level=args.service_profile_level)

    if args.dataset_type in ["random", "fixed"]:
        all_latency_record.sort()
    else:
        # the mean input tokens and output tokens of real dataset will be a little diffferent in defferent batch_size,
        # so just sort by batch_size
        all_latency_record.sort(key=lambda x: x[2])

    os.remove(benchmark_csv)
    save_to_csv(benchmark_head, all_latency_record, benchmark_csv)

    logger.info(benchmark_head)
    for latency_record in all_latency_record:
        logger.info(latency_record)
    logger.info("Benchmark parallel finished")


def main(args: argparse.Namespace):
    logger.info(args)
    if args.run_method == "parallel":
        if args.tokenizer is None:
            raise ValueError("When run_method is parallel, tokenizer must be provided.")
        run_parallel(args)
    else:
        if args.providers_path is None or args.dataset_dir is None or args.benchmark_dir is None:
            raise ValueError("When run_method is climbing, "
                             "providers_path, dataset_dir and benchmark_dir must be provided.")
        run_climbing(args)


if __name__ == "__main__":
    uvloop.install()

    parser = argparse.ArgumentParser(
        description="Benchmark the serving prefill performance.")
    parser.add_argument("--backend", type=str, default="openai-chat",
                        choices=["vllm", "mindspore", "base", "tgi", "openai", "trt", "embedding", "openai-chat"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="9288")
    parser.add_argument("--url", type=str, default="")
    parser.add_argument("--app-code", type=str, default=None)
    parser.add_argument("--tokenizer", type=str,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--best-of", type=int, default=1,
                        help="Generates `best_of` sequences per prompt and "
                             "returns the best one.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                             "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process to synthesize "
                             "the request arrival times.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs.")
    parser.add_argument("--parallel-num", nargs='+', type=int, default=[1, 4, 8],
                        help="Number of parallel request number.")
    parser.add_argument("--output-tokens", nargs='+', type=int, default=[256, 256, 256],
                        help="Max tokens to process.")
    parser.add_argument("--prompt-tokens", nargs='+', type=int, default=[512, 1024, 2048],
                        help="Max tokens to process.")
    parser.add_argument("--benchmark-csv", type=str, default="benchmark_parallel.csv",
                        help="Path to the csv.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--served-model-name", type=str, default=None)
    parser.add_argument("--num-scheduler-steps", type=int, default=1)
    parser.add_argument("--prefix-caching-num", type=int, default=0)
    parser.add_argument("--use-spec-decode", action="store_true")
    parser.add_argument("--num-speculative-tokens", type=int, default=-1,
                        help="the step if spec decode, default -1 for disable accept rate statistic.")
    parser.add_argument("--dataset-type", default="random",
                        choices=["random", "fixed", "sharegpt", "human-eval"])
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--use-real-dataset-output-tokens", action="store_true")
    parser.add_argument("--use-pd-separate", type=bool, default=False)
    parser.add_argument("--profile", action="store_true",
                        help="Use Torch Profiler or Service Profiler."
                             "The vLLM service must be started with env VLLM_SERVICE_PROFILER_DIR set.")
    parser.add_argument("--service-profile-level", type=str, required=False,
                        choices=["Level_none", "Level0", "Level1", "Level2"],
                        help="Set Service Profiler level. Support Level_none, Level0, Level1, Level2."
                             "The default value is Level0.")

    # for climbing script
    parser.add_argument("--run-method", type=str, default="parallel",
                        choices=["parallel", "climbing"])
    parser.add_argument("--providers-path", type=str,
                        help="providers configuration file path")
    parser.add_argument("--dataset-dir", type=str,
                        help="test dataset directory")
    parser.add_argument("--benchmark-dir", type=str,
                        help="save benchmark directory")
    parser.add_argument("--server-prefill-concurrency", type=int, help="Number of prefill request number.")
    parser.add_argument("--server-decode-concurrency", type=int, help="Number of decode request number.")
    parser.add_argument("--control-method", choices=["queue", "round", "pd-adaptive"], default="queue",
                        help="the parallel function")
    parser.add_argument("--growth-rate", type=int)
    parser.add_argument("--use-mtp-accept-rate", type=bool, default=True,
                        help="the accept rate, mtp style accept rate will ignore the score model output token.")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Float that controls the randomness of the sampling. Zero means greedy sampling.")
    parser.add_argument("--top-k", type=int, default=-1,
                        help="Integer that controls the number of top tokens to consider. "
                             "Set to -1 to consider all tokens.")
    parser.add_argument("--top-p", type=float, default=1,
                        help="Float that controls the cumulative probability of the top tokens to consider. "
                             "Must be in (0, 1]. Set to 1 to consider all tokens.")
    parser.add_argument("--server-statistics", action="store_true", help="Enabling Server Information Statistics")
    parser.add_argument("--rounds", type=int, help="the round of batch")
    parser.add_argument("--num-prompts", nargs='+', type=int,
                        help="Number of prompts to process.")
    parser.add_argument("--climbing-period", type=int, default=1000,
                        help="Time of intervals for the climbing, The default value is 1000 milliseconds.")
    args_global = parser.parse_args()
    main(args_global)
