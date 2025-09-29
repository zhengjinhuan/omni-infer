import argparse
import asyncio
import random
import time
from typing import AsyncGenerator, List, Tuple
import sys
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from utils import (get_api_url, get_request_data, generate_random_image)
from benchmark_utils import (generate_str, get_tokenizer, do_request,
                             IS_DEBUG, plot_time_record, save_to_csv,
                             statistics_and_print_performance_data)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sample_requests(
        image_shape: Tuple,
        tokenizer: PreTrainedTokenizerBase,
        prompt_tokens: int,
        output_tokens: int,
) -> List[Tuple[str, int, int, str]]:
    prompt = generate_str(tokenizer, prompt_tokens)
    seed, height, width = image_shape
    image_base64 = generate_random_image(seed, height, width)
    return [(prompt, prompt_tokens, output_tokens, image_base64)]


async def get_request(
        input_requests: List[Tuple[str, int, int, str]],
        request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int, str], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
        image_base64,
        request_latency_record: List,
        api_url: str,
        prompt: str,
        prompt_len: int,
        output_len: int,
        app_code: str = None,
        model: str = None,
        served_model_name: str = None,
        num_scheduler_steps: int = 1,
) -> None:
    headers, pload, confirm_error_output = get_request_data(image_base64,
                                                            prompt,
                                                            output_len,
                                                            app_code,
                                                            model,
                                                            served_model_name)
    # num_scheduler_steps is set to 1 by default, without using multi-step.
    time_record, _ = await do_request(api_url, headers, pload, confirm_error_output, output_len, num_scheduler_steps)
    output_tokens = len(time_record) - 1

    if output_tokens < output_len:
        logger.error(f"output_tokens: %d < output_len: %d", output_tokens, output_len)

    request_latency_record.append((prompt_len, output_len, time_record, None))


async def benchmark(
        request_latency_record: List,
        api_url: str,
        input_requests: List[Tuple[str, int, int, str]],
        request_rate: float,
        parallel_num: int,
        epochs: int,
        app_code: str = None,
        model: str = None,
        served_model_name: str = None, 
        num_scheduler_steps: int = 1,
) -> None:
    input_index = 0
    for ep in tqdm(range(epochs), desc="epoch"):
        input_parallel = []
        for id in range(parallel_num):
            input_parallel.append(input_requests[input_index])
            input_index += 1
            if input_index >= len(input_requests):
                input_index = 0

        tasks: List[asyncio.Task] = []
        async for request in get_request(input_parallel, request_rate):
            prompt, prompt_len, output_len, image_base64 = request
            task = asyncio.create_task(send_request(
                image_base64,
                request_latency_record,
                api_url,
                prompt,
                prompt_len,
                output_len,
                app_code,
                model,
                served_model_name,
                num_scheduler_steps))
            tasks.append(task)
        await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = get_api_url(args.host, args.port, args.url)
    tokenizer = get_tokenizer(args.tokenizer)

    logger.info(f"Warmup ...")
    input_requests = sample_requests((args.seed, args.height, args.width),
                                     tokenizer, args.prompt_tokens[0], args.output_tokens[0])
    asyncio.run(
        benchmark([], api_url, input_requests, args.request_rate,
                  4, 1, args.app_code, args.tokenizer, args.served_model_name, args.num_scheduler_steps)
    )

    all_latency_record = []
    for parallel_num in args.parallel_num:
        for i, prompt_tokens in enumerate(args.prompt_tokens):
            output_tokens = args.output_tokens[i]
            logger.info(
                f"Benchmark running with parallel_num: %d, "
                f"prompt_tokens: %d, output_tokens: %d",
                parallel_num, prompt_tokens, output_tokens
            )

            input_requests = sample_requests((args.seed, args.height, args.width),
                                             tokenizer, prompt_tokens, output_tokens)

            request_latency_record: List[Tuple[int, int, List]] = []
            asyncio.run(benchmark(request_latency_record, api_url, input_requests,
                                  args.request_rate, parallel_num, args.epochs, args.app_code,
                                  args.tokenizer, args.served_model_name, args.num_scheduler_steps))

            statistics_and_print_performance_data(args, prompt_tokens, output_tokens, parallel_num,
                                   request_latency_record, all_latency_record)

    all_latency_record.sort()

    benchmark_head = ["输入长度", "输出长度", "并发数", "image_shape(h,w)", "输出tokens总吞吐",
                      "首tokens时延P90（ms）", "首tokens时延P99（ms）", "最大首tokens时延（ms）", "平均首tokens时延（ms）",
                      "增量时延P90（ms）", "增量时延P99（ms）", "最大增量时延（ms）", "平均增量时延（ms）", "端到端时延P90（s）", "端到端时延p95（s）",
                      "端到端时延P99（s）", "最大端到端时延（s）", "平均端到端时延（s）"]

    save_to_csv(benchmark_head, all_latency_record, args.benchmark_csv)

    logger.info("Benchmark parallel finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the serving prefill performance.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9288)
    parser.add_argument("--url", type=str, default="")
    parser.add_argument("--height", type=int, default=256, help="image height")
    parser.add_argument("--width", type=int, default=256, help="image width")
    parser.add_argument("--app-code", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
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
                        help="Input tokens.")
    parser.add_argument("--prompt-tokens", nargs='+', type=int, default=[512, 1024, 2048],
                        help="Output tokens.")
    parser.add_argument("--benchmark-csv", type=str, default="benchmark_parallel.csv",
                        help="Path to the csv.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--served-model-name", type=str, default=None)
    parser.add_argument("--num-scheduler-steps", type=int, default=1)
    parser.add_argument("--use-pd-separate", type=bool, default=False)

    args_global = parser.parse_args()
    main(args_global)
