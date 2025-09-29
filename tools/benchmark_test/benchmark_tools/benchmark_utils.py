import asyncio
import csv
import json
import logging
import os
import stat
import time
from typing import Union

import aiohttp
import numpy as np
import requests
from tqdm import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
IS_DEBUG = int(os.environ.get("BENCHMARK_DEBUG", 0))
EPOCH_NUM = 10
FAIL_RATE = 0
DEFAULT_TPOT = 0
# The size of the data block returned in each iteration is not greater than 8192. Therefore, chunk_size is 8192.
CHUNK_SIZE = 8192
TIMEOUT = int(os.environ.get("BENCHMARK_TIMEOUT", 5 * 3600))
SLEEP_TIME = 10
MS_SCALE = 1000
LATENCY_RESERVATION_BITS = 3
THROUGHPUT_RESERVATION_BITS = 2
TP75 = 75
TP90 = 90
TP95 = 95
TP99 = 99

if IS_DEBUG:
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt


def get_tokenizer(
        transformer_tokenizer_path: str,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    tokenizer = AutoTokenizer.from_pretrained(transformer_tokenizer_path, trust_remote_code=True)

    return tokenizer


def generate_hello_str(tokenizer, length, hello_token="Hello"):
    text = hello_token * (length - 1)
    completion_token_ids = tokenizer([text]).input_ids
    while len(completion_token_ids[0]) < length:
        text += hello_token
        completion_token_ids = tokenizer([text]).input_ids

    return text


def extract_str(tokenizer, origin_text, length):
    text = origin_text[0:length]
    completion_token_ids = tokenizer([text]).input_ids
    if IS_DEBUG:
        logger.info("len(completion_token_ids[0]) %d, length %d ", len(completion_token_ids[0]), length)

    epoch = EPOCH_NUM
    while len(completion_token_ids[0]) != length and epoch > 0:
        while len(completion_token_ids[0]) > length:
            diff = len(completion_token_ids[0]) - length
            end = len(text) - diff
            text = origin_text[0:end]
            completion_token_ids = tokenizer([text]).input_ids
            if IS_DEBUG:
                logger.info("len(completion_token_ids[0]) %d, %d ", len(completion_token_ids[0]), length)

        while len(completion_token_ids[0]) < length:
            diff = length - len(completion_token_ids[0])
            end = len(text) + diff
            if end > len(origin_text):
                origin_text = origin_text * 2
            text = origin_text[0:end]
            completion_token_ids = tokenizer([text]).input_ids
            if IS_DEBUG:
                logger.info("len(completion_token_ids[0]) %d, %d ", len(completion_token_ids[0]), length)

        epoch -= 1
    if len(completion_token_ids[0]) != length:
        text = generate_hello_str(tokenizer, length)

    if IS_DEBUG:
        logger.info(text)
    return text


def generate_str(tokenizer, length):
    vocab_size = tokenizer.vocab_size
    ids = np.random.randint(vocab_size / 4, vocab_size / 3, length)
    text = tokenizer.decode(ids)

    completion_token_ids = tokenizer([text]).input_ids
    if IS_DEBUG:
        logger.info("len(completion_token_ids[0]) %d, length %d ", len(completion_token_ids[0]), length)

    epoch = EPOCH_NUM
    while len(completion_token_ids[0]) != length and epoch > 0:
        while len(completion_token_ids[0]) > length:
            diff = len(completion_token_ids[0]) - length
            now_length = ids.shape[0] - diff
            ids = ids[:now_length]
            text = tokenizer.decode(ids)
            completion_token_ids = tokenizer([text]).input_ids
            if IS_DEBUG:
                logger.info("len(completion_token_ids[0]) %d, %d ", len(completion_token_ids[0]), length)

        while len(completion_token_ids[0]) < length:
            diff = length - len(completion_token_ids[0])
            diff_ids = np.random.randint(vocab_size / 4, vocab_size / 3, diff)
            ids = np.append(ids, diff_ids)
            text = tokenizer.decode(ids)
            completion_token_ids = tokenizer([text]).input_ids
            if IS_DEBUG:
                logger.info("len(completion_token_ids[0]) %d, %d ", len(completion_token_ids[0]), length)

        epoch -= 1

    if len(completion_token_ids[0]) != length:
        text = generate_hello_str(tokenizer, length)

    if IS_DEBUG:
        logger.info(text)
    return text


def print_data_info(dataset_path, tokenizer):
    with open(dataset_path, "r") as f:
        text_data = [item["input"] for item in json.load(f)]
    length_list = [len(text) for text in text_data]
    token_length_list = [len(tokenizer([text]).input_ids[0]) for text in text_data]

    tp90_length = np.percentile(length_list, 90)
    tp99_length = np.percentile(length_list, 99)
    min_length = np.min(length_list)
    max_length = np.max(length_list)
    avg_length = np.mean(length_list)
    tp90_token_length = np.percentile(token_length_list, 90)
    tp99_token_length = np.percentile(token_length_list, 99)
    min_token_length = np.min(token_length_list)
    max_token_length = np.max(token_length_list)
    avg_token_length = np.mean(token_length_list)

    print(f"\n", flush=True)
    print(f'length: {len(text_data)}')
    print(f'tp90_length: {tp90_length}')
    print(f'tp99_length: {tp99_length}')
    print(f'min_length: {min_length}')
    print(f'max_length: {max_length}')
    print(f'avg_length: {avg_length}')
    print(f"\n", flush=True)
    print(f'tp90_token_length: {tp90_token_length}')
    print(f'tp99_token_length: {tp99_token_length}')
    print(f'min_token_length: {min_token_length}')
    print(f'max_token_length: {max_token_length}')
    print(f'avg_token_length: {avg_token_length}')
    print(f"---------------------------\n", flush=True)


def get_api_url(backend, host, port, url):
    if url is not None and len(url) > 0:
        return url

    if backend == "mindspore":
        api_url = f"http://{host}:{port}/models/llama2/generate"
    elif backend == "base":
        api_url = f"http://{host}:{port}/v1/generate"
    elif backend == "tgi":
        api_url = f"https://{host}:{port}/generate_stream"
    elif backend == "openai":
        api_url = f"http://{host}:{port}/v1/completions"
    elif backend == "openai-chat":
        api_url = f"http://{host}:{port}/v1/chat/completions"
    elif backend == "trt":
        api_url = f"http://{host}:{port}/v2/models/ensemble/generate_stream"
    elif backend == "embedding":
        api_url = f"http://{host}:{port}/v1/embeddings"
    else:
        api_url = f"http://{host}:{port}/generate"
    return api_url


def get_request_data(
        backend: str,
        prompt: str,
        prompt_len: int,
        output_len: int,
        best_of: int,
        use_beam_search: bool,
        app_code: str = None,
        model: str = None,
        served_model_name: str = None,
        use_spec_decode: bool = False,
        temperature: float = 0,
        top_k: int = -1,
        top_p: float = 1,
):
    headers = {"User-Agent": "Benchmark Client"}
    if app_code:
        headers = {"User-Agent": "Benchmark Client",
                   'Content-Type': 'application/json',
                   'X-Apig-AppCode': app_code}
    if served_model_name is None:
        served_name = model
    else:
        served_name = served_model_name
    backend_config = {
        "vllm": {
            "pload": {
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": output_len,
                "ignore_eos": True,
                "stream": True,
            },
            "confirm_error_output": True
        },
        "openai": {
            "pload": {
                "prompt": prompt,
                "temperature": 0,
                "top_p": 0.8,
                "top_k": 5,
                "max_tokens": output_len,
                "ignore_eos": True,
                "model": served_name,
                "stream": True,
                "stream_options": {"include_usage": True,
                                   "continuous_usage_stats": True}
            },
            "confirm_error_output": False
        },
        "openai-chat": {
            "pload": {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": output_len,
                "ignore_eos": True,
                "model": served_name,
                "stream": True,
                "stream_options": {"include_usage": True,
                                   "continuous_usage_stats": True}
            },
            "confirm_error_output": False
        },
        "embedding": {
            "pload": {
                "input": prompt,
                "model": served_name,
            },
            "confirm_error_output": False
        },
        "mindspore": {
            "pload": {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": output_len,
                    "do_sample": False,
                    "ignore_eos": True,
                    "return_full_text": False
                }
            },
            "confirm_error_output": False
        },
        "base": {
            "pload": {
                "prompt": prompt,
                "max_tokens": prompt_len + output_len,
                "model_name": "llama2",
                "do_sample": False,
                "stream": True,
                "debug": 2
            },
            "confirm_error_output": False
        },
        "tgi": {
            "pload": {
                "inputs": prompt,
                "parameters": {
                    "best_of": best_of,
                    "max_new_tokens": output_len,
                    "do_sample": False,
                    "ignore_eos_token": True,
                    "decoder_input_details": False
                }
            },
            "confirm_error_output": True
        },
        "trt": {
            "headers": {"Content-Type": "text/event-stream; charset=utf-8"},
            "pload": {
                "text_input": prompt,
                "parameters": {
                    "max_tokens": output_len,
                    "min_length": output_len,
                    "bad_words": "",
                    "stop_words": "",
                    "ignore_eos": True,
                    "stream": True
                }
            },
            "confirm_error_output": True
        }
    }

    if backend not in backend_config:
        raise ValueError(f"Unknown backend: {backend}")

    config = backend_config[backend]

    if use_spec_decode:
        config['pload']['skip_special_tokens'] = False

    if 'headers' in config:
        headers.update(config['headers'])

    return headers, config['pload'], config['confirm_error_output']


def split_chunk(input):
    res = []
    chunks = input.split(b'\n\n')
    if len(chunks) <= 2:
        return [input]
    for c in chunks:
        if len(c) == 0:
            continue
        res.append(c + b'\n\n')
    return res


async def do_request(api_url, headers, pload, confirm_error_output, output_len, num_scheduler_steps,
                     backend=None, use_spec_decode=False):
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    first_token = True
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(ssl=False)) as session:
        while True:
            last_chunk = None
            prefill_start_time = time.perf_counter()
            time_record = [prefill_start_time]
            chunk_record = []
            async with session.post(api_url, headers=headers, json=pload) as response:
                response.raise_for_status()
                async for chunk_bytes in response.content:
                    # 调用独立函数处理每个 chunk
                    last_chunk, first_token, output_len, chunk_record, time_record = process_chunk(
                        chunk_bytes, first_token, backend, use_spec_decode, output_len, num_scheduler_steps,
                        chunk_record, time_record
                    )

            if confirm_error_output:
                if last_chunk is None:
                    break
                if last_chunk.startswith('data:'):
                    output = last_chunk[5:].strip()
                else:
                    output = last_chunk.strip().rstrip("\0")

                if IS_DEBUG:
                    logger.info(output)
                if output == '[DONE]':
                    break
                try:
                    output = json.loads(output)
                except Exception:
                    logger.error("Exception")
                    break

                # Re-send the request if it failed.
                if "error" not in output:
                    break
                else:
                    logger.error("request failed, %s, retry", output)
                    await asyncio.sleep(SLEEP_TIME)
            else:
                break
        return time_record, chunk_record


def process_chunk(chunk_bytes, first_token, backend, use_spec_decode, output_len, num_scheduler_steps,
                 chunk_record, time_record):
    """
    处理每个响应的 chunk 数据。

    Args:
        chunk_bytes: 当前处理的字节块
        first_token: 是否是第一个处理的 token
        backend: 指定使用的后端类型
        use_spec_decode: 是否进行特殊解码处理
        output_len: 输出的长度限制
        num_scheduler_steps: 调度器的步数
        chunk_record: 存储处理后的内容块
        time_record: 记录处理时间的时间戳列表

    Returns:
        last_chunk: 最后一个处理的 chunk
        first_token: 更新后的 first_token 状态
        output_len: 更新后的输出长度
        chunk_record: 更新后的 chunk 记录列表
        time_record: 更新后的时间记录列表
    """
    chunk_bytes = chunk_bytes.strip()
    if not chunk_bytes:
        return None, first_token, output_len, chunk_record, time_record

    chunk_bytes = chunk_bytes.decode("utf-8")

    # NOTE: Sometimes TGI returns a ping response without any data, we should skip it.
    if chunk_bytes.startswith(":"):
        return None, first_token, output_len, chunk_record, time_record

    # openai-chat 第一个 chunk 内容为空
    if backend == "openai-chat" and first_token:
        json_data = json.loads(chunk_bytes[len("data: "):])
        delta = json_data["choices"][0]["delta"] if len(json_data["choices"]) > 0 else {}
        if "content" in delta and not delta["content"]:
            return None, first_token, output_len, chunk_record, time_record

    if len(chunk_bytes.strip()) > 0:
        last_chunk = chunk_bytes
        return_token_num = 1 if first_token else min(num_scheduler_steps, output_len)
        time_record.extend([time.perf_counter()] * return_token_num)
        first_token = False
        output_len -= return_token_num
    else:
        last_chunk = None

    # for maximum block depth
    if len(chunk_bytes.strip()) > 0 and use_spec_decode:
        chunk_record.append(last_chunk)

    return last_chunk, first_token, output_len, chunk_record, time_record


def statistics_and_print_embedding_performance_data(args, parallel_num, request_latency_record, all_latency_record):
    benchmark_start_time = np.min([time_record[0] for _, _, time_record, _ in request_latency_record])
    benchmark_end_time = np.max([time_record[-1] for _, _, time_record, _ in request_latency_record])
    benchmark_time = benchmark_end_time - benchmark_start_time
    logger.info("所有请求耗时: %.4f s", benchmark_time)

    benchmark_requests = args.epochs * parallel_num / benchmark_time
    logger.info("请求吞吐: %.4f requests/s", benchmark_requests)

    total_prompt_tokens = np.sum([
        prompt_len
        for prompt_len, _, _, _ in request_latency_record
    ])
    total_prompt_token_throughput = total_prompt_tokens / benchmark_time
    logger.info("输入tokens总吞吐: %.4f tokens/s", total_prompt_token_throughput)

    req_latency_list = [
        time_record[-1] - time_record[0]
        for _, _, time_record, _ in request_latency_record
    ]

    p90_req_latency = np.percentile(req_latency_list, 90) * MS_SCALE
    logger.info("请求时延TP90: %.4f ms", p90_req_latency)

    p99_req_latency = np.percentile(req_latency_list, 99) * MS_SCALE
    logger.info("请求时延TP99: %.4f ms", p99_req_latency)

    max_req_latency = np.max(req_latency_list) * MS_SCALE
    logger.info("最大请求时延: %.4f ms", max_req_latency)

    avg_req_latency = np.mean(req_latency_list) * MS_SCALE
    logger.info("平均请求时延: %.4f ms", avg_req_latency)

    avg_prompt_token = np.mean([prompt_len for prompt_len, _, _, _ in request_latency_record])

    latency_record = (avg_prompt_token, parallel_num,
                      benchmark_requests, total_prompt_token_throughput,
                      p90_req_latency, p99_req_latency, max_req_latency, avg_req_latency)

    time.sleep(SLEEP_TIME)

    all_latency_record.append(latency_record)

    return latency_record


def statistics_and_print_performance_data(args, prompt_tokens, output_tokens, parallel_num,
                                          request_latency_record, all_latency_record):
    benchmark_start_time = np.min([time_record[0] for _, _, time_record, _ in request_latency_record])
    benchmark_end_time = np.max([time_record[-1] for _, _, time_record, _ in request_latency_record])
    benchmark_time = round(benchmark_end_time - benchmark_start_time, LATENCY_RESERVATION_BITS)
    logger.info("Total_Time: %.3f s", benchmark_time)

    benchmark_requests = round(args.epochs * parallel_num / benchmark_time, THROUGHPUT_RESERVATION_BITS)
    logger.info("QPS: %.2f requests/s", benchmark_requests)

    total_output_tokens = np.sum([
        output_len
        for _, output_len, _, _ in request_latency_record
    ])
    total_output_token_throughput = round(total_output_tokens / benchmark_time, THROUGHPUT_RESERVATION_BITS)
    logger.info("Output_Token_Throughput: %.2f tokens/s", total_output_token_throughput)

    total_tokens = np.sum([
        prompt_len + output_len
        for prompt_len, output_len, _, _ in request_latency_record
    ])
    total_token_throughput = round(total_tokens / benchmark_time, THROUGHPUT_RESERVATION_BITS)
    logger.info("Total_Token_Throughput: %.2f tokens/s", total_token_throughput)

    prefill_latency_list = [
        time_record[1] - time_record[0]
        for _, _, time_record, _ in request_latency_record
    ]

    p75_prefill_latency = round(np.percentile(prefill_latency_list, TP75), LATENCY_RESERVATION_BITS)
    logger.info("TP75_TTFT: %.3f s", p75_prefill_latency)

    p90_prefill_latency = round(np.percentile(prefill_latency_list, TP90), LATENCY_RESERVATION_BITS)
    logger.info("TP90_TTFT: %.3f s", p90_prefill_latency)

    p95_prefill_latency = round(np.percentile(prefill_latency_list, TP95), LATENCY_RESERVATION_BITS)
    logger.info("TP95_TTFT: %.3f s", p95_prefill_latency)

    p99_prefill_latency = round(np.percentile(prefill_latency_list, TP99), LATENCY_RESERVATION_BITS)
    logger.info("TP99_TTFT: %.3f s", p99_prefill_latency)

    max_prefill_latency = round(np.max(prefill_latency_list), LATENCY_RESERVATION_BITS)
    logger.info("MAX_TTFT: %.3f s", max_prefill_latency)

    avg_prefill_latency = round(np.mean(prefill_latency_list), LATENCY_RESERVATION_BITS)
    logger.info("AVG_TTFT: %.3f s", avg_prefill_latency)

    tpot_list = [
        (time_record[-1] - time_record[1]) / (output_len - 1) if output_len > 1 else DEFAULT_TPOT
        for _, output_len, time_record, _ in request_latency_record
    ]

    p75_tpot = round(np.percentile(tpot_list, TP75), LATENCY_RESERVATION_BITS)
    logger.info("TP75_TPOT from first token: %.3f s", p75_tpot)

    p90_tpot = round(np.percentile(tpot_list, TP90), LATENCY_RESERVATION_BITS)
    logger.info("TP90_TPOT from first token: %.3f s", p90_tpot)

    p95_tpot = round(np.percentile(tpot_list, TP95), LATENCY_RESERVATION_BITS)
    logger.info("TP95_TPOT from first token: %.3f s", p95_tpot)

    p99_tpot = round(np.percentile(tpot_list, TP99), LATENCY_RESERVATION_BITS)
    logger.info("TP99_TPOT from first token: %.3f s", p99_tpot)

    max_tpot = round(np.max(tpot_list), LATENCY_RESERVATION_BITS)
    logger.info("MAX_TPOT from first token: %.3f s", max_tpot)

    avg_tpot = round(np.mean(tpot_list), LATENCY_RESERVATION_BITS)
    logger.info("AVG_TPOT from first token: %.3f s", avg_tpot)

    tpot_second_list = [
        (time_record[-1] - time_record[2]) / (output_len - 2) if output_len > 2 else DEFAULT_TPOT
        for _, output_len, time_record, _ in request_latency_record
    ]
    p90_tpot_second = round(np.percentile(tpot_second_list, TP90), LATENCY_RESERVATION_BITS)
    logger.info("TP90_TPOT from second token: %.3f s", p90_tpot_second)

    p95_tpot_second = round(np.percentile(tpot_second_list, TP95), LATENCY_RESERVATION_BITS)
    logger.info("TP95_TPOT from second token: %.3f s", p95_tpot_second)

    p99_tpot_second = round(np.percentile(tpot_second_list, TP99), LATENCY_RESERVATION_BITS)
    logger.info("TP99_TPOT from second token: %.3f s", p99_tpot_second)

    max_tpot_second = round(np.max(tpot_second_list), LATENCY_RESERVATION_BITS)
    logger.info("MAX_TPOT from second token: %.3f s", max_tpot_second)

    avg_tpot_second = round(np.mean(tpot_second_list), LATENCY_RESERVATION_BITS)
    logger.info("AVG_TPOT from second token: %.3f s", avg_tpot_second)

    time_between_first_and_second_token_list = [
        time_record[2] - time_record[1] if output_len > 1 else DEFAULT_TPOT
        for _, output_len, time_record, _ in request_latency_record
    ]

    p90_time_between_first_and_second_token = round(np.percentile(time_between_first_and_second_token_list, TP90),
                                                    LATENCY_RESERVATION_BITS)
    logger.info("TP90 time between first and second token: %.3f s", p90_time_between_first_and_second_token)

    p95_time_between_first_and_second_token = round(np.percentile(time_between_first_and_second_token_list, TP95),
                                                    LATENCY_RESERVATION_BITS)
    logger.info("TP95 time between first and second token: %.3f s", p95_time_between_first_and_second_token)

    p99_time_between_first_and_second_token = round(np.percentile(time_between_first_and_second_token_list, TP99),
                                                    LATENCY_RESERVATION_BITS)
    logger.info("TP99 time between first and second token: %.3f s", p99_time_between_first_and_second_token)

    min_time_between_first_and_second_token = round(np.min(time_between_first_and_second_token_list),
                                                    LATENCY_RESERVATION_BITS)
    logger.info("Min time between first and second token: %.3f s", min_time_between_first_and_second_token)

    max_time_between_first_and_second_token = round(np.max(time_between_first_and_second_token_list),
                                                    LATENCY_RESERVATION_BITS)
    logger.info("Max time between first and second token: %.3f s", max_time_between_first_and_second_token)

    avg_time_between_first_and_second_token = round(np.mean(time_between_first_and_second_token_list),
                                                    LATENCY_RESERVATION_BITS)
    logger.info("AVG time between first and second token: %.3f s", avg_time_between_first_and_second_token)

    e2e_latency_list = [
        time_record[-1] - time_record[0]
        for _, _, time_record, _ in request_latency_record
    ]

    p75_e2e_latency = round(np.percentile(e2e_latency_list, TP75), LATENCY_RESERVATION_BITS)
    logger.info("TP75_E2E: %.3f s", p75_e2e_latency)

    p90_e2e_latency = round(np.percentile(e2e_latency_list, TP90), LATENCY_RESERVATION_BITS)
    logger.info("TP90_E2E: %.3f s", p90_e2e_latency)

    p95_e2e_latency = round(np.percentile(e2e_latency_list, TP95), LATENCY_RESERVATION_BITS)
    logger.info("TP95_E2E: %.3f s", p95_e2e_latency)

    p99_e2e_latency = round(np.percentile(e2e_latency_list, TP99), LATENCY_RESERVATION_BITS)
    logger.info("TP99_E2E: %.3f s", p99_e2e_latency)

    max_e2e_latency = round(np.max(e2e_latency_list), LATENCY_RESERVATION_BITS)
    logger.info("MAX_E2E: %.3f s", max_e2e_latency)

    avg_e2e_latency = round(np.mean(e2e_latency_list), LATENCY_RESERVATION_BITS)
    logger.info("AVG_E2E: %.3f s", avg_e2e_latency)

    if IS_DEBUG:
        plot_time_record(benchmark_start_time, benchmark_time, request_latency_record,
                         f"{parallel_num}_{prompt_tokens}_{output_tokens}.jpg")

    avg_prompt_token = np.mean([prompt_len for prompt_len, _, _, _ in request_latency_record])
    avg_output_token = np.mean([output_len for _, output_len, _, _ in request_latency_record])

    if getattr(args, "height", None) is None:
        latency_record = (avg_prompt_token, avg_output_token, parallel_num,
                          total_token_throughput, total_output_token_throughput,
                          p75_prefill_latency, p90_prefill_latency, p95_prefill_latency, p99_prefill_latency,
                          max_prefill_latency, avg_prefill_latency,
                          p75_tpot, p90_tpot, p95_tpot, p99_tpot, max_tpot, avg_tpot,
                          p90_tpot_second, p95_tpot_second, p99_tpot_second, max_tpot_second, avg_tpot_second,
                          p90_time_between_first_and_second_token, p95_time_between_first_and_second_token,
                          p99_time_between_first_and_second_token, min_time_between_first_and_second_token,
                          max_time_between_first_and_second_token, avg_time_between_first_and_second_token,
                          p75_e2e_latency, p90_e2e_latency, p95_e2e_latency, p99_e2e_latency, max_e2e_latency,
                          avg_e2e_latency,
                          benchmark_time, benchmark_requests, FAIL_RATE, args.backend, args.temperature, args.top_k,
                          args.top_p)
    else:
        image_shape = str(args.height) + ',' + str(args.width)
        latency_record = (avg_prompt_token, avg_output_token, parallel_num, image_shape,
                          total_output_token_throughput,
                          p90_prefill_latency, p99_prefill_latency, max_prefill_latency, avg_prefill_latency,
                          p90_tpot, p99_tpot, max_tpot, avg_tpot,
                          p90_e2e_latency, p95_e2e_latency, p99_e2e_latency, max_e2e_latency, avg_e2e_latency)

    # If the benchmark backend supports speculative inference, request_latency_record is replaced with output_step,
    # which is an int value. Otherwise, the original chunk_list is retained as a list.
    is_spec_support_backend = isinstance(request_latency_record[0][-1], int)
    if getattr(args, "use_spec_decode", False) and getattr(args, "num_speculative_tokens", -1) >= 0 \
            and is_spec_support_backend:
        if getattr(args, "use_mtp_accept_rate", False):
            accept_rate_list = [(output_len - 1 - (output_step - 1) * args.num_scheduler_steps) / (
                    (output_step - 1) * args.num_speculative_tokens * args.num_scheduler_steps)
                                if output_step > 1 else 0.0
                                for _, output_len, _, output_step in request_latency_record]
        else:
            accept_rate_list = [(output_len - 1) / ((output_step - 1) * (
                    args.num_speculative_tokens + 1) * args.num_scheduler_steps)
                                if output_step > 1 else 0.0
                                for _, output_len, _, output_step in request_latency_record]

        p90_accept_rate = np.percentile(accept_rate_list, 90)
        logger.info("TP90 Speculative acceptance rate: %.4f", p90_accept_rate)

        p99_accept_rate = np.percentile(accept_rate_list, 99)
        logger.info("TP99 Speculative acceptance rate: %.4f", p99_accept_rate)

        max_accept_rate = np.max(accept_rate_list)
        logger.info("MAX Speculative acceptance rate: %.2f", max_accept_rate)

        min_accept_rate = np.min(accept_rate_list)
        logger.info("MIN Speculative acceptance rate: %.2f", min_accept_rate)

        avg_accept_rate = np.mean(accept_rate_list)
        logger.info("AVG Speculative acceptance rate: %.2f", avg_accept_rate)

        accept_rate_record = (p90_accept_rate, p99_accept_rate, max_accept_rate, min_accept_rate, avg_accept_rate)

        latency_record = latency_record + accept_rate_record

    time.sleep(SLEEP_TIME)

    all_latency_record.append(latency_record)

    return latency_record


def plot_time_record(benchmark_start_time, benchmark_time, request_latency_record, name="parallel.jpg"):
    def newline(ax, p1, p2, color='skyblue'):
        line = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=20, markersize=100, marker=".",
                             markerfacecolor=color)
        ax.add_line(line)

    fig_size_x = 256
    fig_size_y = 128
    fig, ax = plt.subplots(1, 1, figsize=(fig_size_x, fig_size_y), facecolor='#f7f7f7', dpi=80)
    time_records = [time_record for _, _, time_record in request_latency_record]
    time_records = (time_records - benchmark_start_time) * MS_SCALE
    for idx, time_record in enumerate(tqdm(time_records, desc="plot_time_record")):
        idx = idx * 1
        newline(ax, [time_record[0], idx], [time_record[1], idx], color='red')
        for start, end in zip(time_record[1:-1], time_record[2:]):
            newline(ax, [start, idx], [end, idx])

    ax.set_facecolor('#f7f7f7')
    ax.set(xlim=(0, (benchmark_time * MS_SCALE) + 10), ylim=(-1, len(time_records) * 1), ylabel='request')
    font_size = round(fig_size_x / 3)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel('time', fontsize=fig_size_x)
    plt.ylabel('request', fontsize=fig_size_x)
    logger.info(f"save fig ...")
    plt.savefig(name)


def get_csv_path(csv_path):
    sprtr_idx = csv_path.rfind('/')
    if sprtr_idx > 0:
        csv_dir = csv_path[0:sprtr_idx]
        if len(csv_dir) > 1:
            os.makedirs(csv_dir, exist_ok=True)

    # add timeStamp
    timestamp = time.strftime("%Y%m%d%H%MXS", time.localtime())
    dot_index = csv_path.rfind(".")
    csv_path = csv_path[0:dot_index] + f'_{timestamp}' + csv_path[dot_index:]

    return csv_path


def save_to_csv(benchmark_head, records, csv_path):
    # 设置文件打开的标志
    flags = os.O_WRONLY | os.O_CREAT
    # 设置文件权限，仅授予文件所有者读写权限
    mode = stat.S_IWUSR | stat.S_IRUSR
    # 使用 os.open 打开文件描述符
    fd = os.open(csv_path, flags, mode)
    with os.fdopen(fd, 'a', encoding='utf-8-sig', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if csv_file.tell() == 0:
            writer.writerow(benchmark_head)
        for items in records:
            to_csv = []
            for item in items:
                if isinstance(item, float):
                    item = round(item, 4)
                to_csv.append(item)
            writer.writerow(to_csv)


def remove_endpoint_from_base_url(url):
    """
    移除url的endpoint，只保留基础 URL
    查找url中第一个出现在第8个字符之后（即从 "http://" 或 "https://" 之后）的斜杠（'/'）的位置，并返回从该位置之前的字符串
    """
    index = url.find('/', 8)
    if index != -1:
        return url[:index]
    return url


def get_profile_level_url_parameter(service_profile_level):
    return f"?level={service_profile_level}" if service_profile_level else ""


def construct_url(host, port, action, service_profile_level, base_url=None):
    if base_url:
        url = f"{base_url}/{action}"
    else:
        url = f"http://{host}:{port}/{action}"
    if action == "start_profile":
        url += get_profile_level_url_parameter(service_profile_level)
    return url


def get_profile_urls(hosts, ports, service_profile_level, base_url=None):
    if base_url:
        profile_url = remove_endpoint_from_base_url(base_url)
        start_profile_url = [construct_url(None, None, "start_profile", service_profile_level,
                                           base_url=profile_url)]
        stop_profile_url = [construct_url(None, None, "stop_profile", service_profile_level,
                                          base_url=profile_url)]
        return [start_profile_url, stop_profile_url]

    hosts = hosts.split(",")
    ports = ports.split(",")
    if len(hosts) == 1:
        hosts = hosts * len(ports)
    elif len(hosts) != len(ports):
        raise ValueError("the number of hosts must be equal to the number of ports")

    start_profile_urls = [construct_url(host, port, "start_profile", service_profile_level) for host, port in
                          zip(hosts, ports)]
    stop_profile_urls = [construct_url(host, port, "stop_profile", service_profile_level) for host, port in
                         zip(hosts, ports)]
    return [start_profile_urls, stop_profile_urls]


async def send_profile_request(api_url, app_code=None, api_key=None):
    headers = {'Content-Type': 'application/json'}
    if app_code:
        headers['X-Apig-AppCode'] = f'{app_code}'
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    await do_request(api_url, headers, None, None, 1, 1, "")


async def do_operate_profile(action, app_code=None, api_key=None, hosts=None, ports=None,
                             base_url=None, service_profile_level=None):
    action_message = "Start profiler..." if action == "start" else "Stop profiler..."
    logger.warning("Request has enabled service profile mode. Action: %s", action_message)

    start_profile_urls, stop_profile_urls = get_profile_urls(hosts, ports, service_profile_level, base_url)
    api_urls = start_profile_urls if action == "start" else stop_profile_urls
    tasks = [asyncio.create_task(send_profile_request(api_url, app_code=app_code, api_key=api_key)) for api_url in
             api_urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            logger.error("Error occurred during profile operation: %s", result)
        else:
            logger.info("Profile operation successful.")


def operate_profile(profile, action, app_code=None, api_key=None, hosts=None, ports=None, base_url=None, level=None):
    if not profile:
        return
    asyncio.run(do_operate_profile(action, app_code=app_code, api_key=api_key, hosts=hosts, ports=ports,
                                   base_url=base_url, service_profile_level=level))
