import argparse
import asyncio
import concurrent.futures
import csv
import functools
import gc
import json
import logging
import os
import random
import sys
import threading
import time
import uuid
import weakref
from queue import Queue

gc.disable()

import numpy as np
import requests
import yaml
from tqdm import tqdm

from benchmark_utils import (LATENCY_RESERVATION_BITS, THROUGHPUT_RESERVATION_BITS, TP90, TP95, TP99, TP75,
                             operate_profile)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
IS_DEBUG = int(os.environ.get("BENCHMARK_DEBUG", 0))
CHUNK_SIZE = 1024
CHUNK_LINE_START_PREFIX = "data: "
CHUNK_LINE_END_TAG = "[DONE]"
# 在启动线程池前，程序存在两个线程：主线程 和 tqdm
INITIAL_THREADS = 2
# max_tokens=1时存在特殊情况，content和reasoning_content都为空
MAX_TOKENS_1 = 1
FAKE_REASONING_PIECE = "fake_reasoning_piece"


class NoReuseThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    # 该函数继承自concurrent/futures/thread.py的ThreadPoolExecutor
    # _adjust_thread_count 重写是为了规避线程复用导致的并发量爬坡时偏小的问题
    def _adjust_thread_count(self):
        # When the executor gets lost, the weakref callback will wake up
        # the worker threads.
        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads >= self._max_workers:
            return
        thread_name = '%s_%d' % (self._thread_name_prefix or self,
                                 num_threads)
        t = threading.Thread(name=thread_name, target=concurrent.futures.thread._worker,
                             args=(weakref.ref(self, weakref_cb),
                                   self._work_queue,
                                   self._initializer,
                                   self._initargs))
        t.daemon = True
        t.start()
        self._threads.add(t)
        concurrent.futures.thread._threads_queues[t] = self._work_queue


def adapt_suffix_slash(url):
    if url.endswith('/'):
        url = url.rstrip('/')
    return url


class RequestHandler:
    def __init__(self, args, provider, concurrency, input_length, output_length, test_data):
        self.args = args

        provider["base_url"] = adapt_suffix_slash(provider.get("base_url"))
        self.provider = {}
        for key in provider:
            value = provider.get(key)
            if key == "name":
                self.provider["provider_name"] = value
            else:
                self.provider[key] = value

        self.sampling_params = {
            'temperature': args.temperature,
            'top_k': args.top_k,
            'top_p': args.top_p,
            'max_tokens': output_length,
        }

        self.input_length = input_length
        self.test_data = test_data

        # pd-adaptive 场景下连接池最大值按照 decode请求数+prefill请求数计算
        self.concurrency = concurrency if not self.args.control_method == "pd-adaptive" \
            else args.server_decode_concurrency + args.server_prefill_concurrency
        self.prefill_count = 0
        self.decode_count = 0
        if self.args.control_method == "pd-adaptive":
            self.lock = threading.Lock()

        # 构建client
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=self.concurrency, pool_maxsize=self.concurrency)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def get_cur_rest_req_count(self):
        with self.lock:
            prefill_rest = self.args.server_prefill_concurrency - self.prefill_count
            decode_rest = self.args.server_decode_concurrency - self.decode_count
            cur_rest = min(prefill_rest, decode_rest)
            return cur_rest

    def add_count(self):
        with self.lock:
            self.prefill_count += 1

    def _send_request(self):
        prompt = random.choice(self.test_data)
        self.x_span_id = str(uuid.uuid1())
        metrics = {
            "start_time": time.time(),
            "reasoning_piece": None,
            "content_piece": None,
            "time_to_first_token": None,
            "reasoning_start_time": None,
            "reasoning_end_time": None,
            "content_start_time": None,
            "content_end_time": None,
            "first_decode_token_start_time": None,
            "second_decode_token_start_time": None,
            "second_decode_completion_tokens": 2,
            "prompt_tokens": 0,
            "reasoning_tokens": 0,
            "content_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "server_time_to_first_token": 0,
            "server_time_per_output_token": 0,
            "server_total_time": 0,
            "trace_id": 0,
            "x_span_id": 0,
            "prefill_add": True,
            "decode_add": False,
            "output_steps": 0
        }

        headers = self._prepare_headers()
        if self.args.server_statistics:
            metrics["x_span_id"] = headers["x-span-id"]
        url, data = self._prepare_request_data(prompt)

        try:
            with self.session.post(
                    url,
                    headers=headers,
                    json=data,
                    stream=True,
                    verify=False,
                    timeout=None
            ) as response:
                response.raise_for_status()
                return self._process_response(metrics, response)
        except Exception as e:
            self._log_error(e)
            if self.args.control_method == "pd-adaptive":
                with self.lock:
                    if metrics["prefill_add"]:
                        self.prefill_count -= 1
                    if metrics["decode_add"]:
                        self.decode_count -= 1
            return None

    def _prepare_headers(self):
        headers = {'Content-Type': 'application/json'}
        if api_key := self.provider.get("api_key"):
            headers['Authorization'] = f'Bearer {api_key}'
        if self.args.server_statistics:
            headers['x-span-id'] = self.x_span_id

        return headers

    def _prepare_request_data(self, prompt):
        data = {
            'model': self.provider.get("model_name"),
            **self.sampling_params,
            'stream': True,
            'stream_options': {"include_usage": True,
                               "continuous_usage_stats": True},
            'ignore_eos': True
        }
        url = self.provider.get("base_url")
        backend = self.args.backend
        if backend == 'openai-chat':
            url = f"{url}/chat/completions"
            data["messages"] = [{"role": "user", "content": prompt}]
        elif backend == 'openai':
            url = f"{url}/completions"
            data["prompt"] = prompt
        else:
            raise ValueError("backend 只支持 openai 或 openai-chat")
        return url, data

    def _process_response(self, metrics, response):
        buffer = ""
        for chunk_bytes in response.iter_content(chunk_size=CHUNK_SIZE, decode_unicode=True):
            buffer += chunk_bytes
            chunk_bytes_list = buffer.split("\n\n")
            if len(chunk_bytes_list) == 1:
                # 当前trunk没有完整的event response
                continue
            else:
                # 当前trunk已经包含至少一个完整event response,按照\n\n切分后，最后一个item可能是""或者不完整的event
                buffer = chunk_bytes_list[-1]

            self._process_chunk(chunk_bytes_list[:-1], metrics)

        return self._finalize_metrics(metrics)

    def _process_chunk(self, chunk, metrics):
        for line in chunk:
            if not line.startswith(CHUNK_LINE_START_PREFIX) or line.endswith(CHUNK_LINE_END_TAG):
                continue

            json_data = json.loads(line[len(CHUNK_LINE_START_PREFIX):])
            if "choices" not in json_data:
                continue
            cur_time = time.time()
            backend = self.args.backend
            if backend == 'openai-chat':
                delta = json_data["choices"][0]["delta"] if len(json_data["choices"]) > 0 else {}
                self._update_reasoning_and_content(delta, cur_time, metrics)
            else:
                content = json_data["choices"][0] if json_data["choices"] else {}
                if "text" in content:
                    if metrics["content_start_time"] is None:
                        metrics["content_start_time"] = cur_time
                    metrics["content_end_time"] = cur_time
                    metrics["content_piece"] = content["text"]
                    metrics["content_tokens"] += 1
            self._update_metrics_from_chunk(json_data, metrics, cur_time)

    def _update_metrics_from_chunk(self, json_data, metrics, cur_time):
        self._update_steps(metrics)
        self._update_token_counts(json_data, metrics)
        self._update_token_times(cur_time, metrics)

    def _update_steps(self, metrics):
        metrics["output_steps"] += 1

    def _update_token_times(self, cur_time, metrics):
        if metrics["time_to_first_token"] is None:
            first_token_end_time = cur_time
            metrics["time_to_first_token"] = first_token_end_time - metrics["start_time"]
            metrics["first_decode_token_start_time"] = first_token_end_time
            if self.args.control_method == "pd-adaptive":
                with self.lock:
                    self.prefill_count -= 1
                    metrics["prefill_add"] = False
        elif (metrics["first_decode_token_start_time"] is not None) and (
                metrics["second_decode_token_start_time"] is None):
            metrics["second_decode_token_start_time"] = cur_time
            metrics["second_decode_completion_tokens"] = metrics["completion_tokens"]
            if self.args.control_method == "pd-adaptive":
                with self.lock:
                    self.decode_count += 1
                    metrics["decode_add"] = True

    def _update_reasoning_and_content(self, delta, cur_time, metrics):
        if self.sampling_params['max_tokens'] == MAX_TOKENS_1:
            metrics["reasoning_start_time"] = cur_time
            metrics["reasoning_end_time"] = cur_time
            metrics["reasoning_piece"] = FAKE_REASONING_PIECE
            metrics["reasoning_tokens"] += 1
            return
        if "reasoning_content" in delta and delta["reasoning_content"]:
            if metrics["reasoning_start_time"] is None:
                metrics["reasoning_start_time"] = cur_time
            metrics["reasoning_end_time"] = cur_time
            metrics["reasoning_piece"] = delta["reasoning_content"]
            metrics["reasoning_tokens"] += 1
        if "content" in delta and delta["content"]:
            if metrics["content_start_time"] is None:
                metrics["content_start_time"] = cur_time
            metrics["content_end_time"] = cur_time
            metrics["content_piece"] = delta["content"]
            metrics["content_tokens"] += 1

    def _update_token_counts(self, json_data, metrics):
        # 有usage不存在的情况
        if usage := json_data.get("usage"):
            if isinstance(usage, list) and len(usage) > 0:
                usage = usage[0]
            if self.args.server_statistics and len(json_data["choices"]) == 0:
                # 配置--server-statistics参数，但服务端无信息返回，终止任务
                if "ttft" not in usage or "tpot" not in usage or "latency" not in usage:
                    logger.error("The --server-statistics parameter is set, but no server "
                                 "information ('ttft', 'tpot', 'latency') is returned. "
                                 "Check the returned information on the server or cancel "
                                 "the --server-statistics parameter.")
                    sys.exit(1)
                metrics.update({
                    "server_time_to_first_token": usage["ttft"],
                    "server_time_per_output_token": usage["tpot"],
                    "server_total_time": usage["latency"],
                    "trace_id": json_data.get('id'),
                })
            metrics.update({
                "completion_tokens": usage['completion_tokens'],
                "prompt_tokens": usage['prompt_tokens'],
                "total_tokens": usage['total_tokens']
            })

    def _finalize_metrics(self, metrics):
        if metrics["completion_tokens"] < self.sampling_params.get("max_tokens"):
            logger.warning(f"请求返回tokens不符合要求，响应tokens数: {metrics['completion_tokens']}")
        metrics["end_time"] = time.time()
        metrics["total_time"] = metrics["end_time"] - metrics["start_time"]
        if self.args.control_method == "pd-adaptive":
            with self.lock:
                self.decode_count -= 1
                metrics["decode_add"] = False
                if metrics["prefill_add"]:
                    self.prefill_count -= 1
                    metrics["prefill_add"] = False
        self._validate_required_times(metrics)
        self._calculate_derived_metrics(metrics)
        return metrics

    def _validate_required_times(self, metrics):
        if metrics["first_decode_token_start_time"] is None:
            raise ValueError("first_decode_token_start_time is None")
        if metrics["second_decode_token_start_time"] is None:
            raise ValueError("second_decode_token_start_time is None")

    def _calculate_derived_metrics(self, metrics):
        metrics["time_between_first_and_second_token"] = metrics["second_decode_token_start_time"] - metrics[
            "first_decode_token_start_time"]
        metrics["decode_total_time_from_first_decode_token"] = metrics["end_time"] - metrics[
            "first_decode_token_start_time"]
        metrics["decode_total_time_from_second_decode_token"] = metrics["end_time"] - metrics[
            "second_decode_token_start_time"]

        metrics["time_per_output_token"] = (metrics["decode_total_time_from_first_decode_token"] / (
                metrics["completion_tokens"] - 1)) if (metrics["completion_tokens"] > 1) else 0

        if metrics["completion_tokens"] > metrics["second_decode_completion_tokens"]:
            metrics["time_per_output_token_from_second_decode_token"] = (
                    metrics["decode_total_time_from_second_decode_token"] / (
                    metrics["completion_tokens"] - metrics["second_decode_completion_tokens"]))
        else:
            metrics["time_per_output_token_from_second_decode_token"] = 0

        if metrics["reasoning_start_time"] and metrics["reasoning_end_time"]:
            metrics["reasoning_time"] = (metrics["reasoning_end_time"] - metrics["reasoning_start_time"])
        else:
            metrics["reasoning_time"] = 0

        if metrics["content_start_time"] and metrics["content_end_time"]:
            metrics["content_time"] = (metrics["content_end_time"] - metrics["content_start_time"])
        else:
            metrics["content_time"] = 0

        metrics["total_decode_time"] = metrics["total_time"] - metrics["time_to_first_token"]

    def _log_error(self, error):
        logger.error(
            f"Thread name: {threading.current_thread().name}, "
            f"provider name: {self.provider.get('name')}, "
            f"concurrency num: {self.concurrency}, "
            f"input length: {self.input_length}, "
            f"error message: {error}"
        )


def write_raw_results_to_csv(args, provider, concurrency, rounds, num_prompts, input_length, output_length, results):
    current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    thread = "thread" if concurrency == 1 else "threads"
    save_dir = f'{args.benchmark_dir}/raw'
    os.makedirs(save_dir, exist_ok=True)
    save_file = f"{provider.get('name')}_{provider.get('model_category')}_{input_length}in_{output_length}out_" \
                f"{concurrency}{thread}_{args.growth_rate}growth_{rounds}rounds_{num_prompts}num_prompts_" \
                f"{current_time}.csv"

    # 写入CSV文件
    with open(f'{save_dir}/{save_file}', "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Start_Time",
            "End_Time",
            "Prompt_Tokens",
            "Reasoning_Tokens",
            "Content_Tokens",
            "Completion_Tokens",
            "Total_Tokens",
            "TTFT",
            "TPOT",
            "TPOT_From_Second_Decode_Token",
            "TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s)",
            "Reasoning_Time",
            "Content_Time",
            "Total_Time(s)",
            "Total_Decode_Time(s)",
            "Backend",
            "Temperature",
            "Top_K",
            "Top_P",
            "Control_Method",
            "Growth_Rate",
            "Rounds",
            "Num_Prompts",
            "x_span_id",
            "trace_id",
            "SERVER_TTFT(ms)",
            "SERVER_TPOT(ms)",
            "Output_Steps",
        ])
        # 写入数据
        for result in results:
            writer.writerow([
                result["start_time"],
                result["end_time"],
                result["prompt_tokens"],
                result["reasoning_tokens"],
                result["content_tokens"],
                result["completion_tokens"],
                result["total_tokens"],
                round(result["time_to_first_token"], LATENCY_RESERVATION_BITS),
                round(result["time_per_output_token"], LATENCY_RESERVATION_BITS),
                round(result["time_per_output_token_from_second_decode_token"], LATENCY_RESERVATION_BITS),
                round(result["time_between_first_and_second_token"], LATENCY_RESERVATION_BITS),
                round(result["reasoning_time"], LATENCY_RESERVATION_BITS),
                round(result["content_time"], LATENCY_RESERVATION_BITS),
                round(result["total_time"], LATENCY_RESERVATION_BITS),
                round(result["total_decode_time"], LATENCY_RESERVATION_BITS),
                args.backend,
                args.temperature,
                args.top_k,
                args.top_p,
                args.control_method,
                args.growth_rate,
                rounds,
                num_prompts,
                result["x_span_id"],
                result["trace_id"],
                round(result["server_time_to_first_token"]),
                round(result["server_time_per_output_token"]),
                result["output_steps"],
            ])


def calculate_metrics(args, results, concurrency, rounds, num_prompts):
    # 计算总处理时间
    first_start = min([result['start_time'] for result in results])
    last_end = max([result['end_time'] for result in results])
    total_time = last_end - first_start

    # 计算输出的总 token 数
    all_completion_tokens = [result['completion_tokens'] for result in results]
    tp90_completion_tokens = np.percentile(all_completion_tokens, TP90)
    tp95_completion_tokens = np.percentile(all_completion_tokens, TP95)
    tp99_completion_tokens = np.percentile(all_completion_tokens, TP99)
    min_completion_tokens = np.min(all_completion_tokens)
    avg_completion_tokens = np.mean(all_completion_tokens)
    total_output_tokens = sum(all_completion_tokens)

    # 计算avg reasoning content length
    all_prompt_tokens = [result['prompt_tokens'] for result in results]
    all_reasoning_tokens = [result['reasoning_tokens'] for result in results]
    all_content_tokens = [result['content_tokens'] for result in results]
    avg_prompt_tokens = np.mean(all_prompt_tokens)
    avg_reasoning_tokens = np.mean(all_reasoning_tokens)
    avg_content_tokens = np.mean(all_content_tokens)

    # 计算 TPS (token per second)
    tps = total_output_tokens / total_time if total_time else 0
    # total token per second
    total_tokens_list = [result['total_tokens'] for result in results]
    ttps = sum(total_tokens_list) / total_time if total_time else 0

    # 计算 TTFT
    ttft_list = [result['time_to_first_token'] for result in results]
    tp75_ttft = np.percentile(ttft_list, TP75)
    tp90_ttft = np.percentile(ttft_list, TP90)
    tp95_ttft = np.percentile(ttft_list, TP95)
    tp99_ttft = np.percentile(ttft_list, TP99)
    max_ttft = np.max(ttft_list)
    avg_ttft = np.mean(ttft_list)
    server_ttft = [result["server_time_to_first_token"] for result in results]
    tp75_server_ttft = np.percentile(server_ttft, TP75)
    tp90_server_ttft = np.percentile(server_ttft, TP90)
    tp95_server_ttft = np.percentile(server_ttft, TP95)
    tp99_server_ttft = np.percentile(server_ttft, TP99)
    max_server_ttft = np.max(server_ttft)
    avg_server_ttft = np.mean(server_ttft)

    # queries_per_second
    queries_per_second = num_prompts / total_time
    # failure_times
    request_num = num_prompts
    success_times = len(results)
    failure_times = request_num - success_times
    fail_rate = failure_times / request_num

    # 计算 TPOT from first decode time
    tpot_list = [result['time_per_output_token'] for result in results]
    tp75_tpot = np.percentile(tpot_list, TP75)
    tp90_tpot = np.percentile(tpot_list, TP90)
    tp95_tpot = np.percentile(tpot_list, TP95)
    tp99_tpot = np.percentile(tpot_list, TP99)
    max_tpot = np.max(tpot_list)
    avg_tpot = np.mean(tpot_list)
    tpot_server_list = [result["server_time_per_output_token"] for result in results]
    tp75_server_tpot = np.percentile(tpot_server_list, TP75)
    tp90_server_tpot = np.percentile(tpot_server_list, TP90)
    tp95_server_tpot = np.percentile(tpot_server_list, TP95)
    tp99_server_tpot = np.percentile(tpot_server_list, TP99)
    max_server_tpot = np.max(tpot_server_list)
    avg_server_tpot = np.mean(tpot_server_list)

    # 计算 TPOT from second decode time
    tpots_second = [result['time_per_output_token_from_second_decode_token'] for result in results]
    tp90_tpot_second = np.percentile(tpots_second, TP90)
    tp95_tpot_second = np.percentile(tpots_second, TP95)
    tp99_tpot_second = np.percentile(tpots_second, TP99)
    max_tpot_second = np.max(tpots_second)
    avg_tpot_second = np.mean(tpots_second)

    # 计算首token和第一个增量token之间的时间
    time_between_first_and_second_token = [result['time_between_first_and_second_token'] for result in results]
    tp90_time_between_first_and_second_token = np.percentile(time_between_first_and_second_token, TP90)
    tp95_time_between_first_and_second_token = np.percentile(time_between_first_and_second_token, TP95)
    tp99_time_between_first_and_second_token = np.percentile(time_between_first_and_second_token, TP99)
    min_time_between_first_and_second_token = np.min(time_between_first_and_second_token)
    max_time_between_first_and_second_token = np.max(time_between_first_and_second_token)
    avg_time_between_first_and_second_token = np.mean(time_between_first_and_second_token)

    # 计算e2e
    e2e_list = [result['total_time'] for result in results]
    tp75_e2e = np.percentile(e2e_list, TP75)
    tp90_e2e = np.percentile(e2e_list, TP90)
    tp95_e2e = np.percentile(e2e_list, TP95)
    tp99_e2e = np.percentile(e2e_list, TP99)
    max_e2e = np.max(e2e_list)
    avg_e2e = np.mean(e2e_list)
    e2e_server_list = [result["server_total_time"] for result in results]
    tp75_server_e2e = np.percentile(e2e_server_list, TP75)
    tp90_server_e2e = np.percentile(e2e_server_list, TP90)
    tp95_server_e2e = np.percentile(e2e_server_list, TP95)
    tp99_server_e2e = np.percentile(e2e_server_list, TP99)
    max_server_e2e = np.max(e2e_server_list)
    avg_server_e2e = np.mean(e2e_server_list)

    # 计算接收率
    tp90_mtp_accept_rate, tp95_mtp_accept_rate, tp99_mtp_accept_rate = None, None, None
    max_mtp_accept_rate, min_mtp_accept_rate, avg_mtp_accept_rate = None, None, None
    if getattr(args, "use_spec_decode", False) and getattr(args, "num_speculative_tokens", -1) >= 0:
        if getattr(args, "use_mtp_accept_rate", False):
            mtp_accept_rate_list = [(result["completion_tokens"] - 1 - (result["output_steps"] - 1) *
                                     args.num_scheduler_steps) / (
                                            (result["output_steps"] - 1) * args.num_speculative_tokens * args.num_scheduler_steps)
                                    if result["output_steps"] > 1 else 0.0 for result in results]
        else:
            mtp_accept_rate_list = [(result["completion_tokens"] - 1) / ((result["output_steps"] - 1) * (
                    args.num_speculative_tokens + 1) * args.num_scheduler_steps) if result["output_steps"] > 1
                                    else 0.0 for result in results]

        tp90_mtp_accept_rate = np.percentile(mtp_accept_rate_list, TP90)
        tp95_mtp_accept_rate = np.percentile(mtp_accept_rate_list, TP95)
        tp99_mtp_accept_rate = np.percentile(mtp_accept_rate_list, TP99)
        max_mtp_accept_rate = np.max(mtp_accept_rate_list)
        min_mtp_accept_rate = np.min(mtp_accept_rate_list)
        avg_mtp_accept_rate = np.mean(mtp_accept_rate_list)

    logger.info(f"\n")
    logger.info(f"TP90_COMPLETION_TOKENS: {tp90_completion_tokens:.2f} tokens")
    logger.info(f"TP95_COMPLETION_TOKENS: {tp95_completion_tokens:.2f} tokens")
    logger.info(f"TP99_COMPLETION_TOKENS: {tp99_completion_tokens:.2f} tokens")
    logger.info(f"MIN_COMPLETION_TOKENS: {min_completion_tokens} tokens")
    logger.info(f"AVG_COMPLETION_TOKENS: {avg_completion_tokens:.2f} tokens")
    logger.info(f"AVG_REASONING_TOKENS: {avg_reasoning_tokens:.2f} tokens")
    logger.info(f"AVG_CONTENT_TOKENS: {avg_content_tokens:.2f} tokens")
    logger.info(f"AVG_PROMPT_TOKENS: {avg_prompt_tokens:.2f} tokens")
    logger.info(f"Output_Token_Throughput: {tps:.2f} tokens/s")
    logger.info(f"Total_Token_Throughput: {ttps:.2f} tokens/s")
    if args.server_statistics:
        logger.info(f"TP75_SERVER_TTFT: {tp75_server_ttft:.3f} ms")
        logger.info(f"TP90_SERVER_TTFT: {tp90_server_ttft:.3f} ms")
        logger.info(f"TP95_SERVER_TTFT: {tp95_server_ttft:.3f} ms")
        logger.info(f"TP99_SERVER_TTFT: {tp99_server_ttft:.3f} ms")
        logger.info(f"MAX_SERVER_TTFT: {max_server_ttft:.3f} ms")
        logger.info(f"AVG_SERVER_TTFT: {avg_server_ttft:.3f} ms")
        logger.info(f"TP75_SERVER_TPOT from first token: {tp75_server_tpot:.3f} ms")
        logger.info(f"TP90_SERVER_TPOT from first token: {tp90_server_tpot:.3f} ms")
        logger.info(f"TP95_SERVER_TPOT from first token: {tp95_server_tpot:.3f} ms")
        logger.info(f"TP99_SERVER_TPOT from first token: {tp99_server_tpot:.3f} ms")
        logger.info(f"MAX_SERVER_TPOT from first token: {max_server_tpot:.3f} ms")
        logger.info(f"AVG_SERVER_TPOT from first token: {avg_server_tpot:.3f} ms")
        logger.info(f"TP75_SERVER_E2E: {tp75_server_e2e:.3f} ms")
        logger.info(f"TP90_SERVER_E2E: {tp90_server_e2e:.3f} ms")
        logger.info(f"TP95_SERVER_E2E: {tp95_server_e2e:.3f} ms")
        logger.info(f"TP99_SERVER_E2E: {tp99_server_e2e:.3f} ms")
        logger.info(f"MAX_SERVER_E2E: {max_server_e2e:.3f} ms")
        logger.info(f"AVG_SERVER_E2E: {avg_server_e2e:.3f} ms")
    logger.info(f"TP75_TTFT: {tp75_ttft:.3f} s")
    logger.info(f"TP90_TTFT: {tp90_ttft:.3f} s")
    logger.info(f"TP95_TTFT: {tp95_ttft:.3f} s")
    logger.info(f"TP99_TTFT: {tp99_ttft:.3f} s")
    logger.info(f"MAX_TTFT: {max_ttft:.3f} s")
    logger.info(f"AVG_TTFT: {avg_ttft:.3f} s")
    logger.info(f"TP75_TPOT from first token: {tp75_tpot:.3f} s")
    logger.info(f"TP90_TPOT from first token: {tp90_tpot:.3f} s")
    logger.info(f"TP95_TPOT from first token: {tp95_tpot:.3f} s")
    logger.info(f"TP99_TPOT from first token: {tp99_tpot:.3f} s")
    logger.info(f"MAX_TPOT from first token: {max_tpot:.3f} s")
    logger.info(f"AVG_TPOT from first token: {avg_tpot:.3f} s")
    logger.info(f"TP90_TPOT from second token: {tp90_tpot_second:.3f} s")
    logger.info(f"TP95_TPOT from second token: {tp95_tpot_second:.3f} s")
    logger.info(f"TP99_TPOT from second token: {tp99_tpot_second:.3f} s")
    logger.info(f"MAX_TPOT from second token: {max_tpot_second:.3f} s")
    logger.info(f"AVG_TPOT from second token: {avg_tpot_second:.3f} s")
    logger.info(f"TP90 time between first and second token: {tp90_time_between_first_and_second_token:.3f} s")
    logger.info(f"TP95 time between first and second token: {tp95_time_between_first_and_second_token:.3f} s")
    logger.info(f"TP99 time between first and second token: {tp99_time_between_first_and_second_token:.3f} s")
    logger.info(f"Min time between first and second token: {min_time_between_first_and_second_token:.3f} s")
    logger.info(f"Max time between first and second token: {max_time_between_first_and_second_token:.3f} s")
    logger.info(f"AVG time between first and second token: {avg_time_between_first_and_second_token:.3f} s")
    if getattr(args, "use_spec_decode", False) and getattr(args, "num_speculative_tokens", -1) >= 0:
        logger.info(f"TP90 mtp accept rate: {tp90_mtp_accept_rate:.3f} s")
        logger.info(f"TP95 mtp accept rate: {tp95_mtp_accept_rate:.3f} s")
        logger.info(f"TP99 mtp accept rate: {tp99_mtp_accept_rate:.3f} s")
        logger.info(f"Min mtp accept rate: {max_mtp_accept_rate:.3f} s")
        logger.info(f"Max mtp accept rate: {min_mtp_accept_rate:.3f} s")
        logger.info(f"AVG mtp accept rate: {avg_mtp_accept_rate:.3f} s")
    logger.info(f"TP75_E2E: {tp75_e2e:.3f} s")
    logger.info(f"TP90_E2E: {tp90_e2e:.3f} s")
    logger.info(f"TP95_E2E: {tp95_e2e:.3f} s")
    logger.info(f"TP99_E2E: {tp99_e2e:.3f} s")
    logger.info(f"MAX_E2E: {max_e2e:.3f} s")
    logger.info(f"AVG_E2E: {avg_e2e:.3f} s")
    logger.info(f"QPS: {queries_per_second:.2f}")
    logger.info(f"Total_Time: {total_time:.3f} s")
    logger.info(f"Failure_Times: {failure_times}")
    logger.info(f"Total_Times: {request_num}")
    logger.info(f"Fail_Rate: {fail_rate:.4f}")
    logger.info(f"---------------------------\n")

    return tp90_completion_tokens, tp95_completion_tokens, tp99_completion_tokens, \
        min_completion_tokens, avg_completion_tokens, \
        avg_reasoning_tokens, avg_content_tokens, avg_prompt_tokens, \
        tps, ttps, \
        tp75_ttft, tp90_ttft, tp95_ttft, tp99_ttft, max_ttft, avg_ttft, \
        tp75_server_ttft, tp90_server_ttft, tp95_server_ttft, tp99_server_ttft, max_server_ttft, avg_server_ttft, \
        tp75_tpot, tp90_tpot, tp95_tpot, tp99_tpot, max_tpot, avg_tpot, \
        tp75_server_tpot, tp90_server_tpot, tp95_server_tpot, tp99_server_tpot, max_server_tpot, avg_server_tpot, \
        tp90_tpot_second, tp95_tpot_second, tp99_tpot_second, max_tpot_second, avg_tpot_second, \
        tp90_time_between_first_and_second_token, tp95_time_between_first_and_second_token, \
        tp99_time_between_first_and_second_token, min_time_between_first_and_second_token, \
        max_time_between_first_and_second_token, avg_time_between_first_and_second_token, \
        tp90_mtp_accept_rate, tp95_mtp_accept_rate, tp99_mtp_accept_rate, max_mtp_accept_rate, min_mtp_accept_rate, \
        avg_mtp_accept_rate, \
        tp75_e2e, tp90_e2e, tp95_e2e, tp99_e2e, max_e2e, avg_e2e, \
        tp75_server_e2e, tp90_server_e2e, tp95_server_e2e, tp99_server_e2e, max_server_e2e, avg_server_e2e, \
        queries_per_second, total_time, fail_rate


def write_summary_metrics_to_csv(args, provider, rounds, all_metrics):
    benchmark_dir = args.benchmark_dir
    control_method = args.control_method
    growth_rate = args.growth_rate
    current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    os.makedirs(benchmark_dir, exist_ok=True)

    with open(f'{benchmark_dir}/summary_{provider.get("name")}_{provider.get("model_category")}_'
              f'{control_method}_{growth_rate}growth_{current_time}.csv',
              "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Execution_Time",
            "Input_Length",
            "Output_Length",
            "Concurrency",
            "Total_Token_Throughput(tokens/s)",
            "Output_Token_Throughput(tokens/s)",
            "TP75_TTFT(s)",
            "TP90_TTFT(s)",
            "TP95_TTFT(s)",
            "TP99_TTFT(s)",
            "MAX_TTFT(s)",
            "AVG_TTFT(s)",
            "TP75_TPOT(s)",
            "TP90_TPOT(s)",
            "TP95_TPOT(s)",
            "TP99_TPOT(s)",
            "MAX_TPOT(s)",
            "AVG_TPOT(s)",
            "TP90_TPOT_SEC(s)",
            "TP95_TPOT_SEC(s)",
            "TP99_TPOT_SEC(s)",
            "MAX_TPOT_SEC(s)",
            "AVG_TPOT_SEC(s)",
            "TP90_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s)",
            "TP95_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s)",
            "TP99_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s)",
            "MIN_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s)",
            "MAX_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s)",
            "AVG_TIME_BETWEEN_FIRST_AND_SECOND_TOKEN(s)",
            "TP75_E2E(s)",
            "TP90_E2E(s)",
            "TP95_E2E(s)",
            "TP99_E2E(s)",
            "MAX_E2E(s)",
            "AVG_E2E(s)",
            "Total_Time(s)",
            "QPS",
            "Fail_Rate",
            "Backend",
            "Temperature",
            "Top_K",
            "Top_P",
            "Control_Method",
            "Growth_Rate",
            "Rounds",
            "Num_Prompts",
            "Provider",
            "TP90_COMPLETION_TOKENS",
            "TP95_COMPLETION_TOKENS",
            "TP99_COMPLETION_TOKENS",
            "MIN_COMPLETION_TOKENS",
            "AVG_COMPLETION_TOKENS",
            "AVG_REASONING_TOKENS",
            "AVG_CONTENT_TOKENS",
            "AVG_PROMPT_TOKENS",
            "TP75_SERVER_TTFT(ms)",
            "TP90_SERVER_TTFT(ms)",
            "TP95_SERVER_TTFT(ms)",
            "TP99_SERVER_TTFT(ms)",
            "MAX_SERVER_TTFT(ms)",
            "AVG_SERVER_TTFT(ms)",
            "TP75_SERVER_TPOT(ms)",
            "TP90_SERVER_TPOT(ms)",
            "TP95_SERVER_TPOT(ms)",
            "TP99_SERVER_TPOT(ms)",
            "MAX_SERVER_TPOT(ms)",
            "AVG_SERVER_TPOT(ms)",
            "TP75_SERVER_E2E(ms)",
            "TP90_SERVER_E2E(ms)",
            "TP95_SERVER_E2E(ms)",
            "TP99_SERVER_E2E(ms)",
            "MAX_SERVER_E2E(ms)",
            "AVG_SERVER_E2E(ms)",
            "TP90_SPEC_ACCEPT_RATE",
            "TP95_SPEC_ACCEPT_RATE",
            "TP99_SPEC_ACCEPT_RATE",
            "MIN_SPEC_ACCEPT_RATE",
            "MAX_SPEC_ACCEPT_RATE",
            "AVG_SPEC_ACCEPT_RATE",
        ])
        for metric in all_metrics:
            writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                metric['input_length'],
                metric['output_length'],
                metric['concurrency'],
                round(metric['ttps'], THROUGHPUT_RESERVATION_BITS),
                round(metric['tps'], THROUGHPUT_RESERVATION_BITS),
                round(metric['tp75_ttft'], LATENCY_RESERVATION_BITS),
                round(metric['tp90_ttft'], LATENCY_RESERVATION_BITS),
                round(metric['tp95_ttft'], LATENCY_RESERVATION_BITS),
                round(metric['tp99_ttft'], LATENCY_RESERVATION_BITS),
                round(metric['max_ttft'], LATENCY_RESERVATION_BITS),
                round(metric['avg_ttft'], LATENCY_RESERVATION_BITS),
                round(metric['tp75_tpot'], LATENCY_RESERVATION_BITS),
                round(metric['tp90_tpot'], LATENCY_RESERVATION_BITS),
                round(metric['tp95_tpot'], LATENCY_RESERVATION_BITS),
                round(metric['tp99_tpot'], LATENCY_RESERVATION_BITS),
                round(metric['max_tpot'], LATENCY_RESERVATION_BITS),
                round(metric['avg_tpot'], LATENCY_RESERVATION_BITS),
                round(metric['tp90_tpot_second'], LATENCY_RESERVATION_BITS),
                round(metric['tp95_tpot_second'], LATENCY_RESERVATION_BITS),
                round(metric['tp99_tpot_second'], LATENCY_RESERVATION_BITS),
                round(metric['max_tpot_second'], LATENCY_RESERVATION_BITS),
                round(metric['avg_tpot_second'], LATENCY_RESERVATION_BITS),
                round(metric['tp90_time_between_first_and_second_token'], LATENCY_RESERVATION_BITS),
                round(metric['tp95_time_between_first_and_second_token'], LATENCY_RESERVATION_BITS),
                round(metric['tp99_time_between_first_and_second_token'], LATENCY_RESERVATION_BITS),
                round(metric['min_time_between_first_and_second_token'], LATENCY_RESERVATION_BITS),
                round(metric['max_time_between_first_and_second_token'], LATENCY_RESERVATION_BITS),
                round(metric['avg_time_between_first_and_second_token'], LATENCY_RESERVATION_BITS),
                round(metric['tp75_e2e'], LATENCY_RESERVATION_BITS),
                round(metric['tp90_e2e'], LATENCY_RESERVATION_BITS),
                round(metric['tp95_e2e'], LATENCY_RESERVATION_BITS),
                round(metric['tp99_e2e'], LATENCY_RESERVATION_BITS),
                round(metric['max_e2e'], LATENCY_RESERVATION_BITS),
                round(metric['avg_e2e'], LATENCY_RESERVATION_BITS),
                round(metric["total_time"], LATENCY_RESERVATION_BITS),
                round(metric["queries_per_second"], THROUGHPUT_RESERVATION_BITS),
                round(metric["fail_rate"], THROUGHPUT_RESERVATION_BITS),
                args.backend,
                args.temperature,
                args.top_k,
                args.top_p,
                control_method,
                growth_rate,
                rounds,
                metric['num_prompts'],
                metric['provider'],
                metric['tp90_completion_tokens'],
                metric['tp95_completion_tokens'],
                metric['tp99_completion_tokens'],
                metric['min_completion_tokens'],
                metric['avg_completion_tokens'],
                metric['avg_reasoning_tokens'],
                metric['avg_content_tokens'],
                metric['avg_prompt_tokens'],
                round(metric['tp75_server_ttft'], LATENCY_RESERVATION_BITS),
                round(metric['tp90_server_ttft'], LATENCY_RESERVATION_BITS),
                round(metric['tp95_server_ttft'], LATENCY_RESERVATION_BITS),
                round(metric['tp99_server_ttft'], LATENCY_RESERVATION_BITS),
                round(metric['max_server_ttft'], LATENCY_RESERVATION_BITS),
                round(metric['avg_server_ttft'], LATENCY_RESERVATION_BITS),
                round(metric['tp75_server_tpot'], LATENCY_RESERVATION_BITS),
                round(metric['tp90_server_tpot'], LATENCY_RESERVATION_BITS),
                round(metric['tp95_server_tpot'], LATENCY_RESERVATION_BITS),
                round(metric['tp99_server_tpot'], LATENCY_RESERVATION_BITS),
                round(metric['max_server_tpot'], LATENCY_RESERVATION_BITS),
                round(metric['avg_server_tpot'], LATENCY_RESERVATION_BITS),
                round(metric['tp75_server_e2e'], LATENCY_RESERVATION_BITS),
                round(metric['tp90_server_e2e'], LATENCY_RESERVATION_BITS),
                round(metric['tp95_server_e2e'], LATENCY_RESERVATION_BITS),
                round(metric['tp99_server_e2e'], LATENCY_RESERVATION_BITS),
                round(metric['max_server_e2e'], LATENCY_RESERVATION_BITS),
                round(metric['avg_server_e2e'], LATENCY_RESERVATION_BITS),
                round(metric["tp90_mtp_accept_rate"], LATENCY_RESERVATION_BITS) if metric["tp90_mtp_accept_rate"] else "None",
                round(metric['tp95_mtp_accept_rate'], LATENCY_RESERVATION_BITS) if metric["tp95_mtp_accept_rate"] else "None",
                round(metric['tp99_mtp_accept_rate'], LATENCY_RESERVATION_BITS) if metric["tp99_mtp_accept_rate"] else "None",
                round(metric['max_mtp_accept_rate'], LATENCY_RESERVATION_BITS) if metric["max_mtp_accept_rate"] else "None",
                round(metric['min_mtp_accept_rate'], LATENCY_RESERVATION_BITS) if metric["min_mtp_accept_rate"] else "None",
                round(metric['avg_mtp_accept_rate'], LATENCY_RESERVATION_BITS) if metric["avg_mtp_accept_rate"] else "None",
            ])


def future_done_callback(pbar, future):
    pbar.update(1)


def test_concurrent_performance(args, provider, concurrency=1, rounds=2, num_prompts=2,
                                input_length=256, output_length=256):
    with open(f'{args.dataset_dir}/{input_length}.json', "r", encoding="utf-8") as f:
        test_data = [item["input"] for item in json.load(f)]

    handler = RequestHandler(args, provider, concurrency, input_length, output_length, test_data)
    # 总请求次数
    futures = []
    if args.control_method == "pd-adaptive":
        futures = _test_adaptive_concurrency(args.server_prefill_concurrency, args.server_decode_concurrency,
                                             num_prompts, handler, futures, args.climbing_period)
    else:
        with NoReuseThreadPoolExecutor() as executor:
            current_connections = 0
            # 先处理并发数是否爬坡，再提交实际请求
            # 如果 control_method 是 round，直接设置最大连接数
            if args.control_method == 'round':
                executor._max_workers = concurrency
                # 使用轮询方式提交并发任务
                for _ in range(num_prompts):
                    future = executor.submit(handler._send_request)
                    futures.append(future)
            else:
                pbar = tqdm(total=num_prompts)
                # 每秒增加 growth_rate 个连接
                while current_connections < concurrency:
                    # 计算需要增加的连接数
                    add_connections = min(args.growth_rate, concurrency - current_connections)
                    current_connections += add_connections
                    executor._max_workers = current_connections

                    if IS_DEBUG:
                        thread_num = len(threading.enumerate())
                        logger.info("线程数量是%d" % thread_num)
                        logger.info(threading.enumerate())

                    # 提交到 _max_workers 个任务
                    while executor._max_workers > (threading.active_count() - INITIAL_THREADS) and len(
                            futures) < concurrency:
                        future = executor.submit(handler._send_request)
                        future.add_done_callback(functools.partial(future_done_callback, pbar))
                        futures.append(future)

                    # 等待 1 秒, 打印爬坡过程中线程数量的变化
                    logger.info(
                        f"active connection: {threading.active_count() - INITIAL_THREADS}; "
                        f"pending:{executor._work_queue.qsize()}")
                    time.sleep(args.climbing_period / 1000)

                q = Queue()
                for _ in range(num_prompts - concurrency):
                    q.put(handler._send_request)
                while not q.empty():
                    future = executor.submit(q.get())
                    future.add_done_callback(functools.partial(future_done_callback, pbar))
                    futures.append(future)
                    q.task_done()

            # 打印当前活跃线程数
            while executor._work_queue.qsize():
                time.sleep(5)
                logger.info(f"active connection: {threading.active_count() - INITIAL_THREADS}; "
                            f"pending:{executor._work_queue.qsize()}")

    # 等待所有任务完成
    concurrent.futures.wait(futures)
    if args.control_method == 'queue':
        pbar.close()
    results = [future.result()
               for future in futures if future.result() is not None]

    return results


def _test_adaptive_concurrency(server_prefill_concurrency, server_decode_concurrency,
                               num_prompts, handler, futures, climbing_period):
    # 总请求次数
    total_reqs = num_prompts
    with NoReuseThreadPoolExecutor() as executor:
        executor._max_workers = server_decode_concurrency + server_prefill_concurrency  # 线程池预留

        while True:
            rest_reqs = total_reqs - len(futures)
            if rest_reqs == 0:
                break
            cur_reqs = handler.get_cur_rest_req_count()
            if len(futures) >= server_decode_concurrency:
                cur_reqs = int(cur_reqs * 1.1)  # 客户端往服务端发送请求 factory 的经验值
            for _ in range(min(cur_reqs, rest_reqs)):
                handler.add_count()
                future = executor.submit(handler._send_request)
                futures.append(future)
            time.sleep(climbing_period / 2000)
    return futures


def warm_up(args, provider, dataset_dir, concurrency=4, num_prompts=8, prompt_tokens=128, output_tokens=128):
    logger.info('warm up...')
    with open(f'{dataset_dir}/{prompt_tokens}.json', "r", encoding="utf-8") as f:
        test_data = [item["input"] for item in json.load(f)]
    handler: RequestHandler = RequestHandler(args, provider, concurrency, prompt_tokens, output_tokens, test_data)
    results = []
    with NoReuseThreadPoolExecutor() as executor:
        executor._max_workers = concurrency
        futures = []
        for _ in tqdm(range(num_prompts)):
            future = executor.submit(handler._send_request)
            futures.append(future)
    concurrent.futures.wait(futures)
    future_results = [future.result() for future in futures if future.result() is not None]
    results.append(future_results)
    logger.info('warm up end.')


def process_num_prompts(num_prompts, rounds, concurrencies):
    # 如果num_prompts参数和rounds参数都没有传入，则抛出异常
    if not num_prompts and not rounds:
        raise ValueError("At least one of num_prompts and rounds must be set.")

    # 如果同时传入num_prompts和rounds参数，打印警告日志，优先使用num_prompts参数
    if rounds and num_prompts:
        logger.warning("Both num_prompts and rounds are set, num_prompts will be used.")

    if num_prompts:
        if len(num_prompts) != len(concurrencies):
            raise ValueError("concurrencies length should equal num_prompts length.")

        # num_prompts列表中的每个元素均要大于等于concurrencies列表中的每个元素
        if any(num_prompt < concurrency for num_prompt, concurrency in zip(num_prompts, concurrencies)):
            logger.warning("values in num_prompts are smaller than values in concurrencies.")

    else:
        # round参数小于1提示用户报错
        if rounds <= 0:
            raise ValueError("rounds must be greater than 0.")

        return list(map(lambda concurrency: concurrency * rounds, concurrencies))

    return num_prompts


def run_climbing(args: argparse.Namespace):
    # 读取 providers 配置
    providers_path = os.path.realpath(args.providers_path)
    with open(providers_path, "r", encoding="utf-8") as f:
        providers = yaml.safe_load(f)
    # 过滤掉 base_url 和 model_name 同时为空的服务提供商(provider)
    # api_key 为原openai的api_key, 现可用于MAAS的认证
    # 实际也承担着启动的开关. 只有不为空字符时服务提供商才会被启用
    providers = [provider for provider in providers["providers"]
                 if (len(provider["api_key"]) > 0
                     and len(provider["base_url"]) > 0
                     and len(provider["model_name"]) > 0)
                 ]
    if len(providers) == 0:
        raise ValueError("One valid provider is required at least. "
                         "A valid provider must have api_key, base_url and model_name.")

    # 组合 input_output_tokens_length
    if len(args.prompt_tokens) != len(args.output_tokens):
        raise ValueError("prompt_tokens length should equal output_tokens length.")
    input_output_lengths = [(args.prompt_tokens[i], args.output_tokens[i]) for i in range(len(args.prompt_tokens))]

    # 爬坡并发
    server_prefill_concurrency = args.server_prefill_concurrency
    server_decode_concurrency = args.server_decode_concurrency
    if args.control_method == "pd-adaptive":
        if not args.server_prefill_concurrency and not args.server_decode_concurrency:
            raise ValueError("In the pd-adaptive scenario, server_decode_concurrency and "
                             "server_decode_concurrency must be set.")
        # 并发数按prefill与decode请求数中取小计算
        concurrencies = [min(server_prefill_concurrency, server_decode_concurrency)]
    else:
        concurrencies = args.concurrencies if hasattr(args, "concurrencies") else args.parallel_num
    rounds = args.rounds if args.rounds is not None else args.epochs
    control_method = args.control_method
    growth_rate = args.growth_rate

    if control_method == "round" and growth_rate > 0:
        logger.warning("growth-rate doesn't work when control-method is round")

    if control_method == "queue" and growth_rate <= 0:
        raise ValueError("growth-rate must be greater than 0 when control-method is queue")

    if args.climbing_period <= 0:
        raise ValueError("climbing-period must be greater than 0")

    num_prompts = process_num_prompts(args.num_prompts, rounds, concurrencies)

    warm_up(args, providers[0], dataset_dir='./built_in_dataset')

    # 循环对每个服务商进行测试
    for provider in providers:
        # 将 concurrency 循环抽离成一个函数
        all_metrics = test_concurrency_performance(
            args,
            provider,
            concurrencies,
            rounds,
            num_prompts,
            input_output_lengths
        )

        write_summary_metrics_to_csv(args, provider, rounds, all_metrics)


def test_concurrency_performance(
        args,
        provider,
        concurrencies,
        rounds,
        num_prompts,
        input_output_lengths
):
    control_method = args.control_method
    provider_name = provider.get("name")
    model_category = provider.get("model_category")

    # profile start.
    operate_profile(args.profile, "start", api_key=provider.get("api_key"), base_url=provider.get("base_url"),
                    level=args.service_profile_level)

    all_metrics = []

    for i, concurrency in enumerate(concurrencies):
        for input_length, output_length in input_output_lengths:
            logger.info(f"\n---------------------------")
            logger.info(f"开始测试服务商：{provider_name}")
            logger.info(f"模型类型：{model_category}")
            logger.info(f"测试方法：{control_method}")
            logger.info(f"并发数： {concurrency}")
            logger.info(f"轮数： {rounds}")
            logger.info(f"输入tokens长度： {input_length}")
            logger.info(f"输出tokens长度： {output_length}")

            results = []
            results.extend(
                test_concurrent_performance(args, provider,
                                            concurrency, rounds, num_prompts[i], input_length, output_length)
            )
            write_raw_results_to_csv(args, provider, concurrency, rounds, num_prompts[i], input_length,
                                     output_length, results)
            tp90_completion_tokens, tp95_completion_tokens, tp99_completion_tokens, \
                min_completion_tokens, avg_completion_tokens, \
                avg_reasoning_tokens, avg_content_tokens, avg_prompt_tokens, \
                tps, ttps, \
                tp75_ttft, tp90_ttft, tp95_ttft, tp99_ttft, max_ttft, avg_ttft, \
                tp75_server_ttft, tp90_server_ttft, \
                tp95_server_ttft, tp99_server_ttft, max_server_ttft, avg_server_ttft, \
                tp75_tpot, tp90_tpot, tp95_tpot, tp99_tpot, max_tpot, avg_tpot, tp75_server_tpot, tp90_server_tpot, \
                tp95_server_tpot, tp99_server_tpot, max_server_tpot, avg_server_tpot, \
                tp90_tpot_second, tp95_tpot_second, tp99_tpot_second, max_tpot_second, avg_tpot_second, \
                tp90_time_between_first_and_second_token, tp95_time_between_first_and_second_token, \
                tp99_time_between_first_and_second_token, \
                min_time_between_first_and_second_token, max_time_between_first_and_second_token, \
                avg_time_between_first_and_second_token, \
                tp90_mtp_accept_rate, tp95_mtp_accept_rate, tp99_mtp_accept_rate, \
                max_mtp_accept_rate, min_mtp_accept_rate, avg_mtp_accept_rate, \
                tp75_e2e, tp90_e2e, tp95_e2e, tp99_e2e, max_e2e, avg_e2e, \
                tp75_server_e2e, tp90_server_e2e, tp95_server_e2e, tp99_server_e2e, max_server_e2e, avg_server_e2e, \
                queries_per_second, total_time, fail_rate \
                = calculate_metrics(args, results, concurrency, rounds, num_prompts[i])

            all_metrics.append({
                "provider": provider_name,
                "input_length": input_length,
                "output_length": output_length,
                "concurrency": concurrency,
                "num_prompts": num_prompts[i],
                "tp90_completion_tokens": tp90_completion_tokens,
                "tp95_completion_tokens": tp95_completion_tokens,
                "tp99_completion_tokens": tp99_completion_tokens,
                "min_completion_tokens": min_completion_tokens,
                "avg_completion_tokens": avg_completion_tokens,
                "avg_reasoning_tokens": avg_reasoning_tokens,
                "avg_content_tokens": avg_content_tokens,
                "avg_prompt_tokens": avg_prompt_tokens,
                "tps": tps,
                "ttps": ttps,
                "tp75_server_ttft": tp75_server_ttft,
                "tp90_server_ttft": tp90_server_ttft,
                "tp95_server_ttft": tp95_server_ttft,
                "tp99_server_ttft": tp99_server_ttft,
                "max_server_ttft": max_server_ttft,
                "avg_server_ttft": avg_server_ttft,
                "tp75_server_tpot": tp75_server_tpot,
                "tp90_server_tpot": tp90_server_tpot,
                "tp95_server_tpot": tp95_server_tpot,
                "tp99_server_tpot": tp99_server_tpot,
                "max_server_tpot": max_server_tpot,
                "avg_server_tpot": avg_server_tpot,
                "tp75_server_e2e": tp75_server_e2e,
                "tp90_server_e2e": tp90_server_e2e,
                "tp95_server_e2e": tp95_server_e2e,
                "tp99_server_e2e": tp99_server_e2e,
                "max_server_e2e": max_server_e2e,
                "avg_server_e2e": avg_server_e2e,
                "tp75_ttft": tp75_ttft,
                "tp90_ttft": tp90_ttft,
                "tp95_ttft": tp95_ttft,
                "tp99_ttft": tp99_ttft,
                "max_ttft": max_ttft,
                "avg_ttft": avg_ttft,
                "tp75_tpot": tp75_tpot,
                "tp90_tpot": tp90_tpot,
                "tp95_tpot": tp95_tpot,
                "tp99_tpot": tp99_tpot,
                "max_tpot": max_tpot,
                "avg_tpot": avg_tpot,
                "tp90_tpot_second": tp90_tpot_second,
                "tp95_tpot_second": tp95_tpot_second,
                "tp99_tpot_second": tp99_tpot_second,
                "max_tpot_second": max_tpot_second,
                "avg_tpot_second": avg_tpot_second,
                "tp90_time_between_first_and_second_token": tp90_time_between_first_and_second_token,
                "tp95_time_between_first_and_second_token": tp95_time_between_first_and_second_token,
                "tp99_time_between_first_and_second_token": tp99_time_between_first_and_second_token,
                "min_time_between_first_and_second_token": min_time_between_first_and_second_token,
                "max_time_between_first_and_second_token": max_time_between_first_and_second_token,
                "avg_time_between_first_and_second_token": avg_time_between_first_and_second_token,
                "tp75_e2e": tp75_e2e,
                "tp90_mtp_accept_rate": tp90_mtp_accept_rate,
                "tp95_mtp_accept_rate": tp95_mtp_accept_rate,
                "tp99_mtp_accept_rate": tp99_mtp_accept_rate,
                "max_mtp_accept_rate": max_mtp_accept_rate,
                "min_mtp_accept_rate": min_mtp_accept_rate,
                "avg_mtp_accept_rate": avg_mtp_accept_rate,
                "tp90_e2e": tp90_e2e,
                "tp95_e2e": tp95_e2e,
                "tp99_e2e": tp99_e2e,
                "max_e2e": max_e2e,
                "avg_e2e": avg_e2e,
                "queries_per_second": queries_per_second,
                "total_time": total_time,
                "fail_rate": fail_rate,
            })

    # profile stop.
    operate_profile(args.profile, "stop", api_key=provider.get("api_key"), base_url=provider.get("base_url"),
                    level=args.service_profile_level)

    return all_metrics


def main(args: argparse.Namespace):
    run_climbing(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the DeepSeek latency of processing a single batch of requests till completion.")
    parser.add_argument("--providers-path", type=str,
                        help="providers configuration file path")
    parser.add_argument("--backend", type=str, default="openai-chat", choices=["openai-chat", "openai"],
                        help="it determines which openai interface to use")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Float that controls the randomness of the sampling. Zero means greedy sampling.")
    parser.add_argument("--top-k", type=int, default=-1,
                        help="Integer that controls the number of top tokens to consider. "
                             "Set to -1 to consider all tokens.")
    parser.add_argument("--top-p", type=float, default=1,
                        help="Float that controls the cumulative probability of the top tokens to consider. "
                             "Must be in (0, 1]. Set to 1 to consider all tokens.")
    parser.add_argument("--dataset-dir", type=str,
                        help="test dataset directory")
    parser.add_argument("--benchmark-dir", type=str,
                        help="save benchmark directory")
    parser.add_argument("--server-prefill-concurrency", type=int, help="Number of prefill request number.")
    parser.add_argument("--server-decode-concurrency", type=int, help="Number of decode request number.")
    parser.add_argument("--concurrencies", nargs='+', type=int, default=[1, 8, 16, 32, 64, 96, 128],
                        help="Number of parallel request number.")
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096, 5600],
                        help="Max prompt tokens")
    parser.add_argument("--output-tokens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096, 5600],
                        help="Max output tokens")
    parser.add_argument("--control-method", choices=["queue", "round", "pd-adaptive"], default="queue",
                        help="the parallel function")
    parser.add_argument("--rounds", type=int, default=2, help="the round of batch")
    parser.add_argument("--num-prompts", nargs='+', type=int,
                        help="Number of prompts to process.")
    parser.add_argument("--growth-rate", type=int, help="Number of Climbing Growth Step")
    parser.add_argument("--server-statistics", action="store_true", help="Enabling Server Information Statistics")
    parser.add_argument("--profile", action="store_true",
                        help="Use Torch Profiler or Service Profiler."
                             "The vLLM service must be started with env VLLM_SERVICE_PROFILER_DIR set.")
    parser.add_argument("--service-profile-level", type=str, required=False,
                        choices=["Level_none", "Level0", "Level1", "Level2"],
                        help="Set Service Profiler level. Support Level_none, Level0, Level1, Level2."
                             "The default value is Level0.")
    parser.add_argument("--climbing-period", type=int, default=1000,
                        help="Time of intervals for the climbing, The default value is 1000 milliseconds.")
    parser.add_argument("--num-scheduler-steps", type=int, default=1)
    parser.add_argument("--use-mtp-accept-rate", type=bool, default=True,
                        help="the accept rate, mtp style accept rate will ignore the score model output token.")
    parser.add_argument("--use-spec-decode", action="store_true")
    parser.add_argument("--num-speculative-tokens", type=int, default=-1,
                        help="the step if spec decode, default -1 for disable accept rate statistic.")

    args = parser.parse_args()
    main(args)
