import aiohttp
import argparse
import asyncio
import csv
import glob
import json
import numpy as np
import os
import pandas as pd
import random
import subprocess
import time
from tqdm.asyncio import tqdm_asyncio
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
from transformers import PreTrainedTokenizerBase
from typing import AsyncGenerator, List, Tuple, Union


SECONDS_ONE_MIN = 60


def main(args: argparse.Namespace):
    """
    main entry
    """
    print(args)
    start_time = time.perf_counter()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Benchmark serving with alpaca dataset ...")
    subprocess.run(
        f"python benchmark_serving.py --backend {args.backend} --host {args.host} --port {args.port} "
        f"--tokenizer {args.tokenizer} --dataset {args.alpaca_dataset} --dataset-type alpaca "
        f"--request-rate 0.05 1 2 4 8 10 20 30 --num-prompts 6 100 1000 1000 1000 1000 1000 1000 "
        f"--max-tokens 1024 --max-prompt-tokens 900 "
        f"--benchmark-csv {os.path.join(output_dir, 'benchmark_serving_alpaca.csv')}", check=True)

    benchmark_alpaca_time = (time.perf_counter() - start_time) / SECONDS_ONE_MIN
    print(f"Alpaca dataset cost time {benchmark_alpaca_time:.1f} min")
    print()

    print("Benchmark serving with sharegpt dataset ...")
    sharegpt_start_time = time.perf_counter()
    subprocess.run(
        f"python benchmark_serving.py --backend {args.backend} --host {args.host} --port {args.port} "
        f"--tokenizer {args.tokenizer} --dataset {args.sharegpt_dataset} --dataset-type sharegpt "
        f"--request-rate 0.05 1 2 4 8 10 20 30 --num-prompts 6 1000 1000 1000 1000 1000 1000 1000 "
        f"--max-tokens 1024 --max-prompt-tokens 900 "
        f"--benchmark-csv {os.path.join(output_dir, 'benchmark_serving_sharegpt.csv')}", check=True)
    benchmark_sharegpt_time = (time.perf_counter() - sharegpt_start_time) / SECONDS_ONE_MIN
    print(f"Sharegpt dataset cost time {benchmark_sharegpt_time:.1f} min")
    print()

    print("Benchmark parallel ...")
    parallel_start_time = time.perf_counter()
    benchmark_parallel_csv = os.path.join(output_dir, 'benchmark_parallel.csv')
    subprocess.run(
        f"python benchmark_parallel.py --backend {args.backend} --host {args.host} --port {args.port} "
        f"--tokenizer {args.tokenizer} --epochs 5 --parallel-num 1 2 4 8 "
        f"--output-tokens 256 256  --prompt-tokens 512 1024 --benchmark-csv {benchmark_parallel_csv}", check=True)
    parallel_parallel_time = (time.perf_counter() - parallel_start_time) / SECONDS_ONE_MIN
    print(f"Benchmark parallel cost time {parallel_parallel_time:.1f} min")
    print()

    all_filenames = [i for i in glob.glob(os.path.join(output_dir, 'benchmark_serving_*.csv'))]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    benchmark_serving_csv = os.path.join(output_dir, 'benchmark_serving.csv')
    combined_csv.to_csv(benchmark_serving_csv, index=False, encoding='utf-8-sig')

    try:
        writer = pd.ExcelWriter(os.path.join(output_dir, 'output.xlsx'))
        data1 = pd.read_csv(benchmark_serving_csv, encoding="utf-8-sig")
        data2 = pd.read_csv(benchmark_parallel_csv, encoding="utf-8-sig")
        data1.to_excel(writer, sheet_name='benchmark_serving')
        data2.to_excel(writer, sheet_name='benchmark_parallel')
        writer._save()
        writer.close()
    except Exception:
        print("Excel Exception")

    total_time = (time.perf_counter() - start_time) / SECONDS_ONE_MIN
    print(f"Benchmark all finished, cost time {total_time:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--backend", type=str, default="mindspore",
                        choices=["vllm", "mindspore", "base", "tgi", "openai", "trt"])
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9288)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--app-code", type=str, default=None)
    parser.add_argument("--alpaca-dataset", type=str, required=True,
                        help="Path to the alpaca dataset.")
    parser.add_argument("--sharegpt-dataset", type=str, required=True,
                        help="Path to the alpaca dataset.")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Dir to the output.")

    args_global = parser.parse_args()
    main(args_global)
