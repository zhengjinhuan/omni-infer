import argparse
import asyncio
import json
import os
import random
import stat
import time
from typing import AsyncGenerator, List, Tuple, Union, Optional
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

import logging
import aiohttp
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from benchmark_utils import extract_str, get_tokenizer, print_data_info


INPUT_DATASET: List = []
DATASET: List = []


def sample_requests(
    min_input: int,
    max_input: int,
    avg_input: int,
    std_input: int,
    num_requests: int,
):
    input_samples = np.random.normal(loc=avg_input, scale=std_input, size=num_requests).astype(int)

    input_samples = np.clip(input_samples, min_input, max_input)

    return input_samples


def get_outputs(
    tokenizer,
    input_len: int,
    index: int
) -> None:
    input_str = extract_str(tokenizer, INPUT_DATASET[index]["input"], input_len)

    request = {
        "input": input_str,
        "instruction": "NA"
    }
    DATASET.append(request)


def do_generate(
    tokenizer,
    input_samples,
) -> None:
    for i in tqdm(range(input_samples.shape[0])):
        input_len = input_samples[i]
        get_outputs(tokenizer, input_len, i)


def extract_dataset(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        index = 0
        for line in f:
            chat = json.loads(line)
            output = chat['output']
            INPUT_DATASET.append({
                "index": index,
                "input": output
            })
            index = index + 1


def main(args: argparse.Namespace):
    """
    main entry
    """
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    extract_dataset(args.input_dataset)

    tokenizer = get_tokenizer(args.tokenizer)
    num_requests = args.num_requests if args.num_requests <= len(INPUT_DATASET) else len(INPUT_DATASET)
    input_samples = sample_requests(
        args.min_input,
        args.max_input,
        args.avg_input,
        args.std_input,
        num_requests,
    )
    do_generate(tokenizer, input_samples)

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    mode = stat.S_IWUSR | stat.S_IRUSR

    dataset_path = args.dataset
    # 避免目录不存在报错
    r_slashes_index = dataset_path.rfind('/')
    if r_slashes_index > 0:
        save_dir = dataset_path[0:r_slashes_index]
        os.makedirs(save_dir, exist_ok=True)

    with os.fdopen(os.open(dataset_path, flags, mode), 'w') as f:
        json.dump(DATASET, f, ensure_ascii=False)
    print_data_info(dataset_path, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")

    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--input-dataset", type=str, required=True,
                        help="original dataset.")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--min-input", type=int, default=100,
                        help="Number of min-input to process.")
    parser.add_argument("--max-input", type=int, default=3600,
                        help="Number of max-input to process.")
    parser.add_argument("--avg-input", type=int, default=1800,
                        help="Number of avg-input to process.")
    parser.add_argument("--std-input", type=int, default=500,
                        help="Number of std-input to process.")
    parser.add_argument("--num-requests", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    args_global = parser.parse_args()
    main(args_global)
