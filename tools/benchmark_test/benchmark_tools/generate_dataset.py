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

from benchmark_utils import generate_str, get_tokenizer

logger = logging.getLogger(__name__)

DATASET: List = []


def sample_requests(
    min_input: int,
    max_input: int,
    avg_input: int,
    std_input: int,
    min_output: int,
    max_output: int,
    avg_output: int,
    std_output: int,
    num_requests: int,
):
    input_samples = np.random.normal(loc=avg_input, scale=std_input, size=num_requests).astype(int)

    input_samples = np.clip(input_samples, min_input, max_input)

    output_samples = np.random.normal(loc=avg_output, scale=std_output, size=num_requests).astype(int)

    output_samples = np.clip(output_samples, min_output, max_output)

    return input_samples, output_samples


def get_outputs(
    tokenizer,
    input_len: int,
    output_len: int,
) -> None:
    input_str = generate_str(tokenizer, input_len)
    output_str = generate_str(tokenizer, output_len)

    request = {
        "input": input_str,
        "output": output_str,
        "instruction": "NA"
    }
    DATASET.append(request)


def do_generate(
    tokenizer,
    input_samples,
    output_samples
) -> None:
    for i in tqdm(range(input_samples.shape[0])):
        input_len = input_samples[i]
        output_len = output_samples[i]
        get_outputs(tokenizer, input_len, output_len)


def get_specific_prompt(user_prompt, tokenizer, length):
    tokenizer = get_tokenizer(tokenizer)
    token_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
    total_length = len(token_ids)
    if total_length < length:
        return None

    return tokenizer.decode(token_ids[:length])


def get_specified_dataset(args):
    if not args.input_length:
        raise ValueError("The input_length parameter is required when dataset_type is fixed.")
    expected_json = []
    idx = 0
    for filename in os.listdir(args.dataset):
        if not filename.endswith("jsonl"):
            continue
        sub_task_path = os.path.join(args.dataset, filename)
        with open(sub_task_path, encoding="utf-8") as f:
            for line in tqdm(f, desc=f"{filename}", unit="lines"):
                item = json.loads(line)
                if "context" not in item.keys():
                    raise ValueError("The {filename} does not contain the context parameter"
                                     "or only the long bench dataset is supported.".format(filename=filename))
                context = item["context"]
                expected_user_prompt = get_specific_prompt(context, args.tokenizer, args.input_length)
                if not expected_user_prompt:
                    continue
                expected_json.append({"id": idx, "input": expected_user_prompt})
                idx += 1
                if idx >= args.num_requests:
                    break
            logger.info("{idx} prompts that meet the search criteria have been filtered.".format(idx=idx))
            if idx >= args.num_requests:
                break
    logger.info("{num} prompts are filtered.".format(num=len(expected_json)))
    with os.fdopen(os.open("{input_length}.json".format(input_length=args.input_length),
                           os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                           stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP), "w") as f:
        json.dump(expected_json, f, indent=4)


def main(args: argparse.Namespace):
    """
    main entry
    """
    if args.dataset_type == "fixed":
        get_specified_dataset(args)
    else:
        print(args)
        random.seed(args.seed)
        np.random.seed(args.seed)
        tokenizer = get_tokenizer(args.tokenizer)
        input_samples, output_samples = sample_requests(
            args.min_input,
            args.max_input,
            args.avg_input,
            args.std_input,
            args.min_output,
            args.max_output,
            args.avg_output,
            args.std_output,
            args.num_requests,
        )

        do_generate(tokenizer, input_samples, output_samples)

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        mode = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(args.dataset, flags, mode), 'w') as f:
            json.dump(DATASET, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")

    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--dataset-type", default="random",
                        choices=["random", "fixed"],
                        help="Dataset generation mode, only LongBench dataset is supported."
                             "LongBench open source address:https://huggingface.co/datasets/THUDM/LongBench/tree/main ")
    parser.add_argument("--input-length", type=int,
                        help="Truncates a dataset with a specified length only when dataset-type is fixed.")
    parser.add_argument("--min-input", type=int, default=100,
                        help="Number of min-input to process.")
    parser.add_argument("--max-input", type=int, default=3600,
                        help="Number of max-input to process.")
    parser.add_argument("--avg-input", type=int, default=1800,
                        help="Number of avg-input to process.")
    parser.add_argument("--std-input", type=int, default=500,
                        help="Number of std-input to process.")
    parser.add_argument("--min-output", type=int, default=40,
                        help="Number of min-output to process.")
    parser.add_argument("--max-output", type=int, default=256,
                        help="Number of max_output to process.")
    parser.add_argument("--avg-output", type=int, default=160,
                        help="Number of avg-output to process.")
    parser.add_argument("--std-output", type=int, default=30,
                        help="Number of std-output to process.")
    parser.add_argument("--num-requests", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    args_global = parser.parse_args()
    main(args_global)
