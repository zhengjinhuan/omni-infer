import argparse
import requests
import json
from multiprocessing import Pool
import os
import torch
import functools

def parse_args():
    parser = argparse.ArgumentParser(description="Simple curl")

    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()
    return parser, args


def call_one(vocab_size, input_len, model_name, url, headers):
    input = torch.randint(vocab_size, size=(input_len)).tolist()
    data = {
        "model": model_name,
        "max_tokens": 2,
        "include_stop_str_in_output": True,
        "prompt": input,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()
    else:
        return {"Error": response.status_code}

def run_requests(
        max_concurrency, ip, port, vocab_size, input_len, num_requests, model_name):
    url = f"http://{ip}:{port}/v1/completions"
    headers = {"Content-Type": "application/json"}
    partial_call_one = functools.partial(call_one, input_len=input_len, model_name=model_name, url=url, headers=headers)
    
    with Pool(max_concurrency) as pool:
        results = pool.map(partial_call_one, [vocab_size] * num_requests)
    
    return results

def main():
    parser, args = parse_args()
    results = run_requests(
        max_concurrency=args.max_concurrency,
        ip=args.ip,
        port=args.port,
        vocab_size=args.vocab_size,
        input_len=args.input_len,
        model_name=args.model_name,
    )
    if args.output is not None:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f)
        

if __name__ == "__main__":
    main()

