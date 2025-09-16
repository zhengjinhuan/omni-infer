import json
from multiprocessing import Pool
import os
import requests
import sys
import time
import torch

def get_call_function(ip, port, model_name):
    url = f"http://{ip}:{port}/v1/completions"
    headers = {"Content-Type": "application/json"}
    def make_api_call(input):
        data = {
            "model": model_name,
            "temperature": 0.6,
            "max_tokens": 2,
            "include_stop_str_in_output": True,
            "prompt": input["prompt_token_ids"] + input['output_token_ids']
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))

        return response.status_code == 200
    return make_api_call

def main(ip, port, model_name, n_num, datafile):
    call_function = get_call_function(ip, port, model_name)
    data = torch.load(datafile)
    with Pool(n_num) as pool:
        results = pool.map(call_function, data)
    return results

if __name__ == "__main__":
    ip = "7.150.13.75"
    port = 7000
    model_name = "qwen"
    datafile = sys.argv[1]
    n_num = 64
    
    results = main(ip, port, model_name, n_num, datafile)

    print(f"succeed case : {sum(results)} / {len(results)}")
    

