#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# All rights reserved.

import os
import logging
import torch
import torch_npu


from vllm import LLM, SamplingParams

os.environ["VLLM_USE_V1"] = "1"

#测试单机单卡离线推理
if __name__ == "__main__":
    prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(max_tokens=100, temperature=0.8, top_p=0.95)
    llm = (
        LLM(
            model="/home/ma-user/modelarts/inputs/data_url_0/DeepSeek-V2-Lite",
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=1024,
            gpu_memory_utilization=0.9
        )
    )
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        logging.info(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
