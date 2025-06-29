# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

export GLOO_SOCKET_IFNAME=enp23s0f3
export TP_SOCKET_IFNAME=enp23s0f3
export ASCEND_RT_VISIBLE_DEVICES=15
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_ENABLE_MC2=0
export USING_LCCL_COM=0

export OMNI_USE_QWEN=1

python start_api_servers.py \
        --num-servers 1 \
        --model-path /home/yyx/vllm_ascend/ram_data/Qwen2.5-7B-Instruct \
        --master-ip 7.150.13.168 \
        --tp 1 \
        --master-port 35678 \
        --served-model-name deepseek \
        --log-dir apiserverlog \
        --extra-args "--enforce-eager " \
        --base-api-port 9555