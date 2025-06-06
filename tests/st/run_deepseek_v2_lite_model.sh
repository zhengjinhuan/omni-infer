#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

set -e

# 获取当前脚本的绝对路径并进入
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_PATH
OUTPUT_DIR=${OUTPUT_DIR-"output"}

#单机单卡离线化推理
export VLLM_ENABLE_MC2=0
export USING_LCCL_COM=0
python test_dp2_tp2_ep4.py