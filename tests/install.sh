#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

set -e

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=${OUTPUT_DIR-"output"}
export INPUT_PATH=/home/ma-user/modelarts/inputs/data_url_0
export OUTPUT_PATH=/home/ma-user/modelarts/outputs/train_url_0

source "${SCRIPT_PATH}/log_utils.sh"

# CI -> "level0" | Daily version -> "level0 or level1"
TEST_LEVEL=${TEST_LEVEL-"level0"}

# PD_SEPARATION_FLAG: 0 PD不分离，1 PD分离
PD_SEPARATION_FLAG=${PD_SEPARATION_FLAG-"0"}

if [ ${TEST_LEVEL} = "level0" ]; then
    PD_SEPARATION_FLAG="1"
fi

log_info "TEST_LEVEL: ${TEST_LEVEL}"
log_info "PD_SEPARATION_FLAG: ${PD_SEPARATION_FLAG}"

cd $SCRIPT_PATH

log_info "安装CI依赖 & 卸载旧版本"
pip config set global.timeout 100
pip config set global.retries 10
pip install -U pip -q
pip install -r requirements-ci.txt -q

log_info "拉vllm和vllm_ascend代码"
cd $SCRIPT_PATH/..
if [ ${TEST_LEVEL} = "level0" ]; then
    log_info "Mock model"
    bash build/build.sh --ci "1"
    log_info "DeepSeek-V3-w8a8-0423 模型减层(61 -> 4)"
    sed -i 's#"num_hidden_layers": 61#"num_hidden_layers": 4#g' ${INPUT_PATH}/model/DeepSeek-V3-w8a8-0423/config.json
else
    bash build/build.sh
fi

WHL_PKGS_COUNT=`ls $SCRIPT_PATH/../build/dist/ | wc -l` 
if [ "$WHL_PKGS_COUNT" = "2" ]; then
    log_info "whl包安装vllm和vllm_ascend"
    pip install $SCRIPT_PATH/../build/dist/vllm*
else
    log_info "源码安装vllm"
    cd $SCRIPT_PATH/../infer_engines/vllm
    VLLM_TARGET_DEVICE=empty pip install -e . -q

    log_info "源码安装vllm_ascend"
    cd $SCRIPT_PATH/../infer_engines/vllm_ascend
    pip install -e . -q
fi

log_info "安装高性能torch_npu包"
cd $SCRIPT_PATH/..
mkdir pta
cd pta
tar -xf ${INPUT_PATH}/whl/pytorch_v2.5.1_py310.tar.gz
pip install torch_npu-2.5.1.dev20250519-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
