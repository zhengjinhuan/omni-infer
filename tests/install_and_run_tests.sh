#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

set -e

# 查看当前脚本所在的绝对路径
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_PATH

# 导入日志工具函数
source "$SCRIPT_PATH/log_utils.sh"

export INPUT_PATH=/home/ma-user/modelarts/inputs/data_url_0
export OUTPUT_PATH=/home/ma-user/modelarts/outputs/train_url_0
OUTPUT_DIR=${OUTPUT_DIR-"output"}

log_info "安装CI依赖"
pip config set global.index-url https://mirrors.tools.huawei.com/pypi/simple
pip config set global.trusted-host mirrors.tools.huawei.com
pip config set global.timeout 100
pip config set global.retries 10
pip install -U pip -q
pip install -r $SCRIPT_PATH/requirements-ci.txt -q

log_info "卸载旧版本vllm和vllm_ascend"
pip uninstall vllm -y && pip uninstall vllm_ascend -y

log_info "Applying mock model"
sed -i "s#./bash_install_code.sh#./bash_install_mock.sh#g" $SCRIPT_PATH/../build/build.sh

log_info "拉vllm和vllm_ascend代码构建whl包"
cd $SCRIPT_PATH/..
bash build/build.sh

log_info "安装vllm、vllm-ascend以及UT依赖库"
# pip install $SCRIPT_PATH/../build/dist/*
cd $SCRIPT_PATH/../infer_engines/vllm
VLLM_TARGET_DEVICE=empty pip install -e .
cd $SCRIPT_PATH/../infer_engines/vllm_ascend
pip install -e .

log_info "安装低版本urllib3"
pip install urllib3==1.26.7 -q

log_info "安装高性能torch_npu包"
cd $SCRIPT_PATH/..
mkdir pta && cd pta
tar -xf ${INPUT_PATH}/whl/pytorch_v2.5.1_py310.tar.gz
pip install torch_npu-2.5.1.dev20250519-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

log_info "DeepSeek-V3-w8a8-0423 减层"
sed -i 's#"num_hidden_layers": 61#"num_hidden_layers": 5#g' ${INPUT_PATH}/model/DeepSeek-V3-w8a8-0423/config.json
cd $SCRIPT_PATH/../infer_engines/vllm
git apply ${SCRIPT_PATH}/reduce_the_num_of_hidden_layers_of_deepseek_v3_w8a8.patch

log_info "执行用例"
bash $SCRIPT_PATH/start_tests.sh