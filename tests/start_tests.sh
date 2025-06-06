#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

TEST_LEVEL=${TEST_LEVEL-"level0"}
PD_SEPARATION_FLAG=${PD_SEPARATION_FLAG-"0"} # 0 表示不分离，1 表示分离
# 获取当前脚本的绝对路径并进入
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_PATH
chmod -R +x ./*

# 导入日志工具函数
source "$SCRIPT_PATH/log_utils.sh"

OUTPUT_DIR=${OUTPUT_DIR-"output"}
INPUT_PATH=${INPUT_PATH-"/home/ma-user/modelarts/inputs/data_url_0"}
OUTPUT_PATH=${OUTPUT_PATH-"/home/ma-user/modelarts/outputs/train_url_0"}
TASK_OUTPUT_PATH=${OUTPUT_PATH}/${OUTPUT_DIR}

export PYTHONPATH=${PYTHONPATH}:$SCRIPT_PATH/..:$SCRIPT_PATH/../tools/llm_evaluation/benchmark_tools

# 根据TEST_LEVEL确定跑哪些用例
if [ "${TEST_LEVEL}" = "level0" ]; then
    TEST_DIR="./st" # 门禁
else
    TEST_DIR="./st ./perf_test" # 每日构建
fi

if [ -d "${TASK_OUTPUT_PATH}" ]; then
    log_info "门禁输出目录存在"
else
    log_info "新建门禁输出目录"
    mkdir -p ${TASK_OUTPUT_PATH}
fi

MODEL_PATH="${INPUT_PATH}/model/DeepSeek-V3-w8a8-0423"
log_info "指定模型路径: ${MODEL_PATH}"

log_info "部署vllm在线服务"
touch ${TASK_OUTPUT_PATH}/server.log
# PD不分离模式下，需要调用start_tests.sh启动服务后再执行用例。
# PD分离模式下，需要先启动服务再调用start_tests.sh执行用例。
if [ "${PD_SEPARATION_FLAG}" = "0" ]; then
    source $SCRIPT_PATH/env.sh
    source $SCRIPT_PATH/perf_test/bencnmark/mock_model_env.sh
    nohup bash start_vllm_server/deepseek_dp8_tp1_ep1.sh ${MODEL_PATH}/ &> ${TASK_OUTPUT_PATH}/server.log &

    log_info "进入循环等待服务启动标识符出现"
    cost=0
    interval=10
    endtime=300
    while true; do
        if [ $cost -gt $endtime ]; then
            log_error "等待服务启动时间超过${endtime}秒，退出循环"
            log_error "服务启动日志结尾部分内容:"
            tail -n 50 ${TASK_OUTPUT_PATH}/server.log
            exit 1
        fi
        if grep -q "Application startup complete" ${TASK_OUTPUT_PATH}/server.log; then
            log_info "服务已启动，继续执行用例"
            log_info "服务启动日志开头部分内容:"
            head -n 50 ${TASK_OUTPUT_PATH}/server.log
            break
        elif grep -q "NPU out of memory" ${TASK_OUTPUT_PATH}/server.log; then
            log_error "服务启动失败，NPU out of memory"
            log_error "服务启动日志结尾部分内容:"
            tail -n 50 ${TASK_OUTPUT_PATH}/server.log
            exit 1
        else
            log_warning "服务启动中，等待${interval}秒"
            sleep ${interval}
            cost=$((cost + interval))
        fi
    done
else
    echo "默认PD分离服务已启动"
fi

log_info "执行ST/Perf_Test用例"
cd $SCRIPT_PATH

if [ "${PD_SEPARATION_FLAG}" = "0" ]; then
    pytest -vsra --disable-warnings -m "${TEST_LEVEL}" \
    --ignore=st/test_dp2_tp2_ep4.py \
    --html=${TASK_OUTPUT_PATH}/pytest_report.html \
    --ignore=st/cloud_tests_imp \
    --ignore=st/test_dp2_tp2_ep4.py \
    --ignore=st/test_edit_distance/test_edit_distance.py \
    ${TEST_DIR} \
    --host="127.0.0.1:8000"
else
    pytest -vsra --disable-warnings -m "${TEST_LEVEL}" \
    --ignore=st/test_dp2_tp2_ep4.py \
    --html=${TASK_OUTPUT_PATH}/pytest_report_2p1d.html \
    --ignore=st/cloud_tests_imp \
    --ignore=st/test_dp2_tp2_ep4.py \
    --ignore=st/test_edit_distance/test_edit_distance.py \
    ${TEST_DIR} \
    --host="127.0.0.1:85"
fi

test_result=$?
# 检查退出状态码，如果不为0则表示有测试失败
if [ $test_result -ne 0 ]; then
    log_error "测试失败, 停止vllm服务"
    exit 1
else
    echo "测试例100%通过"
fi

log_info "停止vllm服务"
pkill -f python3

# 关闭Mock model
if [ ${TEST_LEVEL} != "level0" ]; then
    unset RANDOM_MODE
    pytest -vsra --disable-warnings -m ${TEST_LEVEL} ./st/test_edit_distance/test_edit_distance.py

    test_result=$?
    # 检查退出状态码，如果不为0则表示有测试失败
    if [ $test_result -ne 0 ]; then
        log_error "测试失败, 停止vllm服务"
        exit 1
    else
        echo "测试例100%通过"
    fi
fi