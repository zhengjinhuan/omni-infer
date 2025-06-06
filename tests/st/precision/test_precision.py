#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# All rights reserved.

import logging
import json
import requests
import pytest
from tests.mark_utils import arg_mark
from tests.st.scripts.utils import check_service_status


@arg_mark(['platform_ascend910b'], 'level0')
def test_precision(host):

    if check_service_status(host):
        logging.info("Service started successfully.")
    else:
        assert False, "Service started failed."

    url = f"http://{host}/v1/completions"
    headers = {"Content-Type": "application/json"}
    json_data = {
        "model": "deepseek",
        "prompt": ["计算365乘以24"],
        "max_tokens": 50,
        "temperature": 0,
        "top_p": 1,
        "top_k": -1
    }

    response = requests.post(url, headers=headers, json=json_data)

    if response.status_code == 200:
        data = response.json()
        # 检查是否有内容返回
        assert data, "API 返回内容为空"
        if not data:
            assert False, "API 返回内容为空"
        else:
            logging.info(f"API 返回内容不为空,data:{data}")

        # deepseek_v3检查返回内容中是否有 'think' 标签
        # if 'think' in data:
        #     logging.info("返回内容中包含 'think' 标签")
        # else:
        #     assert False, "返回内容中不包含 'think' 标签"
        #
        # # 检查 'reasoning_content' 内容是否为空
        # if 'reasoning_content' in data:
        #     reasoning_content = data['reasoning_content']
        #     if reasoning_content:
        #         logging.info("reasoning_content 不为空")
        #     else:
        #         assert False, "reasoning_content 为空"
        # else:
        #     assert False, "'reasoning_content' 标签不存在"
    else:
        assert False, f"请求失败，状态码: {response.status_code}， 响应结果: {response.content}"


if __name__ == "__main__":
    pytest.main()

