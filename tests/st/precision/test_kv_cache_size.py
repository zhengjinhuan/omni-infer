# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# All rights reserved.

import re

import pytest
from tests.mark_utils import arg_mark


@arg_mark(['platform_ascend910b'], 'level0')
def test_precision_chat_level0(pd_server_log_path, server_mode):
    server_mode = int(server_mode)
    if server_mode == 0:
        target = 93000
    else:
        target = 116224

    def extract_number_from_log(content):
        # 定义正则表达式模式以匹配特定格式的日志行
        pattern = r'GPU KV cache size: (\d{1,3}(?:,\d{3})*) tokens'
        match = re.search(pattern, content)
        if match:
            return match.group(1).replace(',', '')
        else:
            return None

    # 示例日志行
    with open(pd_server_log_path, "r", encoding="utf-8") as f:
        content = f.read()
        number = int(extract_number_from_log(content))

        if number:
            if number < (target * 0.95):
                assert False, f"KV cache size实际值{number}, 小于基准值{target}的95%"
        else:
            assert False, f"日志中未打印GPU KV cache size"


if __name__ == "__main__":
    pytest.main()
