#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import pytest


def pytest_addoption(parser):
    parser.addoption("--output_dir",
                     action="store",
                     default="/home/ma-user/modelarts/outputs/train_url_0/output", help="Parameter for the test")
    parser.addoption("--forward_time", action="store", default="0.01", help="Parameter for mock model infer time")
    parser.addoption("--except_time", action="store", default="0.003", help="Parameter for 3.5k+1k(+-0.005ms) ")
    parser.addoption("--tokenizer",
                     action="store",
                     default="/home/CI/models/DeepSeek-V3-w8a8-0423",
                     help="Parameter for model path")
    parser.addoption("--served_model_name", action="store", default="deepseek", help="Parameter for server name")
    parser.addoption("--host", action="store", default="127.0.0.1:8000", help="Parameter for host")


@pytest.fixture
def output_dir(request):
    return request.config.getoption("--output_dir")


@pytest.fixture
def forward_time(request):
    return request.config.getoption("--forward_time")


@pytest.fixture
def except_time(request):
    return request.config.getoption("--except_time")


@pytest.fixture
def tokenizer(request):
    return request.config.getoption("--tokenizer")


@pytest.fixture
def served_model_name(request):
    return request.config.getoption("--served_model_name")


@pytest.fixture
def host(request):
    return request.config.getoption("--host")
