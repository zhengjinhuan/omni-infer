// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct
{
    const char *input_data;
    size_t input_len;

    char *prompt;
    size_t prompt_buf_size;
    size_t prompt_len;

    int *input_ids;
    size_t input_ids_buf_size;
    size_t input_ids_len;

    int multi_modal_size;
} omni_tokenizer_request;

int omni_tokenizer_init();
void omni_tokenizer_cleanup();
int omni_init_tokenizer(const char *model_path);
int omni_batch_chat_encode(omni_tokenizer_request **requests, size_t num_reqs);
