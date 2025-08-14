// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <stdint.h>

#define NUM_PREFILL_BATCH_METRICS_HIS 10
#define NUM_DECODE_BATCH_METRICS_HIS 10
#define MAX_PREFILL_UPSTREAMS 64
#define MAX_DECODE_UPSTREAMS 512
#define MAX_REQUEST_SLOTS 8192
#define MAX_WORKERS 320

typedef enum omni_proxy_request_phase
{
    PHASE_TOKENIZING,
    PHASE_APC_MATCHING,
    PHASE_PREFILL_WAITING_SCHEDULE,
    PHASE_PREFILL_SCHEDULED,
    PHASE_PREFILLING,
    PHASE_DECODE_WAITING_SCHEDULE,
    PHASE_DECODE_SCHEDULED,
    PHASE_DECODING,
    PHASE_MAX
} omni_proxy_request_phase_t;

typedef struct omni_request_metrics_s
{
    uint32_t prompt_num_tokens;
    uint32_t decoded_tokens;

    uint32_t time_received;
    uint32_t time_contents_received;
    uint32_t time_tokenized;
    uint32_t time_apc_updated;

    uint32_t time_enter_wait_prefill;
    uint32_t time_prefill_scheduled;
    uint32_t time_to_prefill;
    uint32_t time_prefilled;

    uint32_t time_enter_wait_decode;
    uint32_t time_decode_scheduled;
    uint32_t time_to_decode;

    uint32_t time_last_reponse;
    uint16_t time_first_token;
    uint16_t tpot;
} omni_request_metrics_t;

typedef struct omni_request_s
{
    uint16_t in_use;
    uint16_t slot_index;
    void *backend;
    uint32_t last_retry;
    int worker_pid;
    uint16_t upstream_endpoint_idx;
    uint32_t phase_state;
    omni_request_metrics_t metrics;
} omni_req_t;

typedef struct omni_request_pool_s
{
    // Each bit represent a request, 1 for in use.
    uint64_t in_use_bitmap[MAX_REQUEST_SLOTS / 64];
    // The number of request current in use
    uint32_t num_requests;
    uint32_t head;
    omni_req_t slots[MAX_REQUEST_SLOTS];
} omni_request_pool_t;

typedef struct omni_req_info_s
{
    uint32_t in_use;
    uint32_t slot_index;
    double weight;
} omni_req_info_t;

typedef struct omni_req_group_s
{
    uint32_t num_requests;
    uint32_t watermark;
    omni_proxy_request_phase_t phase;
    omni_req_info_t requests[MAX_REQUEST_SLOTS];
} omni_req_group_t;

typedef struct omni_batch_metrics_s
{
    uint32_t num_requests;
    uint32_t num_tokens;
    uint32_t time_taken; // Since the oldest request responded in this batch
    uint32_t first_response_receive_time;
    uint32_t last_response_receive_time;
    uint32_t average_delta;
} omni_batch_metrics_t;

typedef struct omni_batch_metrics_his_s
{
    uint32_t head;
    uint32_t count;
    omni_batch_metrics_t his[NUM_PREFILL_BATCH_METRICS_HIS];
} omni_batch_metrics_his_t;

#define UPSTREAM_NAME_MAX 64
#define UPSTREAM_IP_MAX 16

typedef struct omni_upstream_prefill_s
{
    uint32_t index;
    char name[UPSTREAM_NAME_MAX];
    char ip[UPSTREAM_IP_MAX];
    int port;
    uint32_t num_running;
    uint32_t num_tokens;
    uint32_t last_scheduled_time;
    uint32_t expected_next_schedule_time;
    omni_batch_metrics_his_t his;
} omni_upstream_prefill_t;

typedef struct omni_upstream_decode_s
{
    uint32_t index;
    uint32_t num_running;
    uint32_t generated_tokens;
    uint32_t expected_next_schedule_time;
    omni_batch_metrics_his_t his;
} omni_upstream_decode_t;

typedef enum omni_proxy_pd_policy_s
{
    PD_SEQUENTIAL,
    PD_PARALLEL
} omni_proxy_pd_policy_t;

typedef struct omni_global_state_s
{
    int magic;
    ngx_shmtx_t shmtx;
    ngx_shmtx_sh_t lock;

    omni_proxy_pd_policy_t pd_policy;

    omni_request_pool_t request_pool;
    omni_req_group_t groups[PHASE_MAX];
    uint16_t num_prefill_endpoints;
    uint16_t last_selected_prefill;
    uint16_t num_decode_endpoints;
    uint16_t last_selected_decode;
    uint32_t last_summary;
    uint32_t num_workers;
    int workers[MAX_WORKERS];
    omni_upstream_prefill_t prefill_states[MAX_PREFILL_UPSTREAMS];
    omni_upstream_decode_t decode_states[MAX_DECODE_UPSTREAMS];
} omni_global_state_t;

#define GLOBAL_STATE_SIZE sizeof(omni_global_state_t)
