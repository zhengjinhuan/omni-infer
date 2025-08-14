// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <stdint.h>
#include <assert.h>
#include <ngx_atomic.h>
#include <omni_metrics.h>

typedef struct omni_worker_local_state_s
{
    pid_t pid;
    uint32_t num_prefill_endpoints;
    uint32_t num_decode_endpoints;
    ngx_shmtx_t g_shmtx;
    ngx_event_t omni_proxy_timer_event;
    ngx_http_output_body_filter_pt ngx_http_next_body_filter;
    omni_req_group_t groups[PHASE_MAX];
} omni_worker_local_state_t;

typedef struct
{
    struct sockaddr *sockaddr;
    socklen_t socklen;
    ngx_int_t port; /* port number */
    ngx_str_t text; /* textual host:port (nul-terminated) */
    ngx_str_t index;
} ngx_omni_backend_t;

typedef struct
{
    ngx_str_t name;        /* upstream name */
    ngx_array_t *backends; /* array of ngx_omni_backend_t (allocated in main conf pool) */
} ngx_omni_upstream_entry_t;

/* module main conf: holds parsed upstreams (mapping name -> backends) */
typedef struct
{
    ngx_array_t *entries;                     /* array of ngx_omni_upstream_entry_t */
    ngx_http_upstream_srv_conf_t **upstreams; // reference to upstreams array from ngx_http_upstream_main_conf_t
                                              /* reference to upstreams array from ngx_http_upstream_main_conf_t */
} ngx_http_omni_main_conf_t;

/* location conf: stores the upstream name and rr index */
typedef struct
{
    ngx_str_t upstream_name;
    ngx_uint_t rr_index;
    ngx_http_upstream_srv_conf_t *upstream;
} ngx_http_omni_loc_conf_t;

typedef struct omni_req_context_s
{
    ngx_array_t *backends;
    ngx_http_upstream_conf_t upstream;

    omni_req_t *req;
    u_char *origin_body_data;
    ngx_uint_t origin_body_data_size;
    void *origin_body_tokens;
    int origin_body_tokens_size;
    u_char *prefill_response_body;
    ngx_uint_t prefill_response_body_size;
} omni_req_context_t;
