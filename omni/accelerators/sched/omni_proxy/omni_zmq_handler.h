// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#pragma once

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <zmq.h>

struct ngx_omni_zmq_handler_s;

typedef void (*omni_zmq_msg_callback_t)(struct ngx_omni_zmq_handler_s *handler,
                                        const char *topic,
                                        const void *message,
                                        size_t length);

typedef struct ngx_omni_zmq_handler_s
{
    void *zmq_context;
    void *zmq_socket;
    ngx_int_t active;
    ngx_str_t zmq_address;
    ngx_str_t subscribe_topic;
    omni_zmq_msg_callback_t message_callback;
    ngx_log_t *log;
    ngx_cycle_t *cycle;
    ngx_connection_t *zmq_connection;
    ngx_event_t *zmq_event;
} ngx_omni_zmq_handler_t;

ngx_int_t omni_zmq_handler_reinit(ngx_omni_zmq_handler_t *handler);

ngx_int_t omni_zmq_handler_init(ngx_cycle_t *cycle,
                                ngx_omni_zmq_handler_t *handler,
                                ngx_str_t zmq_address,
                                ngx_str_t subscribe_topic,
                                omni_zmq_msg_callback_t callback);

void omni_zmq_handler_exit(ngx_omni_zmq_handler_t *handler);