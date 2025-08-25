// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "omni_zmq_handler.h"
#include <string.h>

static void omni_zmq_event_handler(ngx_event_t *ev)
{
    omni_zmq_handler_t *handler = ev->data;

    while (1)
    {
        zmq_msg_t topic_msg;
        zmq_msg_t seq_msg;
        zmq_msg_t payload_msg;
        int more;
        size_t more_size = sizeof(more);

        if (zmq_msg_init(&topic_msg) != 0)
        {
            break;
        }

        int rc = zmq_msg_recv(&topic_msg, handler->zmq_socket, ZMQ_DONTWAIT);
        if (rc == -1)
        {
            zmq_msg_close(&topic_msg);
            if (errno == EAGAIN)
            {
                break;
            }
            break;
        }

        zmq_getsockopt(handler->zmq_socket, ZMQ_RCVMORE, &more, &more_size);
        if (!more)
        {
            zmq_msg_close(&topic_msg);
            break;
        }

        if (zmq_msg_init(&seq_msg) != 0)
        {
            zmq_msg_close(&topic_msg);
            break;
        }

        rc = zmq_msg_recv(&seq_msg, handler->zmq_socket, ZMQ_DONTWAIT);
        if (rc == -1)
        {
            zmq_msg_close(&topic_msg);
            zmq_msg_close(&seq_msg);
            break;
        }

        zmq_getsockopt(handler->zmq_socket, ZMQ_RCVMORE, &more, &more_size);
        if (!more)
        {
            zmq_msg_close(&topic_msg);
            zmq_msg_close(&seq_msg);
            break;
        }

        if (zmq_msg_init(&payload_msg) != 0)
        {
            zmq_msg_close(&topic_msg);
            zmq_msg_close(&seq_msg);
            break;
        }

        rc = zmq_msg_recv(&payload_msg, handler->zmq_socket, ZMQ_DONTWAIT);
        if (rc == -1)
        {
            zmq_msg_close(&topic_msg);
            zmq_msg_close(&seq_msg);
            zmq_msg_close(&payload_msg);
            break;
        }

        if (handler->message_callback)
        {
            const char *topic = zmq_msg_data(&topic_msg);
            const void *payload = zmq_msg_data(&payload_msg);
            size_t length = zmq_msg_size(&payload_msg);

            handler->message_callback(handler, topic, payload, length);
        }

        zmq_msg_close(&topic_msg);
        zmq_msg_close(&seq_msg);
        zmq_msg_close(&payload_msg);
    }

    if (ngx_add_event(ev, NGX_READ_EVENT, 0) != NGX_OK)
    {
    }
}

ngx_int_t omni_zmq_handler_reinit(omni_zmq_handler_t *handler)
{
    if (handler->active)
    {
        return NGX_OK;
    }

    char addr_str[256];
    ngx_snprintf((u_char *)addr_str, sizeof(addr_str), "%V", &handler->zmq_address);

    if (zmq_connect(handler->zmq_socket, addr_str) != 0)
    {
        return NGX_ERROR;
    }

    char topic_str[256];
    ngx_snprintf((u_char *)topic_str, sizeof(topic_str), "%V", &handler->subscribe_topic);

    if (zmq_setsockopt(handler->zmq_socket, ZMQ_SUBSCRIBE, topic_str, strlen(topic_str)) != 0)
    {
        return NGX_ERROR;
    }

    int timeout = 100;
    zmq_setsockopt(handler->zmq_socket, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));

    int zmq_fd;
    size_t fd_size = sizeof(zmq_fd);
    zmq_getsockopt(handler->zmq_socket, ZMQ_FD, &zmq_fd, &fd_size);

    if (!handler->zmq_connection)
    {
        handler->zmq_connection = ngx_get_connection(zmq_fd, handler->log);
        if (!handler->zmq_connection)
        {
            return NGX_ERROR;
        }
    }

    if (ngx_nonblocking(zmq_fd) == -1)
    {
        ngx_free_connection(handler->zmq_connection);
        handler->zmq_connection = NULL;
        return NGX_ERROR;
    }

    if (!handler->zmq_event)
    {
        handler->zmq_event = handler->zmq_connection->read;
        handler->zmq_event->handler = omni_zmq_event_handler;
        handler->zmq_event->data = handler;
        handler->zmq_event->log = handler->log;
    }

    if (ngx_add_conn(handler->zmq_connection) != NGX_OK)
    {
        ngx_free_connection(handler->zmq_connection);
        handler->zmq_connection = NULL;
        return NGX_ERROR;
    }

    handler->active = 1;
    return NGX_OK;
}

ngx_int_t omni_zmq_handler_init(ngx_cycle_t *cycle,
                                omni_zmq_handler_t *handler,
                                ngx_str_t zmq_address,
                                ngx_str_t subscribe_topic,
                                omni_zmq_msg_callback_t callback)
{

    handler->log = cycle->log;
    handler->cycle = cycle;
    handler->zmq_address = zmq_address;
    handler->subscribe_topic = subscribe_topic;
    handler->message_callback = callback;

    handler->zmq_context = zmq_ctx_new();
    if (!handler->zmq_context)
    {
        return NGX_ERROR;
    }

    handler->zmq_socket = zmq_socket(handler->zmq_context, ZMQ_SUB);
    if (!handler->zmq_socket)
    {
        return NGX_ERROR;
    }
    return omni_zmq_handler_reinit(handler);
}

void omni_zmq_handler_exit(omni_zmq_handler_t *handler)
{
    if (!handler->active)
    {
        return;
    }

    handler->active = 0;

    if (handler->zmq_connection)
    {
        ngx_del_conn(handler->zmq_connection, 0);
        ngx_free_connection(handler->zmq_connection);
        handler->zmq_connection = NULL;
    }

    if (handler->zmq_socket)
    {
        zmq_close(handler->zmq_socket);
        handler->zmq_socket = NULL;
    }

    if (handler->zmq_context)
    {
        zmq_ctx_destroy(handler->zmq_context);
        handler->zmq_context = NULL;
    }

    handler->zmq_event = NULL;
}