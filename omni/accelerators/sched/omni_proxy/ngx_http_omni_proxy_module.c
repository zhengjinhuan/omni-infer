// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <omni_proxy.h>
#include <omni_pd_body_rewrite.h>
#include <omni_scheduler.h>
#include <omni_utils.h>

ngx_module_t ngx_http_omni_proxy_module;

#define TIMER_INTERVAL 1

static const char *PREFILL_URI = "/prefill_sub";
static const size_t PREFILL_URI_LEN = sizeof("/prefill_sub") - 1;

static omni_global_state_t *g_state;          // In share memory
static omni_worker_local_state_t local_state; // In local process memory space

omni_global_state_t *omni_get_global_state()
{
    return g_state;
}

omni_worker_local_state_t *omni_get_local_state()
{
    return &local_state;
}

static omni_req_t *omni_req_init(ngx_http_request_t *r)
{
    ngx_shmtx_lock(&local_state.g_shmtx);
    omni_req_t *req = omni_allocate_request(&g_state->request_pool, r);
    ngx_shmtx_unlock(&local_state.g_shmtx);

    omni_req_context_t *ctx = ngx_pcalloc(r->pool, sizeof(omni_req_context_t));
    ctx->req = req;
    ngx_http_set_ctx(r, ctx, ngx_http_omni_proxy_module);

    omni_req_enter_phase(req, 0);
    omni_add_req_to_group(req->slot_index, &local_state.groups[0]);

    ngx_shmtx_lock(&g_state->shmtx);
    omni_add_req_to_group(req->slot_index, &g_state->groups[0]);
    ngx_shmtx_unlock(&g_state->shmtx);

    req->worker_pid = ngx_getpid();

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "Allocate Req:%d, at:%d",
                  req->slot_index, req->phase);

    return req;
}

static void omni_req_free(omni_req_t *req)
{
    ngx_http_request_t *r = omni_get_http_request(req);
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "Free Req:%d, at:%x",
                  req->slot_index, req->phase_state);
    omni_free_request(&g_state->request_pool, req);
}

void omni_phase_transition_all(omni_req_t *req, omni_proxy_request_phase_t from, omni_proxy_request_phase_t to)
{
    omni_local_phase_change_to(req, from, to);

    ngx_shmtx_lock(&g_state->shmtx);
    omni_global_phase_change_to(req, from, to);
    omni_req_leave_phase(req, from);
    omni_req_enter_phase(req, to);
    ngx_shmtx_unlock(&g_state->shmtx);
}

static inline omni_req_context_t *omni_get_req_ctx(ngx_http_request_t *r)
{
    return ngx_http_get_module_ctx(r, ngx_http_omni_proxy_module);
}

static inline omni_req_t *omni_get_req(ngx_http_request_t *r)
{
    return omni_get_req_ctx(r)->req;
}

static void omni_proxy_req_body_handler(ngx_http_request_t *r)
{
    omni_req_t *req = omni_get_req(r);
    omni_req_context_t *ctx = omni_get_req_ctx(r);
    req->metrics.time_contents_received = ngx_current_msec;
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[Prefill-%d]: request body received", req->slot_index);

    omni_proxy_save_origin_body(r, ctx);

    req->metrics.prompt_num_tokens = ctx->origin_body_tokens_size;

    omni_phase_transition_all(req, 0, PHASE_PREFILL_WAITING_SCHEDULE);
    req->metrics.time_enter_wait_prefill = ngx_current_msec;

    if (g_state->pd_policy == PD_PARALLEL)
    {
        omni_add_req_to_group(req->slot_index, &local_state.groups[PHASE_DECODE_WAITING_SCHEDULE]);
        req->metrics.time_enter_wait_decode = ngx_current_msec;
    }
}

static void omni_proxy_remove_req_from_groups(omni_req_t *req)
{
    omni_proxy_request_phase_t phases[PHASE_MAX];
    size_t count = 0;
    omni_req_get_phases(req, phases, &count);

    for (size_t i = 0; i < count; ++i)
    {
        omni_proxy_request_phase_t phase = phases[i];
        ngx_shmtx_lock(&g_state->shmtx);
        omni_remove_req_from_group_by_req_index(req->slot_index, &g_state->groups[phases[i]]);

        if (phase == PHASE_PREFILLING)
        {
            g_state->prefill_states[req->prefill_upstream_endpoint_idx].num_running--;
        }
        else if (phase == PHASE_DECODING)
        {
            g_state->decode_states[req->decode_upstream_endpoint_idx].num_running--;
        }
        ngx_shmtx_unlock(&g_state->shmtx);

        omni_remove_req_from_group_by_req_index(req->slot_index, &local_state.groups[phase]);
    }
}

static void omni_proxy_main_req_cleanup(void *data)
{
    omni_req_t *req = data;
    omni_proxy_remove_req_from_groups(req);

    ngx_log_error(NGX_LOG_INFO, omni_get_http_request(req)->connection->log, 0,
                  "[Decode-%d]: Done from %d.",
                  req->slot_index, req->decode_upstream_endpoint_idx);

    omni_req_free(req);
}

/* find entry in main conf by name */
static ngx_omni_upstream_entry_t *
ngx_http_omni_find_entry(ngx_http_omni_main_conf_t *mcf, ngx_str_t *name)
{
    if (mcf == NULL || mcf->entries == NULL)
    {
        return NULL;
    }

    ngx_uint_t i;
    ngx_omni_upstream_entry_t *entries = mcf->entries->elts;
    for (i = 0; i < mcf->entries->nelts; i++)
    {
        if (entries[i].name.len == name->len && ngx_strncmp(entries[i].name.data, name->data, name->len) == 0)
        {
            return &entries[i];
        }
    }
    return NULL;
}

static ngx_omni_backend_t *ngx_http_omni_choose_backend(ngx_http_request_t *r)
{
    omni_req_context_t *ctx = omni_get_req_ctx(r);
    if (ctx == NULL || ctx->backends == NULL)
    {
        return NULL;
    }

    ngx_array_t *arr = ctx->backends;
    if (arr->nelts == 0)
    {
        return NULL;
    }

    omni_req_t *req = omni_get_req(r);
    ngx_uint_t idx = req->decode_upstream_endpoint_idx;
    if (idx >= arr->nelts)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "omni_proxy: upstream endpoint index %d out of bounds for backend array size %d",
                      idx, arr->nelts);
        return NULL;
    }

    ngx_omni_backend_t *b = arr->elts;
    return &b[idx];
}

static ngx_int_t omni_proxy_handler(ngx_http_request_t *r)
{
    if (r->parent != NULL)
    {
        return NGX_DECLINED;
    }

    omni_req_t *req = omni_req_init(r);
    if (req == NULL)
    {
        ngx_http_finalize_request(r, NGX_ERROR);
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    ngx_http_omni_loc_conf_t *olcf;
    ngx_http_omni_main_conf_t *mcf;
    ngx_omni_upstream_entry_t *entry;

    olcf = ngx_http_get_module_loc_conf(r, ngx_http_omni_proxy_module);
    if (olcf == NULL || olcf->upstream_name.len == 0)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "omni_proxy: no upstream_name in loc conf");
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    g_state->pd_policy = olcf->pd_policy;
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "[Prefill-%d]: enter main request handler, policy:%d",
                  req->slot_index, g_state->pd_policy);

    /* find parsed upstream entry from main conf */
    mcf = ngx_http_get_module_main_conf(r, ngx_http_omni_proxy_module);
    if (mcf == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "omni_proxy: main conf missing");
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    entry = ngx_http_omni_find_entry(mcf, &olcf->upstream_name);
    if (entry == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "omni_proxy: upstream \"%V\" not found in configuration", &olcf->upstream_name);
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    omni_req_context_t *ctx = omni_get_req_ctx(r);

    ctx->backends = entry->backends;

    req->metrics.time_received = ngx_current_msec;

    ngx_http_cleanup_t *cleanup = ngx_http_cleanup_add(r, 0);
    cleanup->handler = omni_proxy_main_req_cleanup;
    cleanup->data = req;

    ngx_int_t rc = ngx_http_read_client_request_body(r, omni_proxy_req_body_handler);
    if (rc == NGX_AGAIN)
    {
        return NGX_DONE;
    }
    else if (rc >= NGX_HTTP_SPECIAL_RESPONSE)
    {
        return rc;
    }

    return NGX_DONE;
}

static ngx_int_t ngx_http_prefill_post_subrequest(ngx_http_request_t *subr, void *data, ngx_int_t rc)
{
    omni_req_t *req = data;
    ngx_chain_t *cl;
    size_t total = 0;
    u_char *p;
    ngx_http_request_t *r = omni_get_http_request(req);
    omni_req_context_t *ctx = omni_get_req_ctx(subr);
    req->metrics.time_prefilled = ngx_current_msec;

    ngx_connection_t *c;

    c = r->connection;
    ctx = ngx_http_get_module_ctx(r, ngx_http_omni_proxy_module);

    if (c->read->timer_set)
    {
        ngx_del_timer(c->read);
    }
    if (c->write->timer_set)
    {
        ngx_del_timer(c->write);
    }

    r->read_event_handler = ngx_http_request_empty_handler;
    r->write_event_handler = ngx_http_request_empty_handler;

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "[Prefill-%d] Done from %d", req->slot_index, req->decode_upstream_endpoint_idx);

    if (rc != NGX_OK)
    {
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                      "[Prefill-%d] subrequest failed with code %i", req->slot_index, rc);
        ngx_http_finalize_request(r->main, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return rc;
    }

    if (subr->headers_out.status != NGX_HTTP_OK)
    {
        r->headers_out.status = subr->headers_out.status;
        ngx_http_finalize_request(r, subr->headers_out.status);
        return NGX_OK;
    }

    for (cl = subr->out; cl; cl = cl->next)
    {
        total += ngx_buf_size(cl->buf);
    }

    ctx->prefill_response_body = ngx_palloc(r->main->pool, total + 1);
    if (ctx->prefill_response_body == NULL)
    {
        ngx_http_finalize_request(r, NGX_ERROR);
        ngx_log_error(
            NGX_LOG_ERR, r->connection->log, 0,
            "done prefill subrequest, malloc prefill response body buffer failed");
        return rc;
    }

    p = ctx->prefill_response_body;
    for (cl = subr->out; cl; cl = cl->next)
    {
        size_t buf_size = ngx_buf_size(cl->buf);
        if (buf_size > 0)
        {
            p = ngx_cpymem(p, cl->buf->pos, buf_size);
        }
    }
    *p = '\0';
    ctx->prefill_response_body_size = total;

    omni_upstream_prefill_t *us = &g_state->prefill_states[req->decode_upstream_endpoint_idx];
    us->num_running--;

    omni_batch_metrics_t *batch = &us->his.his[us->his.head];
    uint32_t delta = ngx_current_msec - batch->last_response_receive_time;

    // Need a smarter value from statistics or work out by the number of tokens scheduled
    if (delta > 20)
    {
        // An new batch comes back
        if (us->his.count < NUM_PREFILL_BATCH_METRICS_HIS - 1)
        {
            us->his.count++;
        }

        us->his.head++;
        if (us->his.head == NUM_PREFILL_BATCH_METRICS_HIS)
        {
            us->his.head = 0;
        }

        batch = &us->his.his[us->his.head];
        ngx_memzero(batch, sizeof(omni_batch_metrics_t));

        batch->first_response_receive_time = ngx_current_msec;
        batch->last_response_receive_time = ngx_current_msec;
        batch->num_requests = 1;
        batch->num_tokens = req->metrics.prompt_num_tokens;
    }
    else
    {
        batch->average_delta = batch->average_delta * (batch->num_requests - 1) +
                               delta / (batch->num_requests);
        batch->last_response_receive_time = ngx_current_msec;
        batch->num_requests++;
        batch->num_tokens += req->metrics.prompt_num_tokens;
    }

    // check policy
    if (g_state->pd_policy == PD_SEQUENTIAL)
    {
        omni_phase_transition_all(req, PHASE_PREFILLING, PHASE_DECODE_WAITING_SCHEDULE);
        req->metrics.time_enter_wait_decode = ngx_current_msec;
    }

    return NGX_DONE;
}

static ngx_int_t ngx_http_prefill_wakeup(omni_req_t *req)
{
    ngx_http_request_t *r = omni_get_http_request(req);

    size_t prelen = PREFILL_URI_LEN + r->uri.len;
    if (r->args.len)
    {
        prelen += 1 + r->args.len;
    }
    u_char *prefill_uri = ngx_pnalloc(r->pool, prelen + 1);
    if (prefill_uri == NULL)
    {
        ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    u_char *p = prefill_uri;
    p = ngx_cpymem(p, PREFILL_URI, PREFILL_URI_LEN);
    p = ngx_cpymem(p, r->uri.data, r->uri.len);
    if (r->args.len)
    {
        *p++ = '?';
        p = ngx_cpymem(p, r->args.data, r->args.len);
    }
    *p = '\0';

    ngx_str_t sub_uri;
    sub_uri.len = ngx_strlen(prefill_uri);
    sub_uri.data = prefill_uri;

    ngx_http_post_subrequest_t *psr = ngx_pcalloc(r->pool, sizeof(ngx_http_post_subrequest_t));
    psr->handler = ngx_http_prefill_post_subrequest;
    psr->data = req;

    ngx_str_t args = ngx_null_string;
    ngx_http_request_t *sr;

    ngx_int_t rc = ngx_http_subrequest(r, &sub_uri, &args, &sr, psr,
                                       NGX_HTTP_SUBREQUEST_IN_MEMORY);
    if (rc != NGX_OK)
    {
        ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return rc;
    }
    sr->method = r->method;
    sr->method_name = r->method_name;

    omni_req_context_t *ctx = omni_get_req_ctx(r);
    ngx_http_set_ctx(sr, ctx, ngx_http_omni_proxy_module);

    omni_proxy_prepare_prefill_subrequest(r, sr, ctx);

    omni_phase_transition_all(req, PHASE_PREFILL_SCHEDULED, PHASE_PREFILLING);

    req->metrics.time_to_prefill = ngx_current_msec;

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[Prefill-%d] Submit to:%d",
                  req->slot_index, req->prefill_upstream_endpoint_idx);

    // ngx_http_core_run_phases(sr);
    ngx_http_run_posted_requests(r->connection);

    return NGX_OK;
}

static ngx_int_t omni_proxy_get_peer(ngx_peer_connection_t *pc,
                                     void *data)
{
    ngx_http_upstream_rr_peer_data_t *rrp = data;
    ngx_http_request_t *r = (ngx_http_request_t *)rrp->data;
    omni_req_t *req = omni_get_req(r);
    ngx_http_upstream_rr_peers_t *peers = rrp->peers;

    assert(omni_req_is_in_phase(req, PHASE_PREFILLING));

    ngx_uint_t idx = req->prefill_upstream_endpoint_idx;
    // TODO: Need to consider the liveness of the upstream in case it crashed or lag
    assert(idx < rrp->peers->number);

    pc->sockaddr = peers->peer[idx].sockaddr;
    pc->socklen = peers->peer[idx].socklen;
    pc->name = &peers->peer[idx].name;
    rrp->current = &peers->peer[idx];

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "[Prefill-%d] Upstream set: %d, running:%d",
                  req->slot_index, idx, g_state->prefill_states[idx].num_running);

    return NGX_OK;
}

static ngx_int_t omni_proxy_upstream_init(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *uscf)
{

    omni_req_t *req = omni_get_req(r);
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[-%d] init upstream at phase: %x",
                  req->slot_index, req->phase_state);

    ngx_http_upstream_t *u = r->upstream;
    ngx_http_upstream_rr_peer_data_t *rrp;
    if (ngx_http_upstream_init_round_robin_peer(r, uscf) != NGX_OK)
    {
        return NGX_ERROR;
    }
    rrp = u->peer.data;

    u->peer.get = omni_proxy_get_peer;
    u->peer.data = rrp;
    rrp->data = (uintptr_t)r;

    return NGX_OK;
}

static void omni_proxy_update_decode_stats(ngx_http_request_t *r, ngx_buf_t *buf, ssize_t bytes)
{
    omni_req_t *req = omni_get_req(r);

    // Update request level statistics

    if (req->metrics.time_last_reponse)
    {
        req->metrics.decoded_tokens++;
        if (++req->metrics.decoded_tokens != 0)
        {
            req->metrics.tpot =
                ((req->metrics.tpot * req->metrics.decoded_tokens) +
                 ngx_current_msec - req->metrics.time_last_reponse) /
                req->metrics.decoded_tokens;
        }
    }
    else
    {
        req->metrics.time_first_token = req->metrics.tpot = ngx_current_msec - req->metrics.time_to_decode;
    }

    req->metrics.time_last_reponse = ngx_current_msec;

    omni_upstream_decode_t *us = &g_state->decode_states[req->decode_upstream_endpoint_idx];
    // Update batch level statistics
    omni_batch_metrics_t *batch = &us->his.his[us->his.head];
    uint32_t delta = ngx_current_msec - batch->last_response_receive_time;

    // Need a smarter value from statistics or work out by the number of tokens scheduled
    if (delta > 10)
    {
        // An new batch comes back
        if (us->his.count < NUM_PREFILL_BATCH_METRICS_HIS - 1)
        {
            us->his.count++;
        }

        us->his.head++;
        if (us->his.head == NUM_PREFILL_BATCH_METRICS_HIS)
        {
            us->his.head = 0;
        }

        batch = &us->his.his[us->his.head];
        ngx_memzero(batch, sizeof(omni_batch_metrics_t));

        batch->first_response_receive_time = ngx_current_msec;
        batch->last_response_receive_time = ngx_current_msec;
        batch->num_requests = 1;
        batch->num_tokens = req->metrics.prompt_num_tokens + req->metrics.decoded_tokens;
    }
    else
    {
        batch->average_delta = batch->average_delta * (batch->num_requests - 1) +
                               delta / (batch->num_requests);
        batch->last_response_receive_time = ngx_current_msec;
        batch->num_requests++;
        batch->num_tokens += req->metrics.prompt_num_tokens + req->metrics.decoded_tokens;
    }
}

static ngx_int_t ngx_http_omni_create_request(ngx_http_request_t *r)
{
    ngx_chain_t *cl;
    size_t body_len = 0;
    ngx_str_t host;
    omni_req_context_t *ctx = ngx_http_get_module_ctx(r, ngx_http_omni_proxy_module);

    omni_proxy_prepare_decode_request_body(r, ctx);

    if (r->request_body == NULL || r->request_body->bufs == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "ngx_http_omni_create_request: request body is NULL");
        return NGX_ERROR;
    }

    for (cl = r->request_body->bufs; cl; cl = cl->next)
    {
        body_len += ngx_buf_size(cl->buf);
    }

    if (r->headers_in.host && r->headers_in.host->value.len)
    {
        host = r->headers_in.host->value;
    }
    else
    {
        host = r->headers_in.server;
    }

    ngx_buf_t *hdr = ngx_create_temp_buf(r->pool, 256 + r->uri.len + host.len);
    if (hdr == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "ngx_http_omni_create_request: failed to create temp buffer for request header");
        return NGX_ERROR;
    }

    hdr->last = ngx_snprintf(hdr->last, hdr->end - hdr->last,
                             "POST %V HTTP/1.1\r\n", &r->uri);

    ngx_list_part_t *part = &r->headers_in.headers.part;
    ngx_table_elt_t *h = part->elts;
    ngx_uint_t i;

    for (;;)
    {
        for (i = 0; i < part->nelts; i++)
        {
            if (h[i].key.len == sizeof("Content-Length") - 1 &&
                ngx_strncasecmp(h[i].key.data,
                                (u_char *)"Content-Length",
                                h[i].key.len) == 0)
            {
                continue;
            }

            hdr->last = ngx_snprintf(hdr->last, hdr->end - hdr->last,
                                     "%V: %V\r\n", &h[i].key, &h[i].value);
        }

        if (part->next == NULL)
        {
            break;
        }
        part = part->next;
        h = part->elts;
    }

    hdr->last = ngx_snprintf(hdr->last, hdr->end - hdr->last,
                             "Content-Length: %O\r\n\r\n", (off_t)body_len);

    cl = ngx_alloc_chain_link(r->pool);
    if (cl == NULL)
    {
        return NGX_ERROR;
    }

    cl->buf = hdr;
    cl->next = r->request_body->bufs;
    r->upstream->request_bufs = cl;

    return NGX_OK;
}

static ngx_int_t ngx_http_omni_reinit_request(ngx_http_request_t *r)
{
    ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                  "ngx_http_omni_reinit_request(ngx_http_request_t *r): [Line %d]", __LINE__);
    return NGX_OK;
}

static ngx_int_t ngx_http_omni_process_header(ngx_http_request_t *r)
{
    ngx_int_t rc;
    ngx_http_upstream_t *u;
    ngx_table_elt_t *h;

    u = r->upstream;

    for (;;)
    {
        rc = ngx_http_parse_header_line(r, &u->buffer, 1);

        if (rc == NGX_OK)
        {
            h = ngx_list_push(&r->headers_out.headers);
            if (h == NULL)
            {
                return NGX_HTTP_INTERNAL_SERVER_ERROR;
            }

            h->key.len = r->header_name_end - r->header_name_start;
            h->key.data = ngx_pnalloc(r->pool, h->key.len);
            if (h->key.data == NULL)
            {
                return NGX_ERROR;
            }
            ngx_memcpy(h->key.data, r->header_name_start, h->key.len);

            h->value.len = r->header_end - r->header_start;
            h->value.data = ngx_pnalloc(r->pool, h->value.len);
            if (h->value.data == NULL)
            {
                return NGX_ERROR;
            }
            ngx_memcpy(h->value.data, r->header_start, h->value.len);

            h->lowcase_key = ngx_pnalloc(r->pool, h->key.len);
            if (h->lowcase_key == NULL)
            {
                return NGX_ERROR;
            }
            ngx_strlow(h->lowcase_key, h->key.data, h->key.len);

            h->hash = ngx_hash_key_lc(h->lowcase_key, h->key.len);

            continue;
        }

        if (rc == NGX_HTTP_PARSE_HEADER_DONE)
        {
            // Mark as processed to avoid forward to filter below
            u->buffer.last = u->buffer.pos;

            ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                          "omni_process_header: finished parsing header");

            return NGX_OK;
        }

        if (rc == NGX_AGAIN)
        {
            return NGX_AGAIN;
        }

        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "omni_process_header: invalid header from upstream");
        return NGX_HTTP_UPSTREAM_INVALID_HEADER;
    }
}

static ngx_int_t ngx_http_omni_process_status_line(ngx_http_request_t *r)
{
    ngx_int_t rc;
    ngx_http_upstream_t *u = r->upstream;
    ngx_http_status_t st;
    ngx_memzero(&st, sizeof(st));

    rc = ngx_http_parse_status_line(r, &u->buffer, &st);

    if (rc == NGX_AGAIN)
    {
        return NGX_AGAIN;
    }

    if (rc == NGX_ERROR)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "omni_proxy: invalid status line from upstream");
        return NGX_HTTP_UPSTREAM_INVALID_HEADER;
    }

    u->headers_in.status_n = st.code;

    u->headers_in.status_line.len = st.end - st.start;
    u->headers_in.status_line.data = ngx_pnalloc(r->pool, u->headers_in.status_line.len);
    if (u->headers_in.status_line.data == NULL)
    {
        return NGX_ERROR;
    }
    ngx_memcpy(u->headers_in.status_line.data, st.start, u->headers_in.status_line.len);

    return ngx_http_omni_process_header(r);
}

static ngx_int_t ngx_http_omni_input_filter_init(void *data)
{
    return NGX_OK;
}

static ngx_int_t ngx_http_omni_input_filter(void *data, ssize_t bytes)
{
    ngx_http_request_t *r = data;
    ngx_http_upstream_t *u = r->upstream;

    ngx_log_error(NGX_LOG_DEBUG, r->connection->log, 0,
                  "ngx_http_omni_input_filter: bytes %lu", bytes);

    u->buffer.last = u->buffer.pos + bytes;
    if (u->buffer.pos != NULL && u->buffer.pos < u->buffer.last)
    {
        size_t len = u->buffer.last - u->buffer.pos;
        ngx_buf_t *b = ngx_create_temp_buf(r->pool, len);
        if (b == NULL)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                          "ngx_http_omni_input_filter, create buf failed");
            return NGX_ERROR;
        }
        ngx_memcpy(b->pos, u->buffer.pos, len);
        b->last = b->pos + len;
        b->memory = 1;
        b->flush = 1;

        ngx_chain_t *cl = ngx_alloc_chain_link(r->pool);
        if (cl == NULL)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                          "ngx_http_omni_input_filter, allocate chain failed");
            return NGX_ERROR;
        }
        cl->buf = b;
        cl->next = NULL;

        ngx_int_t rc = ngx_http_output_filter(r, cl);
        if (rc == NGX_ERROR)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                          "ngx_http_omni_input_filter, putput failed");
            return NGX_ERROR;
        }
    }

    omni_proxy_update_decode_stats(r, &u->buffer, bytes);

    return NGX_OK;
}

static void ngx_http_omni_finalize_request(ngx_http_request_t *r, ngx_int_t rc)
{
}

static ngx_int_t ngx_http_omni_start_decode_upstream(ngx_http_request_t *r)
{
    ngx_http_omni_loc_conf_t *olcf;

    if (ngx_http_upstream_create(r) != NGX_OK)
    {
        return NGX_ERROR;
    }
    olcf = ngx_http_get_module_loc_conf(r, ngx_http_omni_proxy_module);
    if (olcf->upstream == NULL)
    {
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    ngx_http_upstream_t *u = r->upstream;

    u->conf = olcf->upstream->srv_conf[ngx_http_upstream_module.ctx_index];
    u->conf->buffer_size = 8192;

    u->create_request = ngx_http_omni_create_request;
    u->reinit_request = ngx_http_omni_reinit_request;
    u->process_header = ngx_http_omni_process_status_line;
    u->input_filter_init = ngx_http_omni_input_filter_init;
    u->input_filter = ngx_http_omni_input_filter;
    u->input_filter_ctx = r;
    u->finalize_request = ngx_http_omni_finalize_request;

    u->output.tag = (ngx_buf_tag_t)&ngx_http_omni_proxy_module;

    ngx_omni_backend_t *b = ngx_http_omni_choose_backend(r);
    if (b == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "omni_proxy: no backend available");
        return NGX_ERROR;
    }

    u->resolved = ngx_pcalloc(r->pool, sizeof(ngx_http_upstream_resolved_t));
    if (u->resolved == NULL)
    {
        return NGX_ERROR;
    }

    struct sockaddr *sa = ngx_pcalloc(r->pool, b->socklen);
    if (sa == NULL)
    {
        return NGX_ERROR;
    }
    ngx_memcpy(sa, b->sockaddr, b->socklen);

    u->resolved->sockaddr = sa;
    u->resolved->port = b->port;
    u->resolved->socklen = b->socklen;
    u->resolved->naddrs = 1;

    u->resolved->host.len = b->text.len;
    u->resolved->host.data = ngx_pnalloc(r->pool, b->text.len + 1);
    if (u->resolved->host.data == NULL)
    {
        return NGX_ERROR;
    }
    ngx_memcpy(u->resolved->host.data, b->text.data, b->text.len);
    u->resolved->host.data[b->text.len] = '\0';

    ngx_http_upstream_init(r);

    return NGX_OK;
}

static ngx_int_t ngx_http_decode_wakeup(omni_req_t *req)
{
    ngx_http_request_t *r = omni_get_http_request(req);
    ngx_int_t rc = ngx_http_omni_start_decode_upstream(r);

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[Decode-%d]: wakeup", req->slot_index);

    omni_phase_transition_all(req, PHASE_DECODE_SCHEDULED, PHASE_DECODING);
    req->metrics.time_to_decode = ngx_current_msec;

    if (rc == NGX_OK)
    {
        /* ngx_http_upstream_init will drive request; return NGX_OK from post_subrequest */
        return NGX_OK;
    }

    ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
    return NGX_OK;
}

typedef ngx_int_t (*omni_run_handle_t)(omni_req_t *);

static void omni_proxy_run_group(int phase_from, omni_run_handle_t handle)
{
    omni_req_group_t *group = &local_state.groups[phase_from];

    omni_req_group_t *g_group = &g_state->groups[phase_from];

    for (uint32_t i = 0; i < group->watermark; ++i)
    {
        omni_req_info_t *info = &group->requests[i];
        if (info->in_use)
        {
            omni_req_t *req = omni_info_to_req(info);
            ngx_http_request_t *r = omni_get_http_request(req);
            ngx_int_t rc = handle(req);
            if (rc != NGX_OK)
            {
                // TODO:
            }
        }
    }
}

static void update_prefill_weights(omni_req_group_t *group)
{
    uint32_t max_prompt_tokens = 0;
    uint32_t max_wait_time = 0;
    for (int i = 0; i < group->watermark; i++)
    {
        omni_req_info_t *info = &group->requests[i];
        if (!info->in_use)
        {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);

        if (max_prompt_tokens < req->metrics.prompt_num_tokens)
        {
            max_prompt_tokens = req->metrics.prompt_num_tokens;
        }

        uint32_t waited = ngx_current_msec - req->metrics.time_received;

        if (max_wait_time < waited)
        {
            max_wait_time = waited;
        }
    }

    if (max_wait_time < 50)
    {
        max_wait_time = 50;
    }

    for (int i = 0; i < group->watermark; i++)
    {
        omni_req_info_t *info = &group->requests[i];
        if (!info->in_use)
        {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);
        uint32_t waited = ngx_current_msec - req->metrics.time_received;

        double token_weight = (double)(max_prompt_tokens - req->metrics.prompt_num_tokens) / max_prompt_tokens;
        double time_weight = (double)waited / max_wait_time;

        info->weight = token_weight * 0.8 + time_weight * 0.2;
    }

    omni_sort_compact_group(group);
}

static void omni_proxy_schedule_prefill(omni_global_state_t *gs)
{
    omni_req_group_t *group = &gs->groups[PHASE_PREFILL_WAITING_SCHEDULE];

    // TODO: Check should schedule or wait based on upstream expected come back time

    update_prefill_weights(group);

    for (int i = 0; i < group->num_requests; i++)
    {
        omni_req_info_t *info = &group->requests[i];
        omni_req_t *req = omni_info_to_req(info);
        ngx_http_request_t *r = req->data;

        assert(req->phase == PHASE_PREFILL_WAITING_SCHEDULE);

        uint32_t least_load = UINT32_MAX;
        uint32_t selected = UINT32_MAX;
        for (int j = 0; j < g_state->num_prefill_endpoints; j++)
        {
            if (gs->prefill_states[j].num_tokens < least_load)
            {
                least_load = gs->prefill_states[j].num_tokens;
                selected = j;
                if (least_load == 0)
                {
                    break;
                }
            }
        }

        req->upstream_endpoint_idx = selected;
        gs->prefill_states[selected].num_running++;
        gs->prefill_states[selected].num_tokens += req->metrics.prompt_num_tokens;

        omni_phase_transition_local(PHASE_PREFILL_SCHEDULED, req);
        req->metrics.time_prefill_scheduled = ngx_current_msec;

        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[Prefill-%d] Schedule to: %d",
                      req->slot_index, req->upstream_endpoint_idx);
    }

    // TODO: estimated expected next schedule time
}

static void omni_proxy_schedule_decode(omni_global_state_t *gs)
{
    omni_req_group_t *group = &gs->groups[PHASE_DECODE_WAITING_SCHEDULE];
    // TODO: Check should schedule or wait based on upstream expected come back time
    // TODO: Here we can do some estimation of pull kv finish time to make sure pull kv
    // workloads are balanced

    for (int i = 0; i < group->watermark; i++)
    {
        omni_req_info_t *info = &group->requests[i];
        if (!info->in_use)
        {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);
        assert(req->phase == PHASE_DECODE_WAITING_SCHEDULE);

        uint32_t least_load = UINT32_MAX;
        uint32_t selected = UINT32_MAX;
        for (int j = 0; j < g_state->num_decode_endpoints; j++)
        {
            if (gs->decode_states[j].num_running < least_load)
            {
                least_load = gs->decode_states[j].num_running;
                selected = j;
                if (least_load == 0)
                {
                    break;
                }
            }
        }

        req->upstream_endpoint_idx = selected;
        gs->decode_states[selected].num_running++;

        omni_phase_transition_local(PHASE_DECODE_SCHEDULED, req);
        req->metrics.time_decode_scheduled = ngx_current_msec;

        ngx_log_error(NGX_LOG_INFO, req->data->connection->log, 0, "[Decode-%d] Schedule to: %d",
                      req->slot_index, req->upstream_endpoint_idx);
    }
}

void print_summary()
{
    if (ngx_current_msec - g_state->last_summary < 10000)
    {
        return;
    }
    g_state->last_summary = ngx_current_msec;

    ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                  "Active requests: %d, prefill waiting: %d, prefill running: %d "
                  "decode waiting: %d, decode running: %d, pid: %d\n",
                  g_state->request_pool.num_requests,
                  g_state->groups[PHASE_PREFILL_WAITING_SCHEDULE].num_requests,
                  g_state->groups[PHASE_PREFILLING].num_requests,
                  g_state->groups[PHASE_DECODE_WAITING_SCHEDULE].num_requests,
                  g_state->groups[PHASE_DECODING].num_requests,
                  ngx_pid);
}
static void omni_update_local_waiting(omni_worker_local_state_t *local_state,
                                      omni_req_group_t *group,
                                      omni_proxy_request_phase_t to)
{
    for (uint32_t i = 0; i < group->watermark; ++i)
    {
        omni_req_info_t *info = &group->requests[i];
        omni_req_t *req = omni_info_to_req(info);
        if (req->in_use && omni_req_is_in_phase(req, to))
        {
            omni_remove_from_group_by_req_info(info, group);
            omni_add_req_to_group(req->slot_index,
                                  &local_state->groups[to]);
        }
    }

    omni_sort_compact_group(group);
}

// Global scheduler has changed the req->phase to PHASE_PREFILL_SCHEDULED, here to update local state
// to make sure local state is consistent with global state
static inline void omni_update_local_prefill_waiting(omni_worker_local_state_t *local_state)
{
    omni_update_local_waiting(local_state, &local_state->groups[PHASE_PREFILL_WAITING_SCHEDULE],
                              PHASE_PREFILL_SCHEDULED);
}

static inline void omni_update_local_decode_waiting(omni_worker_local_state_t *local_state)
{
    omni_update_local_waiting(local_state, &local_state->groups[PHASE_DECODE_WAITING_SCHEDULE],
                              PHASE_DECODE_SCHEDULED);
}

static void omni_proxy_schedule(omni_global_state_t *gs)
{
    if (omni_is_master_worker(gs))
    {
        ngx_shmtx_lock(&gs->shmtx);
        omni_proxy_schedule_prefill(gs);
        omni_proxy_schedule_decode(gs);
        ngx_shmtx_unlock(&gs->shmtx);
    }
}

static void omni_proxy_timer_handler(ngx_event_t *ev)
{
    omni_proxy_schedule(g_state);

    // Global state has moved on, local state needs to be updated
    omni_update_local_prefill_waiting(&local_state);
    omni_update_local_decode_waiting(&local_state);

    omni_proxy_run_group(PHASE_PREFILL_SCHEDULED, ngx_http_prefill_wakeup);
    omni_proxy_run_group(PHASE_DECODE_SCHEDULED, ngx_http_decode_wakeup);

    ngx_add_timer(&local_state.omni_proxy_timer_event, TIMER_INTERVAL);

    print_summary();
}

static void omni_proxy_init_req_groups(omni_req_group_t groups[])
{
    for (int i = 0; i < PHASE_MAX; i++)
    {
        groups[i].phase = i;
    }
}

static void omni_proxy_init_req_groups(omni_req_group_t groups[])
{
    if (data)
    {
        zone->data = data;
        return NGX_OK;
    }

    if (zone->shm.addr == NULL)
    {
        return NGX_ERROR;
    }

    g_state = (omni_global_state_t *)zone->shm.addr;
    memset(g_state, 0, GLOBAL_STATE_SIZE);
    g_state->num_decode_endpoints = local_state.num_decode_endpoints;
    g_state->num_prefill_endpoints = local_state.num_prefill_endpoints;

    omni_proxy_init_req_groups(g_state->groups);

    ngx_shmtx_create(&local_state.g_shmtx, &g_state->lock,
                     (u_char *)"omni_proxy_lock");

    return NGX_OK;
}

ngx_int_t omni_proxy_init_global_state(ngx_conf_t *cf)
{
    ngx_str_t name = ngx_string("omni_proxy_state");

    ngx_shm_zone_t *zone = ngx_shared_memory_add(
        cf,
        &name,
        GLOBAL_STATE_SIZE,
        &ngx_http_omni_proxy_module);

    if (zone == NULL)
    {
        return NGX_ERROR;
    }

    zone->init = omni_proxy_global_state_init;

    return NGX_OK;
}

/* -------------------- main conf create/init -------------------- */

static void *
ngx_http_omni_create_main_conf(ngx_conf_t *cf)
{
    ngx_http_omni_main_conf_t *mcf;

    mcf = ngx_pcalloc(cf->pool, sizeof(ngx_http_omni_main_conf_t));
    if (mcf == NULL)
    {
        return NULL;
    }

    mcf->entries = NULL;
    mcf->upstreams = NULL;

    return mcf;
}

static char *
ngx_http_omni_init_main_conf(ngx_conf_t *cf, void *conf)
{
    return NGX_CONF_OK;
}

/* -------------------- loc conf create/merge -------------------- */

static void *
ngx_http_omni_create_loc_conf(ngx_conf_t *cf)
{
    ngx_http_omni_loc_conf_t *conf;

    conf = ngx_pcalloc(cf->pool, sizeof(ngx_http_omni_loc_conf_t));
    if (conf == NULL)
    {
        return NULL;
    }

    conf->upstream_name.len = 0;
    conf->upstream_name.data = NULL;
    conf->rr_index = 0;

    return conf;
}

static char *
ngx_http_omni_merge_loc_conf(ngx_conf_t *cf, void *parent, void *child)
{
    ngx_http_omni_loc_conf_t *prev = parent;
    ngx_http_omni_loc_conf_t *conf = child;

    if (conf->upstream_name.data == NULL)
    {
        conf->upstream_name = prev->upstream_name;
    }

    ngx_conf_merge_uint_value(conf->pd_policy, prev->pd_policy, PD_SEQUENTIAL);

    return NGX_CONF_OK;
}

#define PREFILL_ENDPOINTS "prefill_endpoints"
#define DECODE_ENDPOINTS "decode_endpoints"

static ngx_int_t
ngx_http_omni_init_upstreams(ngx_conf_t *cf)
{
    ngx_http_omni_main_conf_t *mcf;
    ngx_http_upstream_main_conf_t *umcf;
    ngx_uint_t i;

    mcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_omni_proxy_module);
    if (mcf == NULL)
    {
        return NGX_ERROR;
    }

    mcf->entries = ngx_array_create(cf->pool, 4, sizeof(ngx_omni_upstream_entry_t));
    if (mcf->entries == NULL)
    {
        return NGX_ERROR;
    }

    umcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_upstream_module);
    if (umcf == NULL)
    {
        return NGX_OK;
    }

    ngx_array_t *upstreams = &umcf->upstreams;
    ngx_uint_t nupstreams = upstreams->nelts;
    ngx_http_upstream_srv_conf_t **uscfp = upstreams->elts;

    for (i = 0; i < nupstreams; i++)
    {
        ngx_http_upstream_srv_conf_t *uscf = uscfp[i];

        /* create an entry for this upstream name */
        ngx_omni_upstream_entry_t *entry = ngx_array_push(mcf->entries);
        if (entry == NULL)
        {
            return NGX_ERROR;
        }
        ngx_memzero(entry, sizeof(ngx_omni_upstream_entry_t));

        entry->name.len = uscf->host.len;
        entry->name.data = ngx_pnalloc(cf->pool, uscf->host.len + 1);
        if (entry->name.data == NULL)
        {
            return NGX_ERROR;
        }
        ngx_memcpy(entry->name.data, uscf->host.data, uscf->host.len);
        entry->name.data[uscf->host.len] = '\0';

        entry->backends = ngx_array_create(cf->pool, 4, sizeof(ngx_omni_backend_t));
        if (entry->backends == NULL)
        {
            return NGX_ERROR;
        }

        if (uscf->peer.data == NULL)
        {
            continue;
        }

        ngx_http_upstream_rr_peers_t *peers = uscf->peer.data;

        for (; peers; peers = peers->next)
        {
            ngx_uint_t j;
            ngx_http_upstream_rr_peer_t *peer = peers->peer;
            for (j = 0; j < peers->number; j++)
            {
                ngx_omni_backend_t *b = ngx_array_push(entry->backends);
                if (b == NULL)
                {
                    return NGX_ERROR;
                }
                ngx_memzero(b, sizeof(ngx_omni_backend_t));

                struct sockaddr *sa = ngx_pcalloc(cf->pool, peer[j].socklen);
                if (sa == NULL)
                {
                    return NGX_ERROR;
                }
                ngx_memcpy(sa, peer[j].sockaddr, peer[j].socklen);
                b->sockaddr = sa;
                b->socklen = peer[j].socklen;

                struct sockaddr_in *sin = (struct sockaddr_in *)peer->sockaddr;
                b->port = ntohs(sin->sin_port);

                b->text.len = peer[j].name.len;
                b->text.data = ngx_pnalloc(cf->pool, b->text.len + 1);
                if (b->text.data == NULL)
                {
                    return NGX_ERROR;
                }
                ngx_memcpy(b->text.data, peer[j].name.data, b->text.len);
                b->text.data[b->text.len] = '\0';
            }
        }
    }

    return NGX_OK;
}

static ngx_int_t omni_proxy_post_config(ngx_conf_t *cf)
{
    if (omni_proxy_init_global_state(cf) != NGX_OK)
    {
        return NGX_ERROR;
    }

    ngx_memzero(&local_state, sizeof(omni_worker_local_state_t));
    omni_proxy_init_req_groups(local_state.groups);

    ngx_http_upstream_main_conf_t *upcf;
    ngx_http_upstream_srv_conf_t **uscfp;

    ngx_uint_t i;
    upcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_upstream_module);
    if (upcf == NULL)
    {
        return NGX_ERROR;
    }

    uscfp = upcf->upstreams.elts;
    for (i = 0; i < upcf->upstreams.nelts; i++)
    {
        uscfp[i]->peer.init = omni_proxy_upstream_init;

        size_t n = sizeof(PREFILL_ENDPOINTS);
        if (n - 1 == uscfp[i]->host.len && ngx_strncmp(uscfp[i]->host.data, PREFILL_ENDPOINTS, n - 1) == 0)
        {
            printf("Prefill nodes:%ld\n", uscfp[i]->servers->nelts);
            local_state.num_prefill_endpoints = uscfp[i]->servers->nelts;
        }
        else
        {
            printf("Decode nodes:%ld\n", uscfp[i]->servers->nelts);
            local_state.num_decode_endpoints = uscfp[i]->servers->nelts;
        }
    }

    return ngx_http_omni_init_upstreams(cf);
}

static ngx_int_t omni_proxy_init_process(ngx_cycle_t *cycle)
{
    local_state.pid = ngx_pid;
    local_state.worker = ngx_worker;

    local_state.omni_proxy_timer_event.handler = omni_proxy_timer_handler;
    local_state.omni_proxy_timer_event.log = cycle->log;
    local_state.omni_proxy_timer_event.data = NULL;

    omni_register_worker(g_state, &g_state->shmtx);

    printf("Init timer, pid: %u, worker: %lu\n", ngx_pid, ngx_worker);

    ngx_add_timer(&local_state.omni_proxy_timer_event, TIMER_INTERVAL);

    return NGX_OK;
}

static void omni_proxy_exit_process(ngx_cycle_t *cycle)
{
    if (local_state.omni_proxy_timer_event.timer_set)
    {
        ngx_del_timer(&local_state.omni_proxy_timer_event);
    }
}

static char *omni_proxy_init_conf(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_core_loc_conf_t *clcf = ngx_http_conf_get_module_loc_conf(cf, ngx_http_core_module);
    clcf->handler = omni_proxy_handler;

    ngx_http_omni_loc_conf_t *olcf = conf;
    ngx_str_t *value = cf->args->elts;

    // tokens start at value[1]
    for (ngx_uint_t i = 1; i < cf->args->nelts; i++)
    {
        ngx_str_t *v = &value[i];

        // pd_policy=sequential|parallel
        if (v->len >= sizeof("pd_policy=") - 1 &&
            ngx_strncasecmp(v->data, (u_char *)"pd_policy=", sizeof("pd_policy=") - 1) == 0)
        {
            u_char *val = v->data + (sizeof("pd_policy=") - 1);
            size_t len = v->len - (sizeof("pd_policy=") - 1);

            if (len == 0)
            {
                ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                                   "empty value for \"pd_policy\"");
                return NGX_CONF_ERROR;
            }

            if (len == sizeof("sequential") - 1 &&
                ngx_strncasecmp(val, (u_char *)"sequential", sizeof("sequential") - 1) == 0)
            {
                olcf->pd_policy = PD_SEQUENTIAL;
                continue;
            }

            if (len == sizeof("parallel") - 1 &&
                ngx_strncasecmp(val, (u_char *)"parallel", sizeof("parallel") - 1) == 0)
            {
                olcf->pd_policy = PD_PARALLEL;
                continue;
            }

            ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                               "invalid value \"%V\" for \"pd_policy\" "
                               "(expected \"sequential\" or \"parallel\")",
                               v);
            return NGX_CONF_ERROR;
        }

        // Default to be the upstream name
        olcf->upstream_name.len = value[1].len;
        olcf->upstream_name.data = ngx_pstrdup(cf->pool, &value[1]);
        if (olcf->upstream_name.data == NULL)
        {
            return NGX_CONF_ERROR;
        }
    }

    ngx_url_t u;
    ngx_memzero(&u, sizeof(ngx_url_t));
    u.url = value[1];
    u.no_resolve = 1;

    olcf->upstream = ngx_http_upstream_add(cf, &u, 0);
    if (olcf->upstream == NULL)
    {
        return NGX_CONF_ERROR;
    }

    return NGX_CONF_OK;
}

static ngx_command_t omni_proxy_commands[] = {
    {
        ngx_string("omni_proxy"),
        NGX_HTTP_LOC_CONF | NGX_CONF_TAKE2,
        omni_proxy_init_conf,
        NGX_HTTP_LOC_CONF_OFFSET,
        0,
        NULL,
    },
    ngx_null_command};

static ngx_http_module_t omni_proxy_module_ctx = {
    NULL,                           // preconfiguration
    omni_proxy_post_config,         // postconfiguration
    ngx_http_omni_create_main_conf, /* create main configuration */
    ngx_http_omni_init_main_conf,   /* init main configuration */

    NULL, /* create server configuration */
    NULL, /* merge server configuration */

    ngx_http_omni_create_loc_conf, /* create location configuration */
    ngx_http_omni_merge_loc_conf   /* merge location configuration */
};

ngx_module_t ngx_http_omni_proxy_module = {
    NGX_MODULE_V1,
    &omni_proxy_module_ctx,  /* module context */
    omni_proxy_commands,     /* module directives */
    NGX_HTTP_MODULE,         /* module type */
    NULL,                    /* init master */
    NULL,                    /* init module */
    omni_proxy_init_process, /* init process */
    NULL,                    /* init thread */
    NULL,                    /* exit thread */
    omni_proxy_exit_process, /* exit process */
    NULL,                    /* exit master */
    NGX_MODULE_V1_PADDING};