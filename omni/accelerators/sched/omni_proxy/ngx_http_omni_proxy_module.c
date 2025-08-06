#include "omni_proxy.h"
#include "pd_body_rewrite.h"

ngx_module_t ngx_http_omni_proxy_module;

#define TIMER_INTERVAL 2

static const char *prefill_uri = "/prefill_sub";
static const size_t prefill_uri_len = sizeof("/prefill_sub") - 1;
static char *decode_uri = "/decode_sub";
static size_t decode_uri_len = sizeof("/decode_sub") - 1;

static omni_global_state_t *g_state;          // In share memory
static omni_worker_local_state_t local_state; // In local process memory space

static omni_req_t *omni_allocate_req(ngx_http_request_t *r)
{
    ngx_shmtx_lock(&local_state.g_shmtx);
    omni_req_t *req = omni_allocate_request(&g_state->request_pool, r);
    ngx_shmtx_unlock(&local_state.g_shmtx);

    omni_req_context_t *ctx = ngx_pcalloc(r->pool, sizeof(omni_req_context_t));
    ctx->req = req;
    ngx_http_set_ctx(r, ctx, ngx_http_omni_proxy_module);

    omni_add_req_to_group(req->slot_index, &local_state.groups[req->phase]);

    ngx_shmtx_lock(&local_state.g_shmtx);
    omni_add_req_to_group(req->slot_index, &g_state->groups[req->phase]);
    ngx_shmtx_unlock(&local_state.g_shmtx);

    req->worker_pid = ngx_getpid();

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "Allocate Req:%d, at:%d",
                  req->slot_index, req->phase);

    return req;
}

static void omni_free_req(omni_req_t *req)
{
    ngx_http_request_t *r = req->data;
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "Free Req:%d, at:%d",
                  req->slot_index, req->phase);
    omni_free_request(&g_state->request_pool, req);
}

static inline omni_req_t *omni_id_to_req(uint32_t slot)
{
    return &g_state->request_pool.slots[slot];
}

static inline omni_req_t *omni_info_to_req(omni_req_info_t *info)
{
    return omni_id_to_req(info->slot_index);
}

static inline void omni_phase_transition_local(omni_proxy_request_phase_t phase, omni_req_t *req)
{
    omni_remove_req_from_group_by_req_index(
        req->slot_index,
        &local_state.groups[req->phase]);

    ngx_http_request_t *r = req->data;

    omni_add_req_to_group(req->slot_index, &local_state.groups[phase]);

    // TODO: optimize for global
    ngx_shmtx_lock(&local_state.g_shmtx);
    omni_remove_req_from_group_by_req_index(
        req->slot_index,
        &g_state->groups[req->phase]);

    omni_add_req_to_group(req->slot_index, &g_state->groups[phase]);
    ngx_shmtx_unlock(&local_state.g_shmtx);

    req->phase = phase;
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
    req->metrics.time_contents_received = ngx_current_msec;

    ngx_chain_t *chain = r->request_body->bufs;

    while (chain)
    {
        ngx_buf_t *b = chain->buf;
        req->metrics.prompt_num_tokens += ngx_buf_size(b);
        chain = chain->next;
    }

    omni_phase_transition_local(PHASE_PREFILL_WAITING_SCHEDULE, req);
    req->metrics.time_enter_wait_prefill = ngx_current_msec;

    r->count++;
}

static void omni_proxy_remove_req_from_groups(omni_req_t *req)
{
    ngx_shmtx_lock(&local_state.g_shmtx);
    omni_remove_req_from_group_by_req_index(req->slot_index, &g_state->groups[req->phase]);
    g_state->decode_states[req->upstream_endpoint_idx].num_running--;
    ngx_shmtx_unlock(&local_state.g_shmtx);
    omni_remove_req_from_group_by_req_index(req->slot_index, &local_state.groups[req->phase]);
}

static void omni_proxy_main_req_cleanup(void *data)
{
    omni_req_t *req = data;
    omni_proxy_remove_req_from_groups(req);

    ngx_log_error(NGX_LOG_INFO, req->data->connection->log, 0, "[Decode-%d]: Done from %d.",
                  req->slot_index, req->upstream_endpoint_idx);

    omni_free_req(req);
}

static ngx_int_t omni_proxy_handler(ngx_http_request_t *r)
{
    if (r->parent != NULL)
    {
        return NGX_DECLINED;
    }

    omni_req_t *req = omni_allocate_req(r);
    if (req == NULL)
    {
        ngx_http_finalize_request(r, NGX_ERROR);
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }
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
    ngx_http_request_t *r = req->data;
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

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[Prefill-%d] Done from %d", req->slot_index, req->upstream_endpoint_idx);

    if (rc != NGX_OK)
    {
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[Prefill-%d] subrequest failed with code %i", req->slot_index, rc);
        ngx_http_finalize_request(r->main, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return rc;
    }

    if (subr->headers_out.status != NGX_HTTP_OK)
    {
        r->headers_out.status = subr->headers_out.status;
        ngx_http_finalize_request(r, subr->headers_out.status);
        return NGX_OK;
    }

    // Traverse the out_bufs chain to process the entire response body
    for (cl = subr->out; cl; cl = cl->next)
    {
        total += ngx_buf_size(cl->buf);
    }

    // ngx_log_error(
    //     NGX_LOG_INFO, r->connection->log, 0, "done prefill subrequest r:%p %d, status:%i", r, total, req->status);

    // Allocate memory for the temporary body copy + null terminator for cJSON
    ctx->prefill_response_body = ngx_palloc(r->main->pool, total + 1);
    if (ctx->prefill_response_body == NULL)
    {
        ngx_http_finalize_request(r, NGX_ERROR);
        ngx_log_error(
            NGX_LOG_ERR, r->connection->log, 0, "done prefill subrequest, malloc prefill response body buffer failed");
        return rc;
    }

    // Copy main request's buffer chain to a temporary contiguous block
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

    omni_upstream_prefill_t *us = &g_state->prefill_states[req->upstream_endpoint_idx];
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
        batch->average_delta = batch->average_delta * (batch->num_requests - 1) + delta / (batch->num_requests);
        batch->last_response_receive_time = ngx_current_msec;
        batch->num_requests++;
        batch->num_tokens += req->metrics.prompt_num_tokens;
    }

    omni_phase_transition_local(PHASE_DECODE_WAITING_SCHEDULE, req);
    req->metrics.time_enter_wait_decode = ngx_current_msec;

    return NGX_DONE;
}

static ngx_int_t ngx_http_prefill_wakeup(omni_req_t *req)
{
    ngx_http_request_t *r = req->data;
    size_t len = prefill_uri_len + r->uri.len;
    u_char *uri = ngx_pcalloc(r->pool, len);
    ngx_memcpy(uri, prefill_uri, prefill_uri_len);
    ngx_memcpy(uri + prefill_uri_len, r->uri.data, r->uri.len);

    ngx_str_t sub_uri = (ngx_str_t){prefill_uri_len + r->uri.len, uri};
    ngx_http_post_subrequest_t *psr = ngx_pcalloc(r->pool, sizeof(ngx_http_post_subrequest_t));

    psr->handler = ngx_http_prefill_post_subrequest;
    psr->data = req;

    ngx_str_t args = ngx_null_string;
    ngx_http_request_t *sr;

    ngx_int_t rc = ngx_http_subrequest(r, &sub_uri, &args, &sr, psr,
                                       NGX_HTTP_SUBREQUEST_WAITED | NGX_HTTP_SUBREQUEST_IN_MEMORY);
    if (rc != NGX_OK)
    {
        ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return rc;
    }
    sr->method = r->method;
    sr->method_name = r->method_name;

    omni_req_context_t *ctx = omni_get_req_ctx(r);
    ngx_http_set_ctx(sr, ctx, ngx_http_omni_proxy_module);

    omni_proxy_prepare_prefill_subrequest(r, sr, omni_get_req_ctx(req->data));

    omni_phase_transition_local(PHASE_PREFILLING, req);

    req->metrics.time_to_prefill = ngx_current_msec;

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[Prefill-%d] Submit to:%d",
                  req->slot_index, req->upstream_endpoint_idx);

    // sr->phase_handler = 0;

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

    // This get called after change from SCHEDULDED to ...ING
    assert(req->phase == PHASE_PREFILLING || req->phase == PHASE_DECODING);

    // ngx_uint_t idx = req->upstream_endpoint_idx;
    ngx_uint_t idx = req->upstream_endpoint_idx;
    // TODO: Need to consider the liveness of the upstream in case it crashed or lag
    assert(idx < rrp->peers->number);

    pc->sockaddr = peers->peer[idx].sockaddr;
    pc->socklen = peers->peer[idx].socklen;
    pc->name = &peers->peer[idx].name;
    rrp->current = &peers->peer[idx];

    if (req->phase == PHASE_PREFILLING)
    {
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[Prefill-%d] Upstream set: %d, running:%d",
                      req->slot_index, idx, g_state->prefill_states[idx].num_running);
    }
    else
    {
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[Decode-%d] Upstream set: %d, running:%d",
                      req->slot_index, idx, g_state->decode_states[idx].num_running);
    }

    return NGX_OK;
}

static ngx_int_t omni_proxy_upstream_init(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *uscf)
{

    omni_req_t *req = omni_get_req(r);
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[-%d] init upstream at phase: %d",
                  req->slot_index, req->phase);

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

static ngx_int_t omni_proxy_body_filter(ngx_http_request_t *r, ngx_chain_t *in)
{
    ngx_chain_t *cl;

    omni_req_t *req = omni_get_req(r);

    // Update request level statistics
    if (r->main == r && req->phase == PHASE_DECODING)
    {
        if (req->metrics.time_last_reponse)
        {
            req->metrics.tpot =
                ((req->metrics.tpot * req->metrics.decoded_tokens) +
                 ngx_current_msec - req->metrics.time_last_reponse) /
                ++req->metrics.decoded_tokens;
        }
        else
        {
            req->metrics.time_first_token = req->metrics.tpot = ngx_current_msec - req->metrics.time_to_decode;
        }

        req->metrics.time_last_reponse = ngx_current_msec;
    }

    omni_upstream_decode_t *us = &g_state->decode_states[req->upstream_endpoint_idx];
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
        batch->average_delta = batch->average_delta * (batch->num_requests - 1) + delta / (batch->num_requests);
        batch->last_response_receive_time = ngx_current_msec;
        batch->num_requests++;
        batch->num_tokens += req->metrics.prompt_num_tokens + req->metrics.decoded_tokens;
    }

    return local_state.ngx_http_next_body_filter(r, in);
}

ngx_int_t ngx_http_omni_proxy_redirect(ngx_http_request_t *r,
                                       ngx_str_t *uri, ngx_str_t *args)
{
    ngx_http_core_srv_conf_t *cscf;

    r->uri_changes--;

    if (r->uri_changes == 0)
    {
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                      "rewrite or internal redirection cycle "
                      "while internally redirecting to \"%V\"",
                      uri);

        r->main->count++;
        ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return NGX_DONE;
    }

    r->uri = *uri;

    if (args)
    {
        r->args = *args;
    }
    else
    {
        ngx_str_null(&r->args);
    }

    ngx_log_debug2(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "internal redirect: \"%V?%V\"", uri, &r->args);

    ngx_http_set_exten(r);
    omni_req_context_t *ctx = ngx_http_get_module_ctx(r, ngx_http_omni_proxy_module);

    ngx_memzero(r->ctx, sizeof(void *) * ngx_http_max_module);
    ngx_http_set_ctx(r, ctx, ngx_http_omni_proxy_module);

    cscf = ngx_http_get_module_srv_conf(r, ngx_http_core_module);
    r->loc_conf = cscf->ctx->loc_conf;

    ngx_http_update_location_config(r);

#if (NGX_HTTP_CACHE)
    r->cache = NULL;
#endif

    r->internal = 1;
    r->valid_unparsed_uri = 0;
    r->add_uri_to_alias = 0;
    r->main->count++;

    ngx_http_handler(r);

    return NGX_DONE;
}

static ngx_int_t ngx_http_decode_wakeup(omni_req_t *req)
{
    ngx_http_request_t *r = req->data;
    size_t len = decode_uri_len + r->uri.len;
    u_char *uri = ngx_pcalloc(r->pool, len);
    ngx_memcpy(uri, decode_uri, prefill_uri_len);
    ngx_memcpy(uri + decode_uri_len, r->uri.data, r->uri.len);
    ngx_str_t decode_uri = (ngx_str_t){len, uri};

    r->count--;

    omni_proxy_prepare_decode_request_body(r, omni_get_req_ctx(req->data));

    omni_phase_transition_local(PHASE_DECODING, req);
    req->metrics.time_to_decode = ngx_current_msec;

    r->count = 0;
    r->headers_out.status = NGX_HTTP_OK;
    r->headers_out.content_length_n = -1;
    r->headers_out.content_length = NULL;
    r->header_only = 0;

    ngx_http_clear_last_modified(r);
    ngx_http_clear_accept_ranges(r);
    ngx_http_clear_content_length(r);

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[Decode-%d] Submited to: %d",
                  req->slot_index, req->upstream_endpoint_idx);

    return ngx_http_omni_proxy_redirect(r, &decode_uri, NULL);
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
            ngx_http_request_t *r = req->data;
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

    printf("\rActive requests: %d, prefill waiting: %d, prefill running: %d "
           "decode waiting: %d, decode running: %d",
           g_state->request_pool.num_requests,
           g_state->groups[PHASE_PREFILL_WAITING_SCHEDULE].num_requests,
           g_state->groups[PHASE_PREFILLING].num_requests,
           g_state->groups[PHASE_DECODE_WAITING_SCHEDULE].num_requests,
           g_state->groups[PHASE_DECODING].num_requests);
}

static void omni_proxy_schedule(omni_global_state_t *gs)
{
    omni_proxy_schedule_prefill(gs);
    omni_proxy_schedule_decode(gs);
}

static void omni_proxy_timer_handler(ngx_event_t *ev)
{
    omni_proxy_schedule(g_state);

    omni_proxy_run_group(PHASE_PREFILL_SCHEDULED, ngx_http_prefill_wakeup);
    omni_proxy_run_group(PHASE_DECODE_SCHEDULED, ngx_http_decode_wakeup);

    ngx_add_timer(&local_state.omni_proxy_timer_event, TIMER_INTERVAL);

    print_summary();
}

static ngx_int_t omni_proxy_global_state_init(ngx_shm_zone_t *zone, void *data)
{
    if (zone->shm.addr == NULL)
    {
        return NGX_ERROR;
    }

    g_state = (omni_global_state_t *)zone->shm.addr;
    memset(g_state, 0, GLOBAL_STATE_SIZE);
    g_state->num_decode_endpoints = local_state.num_decode_endpoints;
    g_state->num_prefill_endpoints = local_state.num_prefill_endpoints;

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

#define PREFILL_ENDPOINTS "prefill_endpoints"
#define DECODE_ENDPOINTS "decode_endpoints"

static ngx_int_t omni_proxy_post_config(ngx_conf_t *cf)
{
    ngx_http_core_main_conf_t *cmcf;
    ngx_http_handler_pt *h;

    // cmcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_core_module);

    // h = ngx_array_push(&cmcf->phases[NGX_HTTP_CONTENT_PHASE].handlers);
    // if (h == NULL)
    // {
    //     return NGX_ERROR;
    // }

    // *h = omni_proxy_handler;

    local_state.ngx_http_next_body_filter = ngx_http_top_body_filter;
    ngx_http_top_body_filter = omni_proxy_body_filter;

    if (omni_proxy_init_global_state(cf) != NGX_OK)
    {
        return NGX_ERROR;
    }

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

    return NGX_OK;
}

static ngx_int_t omni_proxy_init_process(ngx_cycle_t *cycle)
{
    ngx_log_error(NGX_LOG_INFO, cycle->log, 0, "Initializing shared timer in worker process.");
    printf("Size of global states:%ldK\n", sizeof(omni_global_state_t) / 1024);

    ngx_memzero(&local_state.omni_proxy_timer_event, sizeof(ngx_event_t));
    local_state.omni_proxy_timer_event.handler = omni_proxy_timer_handler;
    local_state.omni_proxy_timer_event.log = cycle->log;
    local_state.omni_proxy_timer_event.data = NULL;

    printf("Init timer\n");

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

    return NGX_CONF_OK;
}

static ngx_command_t omni_proxy_commands[] = {
    {ngx_string("omni_proxy"),
     NGX_HTTP_LOC_CONF | NGX_CONF_NOARGS,
     omni_proxy_init_conf,
     NGX_HTTP_LOC_CONF_OFFSET,
     0,
     NULL},
    ngx_null_command};
static ngx_http_module_t omni_proxy_module_ctx = {
    NULL,                   // preconfiguration
    omni_proxy_post_config, // postconfiguration
    NULL,                   // create main configuration
    NULL,                   // init main configuration

    NULL, // create server configuration
    NULL, // merge server configuration

    NULL, // create location configuration
    NULL  // merge location configuration
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