// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <omni_proxy.h>
#include <omni_scheduler.h>
#include <omni_utils.h>

#define OMNI_NETWORK_DELAY_MS 5
#define OMNI_ASYNC_DELAY_MS 3
#define OMNI_PREFILL_DEFAULT_REQ_MS 50

static inline ngx_msec_t omni_dispatch_overhead()
{
    return OMNI_NETWORK_DELAY_MS + OMNI_ASYNC_DELAY_MS;
}

static ngx_msec_t omni_prefill_base_time(const omni_upstream_prefill_t *state, ngx_msec_t now)
{
    ngx_msec_t base_time = now;

    if (state->batch.current.num_requests > 0 &&
        state->batch.current.finish_time > base_time)
    {
        base_time = state->batch.current.finish_time;
    }

    if (state->batch.last.finish_time > base_time)
    {
        base_time = state->batch.last.finish_time;
    }

    return base_time;
}

static ngx_msec_t omni_prefill_predict_next_finish(const omni_global_state_t *gs,
                                                   const omni_upstream_prefill_t *state,
                                                   uint32_t additional_requests,
                                                   ngx_msec_t now)
{
    ngx_msec_t base_time = omni_prefill_base_time(state, now);
    uint32_t total = state->batch.next.num_requests + additional_requests;

    if (total == 0)
    {
        return base_time;
    }

    ngx_msec_t duration = omni_prefill_model_predict(gs, total);
    return base_time + omni_dispatch_overhead() + duration;
}

static void update_prefill_weights(omni_req_group_t *group)
{
    uint32_t max_prompt_tokens = 0;
    ngx_msec_t max_wait_time = 0;
    for (uint32_t i = 0; i < group->watermark; i++)
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

        ngx_msec_t waited = ngx_current_msec - req->metrics.time_received;

        if (max_wait_time < waited)
        {
            max_wait_time = waited;
        }
    }

    if (max_wait_time < 50)
    {
        max_wait_time = 50;
    }

    if (max_prompt_tokens == 0)
    {
        max_prompt_tokens = 1;
    }

    for (uint32_t i = 0; i < group->watermark; i++)
    {
        omni_req_info_t *info = &group->requests[i];
        if (!info->in_use)
        {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);
        ngx_msec_t waited = ngx_current_msec - req->metrics.time_received;

        double token_weight = (double)(max_prompt_tokens - req->metrics.prompt_num_tokens) / max_prompt_tokens;
        double time_weight = (double)waited / max_wait_time;

        info->weight = token_weight * 0.8 + time_weight * 0.2;
    }

    omni_sort_compact_group(group);

    for (uint32_t idx = 0; idx < group->num_requests; idx++)
    {
        omni_req_info_t *info = &group->requests[idx];
        if (!info->in_use)
        {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);
        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                      "[Prefill-Sort] Order %ui: slot=%ui tokens=%ui weight=%.2f",
                      idx,
                      info->slot_index,
                      req->metrics.prompt_num_tokens,
                      info->weight);
    }
}

static void update_decode_weights(omni_req_group_t *group)
{
    uint32_t max_total_tokens = 1;

    for (uint32_t i = 0; i < group->watermark; i++)
    {
        omni_req_info_t *info = &group->requests[i];
        if (!info->in_use)
        {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);
        uint32_t total = req->metrics.prompt_num_tokens + req->metrics.max_tokens;
        if (total > max_total_tokens)
        {
            max_total_tokens = total;
        }
    }

    for (uint32_t i = 0; i < group->watermark; i++)
    {
        omni_req_info_t *info = &group->requests[i];
        if (!info->in_use)
        {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);
        info->weight = ((double)req->metrics.prompt_num_tokens +
                        (double)req->metrics.max_tokens) /
                       max_total_tokens;
    }

    omni_sort_compact_group(group);

    for (uint32_t idx = 0; idx < group->num_requests; idx++)
    {
        omni_req_info_t *info = &group->requests[idx];
        if (!info->in_use)
        {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);
        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                      "[Decode-Sort] Order %ui: slot=%ui total_tokens=%ui prompt_num_tokens=%ui max_tokens=%ui weight=%.2f",
                      idx,
                      info->slot_index,
                      req->metrics.prompt_num_tokens + req->metrics.max_tokens,
                      req->metrics.prompt_num_tokens,
                      req->metrics.max_tokens,
                      info->weight);
    }
}

void omni_proxy_schedule_prefill(omni_global_state_t *gs, ngx_http_omni_loc_conf_t *olcf)
{
    omni_req_group_t *group = &gs->groups[PHASE_PREFILL_WAITING_SCHEDULE];

    // TODO: Check should schedule or wait based on upstream expected come back time

    update_prefill_weights(group);

    for (uint32_t i = 0; i < group->num_requests; i++)
    {
        omni_req_info_t *info = &group->requests[i];
        omni_req_t *req = omni_info_to_req(info);

        assert(omni_req_is_in_phase(req, PHASE_PREFILL_WAITING_SCHEDULE));

        uint32_t selected = rand() % gs->num_prefill_endpoints;
        uint32_t best_match = 0;
        uint32_t best_load_tokens = UINT32_MAX;
        uint32_t best_running = UINT32_MAX;
        uint32_t best_idx = UINT32_MAX;
        ngx_msec_t best_finish = (ngx_msec_t)(-1);
        ngx_msec_t now = ngx_current_msec;

        for (uint32_t j = 0; j < gs->num_prefill_endpoints; j++)
        {
            uint32_t match_depth = req->match_depths[j];
            omni_upstream_prefill_t *state = &gs->prefill_states[j];

            if (state->num_tokens > olcf->max_batch_num_token ||
                state->num_running > olcf->prefill_max_num_seqs)
            {
                continue;
            }

            ngx_msec_t candidate_finish = omni_prefill_predict_next_finish(gs, state, 1, now);

            if (best_idx == UINT32_MAX ||
                match_depth > best_match ||
                (match_depth == best_match && candidate_finish < best_finish) ||
                (match_depth == best_match && candidate_finish == best_finish && state->num_tokens < best_load_tokens) ||
                (match_depth == best_match && candidate_finish == best_finish && state->num_tokens == best_load_tokens && state->num_running < best_running))
            {
                best_match = match_depth;
                best_load_tokens = state->num_tokens;
                best_running = state->num_running;
                best_idx = j;
                best_finish = candidate_finish;
            }
        }

        if (best_idx != UINT32_MAX)
        {
            selected = best_idx;
            if (best_match > 0)
            {
                ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                              "[Prefill-%d] Prefix cache hit on: %d with match_depth %d",
                              req->slot_index,
                              selected,
                              req->match_depths[selected]);
            }
        }
        else
        {
            uint32_t least_load = UINT32_MAX;
            ngx_flag_t found_available = 0;
            for (uint32_t m = gs->last_selected_prefill;
                 m < gs->num_prefill_endpoints + gs->last_selected_prefill;
                 m++)
            {
                uint32_t j = m % gs->num_prefill_endpoints;
                omni_upstream_prefill_t *state = &gs->prefill_states[j];
                if (state->num_running > olcf->prefill_max_num_seqs ||
                    state->num_tokens > olcf->max_batch_num_token)
                {
                    continue;
                }

                if (state->num_tokens < least_load)
                {
                    least_load = state->num_tokens;
                    selected = j;
                    found_available = 1;
                    if (least_load == 0)
                    {
                        break;
                    }
                }
            }

            if (!found_available)
            {
                for (uint32_t m = gs->last_selected_prefill;
                     m < gs->num_prefill_endpoints + gs->last_selected_prefill;
                     m++)
                {
                    uint32_t j = m % gs->num_prefill_endpoints;
                    omni_upstream_prefill_t *state = &gs->prefill_states[j];
                    if (state->num_tokens < least_load)
                    {
                        least_load = state->num_tokens;
                        selected = j;
                        if (least_load == 0)
                        {
                            break;
                        }
                    }
                }
            }
            best_finish = omni_prefill_predict_next_finish(gs, &gs->prefill_states[selected], 1, now);
            ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                          "[Prefill-%d] No Prefix cache hit, choose least workload Prefill %d with load %d",
                          req->slot_index,
                          selected,
                          least_load);
        }

        omni_upstream_prefill_t *selected_state = &gs->prefill_states[selected];

        req->prefill_upstream_endpoint_idx = selected;
        gs->last_selected_prefill = selected + 1;

        selected_state->num_running++;
        selected_state->num_tokens += req->metrics.prompt_num_tokens;
        selected_state->last_scheduled_time = now;
        selected_state->batch.next.num_requests++;
        selected_state->batch.next.finish_time = omni_prefill_predict_next_finish(gs, selected_state, 0, now);
        selected_state->expected_next_schedule_time = selected_state->batch.next.finish_time;

        if (!selected_state->batch.current_open &&
            selected_state->batch.current.num_requests == 0)
        {
            selected_state->batch.current = selected_state->batch.next;
            selected_state->batch.next.num_requests = 0;
            selected_state->batch.current_start_time = now + omni_dispatch_overhead();

            if (selected_state->batch.current.num_requests > 0)
            {
                ngx_msec_t duration = omni_prefill_model_predict(
                    gs,
                    selected_state->batch.current.num_requests);
                selected_state->batch.current.finish_time =
                    selected_state->batch.current_start_time + duration;
                selected_state->batch.current_open = 1;
                selected_state->expected_next_schedule_time =
                    selected_state->batch.current.finish_time;
            }
            else
            {
                selected_state->batch.current.finish_time =
                    selected_state->batch.current_start_time;
                selected_state->batch.current_open = 0;
            }

            selected_state->batch.next.finish_time =
                selected_state->batch.current.finish_time;
        }

        omni_global_phase_change_to(req, PHASE_PREFILL_WAITING_SCHEDULE, PHASE_PREFILL_SCHEDULED);
        omni_req_leave_phase(req, PHASE_PREFILL_WAITING_SCHEDULE);
        omni_req_enter_phase(req, PHASE_PREFILL_SCHEDULED);

        // If policy is parallel, we can change to DECODE_SCHEDULED directly
        if (gs->pd_policy == PD_PARALLEL)
        {
            req->decode_upstream_endpoint_idx = 0;
            gs->decode_states[selected].num_running++;

            omni_add_req_to_group(req->slot_index, &gs->groups[PHASE_DECODE_SCHEDULED]);
            omni_req_enter_phase(req, PHASE_DECODE_SCHEDULED);
        }

        req->metrics.time_prefill_scheduled = ngx_current_msec;

        struct timeval tv;
        gettimeofday(&tv, NULL);
        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                            "<<<Action: Enter state P scheduled; Timestamp:%d.%06d; RequestID:%s", tv.tv_sec, tv.tv_usec, req->request_id);

        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0, "[Prefill-%d] Schedule to: %d",
                      req->slot_index, req->prefill_upstream_endpoint_idx);
    }

    // TODO: estimated expected next schedule time
}

void omni_proxy_schedule_decode(omni_global_state_t *gs, ngx_http_omni_loc_conf_t *olcf)
{
    omni_req_group_t *group = &gs->groups[PHASE_DECODE_WAITING_SCHEDULE];
    // TODO: Check should schedule or wait based on upstream expected come back time
    // TODO: Here we can do some estimation of pull kv finish time to make sure pull kv
    // workloads are balanced

    update_decode_weights(group);

    for (size_t i = 0; i < group->watermark; i++)
    {
        omni_req_info_t *info = &group->requests[i];
        if (!info->in_use)
        {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);
        assert(omni_req_is_in_phase(req, PHASE_DECODE_WAITING_SCHEDULE));

        uint32_t least_load = UINT32_MAX;
        uint32_t selected = rand() % gs->num_decode_endpoints;
        for (int m = gs->last_selected_decode;
             m < (int)gs->num_decode_endpoints + (int)gs->last_selected_decode;
             m++)
        {
            int j = m % gs->num_decode_endpoints;
            if (gs->decode_states[j].num_tokens < least_load &&
                gs->decode_states[j].num_running < olcf->decode_max_num_seqs)
            {
                least_load = gs->decode_states[j].num_tokens;
                selected = (uint32_t)j;
                if (least_load == 0)
                {
                    break;
                }
            }
        }

        req->decode_upstream_endpoint_idx = selected;
        gs->last_selected_decode = selected + 1;
        gs->decode_states[selected].num_running++;
        gs->decode_states[selected].num_tokens += req->metrics.prompt_num_tokens;

        omni_global_phase_change_to(req, PHASE_DECODE_WAITING_SCHEDULE, PHASE_DECODE_SCHEDULED);
        omni_req_leave_phase(req, PHASE_DECODE_WAITING_SCHEDULE);
        omni_req_enter_phase(req, PHASE_DECODE_SCHEDULED);

        req->metrics.time_decode_scheduled = ngx_current_msec;

        struct timeval tv;
        gettimeofday(&tv, NULL);
        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                      "<<<Action: Enter state D scheduled; Timestamp:%d.%06d; RequestID:%s",
                      tv.tv_sec,
                      tv.tv_usec,
                      req->request_id);

        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                      "[Decode-%d] Schedule to: %d (load=%ui)",
                      req->slot_index,
                      req->decode_upstream_endpoint_idx,
                      gs->decode_states[selected].num_tokens);
    }
}
