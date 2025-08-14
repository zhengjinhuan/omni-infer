
// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <omni_proxy.h>
#include <omni_scheduler.h>
#include <omni_utils.h>

static void update_prefill_weights(omni_req_group_t *group)
{
    uint32_t max_prompt_tokens = 0;
    uint32_t max_wait_time = 0;
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

    for (uint32_t i = 0; i < group->watermark; i++)
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

void omni_proxy_schedule_prefill(omni_global_state_t *gs)
{
    omni_req_group_t *group = &gs->groups[PHASE_PREFILL_WAITING_SCHEDULE];

    // TODO: Check should schedule or wait based on upstream expected come back time

    update_prefill_weights(group);

    for (uint32_t i = 0; i < group->num_requests; i++)
    {
        omni_req_info_t *info = &group->requests[i];
        omni_req_t *req = omni_info_to_req(info);

        assert(omni_req_is_in_phase(req, PHASE_PREFILL_WAITING_SCHEDULE));

        uint32_t least_load = UINT32_MAX;
        uint32_t selected = UINT32_MAX;
        for (int j = 0; j < gs->num_prefill_endpoints; j++)
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

        req->prefill_upstream_endpoint_idx = selected;
        gs->prefill_states[selected].num_running++;
        gs->prefill_states[selected].num_tokens += req->metrics.prompt_num_tokens;

        omni_global_phase_change_to(req, PHASE_PREFILL_WAITING_SCHEDULE, PHASE_PREFILL_SCHEDULED);
        omni_req_leave_phase(req, PHASE_PREFILL_WAITING_SCHEDULE);
        omni_req_enter_phase(req, PHASE_PREFILL_SCHEDULED);
        // If policy is parallel, we can change to DECODE_SCHEDULED directly

        req->metrics.time_prefill_scheduled = ngx_current_msec;

        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0, "[Prefill-%d] Schedule to: %d",
                      req->slot_index, req->prefill_upstream_endpoint_idx);
    }

    // TODO: estimated expected next schedule time
}

void omni_proxy_schedule_decode(omni_global_state_t *gs)
{
    omni_req_group_t *group = &gs->groups[PHASE_DECODE_WAITING_SCHEDULE];
    // TODO: Check should schedule or wait based on upstream expected come back time
    // TODO: Here we can do some estimation of pull kv finish time to make sure pull kv
    // workloads are balanced

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
        uint32_t selected = UINT32_MAX;
        for (int j = 0; j < gs->num_decode_endpoints; j++)
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

        req->decode_upstream_endpoint_idx = selected;
        gs->decode_states[selected].num_running++;

        omni_global_phase_change_to(req, PHASE_DECODE_WAITING_SCHEDULE, PHASE_DECODE_SCHEDULED);
        omni_req_leave_phase(req, PHASE_DECODE_WAITING_SCHEDULE);
        omni_req_enter_phase(req, PHASE_DECODE_SCHEDULED);

        req->metrics.time_decode_scheduled = ngx_current_msec;

        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0, "[Decode-%d] Schedule to: %d",
                      req->slot_index, req->decode_upstream_endpoint_idx);
    }
}