#pragma once

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <stdint.h>
#include <assert.h>
#include <ngx_atomic.h>

#define NUM_PREFILL_BATCH_METRICS_HIS 10
#define NUM_DECODE_BATCH_METRICS_HIS 10
#define MAX_PREFILL_UPSTREAMS 64
#define MAX_DECODE_UPSTREAMS 512
#define MAX_REQUEST_SLOTS 8192

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
    ngx_http_request_t *data;
    ngx_http_request_t *subr;
    uint32_t last_retry;
    pid_t worker_pid;
    uint16_t upstream_endpoint_idx;
    omni_proxy_request_phase_t phase;
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

/* Allocate one free request slot, or return NULL if full */
static inline omni_req_t *omni_allocate_request(omni_request_pool_t *pool, void *data)
{
    if (pool->num_requests >= MAX_REQUEST_SLOTS)
    {
        /* pool is full */
        return NULL;
    }

    /* Scan each 64‐bit word for a zero bit */
    for (size_t word = pool->head / 64; word < (MAX_REQUEST_SLOTS / 64); ++word)
    {
        uint64_t bits = pool->in_use_bitmap[word];
        if (bits != UINT64_MAX)
        {
            /* There's at least one zero bit in this word */
            /* invert bits and find first 1 (i.e. first zero in original) */
            uint64_t inv = ~bits;
            unsigned offset = __builtin_ctzll(inv); /* 0..63 */
            size_t idx = word * 64 + offset;
            assert(idx < MAX_REQUEST_SLOTS);

            /* Mark it in use */
            pool->in_use_bitmap[word] |= (1ULL << offset);
            pool->num_requests++;

            /* Zero‐initialize the slot before returning */
            omni_req_t *r = &pool->slots[idx];
            memset(r, 0, sizeof(*r));

            r->in_use = 1;
            r->data = data;
            r->slot_index = idx;

            pool->head++;

            if (pool->head == MAX_REQUEST_SLOTS)
            {
                pool->head = 0;
            }
            return r;
        }
    }

    /* should never get here if num_requests < MAX_REQUEST_SLOTS */
    return NULL;
}

/* Free a previously‐allocated request slot */
static inline void omni_free_request(omni_request_pool_t *pool, omni_req_t *req)
{
    /* Compute index of req in the pool */
    ptrdiff_t idx = req - pool->slots;
    assert(idx >= 0 && idx < MAX_REQUEST_SLOTS);

    size_t word = (size_t)idx / 64;
    unsigned bit = (unsigned)idx % 64;
    uint64_t mask = (1ULL << bit);

    /* Ensure it was in use */
    assert(pool->in_use_bitmap[word] & mask);

    /* Clear the bit */
    pool->in_use_bitmap[word] &= ~mask;
    pool->num_requests--;

    req->in_use = 0;
}

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
    omni_req_info_t requests[MAX_REQUEST_SLOTS];
} omni_req_group_t;

static inline omni_req_info_t *omni_add_req_to_group(uint32_t req_index, omni_req_group_t *group)
{
    assert(group->watermark < MAX_REQUEST_SLOTS);
    uint32_t idx = group->watermark++;
    group->requests[idx].in_use = 1;
    group->requests[idx].slot_index = req_index;
    group->requests[idx].weight = 0.0; // initial weight; user may adjust later
    group->num_requests++;

    return &group->requests[idx];
}

static inline void omni_remove_from_group_by_req_info(omni_req_info_t *req_info, omni_req_group_t *group)
{
    assert(req_info != NULL);
    req_info->in_use = 0;
    // keep the weight for debug
    // req_info->weight = -1;
    assert(group->num_requests > 0);
    group->num_requests--;
}

static int cmp_req_info_desc(const void *pa, const void *pb)
{
    const omni_req_info_t *a = pa, *b = pb;
    if (a->in_use != b->in_use)
        return (int)b->in_use - (int)a->in_use; // in_use=1 precedes in_use=0
    if (a->weight < b->weight)
        return 1;
    if (a->weight > b->weight)
        return -1;
    return 0;
}

static inline void omni_sort_compact_group(omni_req_group_t *group)
{
    if (group->watermark > 1 && group->num_requests)
    {
        qsort(group->requests,
              group->watermark,
              sizeof(omni_req_info_t),
              cmp_req_info_desc);
    }
    // now the first num_requests entries are in_use==1;
    // drop the rest
    group->watermark = group->num_requests;
}

static inline void omni_remove_req_from_group_by_req_index(uint32_t req_index, omni_req_group_t *group)
{
    for (uint32_t i = 0; i < group->watermark; ++i)
    {
        if (group->requests[i].in_use &&
            group->requests[i].slot_index == req_index)
        {
            omni_remove_from_group_by_req_info(&group->requests[i], group);
            // TODO: performance optimization required
            omni_sort_compact_group(group);
            return;
        }
    }
    // not found ⇒ user error
    assert(!"omni_remove_req_from_group: slot_index not in group");
}

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

typedef struct omni_global_state_s
{
    ngx_shmtx_sh_t lock;
    omni_request_pool_t request_pool;
    omni_req_group_t groups[PHASE_MAX];
    uint16_t num_prefill_endpoints;
    uint16_t last_selected_prefill;
    uint16_t num_decode_endpoints;
    uint16_t last_selected_decode;
    uint32_t last_summary;
    omni_upstream_prefill_t prefill_states[MAX_PREFILL_UPSTREAMS];
    omni_upstream_decode_t decode_states[MAX_DECODE_UPSTREAMS];
} omni_global_state_t;

#define GLOBAL_STATE_SIZE sizeof(omni_global_state_t)

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

typedef struct omni_req_context_s
{
    omni_req_t *req;
    u_char *origin_body_data;
    ngx_uint_t origin_body_data_size;
    void *origin_body_tokens;
    int origin_body_tokens_size;
    u_char *prefill_response_body;
    ngx_uint_t prefill_response_body_size;
} omni_req_context_t;