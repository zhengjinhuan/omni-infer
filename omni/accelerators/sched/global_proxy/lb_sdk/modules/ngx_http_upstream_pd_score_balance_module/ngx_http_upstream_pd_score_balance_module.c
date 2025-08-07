// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <ngx_http_upstream.h>
#include <ngx_atomic.h>
#include <stdlib.h>
#include <float.h>

#define MAX_TOTAL_ACTIVE_REQS 8192
#define MAX_PREDICT_REQS 4
#define MAX_PEER_COUNT 512
#define MAX_P 32
#define MAX_D 512

typedef enum {
    PD_MODE_NONE = 0,
    PD_MODE_PREFILL,
    PD_MODE_DECODE
} pd_score_mode_e;

typedef struct {
    size_t shm_size;
} ngx_http_pd_score_main_conf_t;

typedef struct {
    pd_score_mode_e mode;
    ngx_uint_t n_req_limit;
} ngx_http_pd_score_srv_conf_t;

typedef struct {
    ngx_atomic_t active_requests;
    ngx_atomic_t total_time_cost;
    ngx_atomic_t total_decode_num;
    ngx_atomic_t total_request_length;
    ngx_uint_t active_request_count;
} ngx_http_pd_score_shm_peer_P_t;

typedef struct {
    void *id_ptr;
    ngx_uint_t request_length;
    ngx_uint_t inque_time;
} ngx_http_pd_score_running_request;

typedef struct {
    ngx_atomic_t active_requests;
    ngx_atomic_t total_time_cost;
    ngx_atomic_t total_decode_num;
    ngx_atomic_t total_request_length;
    ngx_uint_t active_request_count;
} ngx_http_pd_score_shm_peer_D_t;

typedef struct {
    ngx_uint_t peer_count;
    ngx_uint_t total_active_request_count;
    ngx_http_pd_score_running_request running_requests_P[MAX_TOTAL_ACTIVE_REQS];
    ngx_http_pd_score_shm_peer_P_t peers_P[MAX_P];
    ngx_http_pd_score_shm_peer_D_t peers_D[MAX_D];
} ngx_http_pd_score_shm_block_t;

typedef struct {
    ngx_http_upstream_rr_peer_data_t *rrp;
    ngx_uint_t chosen;
    double my_time_cost;
    ngx_atomic_t decode_token_count;
    ngx_uint_t request_length;
    int first_chunk;
    ngx_uint_t last_total_tokens;
} ngx_http_pd_score_peer_data_t;

static ngx_shm_zone_t *ngx_http_pd_score_shm_zone = NULL;
static ngx_http_pd_score_shm_block_t *pd_shm = NULL;
static ngx_uint_t ngx_http_pd_score_shm_size = 0;
static ngx_uint_t ngx_http_pd_score_req_lim_D = 0;
static ngx_http_output_body_filter_pt ngx_http_next_body_filter = NULL;

static void *ngx_http_pd_score_create_main_conf(ngx_conf_t *cf);
static void *ngx_http_pd_score_create_srv_conf(ngx_conf_t *cf);
static char *ngx_http_pd_score_set_mode(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);
static ngx_int_t ngx_http_pd_score_init_shm_zone(ngx_shm_zone_t *shm_zone, void *data);
static ngx_int_t ngx_http_pd_score_postconfig(ngx_conf_t *cf);

static ngx_int_t ngx_http_pd_score_upstream_init(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *uscf);
static ngx_int_t ngx_http_pd_score_get_peer(ngx_peer_connection_t *pc, void *data);
static void ngx_http_pd_score_free_peer_P(ngx_peer_connection_t *pc, void *data, ngx_uint_t state);
static void ngx_http_pd_score_free_peer_D(ngx_peer_connection_t *pc, void *data, ngx_uint_t state);
static ngx_int_t ngx_http_pd_score_prefill_strategy(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *uscf);
static ngx_int_t ngx_http_pd_score_decode_strategy(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *uscf);
static ngx_int_t ngx_http_pd_score_body_filter(ngx_http_request_t *r, ngx_chain_t *in);

void ngx_http_pd_score_add_decoded_tokens(ngx_http_request_t *r, ngx_uint_t num_tokens);

static ngx_command_t ngx_http_upstream_pd_score_commands[] = {
    { ngx_string("pd_score_balance_shm_size"),
      NGX_HTTP_MAIN_CONF | NGX_CONF_TAKE1, ngx_conf_set_size_slot,
      NGX_HTTP_MAIN_CONF_OFFSET, offsetof(ngx_http_pd_score_main_conf_t, shm_size),
      NULL },

    { ngx_string("pd_score_balance"),
      NGX_HTTP_UPS_CONF | NGX_CONF_TAKE1,
      ngx_http_pd_score_set_mode, NGX_HTTP_SRV_CONF_OFFSET,
      offsetof(ngx_http_pd_score_srv_conf_t, mode), NULL },

    { ngx_string("pd_score_balance_decode_req_limit"),
      NGX_HTTP_UPS_CONF | NGX_CONF_TAKE1,
      ngx_conf_set_num_slot, NGX_HTTP_SRV_CONF_OFFSET,
      offsetof(ngx_http_pd_score_srv_conf_t, n_req_limit),
      NULL },

    ngx_null_command
};

static ngx_http_module_t ngx_http_upstream_pd_score_balance_module_ctx = {
    NULL,
    ngx_http_pd_score_postconfig,
    ngx_http_pd_score_create_main_conf,
    NULL,
    ngx_http_pd_score_create_srv_conf,
    NULL,
    NULL,
    NULL
};

ngx_module_t ngx_http_upstream_pd_score_balance_module = {
    NGX_MODULE_V1,
    &ngx_http_upstream_pd_score_balance_module_ctx,
    ngx_http_upstream_pd_score_commands,
    NGX_HTTP_MODULE,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NGX_MODULE_V1_PADDING
};

static void *ngx_http_pd_score_create_main_conf(ngx_conf_t *cf) {
    ngx_http_pd_score_main_conf_t *conf = ngx_pcalloc(cf->pool, sizeof(*conf));
    if (conf == NULL) {
        ngx_log_error(NGX_LOG_EMERG, cf->log, 0,
                      "Failed to allocate pd_score_balance main conf");
        return NULL;
    }

    conf->shm_size = NGX_CONF_UNSET_SIZE;
    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, cf->log, 0,
                   "pd_score_balance main conf created");
    return conf;
}

static char *ngx_http_pd_score_set_mode(ngx_conf_t *cf, ngx_command_t *cmd,
                                        void *conf) {
    ngx_str_t *value = cf->args->elts;

    ngx_http_pd_score_srv_conf_t *c = conf;

    if (ngx_strcmp(value[1].data, "prefill") == 0) {
        c->mode = PD_MODE_PREFILL;
    } else if (ngx_strcmp(value[1].data, "decode") == 0) {
        c->mode = PD_MODE_DECODE;
    } else {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                           "invalid pd_score_balance mode: %V", &value[1]);
        return NGX_CONF_ERROR;
    }
    return NGX_CONF_OK;
}

static void *ngx_http_pd_score_create_srv_conf(ngx_conf_t *cf) {
    ngx_http_pd_score_srv_conf_t *conf = ngx_pcalloc(cf->pool, sizeof(*conf));
    if (conf == NULL) {
        return NULL;
    }
    conf->mode = PD_MODE_NONE;
    conf->n_req_limit = NGX_CONF_UNSET_UINT;
    return conf;
}

static ngx_int_t ngx_http_pd_score_init_shm_zone(ngx_shm_zone_t *shm_zone,
                                                 void *data) {
    ngx_slab_pool_t *shpool;
    ngx_http_pd_score_shm_block_t *shm_block;
    ngx_uint_t i, n;
    if (data) {
        shm_zone->data = data;
        pd_shm = data;
        return NGX_OK;
    }
    shpool = (ngx_slab_pool_t *)shm_zone->shm.addr;

    n = MAX_PEER_COUNT;

    size_t sz = sizeof(ngx_http_pd_score_shm_block_t);
    shm_block = ngx_slab_alloc(shpool, sz);

    if (!shm_block) {
        return NGX_ERROR;
    }

    shm_block->peer_count = n;
    shm_block->total_active_request_count = 0;
    for (i = 0; i < MAX_P; i++) {
        shm_block->peers_P[i].active_requests = 0;
        shm_block->peers_P[i].total_time_cost = 0;
        shm_block->peers_P[i].total_decode_num = 0;
        shm_block->peers_P[i].total_request_length = 0;
        shm_block->peers_P[i].active_request_count = 0;
    }
    for (i = 0; i < MAX_D; i++) {
        shm_block->peers_D[i].active_requests = 0;
        shm_block->peers_D[i].total_time_cost = 0;
        shm_block->peers_D[i].total_decode_num = 0;
        shm_block->peers_D[i].total_request_length = 0;
        shm_block->peers_D[i].active_request_count = 0;
    }
    shm_zone->data = shm_block;
    pd_shm = shm_block;
    return NGX_OK;
}

static ngx_int_t ngx_http_pd_score_postconfig(ngx_conf_t *cf) {
    ngx_http_pd_score_main_conf_t *pmcf = ngx_http_conf_get_module_main_conf(
        cf, ngx_http_upstream_pd_score_balance_module);
    if (pmcf == NULL) {
        ngx_log_error(NGX_LOG_EMERG, cf->log, 0, "Failed to get main conf");
        return NGX_ERROR;
    }
    if (pmcf->shm_size == 0 || pmcf->shm_size == NGX_CONF_UNSET_SIZE) {
        pmcf->shm_size = 256 * ngx_pagesize;
    }
    ngx_http_pd_score_shm_size = pmcf->shm_size;
    ngx_log_error(NGX_LOG_WARN, cf->log, 0, "Set shm_size: %uz bytes", ngx_http_pd_score_shm_size);

    ngx_http_next_body_filter = ngx_http_top_body_filter;
    ngx_http_top_body_filter = ngx_http_pd_score_body_filter;

    ngx_str_t *shm_name = ngx_palloc(cf->pool, sizeof(*shm_name));
    shm_name->len = sizeof("pd_score_balance") - 1;
    shm_name->data = (u_char *)"pd_score_balance";
    ngx_http_pd_score_shm_zone =
        ngx_shared_memory_add(cf, shm_name, ngx_http_pd_score_shm_size,
                              &ngx_http_upstream_pd_score_balance_module);
    if (ngx_http_pd_score_shm_zone == NULL) {
        return NGX_ERROR;
    }

    ngx_http_pd_score_shm_zone->init = ngx_http_pd_score_init_shm_zone;

    ngx_http_upstream_main_conf_t *upcf;
    ngx_http_upstream_srv_conf_t **uscfp;
    ngx_http_pd_score_srv_conf_t *conf;
    ngx_uint_t i;
    upcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_upstream_module);
    if (upcf == NULL) {
        return NGX_OK;
    }
    uscfp = upcf->upstreams.elts;
    for (i = 0; i < upcf->upstreams.nelts; i++) {
        conf = ngx_http_conf_upstream_srv_conf(
            uscfp[i], ngx_http_upstream_pd_score_balance_module);
        if (conf->mode != PD_MODE_NONE) {
            uscfp[i]->peer.init = ngx_http_pd_score_upstream_init;
        }
        if (conf->mode == PD_MODE_DECODE) {
            ngx_http_pd_score_req_lim_D = conf->n_req_limit;
            ngx_log_error(NGX_LOG_WARN, cf->log, 0,
                          "[PDScoreBalance] request limit set to %ui", ngx_http_pd_score_req_lim_D);
        }
        ngx_log_error(NGX_LOG_WARN, cf->log, 0,
                      "[PDScoreBalance] upstream[%ui] mode=%d", i, conf->mode);
    }
    return NGX_OK;
}

void ngx_http_pd_score_add_decoded_tokens(ngx_http_request_t *r, ngx_uint_t num_tokens) {
    ngx_http_pd_score_peer_data_t *pdata = 
    r->upstream ? r->upstream->peer.data : NULL;
    ngx_slab_pool_t *shpool;
    if (pd_shm == NULL || pdata == NULL) {
        return;
    }
    shpool = (ngx_slab_pool_t *)ngx_http_pd_score_shm_zone->shm.addr;
    ngx_shmtx_lock(&shpool->mutex);

    ngx_atomic_fetch_add(&pd_shm->peers_D[pdata->chosen].total_decode_num,
                         (ngx_atomic_int_t)num_tokens);
    ngx_atomic_fetch_add(&pdata->decode_token_count,
                         (ngx_atomic_int_t)num_tokens);

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "[PDScore] peer=%ui request=%p decode_token +%ui, "
                  "peer_total_decode_token=%uA, request_total_decode_token=%uA",
                  pdata->chosen, r, num_tokens,
                  pd_shm->peers_D[pdata->chosen].total_decode_num,
                  pdata->decode_token_count);
    ngx_shmtx_unlock(&shpool->mutex);
}

static ngx_int_t ngx_http_pd_score_body_filter(ngx_http_request_t *r,
                                               ngx_chain_t *in) {
    ngx_http_pd_score_srv_conf_t *uscf;

    if (r->upstream == NULL) {
        return ngx_http_next_body_filter(r, in);
    }

    uscf = ngx_http_conf_upstream_srv_conf(
        r->upstream->upstream, ngx_http_upstream_pd_score_balance_module);
    if (uscf->mode == PD_MODE_PREFILL) {
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                      "prefill mode, ignore filter");
        return ngx_http_next_body_filter(r, in);
    }
    ngx_chain_t *cl;
    ngx_uint_t total_tokens = 0;

    ngx_http_pd_score_peer_data_t *pdata;
    if (r->upstream == NULL || r->upstream->peer.data == NULL) {
        return ngx_http_next_body_filter(r, in);
    }
    pdata = r->upstream->peer.data;

    if (pdata == NULL) {
        return ngx_http_next_body_filter(r, in);
    }

    for (cl = in; cl; cl = cl->next) {
        if (cl->buf->last > cl->buf->pos) {
            u_char *p = cl->buf->pos;
            u_char *last = cl->buf->last;
            static char key_total[] = "\"total_tokens\":";
            size_t keylen_total = sizeof(key_total) - 1;
            u_char *found = ngx_strnstr(p, key_total, last - p);
            if (found) {
                u_char *num_start = found + keylen_total;
                while (num_start < last &&
                       (*num_start == ' ' || *num_start == '\"')) {
                    num_start++;
                }
                ngx_uint_t val = 0;
                while (num_start < last && *num_start >= '0' &&
                       *num_start <= '9') {
                    val = val * 10 + (*num_start - '0');
                    num_start++;
                }
                total_tokens = val;
            }
        }
    }

    ngx_uint_t added_tokens = 0;

    if (total_tokens > pdata->last_total_tokens) {
        added_tokens = total_tokens - pdata->last_total_tokens;
    }

    pdata->last_total_tokens = total_tokens;

    if (added_tokens > 0) {
        ngx_http_pd_score_add_decoded_tokens(r, added_tokens);
    }
    return ngx_http_next_body_filter(r, in);
}

static int cmp_desc(const void *a, const void *b) {
    ngx_uint_t va = *(const ngx_uint_t *)a;
    ngx_uint_t vb = *(const ngx_uint_t *)b;
    return (vb > va) - (vb < va);
}

static ngx_int_t
ngx_http_pd_score_prefill_strategy(ngx_http_request_t *r,
                                   ngx_http_upstream_srv_conf_t *uscf) {
    ngx_http_upstream_t *u = r->upstream;
    ngx_http_upstream_rr_peer_data_t *rrp;
    ngx_http_pd_score_peer_data_t *pdata;
    ngx_uint_t chosen = 0, i, n;
    double min_load, peer_load, my_time_cost;
    ngx_slab_pool_t *shpool;
    ngx_time_t *tp = ngx_timeofday();
    ngx_uint_t now = tp->sec * 1000 + tp->msec;
    if (ngx_http_upstream_init_round_robin_peer(r, uscf) != NGX_OK) {
        return NGX_ERROR;
    }
    rrp = u->peer.data;

    if (pd_shm == NULL) {
        pd_shm = ngx_http_pd_score_shm_zone->data;
    }
    shpool = (ngx_slab_pool_t *)ngx_http_pd_score_shm_zone->shm.addr;
    n = rrp->peers->number;
    if (n > MAX_P) {
        n = MAX_P;
    }

    my_time_cost = r->request_length;
    ngx_shmtx_lock(&shpool->mutex);

    min_load = DBL_MAX;
    chosen = 0;

    for (i = 0; i < n; i++) {
        peer_load = (double)pd_shm->peers_P[i].total_time_cost + my_time_cost;
        if (peer_load < min_load) {
            min_load = peer_load;
            chosen = i;
        }
    }
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "[PDScore-PREFILL] request(len=%ui) assigned to peer #%ui",
                  (ngx_uint_t)r->request_length, chosen);

    pdata = ngx_pcalloc(r->pool, sizeof(*pdata));
    pdata->rrp = rrp;
    pdata->chosen = chosen;
    pdata->my_time_cost = my_time_cost;
    pdata->decode_token_count = 0;
    pdata->first_chunk = 0;
    pdata->request_length = (ngx_uint_t)r->request_length;
    pdata->last_total_tokens = 0;

    ngx_http_pd_score_shm_peer_P_t *peer = &pd_shm->peers_P[chosen];

    if (pd_shm->total_active_request_count < MAX_TOTAL_ACTIVE_REQS) {
        ngx_uint_t i = pd_shm->total_active_request_count++;
        // find suitable insert position, keep sorted
        while (i > 0 && pd_shm->running_requests_P[i - 1].inque_time > now) {
            pd_shm->running_requests_P[i] = pd_shm->running_requests_P[i - 1];
            i--;
        }
        pd_shm->running_requests_P[i].id_ptr = pdata;
        pd_shm->running_requests_P[i].inque_time = now;
        pd_shm->running_requests_P[i].request_length = (ngx_uint_t)r->request_length;
        
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                      "[PDScore-PREFILL] request %p add to queue", pdata);
    }

    ngx_atomic_fetch_add(&peer->active_requests, 1);
    ngx_atomic_fetch_add(&peer->total_time_cost,
                         (ngx_atomic_int_t)my_time_cost);
    ngx_shmtx_unlock(&shpool->mutex);

    u->peer.data = pdata;
    u->peer.get = ngx_http_pd_score_get_peer;
    u->peer.free = ngx_http_pd_score_free_peer_P;
    return NGX_OK;
}

static ngx_int_t
ngx_http_pd_score_decode_strategy(ngx_http_request_t *r,
                                  ngx_http_upstream_srv_conf_t *uscf) {
    ngx_http_upstream_t *u = r->upstream;
    ngx_http_upstream_rr_peer_data_t *rrp;
    ngx_http_pd_score_peer_data_t *pdata;
    ngx_uint_t chosen = 0, i, n;
    ngx_slab_pool_t *shpool;

    if (ngx_http_upstream_init_round_robin_peer(r, uscf) != NGX_OK) {
        return NGX_ERROR;
    }
    rrp = u->peer.data;

    if (pd_shm == NULL) {
        pd_shm = ngx_http_pd_score_shm_zone->data;
    }

    shpool = (ngx_slab_pool_t *)ngx_http_pd_score_shm_zone->shm.addr;
    n = rrp->peers->number;
    if (n > MAX_D) {
        n = MAX_D;
    }

    ngx_shmtx_lock(&shpool->mutex);

    ngx_uint_t filtered_req_lengths[MAX_PREDICT_REQS + 1];

    ngx_uint_t filtered_count = pd_shm->total_active_request_count < MAX_PREDICT_REQS ?
                                pd_shm->total_active_request_count : MAX_PREDICT_REQS;
    ngx_uint_t j = 0;
    for (j = 0; j < filtered_count; j++) {
        filtered_req_lengths[j] = pd_shm->running_requests_P[j].request_length;
    }
    filtered_req_lengths[filtered_count] = (ngx_uint_t)r->request_length;
    qsort(filtered_req_lengths, filtered_count + 1, sizeof(ngx_uint_t), cmp_desc);
    for (j = 0; j < filtered_count + 1; j++) {
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                      "sorted filtered_req_lengths[%ui]: %ui", j,
                      filtered_req_lengths[j]);
    }

    double peer_loads[MAX_PEER_COUNT];
    for (i = 0; i < n; i++) {
        peer_loads[i] = pd_shm->peers_D[i].total_decode_num * 4;
    }

    chosen = 0;

    for (i = 0; i < filtered_count + 1; i++) {
        double min_load = DBL_MAX;
        double min_load_bak = DBL_MAX;
        ngx_uint_t min_peer = NGX_CONF_UNSET_UINT;
        ngx_uint_t min_peer_bak = NGX_CONF_UNSET_UINT;
        for (ngx_uint_t j = 0; j < n; j++) {
            if (peer_loads[j] < min_load && pd_shm->peers_D[j].active_requests <= ngx_http_pd_score_req_lim_D) {
                min_load = peer_loads[j];
                min_peer = j;
            }
            if (peer_loads[j] < min_load_bak) {
                min_load_bak = peer_loads[j];
                min_peer_bak = j;
            }
        }
        if (min_peer != NGX_CONF_UNSET_UINT) {
            peer_loads[min_peer] += filtered_req_lengths[i];
        } else {
            min_peer = min_peer_bak;
            peer_loads[min_peer_bak] += filtered_req_lengths[i];
        }

        ngx_log_error(
            NGX_LOG_INFO, r->connection->log, 0,
            "[PDScore-DECODE-LPT] predict simul req.%ui dispatched to %ui", i,
            min_peer);
        if (filtered_req_lengths[i] == (ngx_uint_t)r->request_length) {
            chosen = min_peer;
            break;
        }
    }

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "[PDScore-DECODE-LPT] request(len=%ui) assigned to peer #%ui",
                  (ngx_uint_t)r->request_length, chosen);

    ngx_http_pd_score_shm_peer_D_t *peer_D = &pd_shm->peers_D[chosen];

    ngx_atomic_fetch_add(&peer_D->active_requests, 1);
    ngx_atomic_fetch_add(&peer_D->total_request_length,
                         (ngx_atomic_int_t)r->request_length);

    ngx_shmtx_unlock(&shpool->mutex);

    pdata = ngx_pcalloc(r->pool, sizeof(*pdata));
    pdata->rrp = rrp;

    pdata->chosen = chosen;

    pdata->my_time_cost = 0;
    pdata->decode_token_count = 0;
    pdata->first_chunk = 0;
    pdata->request_length = (ngx_uint_t)r->request_length;
    pdata->last_total_tokens = 0;
    u->peer.data = pdata;
    u->peer.get = ngx_http_pd_score_get_peer;
    u->peer.free = ngx_http_pd_score_free_peer_D;
    return NGX_OK;
}

static ngx_int_t ngx_http_pd_score_get_peer(ngx_peer_connection_t *pc,
                                            void *data) {
    ngx_http_pd_score_peer_data_t *pdata = data;
    ngx_http_upstream_rr_peer_data_t *rrp = pdata->rrp;
    ngx_http_upstream_rr_peers_t *peers = rrp->peers;
    ngx_uint_t idx = pdata->chosen;

    if (idx >= peers->number) {
        return ngx_http_upstream_get_round_robin_peer(pc, rrp);
    }

    if (peers->peer[idx].down) {
        return NGX_BUSY;
    }

    pc->sockaddr = peers->peer[idx].sockaddr;
    pc->socklen = peers->peer[idx].socklen;
    pc->name = &peers->peer[idx].name;
    rrp->current = &peers->peer[idx];

    return NGX_OK;
}

static ngx_int_t ngx_http_pd_score_upstream_init(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *uscf) {
    ngx_http_pd_score_srv_conf_t *conf = ngx_http_conf_upstream_srv_conf(
        uscf, ngx_http_upstream_pd_score_balance_module);
    switch (conf->mode) {
    case PD_MODE_PREFILL:
        return ngx_http_pd_score_prefill_strategy(r, uscf);
    case PD_MODE_DECODE:
        return ngx_http_pd_score_decode_strategy(r, uscf);
    default:
        return NGX_ERROR;
    }
}

static void ngx_http_pd_score_free_peer_P(ngx_peer_connection_t *pc, void *data,
                                          ngx_uint_t state) {
    ngx_log_error(NGX_LOG_INFO, pc->log, 0, "Freeing peer P req.%p", pc->data);
    ngx_http_pd_score_peer_data_t *pdata = data;
    ngx_slab_pool_t *shpool;
    if (pd_shm == NULL) {
        return;
    }
    shpool = (ngx_slab_pool_t *)ngx_http_pd_score_shm_zone->shm.addr;
    ngx_shmtx_lock(&shpool->mutex);
    ngx_http_pd_score_shm_peer_P_t *peer = &pd_shm->peers_P[pdata->chosen];
    ngx_log_error(NGX_LOG_INFO, pc->log, 0, "total active request count: %ui",
                  pd_shm->total_active_request_count);

    for (ngx_uint_t i = 0; i < pd_shm->total_active_request_count; i++) {
        if (pd_shm->running_requests_P[i].id_ptr == (void *)pc->data) {
            ngx_log_error(NGX_LOG_INFO, pc->log, 0,
                          "free P request %p, idx: %ui", pc->data, i);
            // keep sorted
            for (ngx_uint_t j = i; j < pd_shm->total_active_request_count - 1; j++) {
                pd_shm->running_requests_P[j] = pd_shm->running_requests_P[j + 1];
            }
            pd_shm->total_active_request_count--;
            break;
        }
    }
    ngx_atomic_fetch_add(&peer->active_requests, (ngx_atomic_int_t)-1);
    ngx_atomic_fetch_add(&peer->total_request_length,
                         (ngx_atomic_int_t) - (pdata->request_length));
    ngx_atomic_fetch_add(&peer->total_time_cost,
                         (ngx_atomic_int_t) - (pdata->my_time_cost));
    ngx_atomic_fetch_add(&peer->total_decode_num,
                         (ngx_atomic_int_t) - (pdata->decode_token_count));

    ngx_shmtx_unlock(&shpool->mutex);

    ngx_http_upstream_rr_peer_data_t *rrp = pdata->rrp;
    ngx_http_upstream_free_round_robin_peer(pc, rrp, state);
}

static void ngx_http_pd_score_free_peer_D(ngx_peer_connection_t *pc, void *data,
                                          ngx_uint_t state) {
    ngx_log_error(NGX_LOG_INFO, pc->log, 0, "Freeing peer D");
    ngx_http_pd_score_peer_data_t *pdata = data;
    ngx_slab_pool_t *shpool;
    if (pd_shm == NULL) {
        return;
    }
    shpool = (ngx_slab_pool_t *)ngx_http_pd_score_shm_zone->shm.addr;
    ngx_shmtx_lock(&shpool->mutex);

    ngx_http_pd_score_shm_peer_D_t *peer = &pd_shm->peers_D[pdata->chosen];

    ngx_atomic_fetch_add(&peer->active_requests, (ngx_atomic_int_t)-1);
    ngx_atomic_fetch_add(&peer->total_request_length,
                         (ngx_atomic_int_t) - (pdata->request_length));
    ngx_atomic_fetch_add(&peer->total_time_cost,
                         (ngx_atomic_int_t) - (pdata->my_time_cost));
    ngx_atomic_fetch_add(&peer->total_decode_num,
                         (ngx_atomic_int_t) - (pdata->decode_token_count));

    ngx_shmtx_unlock(&shpool->mutex);

    ngx_http_upstream_rr_peer_data_t *rrp = pdata->rrp;
    ngx_http_upstream_free_round_robin_peer(pc, rrp, state);
}