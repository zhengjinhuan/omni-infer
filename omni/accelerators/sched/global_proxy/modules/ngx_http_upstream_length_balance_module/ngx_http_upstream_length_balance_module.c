#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <ngx_atomic.h>

extern ngx_module_t ngx_http_upstream_length_balance_module;

#define NGX_DEFAULT_MERGE_THRESHOLD 64
#define NGX_DEFAULT_MAX_UPSTREAM_SERVERS 512

#define SHM_NAME "length_balance_shm"
#define SHM_SIZE (128 * 1024) // 128K, fixed size for shared memory

typedef struct {
    ngx_atomic_t total_length;
    ngx_atomic_t total_requests;
} ngx_http_length_stats_t;

typedef struct ngx_http_length_peer_s {
    struct sockaddr *sockaddr;
    socklen_t socklen;
    ngx_str_t name;
    ngx_uint_t weight;
    ngx_uint_t down;
    struct ngx_http_length_peer_s *next;
    ngx_uint_t stat_offset;
} ngx_http_length_peer_t;

typedef struct {
    ngx_uint_t number;
    ngx_http_length_peer_t *peer;
    ngx_str_t name;
    ngx_atomic_t local_requests[NGX_DEFAULT_MAX_UPSTREAM_SERVERS];
    ngx_atomic_t local_lengths[NGX_DEFAULT_MAX_UPSTREAM_SERVERS];
    ngx_atomic_t request_counter;
    ngx_uint_t shm_base_idx;
} ngx_http_length_peers_t;

typedef struct {
    ngx_http_length_peers_t *peers;
    ngx_http_length_peer_t *current;
    ngx_http_length_stats_t *stats;
    ngx_http_request_t *request;
    ngx_log_t *log;
} ngx_http_length_peer_data_t;

static ngx_shm_zone_t *length_shm_zone = NULL;
static ngx_http_length_stats_t *global_stats = NULL;
static ngx_str_t shm_name = ngx_string(SHM_NAME);
static ngx_atomic_t next_shm_idx = 0;

// Shared memory initialization callback
static ngx_int_t
ngx_http_length_balance_init_zone(ngx_shm_zone_t *shm_zone, void *data)
{
    if (data) {
        shm_zone->data = data;
        global_stats = (ngx_http_length_stats_t *)data;
        return NGX_OK;
    }
    global_stats = (ngx_http_length_stats_t *)shm_zone->shm.addr;
    ngx_memzero(global_stats, SHM_SIZE);
    shm_zone->data = global_stats;
    return NGX_OK;
}

static char *ngx_http_upstream_length_balance(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);
static ngx_int_t ngx_http_upstream_init_length_balance(ngx_conf_t *cf, ngx_http_upstream_srv_conf_t *us);
static ngx_int_t ngx_http_upstream_init_length_balance_peer(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *us);
static ngx_int_t ngx_http_upstream_get_length_balance_peer(ngx_peer_connection_t *pc, void *data);

// Configuration directives
static ngx_command_t ngx_http_upstream_length_balance_commands[] = {
    {ngx_string("length_balance"),
     NGX_HTTP_UPS_CONF | NGX_CONF_NOARGS,
     ngx_http_upstream_length_balance,
     0,
     0,
     NULL},
    ngx_null_command
};

static ngx_http_module_t ngx_http_upstream_length_balance_module_ctx = {
    NULL, /* preconfiguration */
    NULL, /* postconfiguration */
    NULL, /* create main configuration */
    NULL, /* init main configuration */
    NULL, /* create server configuration */
    NULL, /* merge server configuration */
    NULL, /* create location configuration */
    NULL  /* merge location configuration */
};

ngx_module_t ngx_http_upstream_length_balance_module = {
    NGX_MODULE_V1,
    &ngx_http_upstream_length_balance_module_ctx, /* module context */
    ngx_http_upstream_length_balance_commands,    /* module directives */
    NGX_HTTP_MODULE,                             /* module type */
    NULL,                                        /* init master */
    NULL,                                        /* init module */
    NULL,                                        /* init process */
    NULL,                                        /* exit process */
    NULL,                                        /* exit master */
    NULL,                                        /* reserved 1 */
    NULL,                                        /* reserved 2 */
    NGX_MODULE_V1_PADDING
};

static char *
ngx_http_upstream_length_balance(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    if (length_shm_zone == NULL) {
        length_shm_zone = ngx_shared_memory_add(cf, &shm_name, SHM_SIZE, &ngx_http_upstream_length_balance_module);
        if (length_shm_zone == NULL) {
            return NGX_CONF_ERROR;
        }
        length_shm_zone->init = ngx_http_length_balance_init_zone;
    }

    ngx_http_upstream_srv_conf_t *uscf;
    uscf = ngx_http_conf_get_module_srv_conf(cf, ngx_http_upstream_module);
    uscf->peer.init_upstream = ngx_http_upstream_init_length_balance;
    uscf->peer.init = ngx_http_upstream_init_length_balance_peer;
    return NGX_CONF_OK;
}

static ngx_int_t
ngx_http_upstream_init_length_balance(ngx_conf_t *cf, ngx_http_upstream_srv_conf_t *us)
{
    ngx_uint_t i, n;
    ngx_http_upstream_server_t *server;
    ngx_http_length_peer_t *peer, **peerp;
    ngx_http_length_peers_t *peers;

    if (us->servers == NULL) {
        ngx_log_error(NGX_LOG_EMERG, cf->log, 0, "No upstream servers defined");
        return NGX_ERROR;
    }
    if (!length_shm_zone) {
        ngx_log_error(NGX_LOG_EMERG, cf->log, 0, "length_balance shared memory not initialized");
        return NGX_ERROR;
    }

    server = us->servers->elts;
    n = us->servers->nelts;

    if (n > NGX_DEFAULT_MAX_UPSTREAM_SERVERS) {
        n = NGX_DEFAULT_MAX_UPSTREAM_SERVERS;
    }
    if (n == 0) {
        ngx_log_error(NGX_LOG_EMERG, cf->log, 0, "No valid upstream servers after applying max_upstream_servers limit");
        return NGX_ERROR;
    }

    peers = ngx_pcalloc(cf->pool, sizeof(ngx_http_length_peers_t));
    if (peers == NULL) {
        return NGX_ERROR;
    }
    peer = ngx_pcalloc(cf->pool, sizeof(ngx_http_length_peer_t) * n);
    if (peer == NULL) {
        ngx_pfree(cf->pool, peers);
        return NGX_ERROR;
    }
    peers->number = n;
    peers->name = us->host;
    peerp = &peers->peer;

    if (ngx_atomic_fetch_add(&next_shm_idx, 0) + n > SHM_SIZE / sizeof(ngx_http_length_stats_t)) {
        ngx_log_error(NGX_LOG_EMERG, cf->log, 0,
                      "length_balance shared memory is too small for %ui servers of upstream \"%V\". Total servers %ui, available %D",
                      n, &us->host, ngx_atomic_fetch_add(&next_shm_idx, 0) + n, SHM_SIZE / sizeof(ngx_http_length_stats_t));
        return NGX_ERROR;
    }
    peers->shm_base_idx = ngx_atomic_fetch_add(&next_shm_idx, n);

    for (i = 0; i < n; i++) {
        peer[i].sockaddr = server[i].addrs[0].sockaddr;
        peer[i].socklen = server[i].addrs[0].socklen;
        peer[i].name = server[i].addrs[0].name;
        peer[i].weight = server[i].weight;
        peer[i].down = server[i].down;
        peer[i].next = NULL;
        peer[i].stat_offset = i;
        *peerp = &peer[i];
        peerp = &peer[i].next;
        peers->local_requests[i] = 0;
        peers->local_lengths[i] = 0;
    }
    peers->request_counter = 0;

    us->peer.data = peers;
    us->peer.init = ngx_http_upstream_init_length_balance_peer;

    return NGX_OK;
}

static ngx_int_t
ngx_http_upstream_init_length_balance_peer(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *us)
{
    ngx_http_length_peer_data_t *lp;

    lp = ngx_palloc(r->pool, sizeof(ngx_http_length_peer_data_t));
    if (lp == NULL) {
        return NGX_ERROR;
    }
    lp->peers = us->peer.data;
    lp->current = NULL;
    lp->stats = global_stats;
    if (lp->stats == NULL) {
        ngx_log_error(NGX_LOG_EMERG, r->connection->log, 0, "length_balance global_stats not initialized");
        return NGX_ERROR;
    }
    lp->request = r;
    lp->log = r->connection ? r->connection->log : ngx_cycle->log;

    r->upstream->peer.get = ngx_http_upstream_get_length_balance_peer;
    r->upstream->peer.data = lp;
    r->upstream->peer.tries = lp->peers->number;

    return NGX_OK;
}

static ngx_int_t
ngx_http_upstream_get_length_balance_peer(ngx_peer_connection_t *pc, void *data)
{
    ngx_http_length_peer_data_t *lp = data;
    ngx_http_length_peer_t *peer, *min_peers[NGX_DEFAULT_MAX_UPSTREAM_SERVERS];
    ngx_uint_t i = 0, count = 0;
    ngx_atomic_t min_score = (ngx_atomic_t)-1;
    ngx_http_request_t *r = lp->request;
    ngx_atomic_t req_len = (r && r->headers_in.content_length_n > 0) ? r->headers_in.content_length_n : 0;

    for (peer = lp->peers->peer; peer; peer = peer->next, i++) {
        if (peer->down) {
            continue;
        }

        ngx_uint_t global_idx = lp->peers->shm_base_idx + peer->stat_offset;

        if (global_idx >= SHM_SIZE / sizeof(ngx_http_length_stats_t)) {
            ngx_log_error(NGX_LOG_EMERG, lp->log, 0,
                          "length_balance: global_idx %ui out of bounds (max %D) for peer \"%V\", upstream \"%V\"",
                          global_idx, SHM_SIZE / sizeof(ngx_http_length_stats_t), &peer->name, &lp->peers->name);
            continue;
        }

        ngx_http_length_stats_t *stat = &lp->stats[global_idx];

        ngx_atomic_t current_requests = stat->total_requests + ngx_atomic_fetch_add(&lp->peers->local_requests[i], 0);
        ngx_atomic_t current_length = stat->total_length + ngx_atomic_fetch_add(&lp->peers->local_lengths[i], 0);

        ngx_atomic_t score = current_requests + current_length + req_len;

        if (count == 0 || score < min_score) {
            min_score = score;
            count = 0;
            min_peers[count++] = peer;
        } else if (score == min_score) {
            if (count < NGX_DEFAULT_MAX_UPSTREAM_SERVERS) {
                min_peers[count++] = peer;
            }
        }
    }

    if (count == 0) {
        pc->name = &lp->peers->name;
        ngx_log_error(NGX_LOG_ERR, lp->log, 0,
                      "length_balance: no active peers found for upstream \"%V\"", &lp->peers->name);
        return NGX_BUSY;
    }
    ngx_uint_t rand_idx = ngx_random() % count;
    peer = min_peers[rand_idx];

    pc->sockaddr = peer->sockaddr;
    pc->socklen = peer->socklen;
    pc->name = &peer->name;

    lp->current = peer;
    ngx_uint_t p_local_idx = peer->stat_offset;

    ngx_atomic_fetch_add(&lp->peers->local_requests[p_local_idx], 1);
    ngx_atomic_fetch_add(&lp->peers->local_lengths[p_local_idx], req_len);

    ngx_atomic_t current_req_counter = ngx_atomic_fetch_add(&lp->peers->request_counter, 1) + 1;

    if (current_req_counter >= NGX_DEFAULT_MERGE_THRESHOLD) {
        for (i = 0; i < lp->peers->number; i++) {
            ngx_atomic_t req = ngx_atomic_fetch_add(&lp->peers->local_requests[i], 0);
            ngx_atomic_t len = ngx_atomic_fetch_add(&lp->peers->local_lengths[i], 0);
    
            if (req > 0 || len > 0) {
                ngx_atomic_fetch_add(&lp->peers->local_requests[i], -((ngx_atomic_int_t)req));
                ngx_atomic_fetch_add(&lp->peers->local_lengths[i], -((ngx_atomic_int_t)len));
    
                ngx_uint_t global_idx = lp->peers->shm_base_idx + i;
    
                if (global_idx < SHM_SIZE / sizeof(ngx_http_length_stats_t)) {
                    ngx_atomic_t old_total_req = ngx_atomic_fetch_add(&lp->stats[global_idx].total_requests, 0);
                    ngx_atomic_t old_total_len = ngx_atomic_fetch_add(&lp->stats[global_idx].total_length, 0);
    
                    ngx_atomic_t new_total_req = ((old_total_req >> 1) + req);
                    ngx_atomic_t new_total_len = ((old_total_len >> 1) + len);
    
                    ngx_atomic_cmp_set(&lp->stats[global_idx].total_requests, old_total_req, new_total_req);
                    ngx_atomic_cmp_set(&lp->stats[global_idx].total_length, old_total_len, new_total_len);
                    
                } else {
                    ngx_log_error(NGX_LOG_EMERG, lp->log, 0,
                                  "length_balance: merge global_idx %ui out of bounds (max %D) for local_idx %ui, upstream \"%V\"",
                                  global_idx, SHM_SIZE / sizeof(ngx_http_length_stats_t), i, &lp->peers->name);
                }
            }
        }
        ngx_atomic_fetch_add(&lp->peers->request_counter, -((ngx_atomic_int_t)current_req_counter));
    }

    return NGX_OK;
}