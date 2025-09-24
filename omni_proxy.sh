#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

set -e

nginx_conf_file="/usr/local/nginx/conf/nginx.conf"
listen_port="7150"
core_num="4"
start_core_index="0"
prefill_endpoints=""
decode_endpoints=""
log_file="/tmp/nginx_4p1d.log"
log_level="info"

print_help() {
    echo "Usage:"
    echo "  $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --nginx-conf-file <path>        Path to nginx config file (default: /usr/local/nginx/conf/nginx.conf)"
    echo "  --listen-port <PORT>            Listening port (default: 7150)"
    echo "  --core-num <N>                  Number of CPU cores to use (default: 4)"
    echo "  --start-core-index <N>          Starting CPU core index (default: 0)"
    echo "  --prefill-endpoints <list>      Comma-separated backend servers for prefill"
    echo "  --decode-endpoints <list>       Comma-separated backend servers for decode"
    echo "  --log-file <path>               Log file path (default: /tmp/nginx_4p1d.log)"
    echo "  --log-level <LEVEL>             Log level (default: info)"
    echo "  --help                          Show this help message"
    echo ""
    echo "EXAMPLE:"
    echo "  bash $0 \\"
    echo "      --nginx-conf-file /usr/local/nginx/conf/nginx.conf \\"
    echo "      --listen-port 7150 \\"
    echo "      --core-num 4 \\"
    echo "      --prefill-endpoints 7.150.8.32:9000,7.150.8.47:9001 \\"
    echo "      --decode-endpoints 7.150.10.13:9100,7.150.10.13:9101"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nginx-conf-file)
            nginx_conf_file="$2"
            shift 2
            ;;
        --listen-port)
            listen_port="$2"
            shift 2
            ;;
        --core-num)
            core_num="$2"
            shift 2
            ;;
        --start-core-index)
            start_core_index="$2"
            shift 2
            ;;
        --prefill-servers-list)
            prefill_endpoints="$2"
            shift 2
            ;;
        --decode-servers-list)
            decode_endpoints="$2"
            shift 2
            ;;
        --log-file)
            log_file="$2"
            shift 2
            ;;
        --log-level)
            log_level="$2"
            shift 2
            ;;
        --help|-h)
            print_help
            ;;
        *)
            echo "Unknown argument: $1"
            print_help
            ;;
    esac
done

if [[ -z "$prefill_endpoints" || -z "$decode_endpoints" ]]; then
    echo "Error: --prefill-endpoints and --decode-endpoints are required"
    exit 1
fi

function gen_affinity_masks() {
    local count=$1
    local masks=()
    for ((i=0; i<count; i++)); do
        local mask=""
        for ((j=0; j<count; j++)); do
            if ((j == i)); then
                mask="${mask}1"
            else
                mask="${mask}0"
            fi
        done
        while ((${#mask} < 16)); do
            mask="0${mask}"
        done
        masks+=("$mask")
    done
    echo "${masks[@]}"
}

affinity_masks=$(gen_affinity_masks "$core_num")

function gen_upstream_block() {
    local name="$1"
    local endpoints="$2"
    local block="    upstream $name {\n"
    IFS=',' read -ra list <<< "$endpoints"
    for addr in "${list[@]}"; do
        block+="        server $addr max_fails=3 fail_timeout=10s;\n"
    done
    block+="    }"
    echo -e "$block"
}

cat > "$nginx_conf_file" <<EOF
load_module /usr/local/nginx/modules/ngx_http_omni_proxy_module.so;
load_module /usr/local/nginx/modules/ngx_http_set_request_id_module.so;

env PYTHONHASHSEED;
env TORCH_DEVICE_BACKEND_AUTOLOAD;
env VLLM_PLUGINS;
env LD_LIBRARY_PATH;
user root;

worker_processes $core_num;
worker_rlimit_nofile 102400;
worker_cpu_affinity $affinity_masks;

error_log  $log_file  $log_level;

events {
    use epoll;
    accept_mutex off;
    multi_accept on;
    worker_connections 4096;
}

http {
    proxy_http_version 1.1;
    tcp_nodelay on;
    keepalive_requests 1000;
    keepalive_timeout 300;
    client_max_body_size 10M;
    client_body_buffer_size 1M;

    proxy_read_timeout 14400s;
    proxy_connect_timeout 600s;
    proxy_send_timeout 600s;

$(gen_upstream_block "prefill_endpoints" "$prefill_endpoints")
    
$(gen_upstream_block "decode_endpoints" "$decode_endpoints")

    server {
        listen $listen_port reuseport;
        server_name localhost;

        location /v1 {
            set_request_id on;
            omni_proxy decode_endpoints;
            omni_proxy_pd_policy sequential;
            chunked_transfer_encoding off;
            proxy_buffering off;
            send_timeout 1h;
            postpone_output 0;
        }

        location = /omni_proxy/metrics {
            omni_proxy_metrics on;
            default_type text/plain;
        }

        location ~ ^/prefill_sub(?<orig>/.*)\$ {
            internal;
            proxy_pass http://prefill_endpoints\$orig\$is_args\$args;
            subrequest_output_buffer_size 1M;
        }
    }
}
EOF

echo "nginx.conf generated at $nginx_conf_file"