#!/bin/bash

set -e

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
echo "$WORKDIR"

NGINX_VERSION="${NGINX_VERSION:-1.28.0}"
echo "NGINX_VERSION is $NGINX_VERSION"

NGINX_SBIN_PATH="${NGINX_SBIN_PATH:-/usr/local/nginx}"
echo "NGINX_SBIN_PATH is $NGINX_SBIN_PATH"

if [ ! -d nginx-${NGINX_VERSION} ]; then
	wget --no-check-certificate https://mirrors.tools.huawei.com/nginx/nginx-${NGINX_VERSION}.tar.gz
	tar -zxf nginx-${NGINX_VERSION}.tar.gz
fi

yum install -y pcre gcc gcc-c++ make zlib zlib-devel pcre pcre-devel openssl-devel

cd nginx-${NGINX_VERSION}
CFLAGS="-O2" ./configure --sbin-path=${NGINX_SBIN_PATH} \
    --add-dynamic-module=$WORKDIR/modules/ngx_http_prefill_module \
    --add-dynamic-module=$WORKDIR/modules/ngx_http_set_request_id_module \
    --add-dynamic-module=$WORKDIR/lb_sdk/modules/ngx_http_upstream_length_balance_module \
    --add-dynamic-module=$WORKDIR/lb_sdk/modules/ngx_http_upstream_greedy_timeout_module \
    --add-dynamic-module=$WORKDIR/lb_sdk/modules/ngx_http_upstream_prefill_score_balance_module \
    --add-dynamic-module=$WORKDIR/lb_sdk/modules/ngx_http_upstream_weighted_least_active_module \
    --without-http_gzip_module \
    --with-ld-opt="-lm"
make -j16
make install