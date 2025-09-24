#!/bin/bash

set -e

cd ../

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
echo "$WORKDIR"

NGINX_VERSION="${NGINX_VERSION:-1.28.0}"
echo "NGINX_VERSION is $NGINX_VERSION"

MSGPACK_VERSION="${MSGPACK_VERSION:-6.1.0}"
echo "MSGPACK_VERSION is $MSGPACK_VERSION"

PYTHON_VERSION="${PYTHON_VERSION:-3.11.9}"
echo "PYTHON_VERSION is $PYTHON_VERSION"

NGINX_SBIN_PATH="${NGINX_SBIN_PATH:-/usr/local/nginx}"
echo "NGINX_SBIN_PATH is $NGINX_SBIN_PATH"


yum install -y pcre gcc gcc-c++ make zlib zlib-devel pcre pcre-devel openssl-devel zeromq zeromq-devel boost-devel

if [ ! -d nginx-${NGINX_VERSION} ]; then
	wget --no-check-certificate https://nginx.org/download/nginx-${NGINX_VERSION}.tar.gz
	tar -zxf nginx-${NGINX_VERSION}.tar.gz
fi

if [ ! -d msgpack-c-${MSGPACK_VERSION} ]; then
    wget --no-check-certificate https://github.com/msgpack/msgpack-c/releases/download/c-${MSGPACK_VERSION}/msgpack-c-${MSGPACK_VERSION}.tar.gz
	tar -zxf msgpack-c-${MSGPACK_VERSION}.tar.gz
    cd msgpack-c-${MSGPACK_VERSION}
    mkdir build && cd build
    cmake .. \
        -DMSGPACK_BUILD_EXAMPLES=OFF \
        -DMSGPACK_BUILD_TESTS=OFF \
        -DMSGPACK_USE_BOOST=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr
    make -j$(nproc)
    make install
    cd ../../
fi

if [ ! -d Python-${PYTHON_VERSION} ]; then
    wget --no-check-certificate https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
    tar -zxf Python-${PYTHON_VERSION}.tgz
    cd Python-${PYTHON_VERSION}

    ./configure \
        --enable-optimizations \
        --enable-shared \
        --prefix=/usr/local

    make -j$(nproc)
    make install

    echo "/usr/local/lib" > /etc/ld.so.conf.d/python3.11.conf
    ldconfig
    
    cd ..
fi

cd nginx-${NGINX_VERSION}
CFLAGS="-O0 -g" ./configure --sbin-path=${NGINX_SBIN_PATH} \
    --add-dynamic-module=$WORKDIR/omni_proxy/modules \
    --add-dynamic-module=$WORKDIR/global_proxy/modules/ngx_http_set_request_id_module \
    --with-http_v2_module \
    --with-stream
make -j16
make install

ln -s "$WORKDIR"/omni_proxy/modules/omni_tokenizer.py .
export PYTHONHASHSEED=123