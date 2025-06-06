#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

version=$(date +%Y%m%d)
source_cmd="source /usr/local/Ascend/CANN-7.7/bin/setenv.bash && source /usr/local/Ascend/nnal/atb/set_env.sh"
# 查看当前脚本所在的绝对路径
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_PATH

is_image_exists=`docker images | grep omni_infer_v1 | grep ${version}`
if [[ $is_image_exists == '' ]]; then
  echo "镜像不存在，开始构建镜像"
  docker build -f Dockerfile -t omni_infer_v1:${version} .
else
  echo "镜像已存在，跳过构建直接启动"
fi

is_image_exists=`docker images | grep omni_infer_v1 | grep ${version}`
if [[ $is_image_exists == '' ]]; then
  echo "镜像构建失败"
  exit 1
else
  echo "镜像构建成功"
fi

is_container_exists=`docker ps |grep omni_infer_v1_${version} | grep omni_infer_v1:${version}`
if [[ $is_container_exists == '' ]]; then
  echo "容器不存在，正常拉起容器"
  container_id=`bash start_docker.sh omni_infer_v1:${version} omni_infer_v1_${version}`
  docker ps
else
  echo "容器已存在，正常拉起脚本"
  container_id=`docker ps |grep omni_infer_v1_${version} | grep omni_infer_v1:${version} | awk '{print $1}'`
fi

# 容器中代码目录在/home/ma-user/omni_infer

echo "启动服务"
docker exec -i ${container_id} bash -c "${source_cmd} && cd /home/ma-user/omni_infer/tests && OUTPUT_DIR=${version} bash start_tests.sh"

echo "关闭容器"
docker stop ${container_id}