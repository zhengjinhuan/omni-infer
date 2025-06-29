# 环境准备

硬件：目前仅支持CloudMatrix384推理卡
操作系统：Linux
Python: >=3.9, <= 3.11

# 安装

目前仅支持基于docker 镜像进行安装，docker镜像中安装好了依赖的CANN和Torch-NPU包，并且已经预装了一个可直接运行的omni-infer和vllm包。

## 新环境检查

### Ascend NPU固件和驱动检查

使用如下命令检查Ascend NPU固件和驱动是否正确安装
npu-smi info
正常显示如下图所示

![image](./figures/432644be-20cf-4163-95f4-72bfde0eff90.png)

![image](./figures/d00adf38-6e60-42ad-94be-a74964c47694.png)

### 网络连通性检查

若要PD分离部署，确保部署PD分离部署的CloudMatrix384机器网络是联通的，可以在其中一台机器上使用ssh命令进行尝试，命令示例

```
ssh root@192.168.1.100
```

## docker镜像下载

```
docker pull swr.cn-southwest-2.myhuaweicloud.com/omni-ai/omniinfer:202506272026
```

## docker镜像拉起命令

镜像拉起脚本，可以参考如下脚本，第一个参数是镜像image_id，第二个参数是待启动的容器名称

```
IMAGES_ID=$1
NAME=$2
if [ $# -ne 2 ]; then
    echo "error: need one argument describing your container name."
    exit 1
fi
docker run --name ${NAME} -it -d  --shm-size=500g \
    --net=host \
    --privileged=true \
    -u root \
    -w /home \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --entrypoint=bash \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
    -v /home/:/home/\
    -v /data:/data \
    -v /tmp:/tmp \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    ${IMAGES_ID}

```

## omni_infer包检查

进入镜像后，使用如下命令检查omni_infer包是否安装

```
pip list | grep omni_infer
```

![image](./figures/7bf10117-2c1a-4ec7-a7b6-2ce29c37fda1.png)

## 混布

目前支持Qwen2.5系列模型TP>=1, DP=1
以Qwen2.5-0.5B-Instruct为例，示例中的--model填充实际的模型存放目录

```
#!/bin/bash
set -e

export GLOO_SOCKET_IFNAME=enp23s0f3
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_ENABLE_MC2=0
export USING_LCCL_COM=0
export VLLM_LOGGING_LEVEL=DEBUG
export ASCEND_RT_VISIBLE_DEVICES=0

python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8300 \
    --model /data/Qwen2.5-0.5B-Instruct \
    --data-parallel-size 1 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --trust_remote_code \
    --gpu_memory_utilization 0.9 \
    --enforce-eager \
    --block_size 128 \
    --served-model-name qwen \
    --distributed-executor-backend mp \
    --max-num-batched-tokens 20000 \
    --max-num-seqs 128
```

拉起成功后，可以通过curl命令进行测试

```
curl -X POST http://127.0.0.1:8300/v1/completions -H "Content-Type:application/json" -d '{"model": "qwen","temperature":0,"max_tokens":50,"prompt": "how are you?", "stream":true,"stream_options": {"include_usage": true,"continuous_usage_stats": true}}'
```

## PD分离自动化部署

当前限制说明：

目前仅支持Deepseek-R1-W8A8模型，权重下载地址（TODO）
目前仅支持支持一个 D 的场景，如支持 4P1D、8P1D 等场景，不支持多个 D，如 4P2D、8P4D 等场景不支持；
目前不支持 P 一主一从的配置，仅支持每个 P 只有一主

### 部署框架介绍

以**4机2P1D**进行示例

![image](./figures/ab1a606f-20cd-417f-a0d6-fec4b3d26d27.png)

ansible 详细说明参考：**tools**/**ansible**/**README.md**

### omni_cli一键部署

该工具目前仅支持拉起**MTP+入图**的服务配置，若要修改请参考**通过ansible部署**章节；
提供的docker镜像中默认安装omni_cli工具，进入容器后，通过以下命令查看是否携带。

```
omni_cli --help
```

![image](./figures/53e799d4-b756-46d0-8cc8-8ff2c99c1dd8.png)

#### 配置文件说明

`cd /your_path/omni_infer/omni/cli` 进入配置文件 `omni_infer_deployment.yml` 所在目录；
配置文件是一个 4P1D 的模板，如果需要增加 P，在 `prefill` 中增加一个 `group5` ，其他配置按实际情况配置即可，需要继续增加 P，以此类推；`group` 中的各个字段说明如下：

| 字段                            | 含义                                                                                                                               |
| :-------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| `user`                      | 远程目标机的用户名， 如 user 等                                                                                                    |
| `hosts`                     | 目标机 IP 地址                                                                                                                     |
| `master_port`               | 即 Prefill 和 Decode 的实际 `master-port`                                                                                      |
| `base_api_port`             | API Server 的端口号                                                                                                                |
| `private_key`               | 连接目标机的私钥文件路径。也可以使用密码登录目标机的方式， 则使用 `password` 字段并将密码填入， 如：password: "passwod"        |
| `ascend_rt_visible_devices` | 每个 Prefill 或 Decode 实例需要使用的卡号， 参数值需要严格按照以下格式: "x,x,x,x" (用英文逗号分隔的连续值) ， 不能有多余逗号和空格 |

如果需要增加 D 的从机器，在 `decode` 的 `group1` 中的 `hosts` 增加对应的机器 ip 即可。

`services` 中的各个字段说明如下：

| 字段                         | 含义                                                                                         |
| :----------------------------- | :--------------------------------------------------------------------------------------------- |
| `model_path`             | 加载的模型路径， 要求 Prefill 和 Decode 所有实例所在的节点提前拷贝好模型并且模型路径保持一致 |
| `prefill: max_model_len` | Prefill 侧模型的最大生成长度， 包含 prompt 长度和 generated 长度， 默认值为30000             |
| `decode: max_model_len`  | Decode 侧模型的最大生成长度， 包含 prompt 长度和 generated 长度， 默认值为16384              |

注意到 D 中所有机器的配置都是统一的，但是多数情况下机器的配置是各异的，比如登录目标机的私钥文件路径可能就不一样，所以，如果需要修改 D 的某一台机器的配置，可以参照如下，**注意缩进**即可：

```
group1:
  user: "root"
  hosts: "127.0.0.5,127.0.0.6,127.0.0.7,127.0.0.8" # The first IP address of hosts must be the IP address of the master decode instance. 
  master_port: "8080"
  base_api_port: "9000"
  private_key: "/workspace/pem/keypair.pem"
  ascend_rt_visible_devices: "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
  
  # 修改 IP 为 127.0.0.6 的机器的配置
  host 127.0.0.6:
    private_key: "/workspace/new_pem/keypair.pem"
    base_api_port: "9100"

```

#### 命令执行

```
# 进入到你的工作目录下
cd /your_workspace

# 拉起 服务
omni_cli serve /your_path/omni_infer/omni/cli/omni_infer_deployment.yml

# 查看服务是否拉起成功
omni_cli status

```

### 通过ansible部署（高阶）

#### 环境准备

**在执行机安装 ansible和openssh:**

```
yum install ansible
yum install openssh-server

```

**修改配置文件：**

在 **omni\_infer\_inventory.yml** 中， 目标机信息；
以2P1D为例，修改tools/ansible/template/omni_infer_inventory_used_for_2P1D.yml文件中的执行机IP，p0/p1/d0/d1/c0下面的`ansible_host:`值，其中p0/p1表示用来部署P的2台CloudMatrix384机器信息，d0/d1表示用来部署D的2台A3机器信息，c0表示用来部署globalproxy的机器信息，可以使用p0/p1/d0/d1中的任意一台;

![image](./figures/79f4a480-e13b-45a3-bc9e-080f27ea3995.png)

![image](./figures/4d4eb6e0-2af8-4ce2-8233-b8ccebc7c4a4.png)
在 **omni\_infer\_server.yml** 中， 修改task任务依赖的环境变量。

**修改完命令执行**

```
# 进入到文件目录下执行
cd ./omni_infer/tools/ansible
ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml

```

#### PD分离部署

相关配置：

![image](./figures/2160b5d6-6c54-4fab-8ec0-8df5ef0687b9.png)

#### global\_proxy部署

相关配置：

![image](./figures/871cdf50-4402-404a-af8b-01d660b36f5d.png)

#### 代码同步更新

执行机代码存放路径：/data/omni\_infer (可以修改成自己的代码路径，下图synchronize操作就是把执行机的代码同步到目标机上)

![image](./figures/07a40404-2aa6-4165-b43d-6e386da57777.png)

#### Task任务

![image](./figures/9c9a720a-8936-43e1-884a-97ee23d1c264.png)

**omni\_infer\_server.yml** 主要放的是自动化的一些操作任务。

当前已有的task:

```
    - run_docker
    - ranktable
    - run_server
    - run_proxy
    - sync_code
```

配置完相关配置信息后拉起服务：

**第一次拉起环境，执行命令：ansible-playbook -i omni\_infer\_inventory.yml omni\_infer\_server.yml**

其他相关操作：

```

ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml --》默认按照task全部任务顺序执行
ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml --tags run_server --》只执行pd分离服务拉起
ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml --tags run_proxy --》只执行global_proxy分离服务拉起
ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml --tags sync_code --》只执行代码同步更新任务
ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml --skip-tags sync_code --》过滤sync

```

## 源码安装omni_infer

1. 拉取vllm v0.9.0代码
2. 将vllm代码放入infer_engines目录下
   infer_engines下的目录结构如下
   
   ![image](./figures/9f66a7d7-2b5a-470c-a816-b0b9064854d2.png)
3. 卸载镜像中已有的omni_infer包
   pip uninstall vllm -y
   pip uninstall omni_infer -y
   pip uninstall omni_placement -y
4. 编译omni_infer
   cd omni
   build/build.sh
   可以看到最终在build/dist目录下有生成whl包
5. 通过whl包安装
   pip install vllm.whl
   pip install omni_infer.whl
   pip install omni_placement.whl
   安装成后，可以通过pip list查看是否有安装成功


