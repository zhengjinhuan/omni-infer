# PD分离一般性部署流程

本文档介绍如何在复杂场景多步骤拉起PD分离部署推理，以QwQ 32B为例，支持2机3P1D。

## 硬件要求

**硬件：** CloudMatrix384推理卡

**操作系统：** Linux

**镜像版本：** swr.cn-east-4.myhuaweicloud.com/omni-ci/daily_omniinfer:20250722_26

[**驱动检查**](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md#ascend-npu%E5%9B%BA%E4%BB%B6%E5%92%8C%E9%A9%B1%E5%8A%A8%E6%A3%80%E6%9F%A5): `npu-smi info` 检查Ascend NPU固件和驱动是否正确安装。

**网络联通：** 使用[ssh命令](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md#%E7%BD%91%E7%BB%9C%E8%BF%9E%E9%80%9A%E6%80%A7%E6%A3%80%E6%9F%A5)确认机器互连。

## 模型准备

基于开源的 QwQ-32B 模型。

## 部署

### 镜像及源码准备

```bash
docker pull swr.cn-east-4.myhuaweicloud.com/omni-ci/daily_omniinfer:20250722_26
git clone https://gitee.com/omniai/omniinfer.git
git clone https://github.com/vllm-project/vllm.git omniinfer/infer_engines/vllm
git checkout XXX (commit ID必须与镜像中安装的omni相同)
```
#### 注：[**无法获取代码**](#无法获取源码时的-ansible-配置调整)或[**昇腾 + x86**](#昇腾--x86平台网络配置)等情况，参见下一节ansible修改。

### 部署

#### 环境准备

安装ansible，参考[文档](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md#%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87-1)。

#### 修改配置文件

需要修改`omni_infer_inventory_used_for_qwq_3P1D.yml`和 `omni_infer_server_template_qwq.yml` 两处配置文件，位于`omniinfer/tools/ansible/templete/`路径下。以3P1D为例:

1. **omni_infer_inventory_used_for_qwq_3P1D.yml**

   将`p0/p1/p2/d0/c0`下面的`ansible_host` 与 `host_ip` 值改为对应的IP：

   ```YAML
   children:
     P:
       hosts:
         p0:
           ansible_host: "127.0.0.1"  # P0节点的IP
           ...
           host_ip: "127.0.0.1"  # P0节点的IP
           ascend_rt_visible_devices: "0,1,2,3,4,5,6,7"
           ...

         p1:
           ansible_host: "127.0.0.1"  # P1节点的IP
           ...
           host_ip: "127.0.0.1"  # P1节点的IP
           ascend_rt_visible_devices: "8,9,10,11,12,13,14,15"
           ...
           
         p2:
           ansible_host: "127.0.0.1"  # P2节点的IP
           ...
           host_ip: "127.0.0.2"  # P2节点的IP
           ascend_rt_visible_devices: "0,1,2,3,4,5,6,7"
           ...
     D:
       hosts:
         d0:
           ansible_host: "127.0.0.2"  # D0 节点的IP
           ...
           host_ip: "127.0.0.2"       # D0 节点的IP
           ascend_rt_visible_devices: "8,9,10,11,12,13,14,15"
           ...

     C:
       hosts:
         c0:
           ansible_host: "127.0.0.1"  # C0 节点的IP，即 Global Proxy 节点
           ...

   ```

   生成私钥文件，参考[文档](https://gitee.com/omniai/omniinfer/blob/master/tools/ansible/README.md#%E5%AF%86%E9%92%A5%E6%96%87%E4%BB%B6%E7%9A%84%E5%87%86%E5%A4%87)。将`ansible_ssh_private_key_file:`修改为私钥文件路径：

   ```YAML
    all:
      vars:
        ...
        ansible_ssh_private_key_file: /path/to/key.pem  # 私钥文件路径
        ...
   ```

2. **omni_infer_server_template_qwq.yml**

    修改以下环境变量
    ```yaml
    environment:
        # Global Configuration
        LOG_PATH: "/data/log_path"
        MODEL_PATH: "/data/model/qwen"  #模型文件路径
        MODEL_LEN_MAX_PREFILL: "61244" #P节点max_model_len设置，同时相应修改`run_vllm_server_prefill_cmd`中的--max-num-batched-tokens, 要大于等于max_model_len
        MODEL_LEN_MAX_DECODE: "61244"  #D节点max_model_len设置，同时相应修改`run_vllm_server_decode_cmd`中的--max-num-batched-tokens, 要大于等于max_model_len
        LOG_PATH_IN_EXECUTOR: "/data/log_path_in_executor"
        CODE_PATH: "/data/local_code_path"  # omniinfer本地代码路径
        HTTP_PROXY: ""  # 下载nginx的HTTP代理地址，如果不需要代理可以留空

        # Configuration for containers
        DOCKER_IMAGE_ID: "REPOSITORY:TAG" #镜像与标签
        DOCKER_NAME_P: "you_name_omni_infer_prefill" # P容器名称
        DOCKER_NAME_D: "you_name_omni_infer_decode"  # D容器名称
        DOCKER_NAME_C: "you_name_omni_infer_proxy"   # Proxy 容器名称
    ```

### 无法获取源码时的 Ansible 配置调整
若因网络或权限问题无法拉取代码，可按以下步骤修改 Ansible 配置：

1. 修改 **CODE_PATH** 变量
    - 在 **omni_infer_server_template_qwq.yml** 中，将 `CODE_PATH` 设置为 `/workspace`，这样可以避免在执行机上拉取代码，直接使用容器内的代码：
      ```yaml
      CODE_PATH: "/workspace"
      ```
2. 修改 docker 启动命令
    - 在 **omni_infer_server_template_qwq.yml** 中，删除容器挂载代码的部分，这样可以避免因容器内代码路径不可访问而导致的部署失败。即删除以下部分：
      ```yaml
      -v $CODE_PATH:$CODE_PATH
      ```

### 昇腾 + x86平台网络配置
在昇腾 + x86 平台，需要根据主机网卡动态设置接口名：

1. **查找主机 IP 对应网卡**
    ```bash
    ifconfig
    ```
    在输出中找到主机 IP 所在的网卡名称（例如：virbr0）。

2. **替换脚本中的网卡名称**
   
   在所有相关脚本或配置文件中，将：
    ```bash
    GLOO_SOCKET_IFNAME="enp23s0f3"
    TP_SOCKET_IFNAME="enp23s0f3"
    ```
    修改为：
    ```bash
    GLOO_SOCKET_IFNAME="virbr0"
    TP_SOCKET_IFNAME="virbr0"
    ```
3. **在 ansible 的 **omni_infer_server_template_qwq.yml** 中，`run_vllm_server_prefill_cmd`和`run_vllm_server_decode_cmd`，新增加环境变量**
    ```bash
    export HCCL_SOCKET_IFNAME=virbr0
    ```
#### 执行命令

在执行机上，一般为p0节点，执行以下命令：
```bash
cd omniinfer/tools/ansible/template
ansible-playbook -i omni_infer_inventory_used_for_qwq_3P1D.yml omni_infer_server_template_qwq.yml  --tags clean_up
ansible-playbook -i omni_infer_inventory_used_for_qwq_3P1D.yml omni_infer_server_template_qwq.yml  --tags run_docker
ansible-playbook -i omni_infer_inventory_used_for_qwq_3P1D.yml omni_infer_server_template_qwq.yml  --tags ranktable
ansible-playbook -i omni_infer_inventory_used_for_qwq_3P1D.yml omni_infer_server_template_qwq.yml  --tags run_proxy
ansible-playbook -i omni_infer_inventory_used_for_qwq_3P1D.yml omni_infer_server_template_qwq.yml  --tags run_server
```

#### curl 测试

拉起成功后，可以通过curl命令进行测试：

```bash
curl -X POST http://127.0.0.1:7000/v1/completions -H "Content-Type:application/json" -d '{"model": "qwen","max_tokens":50,"prompt": "how are you?"}'
```
