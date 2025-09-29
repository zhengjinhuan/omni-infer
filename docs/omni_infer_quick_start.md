# DeepSeek-V3.2 PD分离快速部署

本文档介绍基于8机Atlas 800IA3如何快速拉起DeepSeek-V3.2 PD分离部署推理。
本次提供的镜像和权重下载路径：https://gitee.com/omniai/omniinfer/releases/tag/v0.4.2

## 硬件要求

**硬件：** Atlas 800I A3

**操作系统：** Linux

**镜像版本：** swr.cn-east-4.myhuaweicloud.com/omni/omni_infer-a3-arm:release_v0.4.2

[**驱动检查**](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md#ascend-npu%E5%9B%BA%E4%BB%B6%E5%92%8C%E9%A9%B1%E5%8A%A8%E6%A3%80%E6%9F%A5): `npu-smi info` 检查Ascend NPU固件和驱动是否正确安装。

**驱动版本：** Ascend HDK 25.2.1 
https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/266220744?idAbsPath=fixnode01|23710424|251366513|254884019|261408772|252764743

**网络联通：** 
1、使用[ssh命令](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md#%E7%BD%91%E7%BB%9C%E8%BF%9E%E9%80%9A%E6%80%A7%E6%A3%80%E6%9F%A5)确认机器互连。
2、服务机器ROCE面联通

## 模型准备

权重路径需要与下文中 omni_infer_server_template_a3_ds.yml模型文件路径保持一致。

## 部署

### 镜像及源码准备

```bash
docker pull swr.cn-east-4.myhuaweicloud.com/omni/omni_infer-a3-arm:release_v0.4.2
git clone https://gitee.com/omniai/omniinfer.git
```


### 环境准备
目标机：8台 Atlas 800I A3
执行机：1台linux服务器

用户执行机需要安装ansible
```bash
yum install ansible
yum install openssh-server
```

目标机安装libselinux-python3
```bash
yum install libselinux-python3
```


### 修改配置文件

需要修改`omniinfer/tools/ansible/templete/`路径下得2个配置文件。`omni_infer_inventory_used_for_1P32_1D32.yml`和 `omni_infer_server_template_a3_ds.yml` 。
以1P1D(4机组P,4机组D)部署DeepSeek-V3.2 BF16为例:

1. **tools/ansible/template/omni_infer_inventory_used_for_1P32_1D32.yml**

   将`p/d/c`下面的`ansible_host` 与 `host_ip` 值改为对应的IP。<span style="color:red; font-weight:bold">对于多机组 P/D 的场景，所有 P/D 节点的 `host_ip` 为主节点 d0 的 IP。</span>


   ```YAML
   children:
     P:
       hosts:
         p0:
           ansible_host: "127.0.0.1"  # P0节点的IP
           ...
           host_ip: "127.0.0.1"  # P0节点的IP
           ...

         p1:
           ansible_host: "127.0.0.2"  # P1节点的IP
           ...
           host_ip: "127.0.0.1"  # P0节点的IP, 即 P 节点的主节点 IP
           ...
        ...

     D:
       hosts:
         d0:
           ansible_host: "127.0.0.5"  # D0 节点的IP
           ...
           host_ip: "127.0.0.5"       # D0 节点的IP
           ...

         d1:
           ansible_host: "127.0.0.6"  # D1 节点的IP
           ...
           host_ip: "127.0.0.5"       # D0 节点的IP, 即 D 节点的主节点 IP
           ...
        ...

     C:
       hosts:
         c0:
           ansible_host: "127.0.0.1"  # C0 节点的IP，即 Global Proxy 节点
           ...

   ```

生成私钥文件。将`ansible_ssh_private_key_file:`修改为私钥文件路径：
```bash
# 首先在执行机生成密钥对:
ssh-keygen -t ed25519 -C "Your SSH key comment" -f ~/.ssh/my_key  # -t 指定密钥类型（推荐ed25519）， -f 指定文件名
# 密钥文件默认存放位置为: 私钥：~/.ssh/id_ed25519 公钥：~/.ssh/id_ed25519.pub. 设置密钥文件权限:
chmod 700 ~/.ssh
chmod 600 ~/.ssh/my_key   # 私钥必须设为 600
chmod 644 ~/.ssh/my_key.pub
# 部署公钥到远程目标机:以下例子是通过密码去传输密钥文件到远程目标机
ssh-copy-id -i ~/.ssh/my_key.pub user@remote-host
```

   ```YAML
    all:
      vars:
        ...
        ansible_ssh_private_key_file: /path/to/key.pem  # 私钥文件路径
        ...
   ```

2. **tools/ansible/template/omni_infer_server_template_a3_ds.yml**

    修改以下环境变量
    ```yaml
    environment:
        # Global Configuration
        LOG_PATH: "/data/log_path"  # 服务日志路径
        MODEL_PATH: "/data/models/origin/bf16"  # 模型文件路径
        LOG_PATH_IN_EXECUTOR: "/data/log_path_in_executor"
        CODE_PATH: "/data/local_code_path"  # [可选配置]omniinfer本地代码路径
        HTTP_PROXY: ""  # 下载nginx的HTTP代理地址，如果不需要代理可以留空

        # Configuration for containers
        DOCKER_IMAGE_ID: "REPOSITORY:TAG" # 镜像与标签
        DOCKER_NAME_P: "you_name_omni_infer_prefill" # P容器名称
        DOCKER_NAME_D: "you_name_omni_infer_decode"  # D容器名称
        DOCKER_NAME_C: "you_name_omni_infer_proxy"   # Proxy 容器名称
    ```


### 执行命令

```bash
cd omniinfer/tools/ansible/template
ansible-playbook -i omni_infer_inventory_1p32_1d32.yml omni_infer_server_template_a3_ds.yml
```
提示：建议起服务前清理一下全部节点的环境，例如：
```bash
ps aux | grep "python" | grep -v "grep" | awk '{print $2}' | xargs kill -9
```
### 服务拉起成功
查看服务启动日志, 配置文件设置的LOG_PATH

### curl 测试

拉起成功后，可以通过curl命令进行测试：

```bash
curl -X POST http://127.0.0.1:7000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek",
    "temperature": 0,
    "max_tokens": 50,
    "prompt": "how are you?",
    "stream": true,
    "stream_options": {
      "include_usage": true,
      "continuous_usage_stats": true
    }
  }'
```

更多ansible部署指导详见[ansible部署文档](https://gitee.com/omniai/omniinfer/blob/master/tools/ansible/README.md)
配置文件详细解释说明请参考[文档](https://gitee.com/omniai/omniinfer/blob/master/tools/ansible/template/README.md#%E7%9B%B8%E5%85%B3%E6%96%87%E4%BB%B6%E8%A7%A3%E9%87%8A%E8%AF%B4%E6%98%8E)