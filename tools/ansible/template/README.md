[TOC]

# 脚本介绍
本目录 (tools/ansible/template) 下的脚本实现一键式部署多机PD分离服务和 Global Proxy 服务， 用户只需按要求进行部分配置的修改， 即可实现容器自动启动， 自动生成全局 ranktable 以及拉起 vllm 服务和 Proxy 服务。
脚本文件目录结构如下：
```bash
└── template
    ├── omni_infer_inventory_used_for_2P1D.yml
    ├── omni_infer_inventory_used_for_4P1D.yml
    ├── omni_infer_server_template.yml
    └── README.md
```

# 相关文件解释说明
## omni_infer_inventory_used_for_2P1D.yml 和 omni_infer_inventory_used_for_4P1D.yml
两个文件一个用于 2P1D 四机场景，一个用于 4P1D 八机场景，都是用于定义被管理目标机的配置信息；文件中参数配置说明如下: 

* `ansible_user`: 远程目标机的用户名， 如 user 等。

* `global_port_base`: 基础端口号， 用于作为 `master-port` 的基准。

* `base_api_port`:  API 服务的基础端口号， 用于作为 `base-api-port` 的基准。

* `proxy_port`: global proxy 的监听基础端口号， Global Proxy 实例的端口： `proxy_port + node_rank`

* `port_offset`: 区分 Prefill 实例端口范围和 Decode 实例的端口范围， 避免在同一节点拉起时端口冲突 （多机场景可以忽略冲突问题）。

* `ansible_host`: 目标机的 IP 。

* `ansible_ssh_private_key_file`: 连接目标机的私钥文件路径。也可以使用密码登录目标机的方式， 则使用 `ansible_password` 字段并将密码填入， 如：`ansible_password: "passwod"` 。

* `node_rank`: Prefill 和 Decode 实例的索引， 一是用于 Prefill 和 Decode 实例的端口区分， 二是用于 prefill 实例 kv_rank 的区分。索引从0开始。

* `node_port`: 即 Prefill 和 Decode 的实际 `master-port`。 
    Prefill 实例的默认端口: `global_port_base + port_offset.P + node_rank`。
    Decode 实例的默认端口: `global_port_base + port_offset.D`。

* `api_port`: 多 API Server 的端口号。
    Prefill 实例的 API Server 默认端口: `base_api_port + port_offset.P + node_rank`。
    Decode 实例的 API Server 默认端口: `base_api_port + port_offset.D + node_rank`。

* `role`: Decode 实例的角色，如 `M` 表示 Master， `S` 表示 Slave。 

* `ascend_rt_visible_devices`: 每个 Prefill 或 Decode 实例需要使用的卡号， 参数值需要严格按照以下格式: `"x,x,x,x"` (用英文逗号分隔的连续值) ， 不能有多余逗号和空格。


## omni_infer_server_template.yml
该文件作用是管理目标节点执行相应的任务；文件中参数配置说明如下: 

* `LOG_PATH`: Decode/Prefill/Global Proxy 实例日志的存放的路径。

* `LOG_PATH_IN_EXECUTOR`: 执行机上存放 Decode/Prefill/Global Proxy 实例日志的路径。

* `LOCAL_CODE_PATH`: 执行机上的代码路径，即用户通过 git clone 拉取的代码存放路径。例如你在 `/workspace/local_code_path` 下 git clone 了代码，那路径就是 `/workspace/local_code_path`。

* `MODEL_PATH`: 加载的模型路径， 要求 Prefill 和 Decode 所有实例所在的节点提前拷贝好模型并且模型路径保持一致。

* `MODEL_LEN_MAX_PREFILL`: Prefill 侧模型的最大生成长度， 包含 prompt 长度和 generated 长度， 默认值为30000。

* `MODEL_LEN_MAX_DECODE`: Decode 侧模型的最大生成长度， 包含 prompt 长度和 generated 长度， 默认值为16384。

* `DOCKER_IMAGE_ID`: omniai 服务实例均在容器里面运行， 用来指定运行的容器镜像， 如: `swr.cn-southwest-2.myhuaweicloud.com/omni-ai/omniinfer:202506272026`，其中 `swr.cn-southwest-2.myhuaweicloud.com/omni-ai/omniinfer` 表示镜像仓地址，`202506272026` 表示镜像版本号，如果远程目标机没有此容器镜像，脚本会自动下载。

* `DOCKER_NAME_P`: Prefill 节点的容器别名， 默认前缀 `omni_infer_prefill_`。 如 `omni_infer_prefill_p0` 表示 Prefill 第一个实例的容器名。

* `DOCKER_NAME_D`: Decode 节点的容器别名， 默认前缀 `omni_infer_decode_`。 如 `omni_infer_prefill_d0` 表示 Decode 第一个实例的容器名。

* `DOCKER_NAME_C`: Proxy 节点的容器别名， 默认前缀 `omni_infer_proxy_`。 如 `omni_infer_proxy_c0` 表示 Proxy 第一个实例的容器名。

* `SCRIPTS_PATH`: ansible 运行过程中自动生成的脚本文件存放路径， 用户可以不关注， ansible 执行过程中会自动生成。

* `CODE_PATH`: 容器内的 omni_infer 源码路径。开发人员通过 ansible 脚本将执行机上的源码同步到容器内，存储路径即 `CODE_PATH`。

* `DECODE_TENSOR_PARALLEL_SIZE`: Decode 实例 tensor parallel 参数 tp， 默认值为1。

* `ranktable_save_path`: ansible 运行过程中， Prefill 和 Decode 实例对应的 ranktable 文件以及它们合并生成的 ranktable 文件存放路径。

# 环境准备
## 在执行机安装 ansible-playbook
```bash
# 安装ansible-playbook
yum install ansible

# 参考open euler系统的公司内部的yum源
rm  /etc/yum.repos.d/*

echo "[openEuler-everything]
name=openEuler-everything
baseurl=http://mirrors.tools.huawei.com/openeuler/openEuler-22.03-LTS-SP4/everything/aarch64/
enabled=1
gpgcheck=0
gpgkey=http://mirrors.tools.huawei.com/openeuler/openEuler-22.03-LTS-SP4/everything/aarch64/RPM-GPG-KEY-openEuler
        
[openEuler-EPOL]
name=openEuler-epol
baseurl=http://mirrors.tools.huawei.com/openeuler/openEuler-22.03-LTS-SP4/EPOL/main/aarch64/
enabled=1
gpgcheck=0
[openEuler-update]
name=openEuler-update
baseurl=http://mirrors.tools.huawei.com/openeuler/openEuler-22.03-LTS-SP4/update/aarch64/
enabled=1
gpgcheck=0" > /etc/yum.repos.d/openeuler.repo
```

## 在执行机安装 sshpass
执行 ansible 依赖 sshpass 链接各个目标机，做远程机器管理
```bash
yum install openssh-server
```

## 密钥文件的准备
请注意，这里的密钥文件仅仅是用于执行机通过 ansible 去登录目标机，如果你已经有登录目标机的密钥文件，放在 omni_infer_inventory.yml 的 ansible_ssh_private_key_file 指定路径即可。
1. 首先在执行机生成密钥对:
    ```bash
    ssh-keygen -t ed25519 -C "Your SSH key comment" -f ~/.ssh/my_key  # -t 指定密钥类型（推荐ed25519）， -f 指定文件名
    ```
2. 密钥文件默认存放位置为: 私钥：~/.ssh/id_ed25519 公钥：~/.ssh/id_ed25519.pub. 设置密钥文件权限:
    ```bash
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/id_ed25519   # 私钥必须设为 600
    chmod 644 ~/.ssh/id_ed25519.pub
    ```
3. 部署公钥到远程目标机:
    ```bash
    # 以下例子是通过密码去传输密钥文件到远程目标机
    ssh-copy-id -i ~/.ssh/id_ed25519.pub user@remote-host
    ```

# 操作步骤

## 修改配置
在 **omni_infer_inventory_used_for_2P1D.yml 和 omni_infer_inventory_used_for_4P1D.yml** 中， 只需修改以下配置项 `ansible_user / ansible_ssh_private_key_file`; 
在 **omni_infer_server_template.yml** 中， 只需修改以下配置项 `MODEL_PATH / DOCKER_IMAGE_ID / CODE_PATH / LOCAL_CODE_PATH`， 就可拉起 omniai 服务。
此外，建议修改 omni_infer_server_template.yml 中的 `LOG_PATH`、`LOG_PATH_IN_EXECUTOR`、`SCRIPTS_PATH` 和 `ranktable_save_path`，防止路径下的文件被其他人覆盖。

## 执行命令
```bash
# 拉取库上最新代码，比如执行机的 /workspace/local_code_path 路径下 clone 源码，执行下列命令即可
cd omniinfer/infer_engines/
bash bash_install_code.sh
# 进入到 ansible 脚本目录下
cd omniinfer/tools/ansible/template
# 如果是部署四机 2P1D，就执行如下命令：
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --skip-tags fetch_log
# 如果是部署八机 4P1D，就执行如下命令：
ansible-playbook -i omni_infer_inventory_used_for_4P1D.yml omni_infer_server_template.yml --skip-tags fetch_log

# 后续修改了执行机上的源码，执行下列命令即可通过修改后的源码拉起服务
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags sync_code
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags run_server

# 基于指定镜像创建并启动新的容器实例
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags run_docker
# 生成ranktable文件
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags ranktable
# 拉起vllm服务
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags run_server
# 拉起nginx服务
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags run_proxy
# 将日志存放在执行机指定路径
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags fetch_log
# 清理残余配置，主要是 stop 容器并且删除容器，以及删除 ranktable 和临时脚本
ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml --tags clean_up
```