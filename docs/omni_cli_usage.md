# omni_cli 使用文档

## 安装

```bash
cd omniinfer/tools; python -m build --wheel      # 无wheel包，源码安装
pip install omni_cli***.wheel --force-reinstall --no-build-isolation --no-deps
```

## 快速开始

omni_cli 会在命令执行路径创建配置文件，并依赖该配置文件。不同用户、模型请在不同路径下执行。add_node第一次执行时会创建该配置文件。以下内容以deepseek 1p1d 三机部署为例。

1. 添加节点
   
   ```bash
   omni_cli add_node --role P --name p0 --host_ip 本机IP --ssh_private_key_file 本机SSH_KEY文件 --docker_image_id 镜像
   omni_cli add_node --role D --name d0 --host_ip 本机IP --ssh_private_key_file 本机SSH_KEY文件 --docker_image_id 镜像
   omni_cli add_node --role D --name d1 --host_ip 本机IP --master_ip 主D节点IP --ssh_private_key_file 本机SSH_KEY文件 --docker_image_id 镜像
   ```

2. 修改配置

   omni_cli具有推断配置的功能。因为不同模型、不同场景参数差别巨大，需要先配置模型路径，omni_cli依据路径名称推断模型，并提供部分场景默认配置。
   
   ```bash
   # 配置所有节点，支持按照Role、节点名称分别配置，参见下文omni_cli cfg详细介绍
   omni_cli cfg --set all env --MODEL_PATH 模型路径
   omni_cli cfg --set all env --LOG_PATH 日志路径
   omni_cli cfg --set all --container_name_prefix  容器名称前缀 # 依据前缀生成所有容器名称，如 prefix_prefill_p0
   ```

3. 服务拉起
   
   ```bash
   omni_cli run_docker  # 如需使用指定容器，必须跳过此步
   omni_cli start       # 服务拉起，包括ranktable生成、global proxy拉起
   omni_cli stop        # 服务停止
   ```

## 命令列表

   ```bash
   # 节点管理
   omni_cli ls          # 开发中
   omni_cli add_node
   omni_cli rm_node
   
   # 配置管理
   omni_cli cfg
   omni_cli inspect
    
   # 服务管理
   omni_cli run_docker
   omni_cli start
   omni_cli stop 
    
   # 开发工具
   omni_cli upgrade     # 开发中
   omni_cli collect_log # 开发中
   omni_cli install_dev
   omni_cli sync_dev
   ```

### 节点管理

1. omni_cli ls

   功能：输出当前的节点，包括role, name, IP。

   使用示例：
   
   ```bash
   omni_cli ls
   ```

2. omni_cli add_node

   功能：在当前工作区初始化或修改servering_profiles.yml文件并向其添加节点。后续支持动态扩缩容

   参数说明：
   
   > --help: 帮助信息
   >
   > --role：节点角色（必需），从 ['P', 'D', 'C'] 中选择，分别代表Prefill，Decode 和 Global Proxy
   >
   > --name：节点名称（必需）
   >
   > --host_ip：主机IP地址（必需）
   >
   > --ssh_private_key_file：SSH私钥文件路径（必需）
   >
   > --docker_image_id：Docker镜像ID（必需） 
   >
   > --user：SSH用户名，默认为 "root"
   >
   > --ssh_common_args：SSH通用参数，默认为"-o StrictHostKeyChecking=no -o IdentitiesOnly=yes"
   >
   >  --master_ip：主节点IP，默认为host_ip值，用于多机组服务，从节点需要配置主节点IP
   
   使用示例：
   
   ```bash
   omni_cli add_node --role P --name p0 --host_ip 127.0.0.1 \
   --ssh_private_key_file /path/to/my/key --docker_image_id myapp:latest
   ```

3. omni_cli rm_node

   功能：从配置文件中删除指定节点。后续支持动态扩缩容

   参数说明：
   
   > --help: 帮助信息
   >
   > --role：节点角色（必需），从 ['P', 'D', 'C'] 中选择
   >
   > --name：要删除的节点名称（必需）
   >
   > --deploy_path：配置文件路径，默认为 $pwd/servering_profiles.yml
   
    使用示例：
   
   ```bash
   omni_cli rm_node --role P --name p0
   ```

### 配置管理

1. omni_cli cfg

2. omni_cli inspect

   功能：根据名称，输出该节点所有配置信息

   参数说明：
   > name: 节点名称，可通过omni_cli ls查看

   使用示例:
   ``` bash
   omni_cli inspect p0
   ```

### 服务管理

1. omni_cli run_docker

   功能：根据 add_node、cfg配置，启动所有容器。

   参数说明：
   
   > --help: 帮助信息
   >
   > --deploy_path：配置文件路径，默认为 $pwd/servering_profiles.yml
   >
   > --dry-run：测试模式，只显示将要执行的操作而不实际执行
   
    使用示例：
   
   ```bash
   omni_cli run_docker
   ```

2. omni_cli start

   功能：依据模板配置和配置推断，在已配置容器中，将推理服务拉起，包括ranktable生成、global proxy拉起。(ranktable生成是临时方案，动态扩缩容会去掉ranktable机制)。

   参数说明：
   
   > --help: 帮助信息
   >
   > --skip-verify-config:  跳过配置校验，避免配置推断修改用户指定配置
   >
   > --normal config_path:  使用指定omni_cli配置文件拉起服务，默认为 $pwd/servering_profiles.yml
   > 
   > --run_dev: 开发者快速拉起，跳过ranktable生成和global proxy拉起。需确保已经进行normal拉起，完成ranktable生成和global proxy拉起。
   
   使用示例：
   
   ```bash
   omni_cli start --normal servering_profiles.yml
   ```

3. omni_cli stop

   功能：在已配置容器中，将服务停止。注：不会停止配置容器之外的服务

   参数说明: 
   
   > --help: 帮助信息
   
   使用示例：
   
   ```bash
   omni_cli stop
   ```

### 开发工具

1. omni_cli upgrade

    功能：执行安装流程，将omniinfer和global proxy安装包安装进入指定容器中。

    参数说明：
   
   > --help: 帮助信息
   >
   > --omni_package:  需要安装的omniinfer包
   >
   > --vllm_package:  需要安装的omniinfer包
   >
   > --proxy_package: 需要安装的omniinfer包
   
    使用示例：
   
   ``` bash
   omni_cli upgrade --omni_package XXXXX.whl  --vllm_package XXXXX.whl --proxy_package XXXXX.rpm
   ```

2. omni_cli collect_log

    功能：将日志从各个容器中收集到执行机路径。

    参数说明：
   
   > --name: 指定需要获取的日志，all, P, D, C分别代表所有日志、所有P日志、所有D日志、Proxy日志
   
    使用示例：
   
   ```bash
   omni_cli collect_log --name all
   ```

3. omni_cli install_dev

    功能：执行安装流程，将容器中/workspace/omniinfer代码采用editable模式安装，替换容器中自带omni_infer和vllm。对于同一个容器，只需执行一次，后续仅同步代码 sync_dev 即可

    参数说明：
   
   > --help: 帮助信息
   >
   > --deploy_path：配置文件路径，默认为 $pwd/servering_profiles.yml
   >
   > --dry_run：测试模式，只显示将要执行的操作而不实际执行 
   
    使用示例：
   
   ```bash
   omni_cli install_dev
   ```

4. omni_cli sync_dev --code_path

    功能：将指定路径源代码同步到所有节点对应的容器中的/workspace/omniinfer。

    参数说明：
   
   > --help: 帮助信息
   >
   > --code_path：要同步的代码路径（必需）。
   >
   > --deploy_path：配置文件路径，默认为 $pwd/servering_profiles.yml
   >
   > --dry_run：测试模式，只显示将要执行的操作而不实际执行
   
    使用示例：
   
   ```bash
   omni_cli sync_dev --code_path /path/to/my/code
   ```