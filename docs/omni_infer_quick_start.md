# PD分离快速部署

本文档介绍如何快速拉起PD分离部署推理，支持3机1P1D、4机2P1D、8机4P1D、16机8P1D和EP144 36机18P1D。

## 硬件要求

**硬件：** CloudMatrix384推理卡

**操作系统：** Linux

**Python:** >=3.9, <=3.11

**PyTorch:** >=2.5.1, < 2.6

[**驱动检查**](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md#ascend-npu%E5%9B%BA%E4%BB%B6%E5%92%8C%E9%A9%B1%E5%8A%A8%E6%A3%80%E6%9F%A5): `npu-smi info`

**网络联通：** 使用ssh命令确认机器互连。

## 模型准备

基于开源的 DeepSeek-V3/R1 进行[权重转换](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md#%E6%9D%83%E9%87%8D%E8%BD%AC%E6%8D%A2)。

## 部署

### 镜像及源码准备

```bash
docker pull IMAGE:TAG
git clone https://gitee.com/omniai/omniinfer.git
```

### 使用ansible一键部署

详见[ansible部署文档](https://gitee.com/omniai/omniinfer/blob/master/tools/ansible/README.md)。以下为快速部署示例。

#### 修改配置文件

需要修改`omni_infer_inventory_used_for_xP1D.yml`和 `omni_infer_server_template` 两处配置文件，位于`omniinfer/tools/ansible/templete/`路径下。以2P1D为例:

1. **omni_infer_inventory_used_for_2P1D.yml**
   
   将`p0/p1/d0/d1/c0`下面的`ansible_host:`值改为对应的IP：
   
   ```YAML
   children:
     P:
       hosts:
         p0:
           ansible_host: "127.0.0.1"  # P0节点的IP
           ...

         p1:
           ansible_host: "127.0.0.2"  # P1节点的IP
           ...
        
     D:
       hosts:
         d0:
           ansible_host: "127.0.0.3"  # D0 节点的IP
           ...
        
         d1:
           ansible_host: "127.0.0.4"  # D1 节点的IP
           ...
     
     C:
       hosts:
         c0:
           ansible_host: "127.0.0.1"  # C0 节点的IP
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

2. **omni_infer_server_template.yml**

    修改以下环境变量
    ```yaml
    environment:
        # Global Configuration 
        LOG_PATH: "/data/log_path"
        MODEL_PATH: "/data/models/DeepSeek-R1-w8a8"  #模型文件路径
        LOG_PATH_IN_EXECUTOR: "/data/log_path_in_executor"

        # Configuration for containers 
        DOCKER_IMAGE_ID: "REPOSITORY:TAG" #镜像与标签
        DOCKER_NAME_P: "you_name_omni_infer_prefill" # P容器名称
        DOCKER_NAME_D: "you_name_omni_infer_decode"  # D容器名称
        DOCKER_NAME_C: "you_name_omni_infer_proxy"   # Proxy 容器名称
    ```

配置文件详细解释说明请参考[文档]()。

#### 执行命令

```bash
cd omniinfer/tools/ansible/template
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml
```

#### curl 测试

拉起成功后，可以通过curl命令进行测试：

```bash
curl -X POST http://127.0.0.1:7000/v1/completions -H "Content-Type:application/json" -d '{"model": "deepseek","temperature":0,"max_tokens":50,"prompt": "how are you?", "stream":true,"stream_options": {"include_usage": true,"continuous_usage_stats": true}}'
```


## 更高性能

可以使能以下特性来获得更高的性能：

**1. 使用图缓存**

首次启动服务时，模型会从头编译。建议首次成功启动后，重新执行以下命令以启用图缓存，提升性能：

```bash
cd omniinfer/tools/ansible/template
ansible-playbook -i omni\_infer\_inventory.yml omni\_infer\_server.yml --tags run_server
```


**2. 调整 proxy batch size**

当前的默认值为 25。建议根据每个 die 的平均并发量大小(n)进行调整。推荐值为(n+1)。进入proxy容器，在 /usr/local/nginx/conf/nginx.conf 文件中修改该值，然后执行 `nginx -c /usr/local/nginx/conf/nginx.conf -s reload` 以应用更改。



**3. 增加 batch size**

ansible部署默认开启 MTP 以优化性能。如需调整 `--max-num-seqs`（batch size），需同步修改所有容器代码 /workspace/omniinfer/tests/test_config/test_config_decode.json中:

```JSON
"decode_gear_list": [batch_size * (1+num_speculative_tokens)]
```
 
 以MTP 1为例，`--max-num-seqs`设置为32，`"decode_gear_list":[64]`。
