# 一、DeepSeek模型部署  

## 1、DeepSeek-R1-int8-A3-4P8-1D32

### 1.1 配置 omniinfer/tests/test_config/config_d.yaml:
    '''
	pattern_path: "../../omni/accelerators/placement/patterns/base_patterns/DSV3_baseline_64_devices_58_MoE_Layers.npy"
	max_redundant_per_expert: 10 # 10
    max_redundant_per_rank: 1 # 1

    '''

### 1.2 配置 omniinfer/tests/test_config/test_config_decode.json:
    '''
	"enable_mc2_v2": true

    '''

### 1.3 配置 omniinfer/tools/ansible/template/omni_infer_server_template.yml:
    '''
	environment:
		MODEL_LEN_MAX_DECODE: "44000"
		
	vars:
		run_vllm_server_decode_cmd: |
            ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1, "use_ge_graph_cached":true}}'

    '''


## 2、DeepSeek-R1-int8-A3-5P8-1D32

### 2.1 配置 omniinfer/tests/test_config/config_d.yaml
    ```
    pattern_path: "../../omni/accelerators/placement/patterns/base_patterns/DSV3_baseline_64_devices_58_MoE_Layers.npy"
    max_redundant_per_expert: 10
    max_redundant_per_rank: 1

    ``` 

### 2.2 配置 omniinfer/tests/test_config/config_p.yaml
    ```
    pattern_path: "../../omni/accelerators/placement/patterns/placement_pattern_20250724_142738_58_redundant_layers_58_layers_16_ranks_epmaxdeploy_200_prefill.npy"
    max_redundant_per_expert: 10 # 10
    max_redundant_per_rank: 1 # 1

    ``` 

### 2.3 配置 omniinfer/tools/scripts/pd_run.sh
    ```
    export OMNI_REUSE_PREFILLED_TOKENS=1
    export OMNI_SKIP_DECODE_TOKENIZE=1
    export TOKENIZER_PROC_POOL=1

    ``` 

### 2.4 配置 omniinfer/tools/ansible/template/omni_infer_server_template.yml
    ```
    vars:
        run_vllm_server_prefill_cmd: |
            bash /workspace/omniinfer/tools/scripts/pd_run.sh \
                --additional-config '{"enable_omni_attn":true}' \
        run_vllm_server_decode_cmd: |
            ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1, "use_ge_graph_cached": true, "enable_omni_attn": true}}'
    tasks:
        - name: Set mooncake & etcd configuration.
            when: "'P' in group_names or 'D' in group_names or 'C' in group_names"
        - name: Kill Mooncake master pod.
            when: "'C' in group_names"
        - name: Kill etcd pod.
            when: "'C' in group_names"
        - name: Generate a mooncake config json.
            when: "'P' in group_names or 'D' in group_names"
        - name: Generate a lmcache + mooncake config yaml.
            when: "'P' in group_names or 'D' in group_names"
        - name: Run etcd for Mooncake.
            when: "'C' in group_names"
        - name: Run master port of Mooncake.
            when: "'C' in group_names"

    ``` 


## 3、DeepSeek-R1-int8-A3-2P8-1D16

### 3.1 配置 omniinfer/tests/test_config/test_config_decode.json:
    '''
	"model_parallel_config": 
                "dp_size": 32

    '''

### 3.2 配置 omniinfer/tools/ansible/template/omni_infer_server_template.yml
    '''
    environment:
        MODEL_LEN_MAX_PREFILL: "67584"
        MODEL_LEN_MAX_DECODE: "67584"

        run_vllm_server_prefill_cmd: |
            EXTRA_ARGS='--max-num-batched-tokens 67584 --enforce-eager --enable-expert-parallel --disable-log-requests --max-num-seqs 16 --no-enable-prefix-caching'
            GPU_UTIL=0.9

        run_vllm_server_decode_cmd: |
            export ASCEND_LAUNCH_BLOCKING=1
            EXTRA_ARGS='--enable-expert-parallel --disable-log-requests --max-num-seqs 32 --no-enable-prefix-caching  --preemption-mode swap'
            GPU_UTIL=0.9

    '''


## 4、DeepSeek-R1-int8-A3-1P16-1D16

### 4.1 配置 omniinfer/tests/test_config/config_d.yaml
    ```
    pattern_path: "../../omni/accelerators/placement/patterns/placement_pattern_20250911_varlen_58_redundant_layers_58_layers_8_ranks_epmaxdeploy_200_prefill.npy"
    max_redundant_per_expert: 10 # 10
    max_redundant_per_rank: 1 # 1

    ``` 

### 4.2 配置 omniinfer/tests/test_config/config_p.yaml
    ```
    # pattern_path: "../../omni/accelerators/placement/patterns/placement_pattern_20250626_221356_58_redundant_layers_58_layers_16_ranks_prefill_step0to100000.npy"
    enable_dynamic: True
    max_redundant_per_expert: 10 # 10
    max_redundant_per_rank: 1 # 1

    ``` 

### 4.3 配置 omniinfer/tests/test_config/test_config_decode.json:
    '''
	"model_parallel_config": 
             "dp_size": 32

    '''

### 4.4 配置 omniinfer/tools/scripts/pd_run.sh
    ```
        export CPU_AFFINITY_CONF=2
        export PROFILING_NAMELIST=/workspace/omniinfer/omni/tools/profiler/proc_marker_namelist.yml

        export OMNI_REUSE_PREFILLED_TOKENS=1
        export OMNI_SKIP_DECODE_TOKENIZE=1
        export TOKENIZER_PROC_POOL=1

    ``` 

### 4.5 配置 omniinfer/tools/ansible/template/omni_infer_server_template.yml
    '''
    environment:
        MODEL_LEN_MAX_PREFILL: "51024"
        MODEL_LEN_MAX_DECODE: "55120"

        run_vllm_server_prefill_cmd: |
            EXTRA_ARGS='--max-num-batched-tokens 128000 --enforce-eager --enable-expert-parallel --disable-log-requests --max-num-seqs 16 --no-enable-prefix-caching --long-prefill-token-threshold 32000'
            --additional-config '{"enable_omni_attn": true, "multi_rank_pull_kv": true}' \

        run_vllm_server_decode_cmd: |
            EXTRA_ARGS='--enable-expert-parallel --disable-log-requests --max-num-seqs 16 --no-enable-prefix-caching'
            ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1, "use_ge_graph_cached": true}, "enable_omni_attn": true, "multi_rank_pull_kv": true, "multi_thread_pull_kv": true}'

        run_proxy_cmd: |
              --decode-max-num-seqs 9

    '''


## 5、DeepSeek-R1-int8-A2-1P16-1D32

### 5.1 配置 omniinfer/tools/ansible/template/omni_infer_server_template.yml
    ```
    environment:
        MODEL_LEN_MAX_PREFILL: "2064"
        MODEL_LEN_MAX_DECODE: "4096"
    vars:
        docker_run_cmd: |
            docker run -it
                -e CODE_PATH=$CODE_PATH \

        generate_prefill_ranktable_cmd: |
            python ${CODE_PATH}/omniinfer/tools/scripts/pd_ranktable_tools.py --mode gen --prefill-server-list "${PREFILL_SERVER_LIST}" --api-server --save-dir ${PREFILL_RANKTABLE_SAVE_PATH}

        generate_decode_ranktable_cmd: |
            python ${CODE_PATH}/omniinfer/tools/scripts/pd_ranktable_tools.py --mode gen --decode-server-list ${DECODE_SERVER_LIST} --save-dir ${DECODE_RANKTABLE_SAVE_PATH}

        generate_global_ranktable_cmd: |
            for i in "${!parts[@]}"; do
                if [ "$comma_count" -ge 1 ]; then
                    python ${CODE_PATH}/omniinfer/tools/scripts/pd_ranktable_tools.py \
            if [ $DECODE_POD_NUM -gt 1 ]; then
                python ${CODE_PATH}/omniinfer/tools/scripts/pd_ranktable_tools.py \
            python ${CODE_PATH}/omniinfer/tools/scripts/pd_ranktable_tools.py \

        run_vllm_server_prefill_cmd: |
            HCCL_BUFFSIZE=20
            MODEL_EXTRA_CFG_PATH="${CODE_PATH}/omniinfer/tests/test_config/test_config_prefill_a2.json"
            EXTRA_ARGS='--max-num-batched-tokens 16896 --enforce-eager --enable-expert-parallel --disable-log-requests --max-num-seqs 8 --no-enable-prefix-caching'
            GPU_UTIL=0.95
            VLLM_ENABLE_MC2=0

            export ASCEND_PLATFORM=A2
            export CPU_AFFINITY_CONF=1,npu0:144-167,npu1:0-23,npu2:144-167,npu3:0-23,npu4:96-119,npu5:48-71,npu6:96-119,npu7:48-71
            export ASCEND_PROCESS_LOG_PATH="/data/log_path"
            pip install pybind11

        run_vllm_server_decode_cmd: |
            HCCL_BUFFSIZE=20 
            MODEL_EXTRA_CFG_PATH="${CODE_PATH}/omniinfer/tests/test_config/test_config_decode_a2.json"
            EXTRA_ARGS='--enable-expert-parallel --disable-log-requests --max-num-seqs 60 --no-enable-prefix-caching'
            GPU_UTIL=0.95
            ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1, "use_ge_graph_cached": false, "block_num_floating_range": 50}}'
            VLLM_ENABLE_MC2=0
            
            export ASCEND_PLATFORM=A2
            export CPU_AFFINITY_CONF=1,npu0:144-167,npu1:0-23,npu2:144-167,npu3:0-23,npu4:96-119,npu5:48-71,npu6:96-119,npu7:48-71
            pip install pybind11

        run_proxy_cmd: |
            cd ${CODE_PATH}/omniinfer/omni/accelerators/sched/global_proxy
            NGINX_VERSION="${NGINX_VERSION:-1.24.0}"
            if [ ! -f "nginx-${NGINX_VERSION}.tar.gz" ]; then
                wget --no-check-certificate "https://nginx.org/download/nginx-${NGINX_VERSION}.tar.gz" > ${LOG_PATH}/install_nginx.log 2>&1
            fi
            tar -zxf "nginx-${NGINX_VERSION}.tar.gz"
            bash build.sh >> ${LOG_PATH}/install_nginx.log 2>&1
            cd ${CODE_PATH}/omniinfer/tools/scripts

        docker_start_proxy_cmd_c: >
            -e http_proxy=$HTTP_PROXY
            -e https_proxy=$HTTP_PROXY
            
        docker_cp_prefill_code_cmd: "docker cp {{ ansible_env.CODE_PATH }}/omniinfer $DOCKER_NAME_P:/workspace/"---删除
        docker_cp_decode_code_cmd: "docker cp {{ ansible_env.CODE_PATH }}/omniinfer $DOCKER_NAME_D:/workspace/"---删除

    tasks:
        - name: Copy the code from the host machine into the container (prefill).---删除
            command: bash -c "{{ docker_cp_prefill_code_cmd }}"---删除
            environment: ---删除
            DOCKER_NAME_P: "{{ ACTUAL_DOCKER_NAME_P }}"---删除
            when: "'P' in group_names"---删除
            tags: sync_code---删除

        - name: Copy the code from the host machine into the container (decode).---删除
            command: bash -c "{{ docker_cp_decode_code_cmd }}"---删除
            environment: ---删除
            DOCKER_NAME_D: "{{ ACTUAL_DOCKER_NAME_D }}"---删除
            when: "'D' in group_names"---删除
            tags: sync_code---删除

        - name: The executor synchronizes the global ranktable file to all instances.
            throttle: 1

        - name: Create a directory on the executor to store the log.
            when: "'P' in group_names or 'D' in group_names"

        - name: Forward logs from all machines to the executor.
            ansible.builtin.synchronize:
                src: "{{ ansible_env.LOG_PATH }}"
            when: "'P' in group_names or 'D' in group_names"
            delegate_to: localhost
    ```       


## 6、DeepSeek-R1-int4-A3-8P4-1D32

### 6.1 配置 omniinfer/tests/test_config/config_d.yaml
    ```
    pattern_path: "../../omni/accelerators/placement/patterns/base_patterns/DSV3_baseline_64_devices_58_MoE_Layers.npy"
    max_redundant_per_expert: 10
    max_redundant_per_rank: 1
    ``` 

### 6.2 配置 omniinfer/tests/test_config/config_p.yaml
    ```
    pattern_path: "../../omni/accelerators/placement/patterns/placement_pattern_20250911_varlen_58_redundant_layers_58_layers_8_ranks_epmaxdeploy_200_prefill.npy"
    max_redundant_per_expert: 10 # 10
    max_redundant_per_rank: 1 # 1
    ``` 

### 6.3 配置 omniinfer/tests/test_config/test_config_prefill.json
    ```
    "operator_optimizition_config": {
        "experts_pruning": true
    }
    ``` 

### 6.4 配置 omniinfer/tools/scripts/pd_run.sh
    ```
    export OMNI_REUSE_PREFILLED_TOKENS=1
    export OMNI_SKIP_DECODE_TOKENIZE=1
    export TOKENIZER_PROC_POOL=1
    ``` 

### 6.5 配置 omniinfer/tools/ansible/template/omni_infer_server_template.yml
    ```
    environment:
        LOG_PATH_A: "/data/log_path/a"
        LOG_PATH_B: "/data/log_path/b"
        MODEL_LEN_MAX_PREFILL: "66000"
        MODEL_LEN_MAX_DECODE: "33792"
        MODEL_PATH: "/data/models/DeepSeek-R1-w8a8-fusion"  ----删除
        MODEL_PATH_P: "/data/models/DeepSeek-R1-sszs50g0a0b4sym1-0915"
        MODEL_PATH_D: "/data/models/DeepSeek-R1-cint8"

        DOCKER_IMAGE_ID: "REPOSITORY:TAG"---删除
        DOCKER_IMAGE_ID_P:"REPOSITORY:TAG"
        DOCKER_IMAGE_ID_D:"REPOSITORY:TAG"

    vars:
        docker_run_cmd: |
            docker run -it --shm-size=500g \
                -e GIT_SSL_NO_VERIFY=1 \
    
        run_vllm_server_prefill_cmd: |
            HCCL_BUFFSIZE=20

            EXTRA_ARGS='--max-num-batched-tokens 65600 --enforce-eager --enable-expert-parallel --disable-log-requests --max-num-seqs 32 --no-enable-prefix-caching'
            GPU_UTIL=0.97

            bash /workspace/omniinfer/tools/scripts/pd_run.sh \
                --vllm-llmdatadist-zmq-port $VLLM_LLMDATADIST_ZMQ_PORT \
                --log-dir "${LOG_PATH}" > ${LOG_PATH}/run_prefill.log 2>&1 &

        run_vllm_server_decode_cmd: |
            ADDITIONAL_CONFIG='{"graph_model_compile_config":{"level":1, "use_ge_graph_cached":true}}'

            pip uninstall omni_placement -y

            if [[ -e "/usr/local/Ascend/ascend-toolkit" ]]; then
                python /workspace/omniinfer/tools/scripts/process_nz_config.py /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json > ${LOG_PATH}/nz.log
            else
                python /workspace/omniinfer/tools/scripts/process_nz_config.py /usr/local/Ascend/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json > ${LOG_PATH}/nz.log
            fi

            bash /workspace/omniinfer/tools/scripts/pd_run.sh \
                --log-dir "${LOG_PATH}" > ${LOG_PATH}/run_decode.log 2>&1 &

        run_proxy_cmd: |
            bash global_proxy.sh \
                --log-file ${LOG_PATH}/nginx_error.log \ 
                --prefill-max-num-seqs 32 \

        start_docker_cmd_p: >     ----删除
            {{ docker_rum_cmd }}    ----删除
            -d --name $DOCKER_NAME_P $DOCKER_IMAGE_ID_P    ----删除

        start_docker_cmd_pa: >
            {{ docker_rum_cmd }}
            -e VLLM_LLMDATADIST_ZMQ_PORT=5570
            -e LOG_PATH=$LOG_PATH_A
            -d --name $DOCKER_NAME_P $DOCKER_IMAGE_ID_P

        start_docker_cmd_pb: >
            {{ docker_rum_cmd }}
            -e VLLM_LLMDATADIST_ZMQ_PORT=5569
            -e LOG_PATH=$LOG_PATH_B
            -d --name $DOCKER_NAME_P $DOCKER_IMAGE_ID_P

        start_docker_cmd_d: >
            -e LOG_PATH=$LOG_PATH
            -d --name $DOCKER_NAME_D $DOCKER_IMAGE_ID_D

        start_docker_cmd_c: >
            -e LOG_PATH=$LOG_PATH
            -d --name $DOCKER_NAME_C $DOCKER_IMAGE_ID_D

        docker_start_vllm_cmd_p: >
            -e MODEL_PATH=$P_MODEL_PATH

        docker_start_vllm_cmd_d: >
            -e MODEL_PATH=$D_MODEL_PATH

    tasks:
        - name: Run container for prefill instances.      ---删除
            command: bash -c "{{ start_docker_cmd_p }}"     ---删除
            environment:      ---删除
                DOCKER_NAME_P: "{{ ACTUAL_DOCKER_NAME_P }}"     ---删除
            when: "'P' in group_names"    ---删除

        - name: Run container for prefill A instances.
            command: bash -c "{{ start_docker_cmd_pa }}"
            environment: 
                DOCKER_NAME_P: "{{ ACTUAL_DOCKER_NAME_P }}"  
            when: "'PA' in group_names"
            tags: run_docker

        - name: Run container for prefill B instances.
            command: bash -c "{{ start_docker_cmd_pb }}"
            environment: 
                DOCKER_NAME_P: "{{ ACTUAL_DOCKER_NAME_P }}"  
            when: "'PB' in group_names"
            tags: run_docker

        - name: Create a directory to store the log.    ---删除
            ansible.builtin.file:    ---删除
                path: "{{ ansible_env.LOG_PATH }}/{{ inventory_hostname }}"    ---删除
                state: directory          ---删除
            when: "'P' in group_names or 'D' in group_names or 'C' in group_names"   ---删除

        - name: Create a directory to store the log.
            ansible.builtin.file:
                path: "{{ ansible_env.LOG_PATH }}"
                state: directory
            when: "'D' in group_names or 'C' in group_names"
            tags: run_docker

        - name: Create a directory to store the log.
            ansible.builtin.file:
                path: "{{ ansible_env.LOG_PATH_A }}"
                state: directory
            when: "'PA' in group_names"
            tags: run_docker

        - name: Create a directory to store the log.
            ansible.builtin.file:
                path: "{{ ansible_env.LOG_PATH_B }}"
                state: directory
            when: "'PB' in group_names"
            tags: run_docker 

        - name: The executor synchronizes code to all instances.
            when : >
                 'PA' in group_names or
                 'P' in group_names or     ---删除

        - name: Register all values.
            set_fact:
                PREFILL_POD_NUM: >-
                    unique |   ----删除
            
        - name: Register values for prefill.
            set_fact:
                MODE_IP_LIST: >-
                {{
                    (groups[ 'PA' ] + groups[ 'PB ]) |
                    groups[ 'P' ] |     ---删除
                }}
               NNODES: >-
                {{
                    (groups[ 'PA' ] + groups[ 'PB' ]) |
                    groups[ 'P' ] |     ---删除
                }}

        - name: Display all values.
            debug:
                msg: |

                    MODE_IP_LIST:{{ MODE_IP_LIST | default('') }}
                    NNODES: {{ NNODES | default('') }}

        - name: Delete the the ranktable files.
            when: "'PA' in group_names or 'D' in group_names"

        - name: Set mooncake & etcd configuration.
            when: "'P' in group_names or 'D' in group_names or 'C' in group_names"

        - name: Kill Mooncake master pod.
            when: "'C' in group_names"

        - name: Kill etcd pod.
            when: "'C' in group_names"

        - name: Generate a mooncake config json.
            when: "'P' in group_names or 'D' in group_names"

        - name: Generate a lmcache + mooncake config yaml.
            when: "'P' in group_names or 'D' in group_names"

        - name: Run etcd for Mooncake.
            when: "'C' in group_names"

        - name: Run master port of Mooncake.
            when: "'C' in group_names"

        - name: Wait 20 seconds.
            when: "'P' in group_names and (NODE_IP_LIST | string).split(',') | length >= 2"

        - name: Wait 45 seconds.
            when: "'P' in group_names and (NODE_IP_LIST | string).split(',') | length == 1"

        - name: Wait for vLLM service to be ready in container D
            ansible.builtin.file:
                cmd: |
                    timeout=300
                    whlie [ $timeout -gt 0 ]; do
                        if docker exec $DOCKER_NAME_D grep -q "Application startup complete" ${LOG_PATH}//server_0.log; then
                    ......

        - name: Forward logs from all machines to the executor.         ---删除
            ansible.builtin.synchronize:          ---删除
                mode: pull           ---删除
                src: "{{ ansible_env.LOG_PATH }}/{{ inventory_hostname }}/"           ---删除
                dest: "{{ ansible_env.LOG_PATH_IN_EXECUTOR }}/{{ inventory_hostname }}/"              ---删除
            when: "'P' in group_names or 'D' in group_names or 'C' in group_names"          ---删除
            tags:              ---删除
                - fetch_log       ----删除

        - name: Forward logs from all machines to the executor.
            ansible.builtin.synchronize:
                mode: pull
                src: "{{ ansible_env.LOG_PATH }}/"
                dest: "{{ ansible_env.LOG_PATH_IN_EXECUTOR }}/{{ inventory_hostname }}/"
            when: "'D' in group_names or 'C' in group_names"
            tags:
                - fetch_log

        - name: Forward logs from all machines to the executor.
            ansible.builtin.synchronize:
                mode: pull
                src: "{{ ansible_env.LOG_PATH_A }}/"
                dest: "{{ ansible_env.LOG_PATH_IN_EXECUTOR }}/{{ inventory_hostname }}/"
            when: "'PA' in group_names or 'C' in group_names"
            tags:
                - fetch_log

        - name: Forward logs from all machines to the executor.
            ansible.builtin.synchronize:
                mode: pull
                src: "{{ ansible_env.LOG_PATH_B }}/"
                dest: "{{ ansible_env.LOG_PATH_IN_EXECUTOR }}/{{ inventory_hostname }}/"
            when: "'PB' in group_names or 'C' in group_names"
            tags:
                - fetch_log
                
    ``` 


## 7、DeepSeek-R1-int4-A2-2P8-1D32

### 7.1 配置 omniinfer/tests/test_config/config_d.yaml
    ```
    enable_dynamic:False

    ``` 

### 7.2 配置 omniinfer/tests/test_config/config_p.yaml
    ```
    pattern_path: "../../tests/test_config/placement_pattern_20250904_2k2k_58_rearrange_layers_58_layers_8_ranks_prefill.npy"

    ``` 

### 7.3 配置 omniinfer/tests/test_config/test_config_prefill_a2_w4a8.json
    ```
    "moe_multi_stream_tune":false,
    "two_stage_comn":false,`

    ``` 

### 7.4 配置 omniinfer/tools/ansible/template/omni_infer_server_template.yml
    ```
    environment:
        MODEL_PATH_P: "/data/models/DeepSeek-R1-sszs50g0a0b4sym1-0915"
        MODEL_PATH_D: "/data/models/DeepSeek-R1-cint8"
        MODEL_LEN_MAX_PREFILL: "66000"
        MODEL_LEN_MAX_DECODE: "4096"
        CODE_PATH:"/workspace"

        DOCKER_IMAGE_ID_P:"registry-cbu.huawei.com/omniai_omniinfer_test/omni_infer-a2-arm:test_0.5.0_20250920_v5"
        DOCKER_IMAGE_ID_D:"registry-cbu.huawei.com/omniai_omniinfer_test/omni_infer-a2-arm:test_0.5.0_20250920_v5"

    vars:
        docker_run_cmd: |
          -e http_proxy=http://p_atlas:proxy%40123@172.18.100.92:8080 \
          -e https_proxy=http://p_atlas:proxy%40123@172.18.100.92:8080 \
          -e no_proxy=7.222.143.19,10.70.113.33,7.242.104.48,10.*.*.*,127.0.0.1,.huawei.com,localhost,local,0.0.0.0,.inhuawei.com \
          -e GIT_SSL_NO_VERIFY=1 \

          -v /efs_guiyang:/efs_guiyang \

        run_vllm_server_prefill_cmd: |
            HCCL_BUFFSIZE=20
            MODEL_EXTRA_CFG_PATH="/workspace/omniinfer/tests/test_config/test_config_prefill_a2_w4a8.json"
            EXTRA_ARGS='--max-num-batched-tokens 66000 --enforce-eager --enable-expert-parallel --disable-log-requests --max-num-seqs 32 --no-enable-prefix-caching'
            GPU_UTIL=0.97
            export ASCEND_PLATFORM=A2

        run_vllm_server_decode_cmd: |
            HCCL_BUFFSIZE=20
            MODEL_EXTRA_CFG_PATH="/workspace/omniinfer/tests/test_config/test_config_decode_a2.json"
            EXTRA_ARGS='--enable-expert-parallel --disable-log-requests --max-num-seqs 92 --no-enable-prefix-caching'
            GPU_UTIL=0.965
            ADDITIONAL_CONFIG='{"graph_model_compile_config":{"level":1, "use_ge_graph_cached":true, "block_num_floating_range":50}, "enable_omni_attn":true}'
            VLLM_ENABLE_MC2=0

            export ASCEND_PLATFORM=A2

            python /workspace/omniinfer/tools/scripts/process_nz_config.py /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json     ----注释掉

        run_proxy_cmd: |
            cd /workspace/omniinfer/omni/accelerators/sched/global_proxy

            bash build.sh

              --prefill-max-num-seqs 32 \

        generate_lmcache_mooncake_config_yml:
        start_docker_cmd_p: >
            {{ docker_rum_cmd }}
            -d --name $DOCKER_NAME_P $DOCKER_IMAGE_ID_P

        start_docker_cmd_d: >
            {{ docker_rum_cmd }}
            -d --name $DOCKER_NAME_D $DOCKER_IMAGE_ID_D

        start_docker_cmd_c: >
            {{ docker_rum_cmd }}
            -e PROXY_NODE_PORT=$NODE_PORT
            -d --name $DOCKER_NAME_D $DOCKER_IMAGE_ID_D

        docker_start_vllm_cmd_p: >
            {{ docker_exec_cmd }}
            -e MODEL_PATH=$MODEL_PATH_P

        docker_start_vllm_cmd_d: >
            {{ docker_exec_cmd }}
            -e MODEL_PATH=$MODEL_PATH_D

        docker_cp_proxy_code_cmd:>    ----注释掉
          [ "${KV_CONNECTOR}"!="LMCacheConnectorV1" ] || { docker cp {{ ansible_env.CODE_PATH }}/omniinfer $DOCKER_NAME_C:/workspace/;}    ----注释掉
        docker_cp_proxy_code_cmd:"docker cp {{ ansible_env.CODE_PATH }}/omniinfer $DOCKER_NAME_C:/workspace/"  

        docker_update_prefill_code_cmd: >
            {{ docker_exec_cmd }}
            -e KV_CONNECTOR=$KV_CONNECTOR
            $DOCKER_NAME_P
            /bin/bash -c '.~/.bashrc && pip config unset global.index-url && cd /workspace/omniinfer/infer_engines && git config --global --add safe.directory /workspace/omniinfer/infer_engines/vllm && cd vllm && git checkout -f && cd .. && bash bash_install_code.sh && pip uninstall vllm -y && pip install omni infer -y && cd vllm && SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e . --no-deps --no-build-isolation && cd ../../ && pip install -e . --no-deps --no-build-isolation && pip uninstall numpy -y && pip uninstall numpy==1.26 --no-deps --no-build-isolation > ${LOG_PATH}/${{inventory_hostname}}/pip.log  && [ "${KV_CONNECTOR}"!="LMCacheConnectorV1" ] || { source ~/.bashrc && cd /workspace/omniinfer/omni/adaptors/lmcache/script && bash install.sh &> $${LOG_PATH}/install.log;}'

        docker_update_decode_code_cmd: >
            {{ docker_exec_cmd }}
            -e KV_CONNECTOR=$KV_CONNECTOR
            $DOCKER_NAME_D
            /bin/bash -c '.~/.bashrc && pip config unset global.index-url && cd /workspace/omniinfer/infer_engines && git config --global --add safe.directory /workspace/omniinfer/infer_engines/vllm && cd vllm && git checkout -f && cd .. && bash bash_install_code.sh && pip uninstall vllm -y && pip install omni infer -y && cd vllm && SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e . --no-deps --no-build-isolation && cd ../../ && pip install -e . --no-deps --no-build-isolation && pip uninstall numpy -y && pip uninstall numpy==1.26 --no-deps --no-build-isolation > ${LOG_PATH}/${{inventory_hostname}}/pip.log  && [ "${KV_CONNECTOR}"!="LMCacheConnectorV1" ] || { source ~/.bashrc && cd /workspace/omniinfer/omni/adaptors/lmcache/script && bash install.sh &> $${LOG_PATH}/install.log;}'

    ``` 


## 8、DeepSeek-R1-BF16-A3-2P32-1D32

### 8.1 配置 omniinfer/tests/test_config/config_p.yaml
      
    ```
    pattern_path: "../../tests/test_config/ep_ds_r1_bf16_a3_2p1d_p.npy"

    Optimizers:
        - expert_balance_optimizer.ExpertsBalanceOptimizer:
            batch_size: 16

    ```

### 8.2 配置 omniinfer/tests/test_config/test_prefill_prefill_bf16.json
    ```
    "model_parallel_config": {
        "dense_mlp_tp_size": 1,
        "o_proj_tp_size": 2,
        "dp_size": 1
    },
    "operator_optimizition_config": {
        "unquant_bmm_nz": true,
        "decode_moe_dispatch_combine": false,
        "prefill_enable_mla_alltoall": true,
        "use_omni_placement": true,
        "omni_placement_config_path": "../../tests/test_config/config_p.yaml",
        "enable_mc2_v2": false,
        "expert_pruning": false
    }

    ```

### 8.3 配置 omniinfer/tests/test_config/test_config_decode_bf16.json                 
    ```
    "model_parallel_config": {
        "dense_mlp_tp_size": 16,
        "o_proj_tp_size": 8,
        "dp_size": 64
    },
    "operator_optimizition_config": {
        "unquant_bmm_nz": true,
        "use_omni_placement": false,
        "prefill_enable_mla_alltoall": false,
        "use_mlaprolog": true,
        "enable_mc2_v2": true,
        "expert_gate_up_prefetch": 30
    }

    ```

### 8.4 配置 omniinfer/tools/scripts/pd_run.sh 
    ```
    export CPU_AFFINITY_CONF=2
    ```
### 8.5 配置 omniinfer/tools/ansible/template/omni_infer_server_template.yml  
    ```
    environment:
        MODEL_LEN_MAX_PREFILL: "70400"
        MODEL_LEN_MAX_DECODE: "40000"
        CODE_PATH: "/workspace"

        PREFILL_LB_SDK: "least_total_load"
        DECODE_LB_SDK: "weighted_least_active"

    vars:
        run_vllm_server_prefill_cmd: |
            HCCL_BUFFSIZE=200
            export HCCL_CONNECT_TIMEOUT=600
            export HCCL_EXEC_TIMEOUT=600
            export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

            MODEL_EXTRA_CFG_PATH="/workspace/omniinfer/tests/test_config/test_prefill_prefill_bf16.json"
            EXTRA_ARGS='--max-num-batched-tokens 70400 --enforce-eager --no-enable-prefix-caching --enable-expert-parallel --disable-log-requests --max-num-seqs 128 --scheduler-cls omni.adaptors.vllm.worker.tfas.tfas_scheduler.TFASScheduler'
            GPU_UTIL=0.90

            export HCCL_OP_EXPANSION_MODE="AIV"

            export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True    ---删除

            ADDITIONAL_CONFIG='{"tfas_scheduler_config": {"adjust_param": 8.708, "token_budget": 38000}}'
            uset https_proxy
            uset http_proxy
            uset proxy
            export HCCL_OP_EXPANSION_MODE="AIV"
            export CPU_AFFINITY_CONF=2
            export OMNI_REUSE_PREFILLED_TOKENS=1
            export OMNI_SKIP_DECODE_TOKENIZE=1
            export TOKENIZER_PROC_POOL=1  

            bash /workspace/omniinfer/tools/scripts/pd_ru.sh \
              --additional-config "$ADDITIONAL_CONFIG" \

        run_vllm_server_decode_cmd: |
            HCCL_BUFFSIZE=200
            export HCCL_CONNECT_TIMEOUT=600
            export HCCL_EXEC_TIMEOUT=600
            export HCCL_DETERMINISTIC=false 

            MODEL_EXTRA_CFG_PATH="/workspace/omniinfer/tests/test_config/test_config_decode_bf16.json"
            EXTRA_ARGS='--enable-expert-parallel --disable-log-requests --max-num-seqs 4 --no-enable-prefix-caching'
            ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1, "use_ge_graph_cached":true}}'

            export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True      
            export HCCL_OP_EXPANSION_MODE="AIV"
            export CPU_AFFINITY_CONF=2
            export OMNI_REUSE_PREFILLED_TOKENS=1
            export OMNI_SKIP_DECODE_TOKENIZE=1
            export TOKENIZER_PROC_POOL=1

        run_proxy_cmd: |

            cd /workspace/omniinfer/omni/accelerators/sched/global_proxy
            sh build.sh

        docker_cp_decode_code_cmd: "docker cp {{ ansible_env.CODE_PATH }}/omniinfer $DOCKER_NAME_D:/workspace/"
        docker_cp_proxy_code_cmd: "docker cp {{ ansible_env.CODE_PATH }}/omniinfer $DOCKER_NAME_C:/workspace/"

        docker_cp_proxy_code_cmd: >    ----删除
           [ "${KV_CONNECTOR}" != "LMCacheConnectorV1" ] || { docker cp {{ ansible_env.CODE_PATH }}/omniinfer $DOCKER_NAME_C:/workspace/; }    ----删除

    ```


## 9、DeepSeek-V3-INT4-A3-8P4-1D32

### 9.1 配置 omniinfer/tests/test_config/config_p.yaml
    ```
    pattern_path: "../../omni/accelerators/placement/patterns/MoE_placement_prefill_node_8_extra_expert_per_node_0_stoptime_1.npy"
    ```
### 9.2 配置 omniinfer/tests/test_config/config_d.yaml
    ```
    pattern_path: "../../omni/accelerators/placement/patterns/base_patterns/DSV3_baseline_64_devices_58_MoE_Layers.npy"
    max_redundant_per_expert: 10 # 10
    max_redundant_per_rank: 1 # 1
    ```
### 9.3 配置 omniinfer/tests/test_config/test_config_decode.json
    ```
     {
        "operator_optimizition_config": {
             "enable_mc2_v2": true
        }
     }
    ```     
### 9.4 配置 omniinfer/tools/ansible/template/omni_infer_server_template_a3_1machine2p.yaml
    ```
    environment:
        MODEL_PATH: "/data/models/DeepSeek-R1-w8a8-fusion"---删除
        MODEL_PATH_P: "/data/models/D/DeepSeek_INT4_P"
        MODEL_PATH_D: "/data/models/D/DeepSeek_INT4_D"
        MODEL_LEN_MAX_PREFILL: "33792"
        MODEL_LEN_MAX_PREFILL: "16284"
        DOCKER_IMAGE_ID: "REPOSITORY:TAG"---删除
        DOCKER_IMAGE_ID_P: "REPOSITORY:TAG"
        DOCKER_IMAGE_ID_D: "REPOSITORY:TAG"
        VLLM_LLMDATADIST_ZMQ_PORT1: 5568
        VLLM_LLMDATADIST_ZMQ_PORT2: 5569
    vars:
        run_vllm_server_prefill_cmd: |
            EXTRA_ARGS='--max-num-batched-tokens 33792 --enforce-eager --enable-expert-parallel --disable-log-requests --max-num-seqs 16 --no--enable-prefix-caching'
            PROFILING_NAMELIST=/workspace/omniinfer/omni/adaptors/vllm/patches/profiler_patches/proc_bind/proc_marker_namelist.yml bash /workspace/omniinfer/tools/scripts/pd_run.sh \---删除
            bash ${CODE_PATH}/omniinfer/tools/scripts/pd_run.sh \
                --vllm-llmdatadist-zmq-port $VLLM_LLMDATADIST_ZMQ_PORT \

        run_vllm_server_decode_cmd: |
            ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1, "use_ge_graph_cached":true}}'
            PROFILING_NAMELIST=/workspace/omniinfer/omni/adaptors/vllm/patches/profiler_patches/proc_bind/proc_marker_namelist.yml bash /workspace/omniinfer/tools/scripts/pd_run.sh \---删除
            bash ${CODE_PATH}/omniinfer/tools/scripts/pd_run.sh \

        start_docker_cmd_pa: >
            -e VLLM_LLMDATADIST_ZMQ_PORT=5570
            -d --name $DOCKER_NAME_P $DOCKER_IMAGE_ID_P
        start_docker_cmd_pb: >
            -e VLLM_LLMDATADIST_ZMQ_PORT=5569
            -d --name $DOCKER_NAME_P $DOCKER_IMAGE_ID_P
        start_docker_cmd_d: >
            -d --name $DOCKER_NAME_D $DOCKER_IMAGE_ID_D
        start_docker_cmd_c: >
            -d --name $DOCKER_NAME_C $DOCKER_IMAGE_ID_P
        docker_start_vllm_cmd_p: >
            -e MODEL_PATH=$MODEL_PATH_P
        docker_start_vllm_cmd_d: >
            {{ docker_exec_cmd }}
            -e MODEL_PATH=$MODEL_PATH_D
        docker_cp_prefill_code_cmd: "docker cp {{ ansible_env.CODE_PATH }}/omniinfer $DOCKER_NAME_P:/workspace/"
    
    tasks:
        - Create a directory to store the log.
            ansible.builtin.file:
                path: "{{ansible_env.LOG_PATH}}"
        - name: Copy the code from the host machine into the container (prefill).
            command: bash -c "{{ docker_cp_prefill_code_cmd }}"
            environment: 
                DOCKER_NAME_P: "{{ ACTUAL_DOCKER_NAME_P }}"
            when: "'P' in group_names"
            tags: sync_code

        - name: Copy the code from the host machine into the container (decode).
            command: bash -c "{{ docker_cp_decode_code_cmd }}"
            environment: 
                DOCKER_NAME_D: "{{ ACTUAL_DOCKER_NAME_D }}"
            when: "'D' in group_names"
            tags: sync_code
        
        - name: docker_update_prefill_code_cmd.
            tags: pip_install
        
        - name: docker_update_decode_code_cmd.
            tags: pip_install
        
        - name: Create a directory on the excutor to store ranktable file.
            tags:
                - ranktable
        - name: Delete the the ranktable files.
            tags:
                -ranktable

        - name: Delete the the ranktable files.
            when: "'PA' in group_names or 'D' in group_names"

        - name: Generate a script to kill all Python process in the container.
            tags: stop_server

        - name: Generate a script to kill all Ray process in the container.
            tags: stop_server

        - name: Kill all Python process in the container of prefill.
            tags: stop_server

        - name: Kill all Ray process in the container of prefill.
            tags: stop_server

        - name: Kill all Ray process in the container of decode.
            tags: stop_server

        - name: Remove proc_trace.txt if it exists (P & D nodes)
            ansible.builtin.file:
                path: /tmp/process/proc_trace.txt
                state: absent
            when: "'P' in group_names or 'D' in group_names"
            tags:
                - run_server
        
        - name: check vLLM is Ready
            shell: |
                timeout=600
                while [ $timeout -gt 0 ]; do
                if grep -q "Application startup complete" {{ ansible_env.LOG_PATH }}/server_0.log; then
                    echo "Service ready"
                    exit 0
                fi
                sleep 20
                timeout=$((timeout - 20))
                done
                echo "Timeout waiting for service" >&2
                exit 0
            delegate_to: "{{ inventory_hostname }}"
            become: yes
            when: 
                - "'P' in group_names or 'D' in group_names"
                - "proc_bind_enabled | default(false)"
            tags:
                - proc_bind
            
        - name: Ensure bind_cpu.sh is executable
            file:
                path: ${CODE_PATH}/omniinfer/tools/scripts/bind_cpu.sh
                mode: "0755"
            become: yes
            when:
                - "proc_bind_enabled | default(false)"
            tags:
                - proc_bind
            
        - name: Prefill cpu_pin
            command: /bin/bash -c "{{ docker_exec_cmd }} $DOCKER_NAME_P /bin/bash -c /workspace/omniinfer/tools/scripts/bind_cpu.sh"
            environment:
                DOCKER_NAME_P: "{{ ACTUAL_DOCKER_NAME_P }}"
                SCRIPTS_PATH: "{{ ansible_env.SCRIPTS_PATH }}"
                ROLE: "P"
            become: yes
            when: 
                - "'P' in group_names"
                - "proc_bind_enabled | default(false)"
            tags:
                - proc_bind
            
        - name: Decode cpu_pin
            command: /bin/bash -c "{{ docker_exec_cmd }} $DOCKER_NAME_D /bin/bash -c /workspace/omniinfer/tools/scripts/bind_cpu.sh"
            environment:
                DOCKER_NAME_D: "{{ ACTUAL_DOCKER_NAME_D }}"
                SCRIPTS_PATH: "{{ ansible_env.SCRIPTS_PATH }}"
                ROLE: "D"
            become: yes
            when: 
                - "'D' in group_names"
                - "proc_bind_enabled | default(false)"
            tags:
                - proc_bind

        - name: Create a directory on the executor to store the log.
            when: "'P' in group_names or 'D' in group_names"

    ```

# 二、Qwen模型部署  

## 1、Qwen2.5-7B-BF16-A2/A3-单机混部  

### 1.1 启动脚本：
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
    export ASCEND_GLOBAL_LOG_LEVEL=3

    python3 -m vllm.entrypoints.openai.api_server \
        --host *.***.*.*** \
        --port ***** \
        --model /***/***/****** \
        --data-parallel-size 1 \
        --tensor-parallel-size 1 \
        --dtype bfloat16 \
        --max-model-len 32768 \
        --trust_remote_code \
        --gpu_memory_utilization 0.9 \
        --enforce-eager \
        --block_size 128 \
        --served-model-name qwen \
        --distributed-executor-backend mp \
        --max-num-batched-tokens 20000 \
        --max-num-seqs 128 \
        --additional-config '{"enable_hybrid_graph_mode": true, "graph_model_compile_config": {"level":1}}' 

    ```


## 2、QwQ-32B-BF16-A2-3P8-1D8/A3-3P4-1D4
### 2.1 配置 omniinfer/tools/ansible/template/omni_infer_server_template.yml   
    ```
    environment:
        MODEL_LEN_MAX_PREFILL: "62144"
        MODEL_LEN_MAX_DECODE: "62144"
        DECODE_TENSOR_PARALLEL_SIZE: "8"
    vars:
        ASCEND_TOOLKIT_HOME: "/usr/local/Ascend/ascend-toolkit/lastest"

        docker_run_cmd: |
            docker run -it --shm-size=500g \
                -e CODE_PATH=$CODE_PATH \
                -v /etc/resolv.conf:/etc/resolv.conf \
    
        generate_prefill_ranktable_cmd: |
            python ${CODE_PATH}/omniinfer/tools/scripts/pd_ranktable_tools.py --mode gen --prefill-server-list "${PREFILL_SERVER_LIST}" --api-server --save-dir ${PREFILL_RANKTABLE_SAVE_PATH}

        generate_decode_ranktable_cmd: |
            python ${CODE_PATH}/omniinfer/tools/scripts/pd_ranktable_tools.py --mode gen --decode-server-list ${DECODE_SERVER_LIST} --save-dir ${DECODE_RANKTABLE_SAVE_PATH}

        generate_global_ranktable_cmd: |
            for i in "${!parts[@]}"; do
                if [ "$comma_count" -ge 1 ]; then
                    python ${CODE_PATH}/omniinfer/tools/scripts/pd_ranktable_tools.py \
            if [ $DECODE_POD_NUM -gt 1 ]; then
                python ${CODE_PATH}/omniinfer/tools/scripts/pd_ranktable_tools.py \
            python ${CODE_PATH}/omniinfer/tools/scripts/pd_ranktable_tools.py \
    
        run_vllm_server_prefill_cmd: |
            export HCCL_CONNECT_TIMEOUT=1800---删除
            export HCCL_EXEC_TIMEOUT=1800---删除
            export LOG_PATH=${LOG_PATH}/${HOST}
            mkdir -p ${LOG_PATH}
            MODEL_EXTRA_CFG_PATH="/workspace/omniinfer/tests/test_config/test_config_prefill.json"---删除
            EXTRA_ARGS='--max-num-batched-tokens 61440 --enforce-eager --disable-log-requests --max-num-seqs 64'
            GPU_UTIL=0.92---删除
            VLLM_ENABLE_MC2=1---删除
            HCCL_OP_EXPANSION_MODE="AIV"---删除
            export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True---删除
            ACCELERATE_ID=cc_kvstore@_@ds_default_ns_001---删除
            if [ $(echo -n "$NODE_IP_LIST" | tr -cd ',' | wc -c) -ge 1 ]; then
                export NNODES=$NNODES---删除
                export NODE_RANK=$---删除
                export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1---删除
                export RAY_CGRAPH_get_timeout=7200---删除
                ray stop --force---删除
                EXTRA_ARGS="${EXTRA_ARGS} --distributed-executor-backend ray"---删除
            pip uninstall omni_placement -y---删除
            cd /workspace/omniinfer/tools/scripts---删除
            PROFILING_NAMELIST=/workspace/omniinfer/omni/adaptors/vllm/patches/profiler_patches/proc_bind/proc_marker_namelist.yml bash /workspace/omniinfer/tools/scripts/pd_run.sh \---删除
                --global-rank-table-path "${RANKTABLE_SAVE_PATH}/global/global_ranktable_merge.json" \---删除
                --rank-table-path ${LOCAL_RANKTABLE_FLIE} \---删除
                --local-decode-server-ip-list "$SERVER_IP_LIST" \---删除
                --global-decode-server-ip-list "$SERVER_IP_LIST" \---删除
                --prefill-pod-num ${PREFILL_POD_NUM} \---删除
                --gloo-socket-ifname ${SOCKET_IFNAME} \---删除
                --tp-socket-ifname ${SOCKET_IFNAME} \---删除
                --model-path ${MODEL_PATH} \---删除
                --master-ip ${HOST_IP} \---删除
                --role "prefill" \---删除
                --kv-role "kv_producer" \---删除
                --max-model-len ${MODEL_LEN_MAX_PREFILL} \---删除
                --master-port ${MASTER_PORT} \---删除
                --base-api-port ${API_PORT} \---删除
                --tp ${PREFILL_TENSOR_PARALLEL_SIZE} \---删除
                --ascend-rt-visible-devices "${PREFILL_SERVER_LIST}" \---删除
                --kv-rank ${KV_RANK} \---删除
                --kv-engine-id ${KV_RANK} \---删除
                --kv-parallel-size ${KV_PARALLEL_SIZE} \---删除
                --model-extra-cfg-path ${MODEL_EXTRA_CFG_PATH} \---删除
                --gpu-util ${GPU_UTIL} \---删除
                --vllm-enable-mc2 ${VLLM_ENABLE_MC2} \---删除
                --extra-args "${EXTRA_ARGS}" \---删除
                --hccl-buffsize "${HCCL_BUFFSIZE}" \---删除
                --hccl-op-expansion-mode "${HCCL_OP_EXPANSION_MODE}" \---删除
                --log-dir "${LOG_PATH}/{{ inventory_hostname }}" > ${LOG_PATH}/{{ inventory_hostname }}/run_prefill.log 2>&1 &---删除
            GPU_UTIL=0.88
            export HCCL_CONNECT_TIMEOUT=1800
            export HCCL_EXEC_TIMEOUT=1800
            export VLLM_LLMDATADIST_ZMQ_PORT=${ZMQ_PORT}
            export ASCEND_RT_VISIBLE_DEVICES=${PREFILL_SERVER_LIST}
            export GLOBAL_RANK_TABLE_FILE_PATH="${RANKTABLE_SAVE_PATH}/global/global_ranktable_merge.json"
            export RANK_TABLE_FILE_PATH=${LOCAL_RANKTABLE_FLIE}
            export LOCAL_DECODE_SERVER_IP_LIST="$SERVER_IP_LIST"
            export GLOBAL_DECODE_SERVER_IP_LIST="$SERVER_IP_LIST"
            export ROLE="prefill"
            export HCCL_INTRA_ROCE_ENABLE=1
            export HCCL_INTRA_PCIE_ENABLE=0
            # export HCCL_DETERMINISTIC=true
            # export CLOSE_MATMUL_K_SHIFT=1
            export ASCEND_LAUNCH_BLOCKING=0
            export PREFILL_POD_NUM=${PREFILL_POD_NUM}
            export DECODE_POD_NUM=1
            source /usr/local/Ascend/nnal/atb/set_env.sh
            export GLOO_SOCKET_IFNAME=${SOCKET_IFNAME}
            export TP_SOCKET_IFNAME=${SOCKET_IFNAME}
            export VLLM_USE_V1=1
            export VLLM_WORKER_MULTIPROC_METHOD=fork
            export VLLM_LOGGING_LEVEL=INFO
            export OMNI_USE_QWEN=1
            export OMNI_REUSE_PREFILLED_TOKENS=1
            export OMNI_SKIP_DECODE_TOKENIZE=1
            ADDITIONAL_CONFIG='{"multi_rank_pull_kv": true}'
            # export VLLM_ENABLE_MC2=1 # Qwen P seem not enable?
            # HCCL_OP_EXPANSION_MODE="AIV"
            # export HCCL_BUFFSIZE=1000
            echo "TP=${PREFILL_TENSOR_PARALLEL_SIZE}" > ${LOG_PATH}/run_prefill.log
            echo "PREFILL_POD_NUM=${PREFILL_POD_NUM}" >> ${LOG_PATH}/run_prefill.log
            echo "DECODE_POD_NUM=${DECODE_POD_NUM}" >> ${LOG_PATH}/run_prefill.log
            cd ${CODE_PATH}/omniinfer/tools/scripts
            python start_api_servers.py \
                --num-servers 1 \
                --model-path ${MODEL_PATH} \
                --master-ip ${HOST_IP} \
                --master-port ${MASTER_PORT} \
                --base-api-port ${API_PORT} \
                --tp ${PREFILL_TENSOR_PARALLEL_SIZE} \
                --served-model-name qwen \
                --max-model-len ${MODEL_LEN_MAX_PREFILL} \
                --no-enable-prefix-caching \
                --no-enable-chunked-prefill \
                --gpu-util $GPU_UTIL \
                --extra-args "${EXTRA_ARGS}" \
                --kv-transfer-config '{
                "kv_connector": "AscendHcclConnectorV1",
                "kv_buffer_device": "npu",
                "kv_role": "kv_producer",
                "kv_rank": '"${KV_RANK}"',
                "engine_id": '"${KV_RANK}"',
                "kv_parallel_size": '"${KV_PARALLEL_SIZE}"'
                }' \
                --log-dir "${LOG_PATH}" >> ${LOG_PATH}/run_prefill.log 2>&1 &

        run_vllm_server_decode_cmd: |
            HCCL_BUFFSIZE=1000---删除
            export HCCL_CONNECT_TIMEOUT=1800---删除
            export HCCL_EXEC_TIMEOUT=1800---删除
            export LOG_PATH=${LOG_PATH}/${HOST}
            mkdir -p ${LOG_PATH}
            dp=$(echo -n "$DECODE_DATA_PARALLEL_SIZE" | tr -cd ',' | wc -c)---删除
            ((dp++))---删除
            EXTRA_ARGS='--max-num-batched-tokens 61440 --disable-log-requests --max-num-seqs 64'
            ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1}, "decode_gear_list": [ 64 ]}'
            DP=1
            MODEL_EXTRA_CFG_PATH="/workspace/omniinfer/tests/test_config/test_config_decode.json"---删除
            EXTRA_ARGS='--enable-expert-parallel --disable-log-requests --max-num-seqs 32 --no-enable-prefix-caching'---删除
            GPU_UTIL=0.92---删除
            ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1}}'---删除
            VLLM_ENABLE_MC2=1---删除
            HCCL_OP_EXPANSION_MODE="AIV"---删除
            export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True---删除
            unset https_proxy---删除
            unset http_proxy---删除
            unset proxy---删除
            pip uninstall omni_placement -y---删除
            python /workspace/omniinfer/tools/scripts/process_nz_config.py /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json---删除
            cd /workspace/omniinfer/tools/scripts---删除
            PROFILING_NAMELIST=/workspace/omniinfer/omni/adaptors/vllm/patches/profiler_patches/proc_bind/proc_marker_namelist.yml bash /workspace/omniinfer/tools/scripts/pd_run.sh \---删除
                --global-rank-table-path "${RANKTABLE_SAVE_PATH}/global/global_ranktable_merge.json" \---删除
                --rank-table-path ${LOCAL_RANKTABLE_FLIE} \---删除
                --local-decode-server-ip-list "$SERVER_IP_LIST" \---删除
                --global-decode-server-ip-list "$SERVER_IP_LIST" \---删除
                --prefill-pod-num ${PREFILL_POD_NUM} \---删除
                --gloo-socket-ifname ${SOCKET_IFNAME} \---删除
                --tp-socket-ifname ${SOCKET_IFNAME} \---删除
                --num-servers ${NUM_SERVERS} \---删除
                --num-dp ${dp} \---删除
                --server-offset ${config_dict[$HOST]:-0} \---删除
                --model-path ${MODEL_PATH} \---删除
                --master-ip ${HOST_IP} \---删除
                --role "decode" \---删除
                --kv-role "kv_consumer" \---删除
                --max-model-len ${MODEL_LEN_MAX_DECODE} \---删除
                --master-port ${MASTER_PORT} \---删除
                --base-api-port ${API_PORT} \---删除
                --tp ${DECODE_TENSOR_PARALLEL_SIZE} \---删除
                --kv-rank ${PREFILL_POD_NUM} \---删除
                --kv-engine-id ${PREFILL_POD_NUM} \---删除
                --kv-parallel-size ${KV_PARALLEL_SIZE} \---删除
                --model-extra-cfg-path ${MODEL_EXTRA_CFG_PATH} \---删除
                --gpu-util ${GPU_UTIL} \---删除
                --additional-config "$ADDITIONAL_CONFIG" \---删除
                --vllm-enable-mc2 ${VLLM_ENABLE_MC2} \---删除
                --extra-args "${EXTRA_ARGS}" \---删除
                --hccl-buffsize "${HCCL_BUFFSIZE}" \---删除
                --hccl-op-expansion-mode "${HCCL_OP_EXPANSION_MODE}" \---删除
                --log-dir "${LOG_PATH}/{{ inventory_hostname }}" > ${LOG_PATH}/{{ inventory_hostname }}/run_decode.log 2>&1 &---删除
            GPU_UTIL=0.88
            export HCCL_CONNECT_TIMEOUT=1800
            export HCCL_EXEC_TIMEOUT=1800
            export VLLM_LLMDATADIST_ZMQ_PORT=${ZMQ_PORT}
            export ASCEND_RT_VISIBLE_DEVICES=${DECODE_SERVER_LIST}
            export GLOBAL_RANK_TABLE_FILE_PATH="${RANKTABLE_SAVE_PATH}/global/global_ranktable_merge.json"
            export RANK_TABLE_FILE_PATH=${LOCAL_RANKTABLE_FLIE}
            source /usr/local/Ascend/nnal/atb/set_env.sh
            export LOCAL_DECODE_SERVER_IP_LIST="$SERVER_IP_LIST"
            export GLOBAL_DECODE_SERVER_IP_LIST="$SERVER_IP_LIST"
            export ROLE="decode"
            export HCCL_INTRA_ROCE_ENABLE=1
            export HCCL_INTRA_PCIE_ENABLE=0
            # export HCCL_DETERMINISTIC=true
            # export CLOSE_MATMUL_K_SHIFT=1
            export ASCEND_LAUNCH_BLOCKING=0
            export PREFILL_POD_NUM=${PREFILL_POD_NUM}
            export DECODE_POD_NUM=1
            export GLOO_SOCKET_IFNAME=${SOCKET_IFNAME}
            export TP_SOCKET_IFNAME=${SOCKET_IFNAME}
            export VLLM_USE_V1=1
            export VLLM_WORKER_MULTIPROC_METHOD=fork
            export VLLM_LOGGING_LEVEL=INFO
            export OMNI_USE_QWEN=1
            export OMNI_REUSE_PREFILLED_TOKENS=1
            export OMNI_SKIP_DECODE_TOKENIZE=1
            # decode enable
            export VLLM_ENABLE_MC2=1
            export HCCL_OP_EXPANSION_MODE="AIV"
            export HCCL_BUFFSIZE=1000
            export MOE_DISPATCH_COMBINE=1
            echo "TP=${DECODE_TENSOR_PARALLEL_SIZE}" > ${LOG_PATH}/run_decode.log
            echo "GPU_UTIL=${GPU_UTIL}" >> ${LOG_PATH}/run_decode.log
            echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}" >> ${LOG_PATH}/run_decode.log
            echo "PREFILL_POD_NUM=${PREFILL_POD_NUM}" >> ${LOG_PATH}/run_decode.log
            echo "DECODE_POD_NUM=${DECODE_POD_NUM}" >> ${LOG_PATH}/run_decode.log
            cd ${CODE_PATH}/omniinfer/tools/scripts
            python start_api_servers.py \
                --num-servers 1 \
                --server-offset ${config_dict[$HOST]:-0} \
                --num-dp ${DP} \
                --model-path ${MODEL_PATH} \
                --master-ip ${HOST_IP} \
                --master-port ${MASTER_PORT} \
                --base-api-port ${API_PORT} \
                --tp ${DECODE_TENSOR_PARALLEL_SIZE} \
                --served-model-name qwen \
                --max-model-len ${MODEL_LEN_MAX_DECODE} \
                --no-enable-prefix-caching \
                --no-enable-chunked-prefill \
                --gpu-util $GPU_UTIL \
                --extra-args "${EXTRA_ARGS}" \
                --additional-config "$ADDITIONAL_CONFIG" \
                --kv-transfer-config '{
                "kv_connector": "AscendHcclConnectorV1",
                "kv_buffer_device": "npu",
                "kv_role": "kv_consumer",
                "kv_rank": '"${PREFILL_POD_NUM}"',
                "engine_id": 0,
                "kv_parallel_size": '"${KV_PARALLEL_SIZE}"'
                }' \
                --log-dir "${LOG_PATH}" >> ${LOG_PATH}/run_decode.log 2>&1 &

        run_proxy_cmd: |
            export LOG_PATH=${LOG_PATH}/${HOST}
            mkdir -p ${LOG_PATH}
            cd ${CODE_PATH}/omniinfer/omni/accelerators/sched/global_proxy
            NGINX_VERSION="${NGINX_VERSION:-1.24.0}"
            if [ ! -f "nginx-${NGINX_VERSION}.tar.gz" ]; then
                wget --no-check-certificate "https://nginx.org/download/nginx-${NGINX_VERSION}.tar.gz" > ${LOG_PATH}/install_nginx.log 2>&1
            fi
            tar -zxf "nginx-${NGINX_VERSION}.tar.gz"
            unset http_proxy
            unset https_proxy
            bash build.sh >> ${LOG_PATH}/install_nginx.log 2>&1
            echo "prefill_result=${prefill_result}" > ${LOG_PATH}/env.log
            echo "prefill_result=${decode_result}" >> ${LOG_PATH}/env.log
            cd ${CODE_PATH}/omniinfer/tools/scripts
            bash global_proxy.sh \
                --log-file ${LOG_PATH}/nginx_error.log \ 

        docker_start_vllm_cmd_p: >
            -e ZMQ_PORT=$ZMQ_PORT
            -e HOST=$HOST

        docker_start_vllm_cmd_d: >
            -e ZMQ_PORT=$ZMQ_PORT

        docker_start_proxy_cmd_c: >
            -e http_proxy=$HTTP_PROXY
            -e https_proxy=$HTTP_PROXY
            -e HOST=$HOST

        docker_cp_prefill_code_cmd: "docker cp {{ ansible_env.CODE_PATH }}/omniinfer $DOCKER_NAME_P:/workspace/"---删除

        docker_cp_decode_code_cmd: "docker cp {{ ansible_env.CODE_PATH }}/omniinfer $DOCKER_NAME_D:/workspace/"---删除

        docker_update_prefill_code_cmd: "{{ docker_exec_cmd }} $DOCKER_NAME_P /bin/bash -c 'cd {{ ansible_env.CODE_PATH }}/omniinfer/infer_engines && git config --global --add safe.directory {{ ansible_env.CODE_PATH }}/omniinfer/infer_engines/vllm && cd vllm && git checkout -f && cd .. && bash bash_install_code.sh > ${LOG_PATH}/${HOST}/pip.log'"
        docker_update_decode_code_cmd: "{{ docker_exec_cmd }} $DOCKER_NAME_D /bin/bash -c 'cd {{ ansible_env.CODE_PATH }}/omniinfer/infer_engines && git config --global --add safe.directory {{ ansible_env.CODE_PATH }}/omniinfer/infer_engines/vllm && cd vllm && git checkout -f && cd .. && bash bash_install_code.sh > ${LOG_PATH}/${HOST}/pip.log'"

        docker_install_prefill_code_cmd: "{{ docker_exec_cmd }} $DOCKER_NAME_P /bin/bash -c 'export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/lastest && cd {{ ansible_env.CODE_PATH }}/omniinfer/infer_engines && pip uninstall vllm -y && pip uninstall omniinfer -y && cd vllm && SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e . && cd ../../ && pip install -e . && pip uninstall numpy -y && pip install numpy==1.26 >> ${LOG_PATH}/${HOST}/pip.log'"
        docker_install_decode_code_cmd: "{{ docker_exec_cmd }} $DOCKER_NAME_D /bin/bash -c 'export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/lastest && cd {{ ansible_env.CODE_PATH }}/omniinfer/infer_engines && pip uninstall vllm -y && pip uninstall omniinfer -y && cd vllm && SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e . && cd ../../ && pip install -e . && pip uninstall numpy -y && pip install numpy==1.26 && >> ${LOG_PATH}/${HOST}/pip.log'"

    tasks:
        - name: Create a directory to store the log.---删除
            ansible.builtin.file:---删除
                path: "{{ ansible_env.LOG_PATH }}/{{ inventory_hostname }}"---删除
                state: directory---删除
            when: "'P' in group_names or 'D' in group_names or 'C' in group_names"---删除
            tags: run_docker---删除

        - name: Group hosts by their IP address
            group_by:
                key: "{{ hostvars[inventory_hostname].ansible_host }}"
            when: "'P' in group_names or 'D' in group_names"
            tags: sync_code

        - name: The executor synchronizes code to all instances.
            when: >
                (
                ('P' in group_names or 'D' in group_names) and
                inventory_hostname == (groups[hostvars[inventory_hostname].ansible_host] | first)
                ) or

        - name: Copy the code from the host machine into the container (prefill).---删除
            command: bash -c "{{ docker_cp_prefill_code_cmd }}"---删除
            environment: ---删除
                DOCKER_NAME_P: "{{ ACTUAL_DOCKER_NAME_P }}"---删除
            when: "'P' in group_names"---删除
            tags: sync_code---删除

        - name: Copy the code from the host machine into the container (decode).---删除
            command: bash -c "{{ docker_cp_decode_code_cmd }}"---删除
            environment: ---删除
                DOCKER_NAME_D: "{{ ACTUAL_DOCKER_NAME_D }}"---删除
            when: "'D' in group_names"---删除
            tags: sync_code---删除

        - name: docker_update_prefill_code_cmd.---删除
            command: bash -c "{{ docker_update_prefill_code_cmd }}"---删除
            environment:---删除
                DOCKER_NAME_P: "{{ ACTUAL_DOCKER_NAME_P }}"---删除
            when: "'P' in group_names"---删除
            tags: pip_install---删除

        - name: docker_update_decode_code_cmd.---删除
            command: bash -c "{{ docker_update_decode_code_cmd }}"---删除
            environment:---删除
                DOCKER_NAME_D: "{{ ACTUAL_DOCKER_NAME_D }}"---删除
            when: "'D' in group_names"---删除
            tags: pip_install---删除

        - name: docker_update_prefill_code_cmd.
            command: bash -c "{{ docker_update_prefill_code_cmd }}"
            environment:
                DOCKER_NAME_P: "{{ ACTUAL_DOCKER_NAME_P }}"
            when: >
                'P' in group_names and
                inventory_hostname == (groups[hostvars[inventory_hostname].ansible_host] | first)
            tags: sync_code

        - name: docker_update_decode_code_cmd.
            command: bash -c "{{ docker_update_decode_code_cmd }}"
            environment:
                DOCKER_NAME_D: "{{ ACTUAL_DOCKER_NAME_D }}"
            when: >
                'D' in group_names and
                inventory_hostname == (groups[hostvars[inventory_hostname].ansible_host] | first)
            tags: sync_code

        - name: docker_install_prefill_code_cmd.
            command: bash -c "{{ docker_install_prefill_code_cmd }}"
            environment:
                DOCKER_NAME_P: "{{ ACTUAL_DOCKER_NAME_P }}"
            when: >
                'P' in group_names
            tags: sync_code

        - name: docker_install_decode_code_cmd.
            command: bash -c "{{ docker_install_decode_code_cmd }}"
            environment:
                DOCKER_NAME_D: "{{ ACTUAL_DOCKER_NAME_D }}"
            when: >
                'D' in group_names
            tags: sync_code

        - name: Register all values.
            set_fact:
                DECODE_API_SERVER_LIST: >-
                {% for host in groups['D']|default([]) %}
                    {% set num=0 %}
                {% endfor %}
                PREFILL_POD_NUM: >-
                {{
                    map(attribute='kv_rank') |
                }}
            

        - name: Register values for prefill.
            set_fact:
                NNODES: >-
                {{
                    selectattr('kv_rank', '==', kv_rank) |
                }}
            
        - name: Display all values.
            debug:
                msg: |
                DECODE_NUM_DP: {{ 1 }}

        - name: Forward the JSON file of the decode instances to the executor.---删除
            register: fetch_result---删除
            - name: Remove proc_trace.txt if it exists (P & D nodes)---删除
            ansible.builtin.file:---删除
                path: /tmp/process/proc_trace.txt---删除
                state: absent---删除
            when: "'P' in group_names or 'D' in group_names"---删除
            tags:---删除
                - run_server---删除

        - name: Remove proc_trace.txt if it exists (P & D nodes)---删除
            ansible.builtin.file:---删除
                path: /tmp/process/proc_trace.txt---删除
                state: absent---删除
            when: "'P' in group_names or 'D' in group_names"---删除
            tags:---删除
                - run_server---删除

        - name: Run the Omniai service for prefill instances.
            environment:
                ZMQ_PORT: "{{ zmq_port }}"
                HOST: "{{ inventory_hostname }}"
            
        - name: Run the Omniai service for decode instances.
            environment:
                ZMQ_PORT: "{{ zmq_port }}"

            - name: Run the Omniai service for prefill instances. 
            environment:
                ZMQ_PORT: "{{ zmq_port }}"
                HOST: "{{ inventory_hostname }}"
                
        - name: check vLLM is Ready---删除
            shell: |---删除
                timeout=600---删除
                while [ $timeout -gt 0 ]; do---删除
                if grep -q "Application startup complete" {{ ansible_env.LOG_PATH }}/server_0.log; then---删除
                    echo "Service ready"---删除
                    exit 0---删除
                fi---删除
                sleep 20---删除
                timeout=$((timeout - 20))---删除
                done---删除
                echo "Timeout waiting for service" >&2---删除
                exit 0---删除
            delegate_to: "{{ inventory_hostname }}"---删除
            become: yes---删除
            when: ---删除
                - "'P' in group_names or 'D' in group_names"---删除
                - "proc_bind_enabled | default(false)"---删除
            tags:---删除
                - proc_bind---删除
            
        - name: Ensure bind_cpu.sh is executable---删除
            file:---删除
                path: ${CODE_PATH}/omniinfer/tools/scripts/bind_cpu.sh---删除
                mode: "0755"---删除
            become: yes---删除
            when:---删除
                - "proc_bind_enabled | default(false)"---删除
            tags:---删除
                - proc_bind---删除
            
        - name: Prefill cpu_pin---删除
            command: /bin/bash -c "{{ docker_exec_cmd }} $DOCKER_NAME_P /bin/bash -c /workspace/omniinfer/tools/scripts/bind_cpu.sh"---删除
            environment:---删除
                DOCKER_NAME_P: "{{ ACTUAL_DOCKER_NAME_P }}"---删除
                SCRIPTS_PATH: "{{ ansible_env.SCRIPTS_PATH }}"---删除
                ROLE: "P"---删除
            become: yes---删除
            when: ---删除
                - "'P' in group_names"---删除
                - "proc_bind_enabled | default(false)"---删除
            tags:---删除
                - proc_bind---删除
            
        - name: Decode cpu_pin---删除
            command: /bin/bash -c "{{ docker_exec_cmd }} $DOCKER_NAME_D /bin/bash -c /workspace/omniinfer/tools/scripts/bind_cpu.sh"---删除
            environment:---删除
                DOCKER_NAME_D: "{{ ACTUAL_DOCKER_NAME_D }}"---删除
                SCRIPTS_PATH: "{{ ansible_env.SCRIPTS_PATH }}"---删除
                ROLE: "D"---删除
            become: yes---删除
            when: ---删除
                - "'D' in group_names"---删除
                - "proc_bind_enabled | default(false)"---删除
            tags:---删除
                - proc_bind---删除

        - name: Run the global proxy server.
            environment:
                HOST: "{{ inventory_hostname }}"
            
        - name: Create a directory on the executor to store the log.---删除
            ansible.builtin.file:---删除
                path: "{{ ansible_env.LOG_PATH_IN_EXECUTOR }}/{{ inventory_hostname }}"---删除
                state: directory---删除
            when: "'P' in group_names or 'D' in group_names or 'C' in group_names"---删除
            delegate_to: localhost---删除
            tags:---删除
                - fetch_log---删除

        - name: Forward logs from all machines to the executor.---删除
            ansible.builtin.synchronize:---删除
                mode: pull---删除
                src: "{{ ansible_env.LOG_PATH }}/{{ inventory_hostname }}/"---删除
                dest: "{{ ansible_env.LOG_PATH_IN_EXECUTOR }}/{{ inventory_hostname }}/"---删除
            when: "'P' in group_names or 'D' in group_names or 'C' in group_names"---删除
            tags:---删除
                - fetch_log---删除
    ```

## 3、Qwen3-235B-int8-A3-2P8-1D32
### 3.1 配置 omniinfer/tests/test_config/test_config_prefill.json  
     ```
        "use_omni_placement": false,
        "experts_pruning": false,    ----删除

     ```
### 3.2 配置 omniinfer/tests/test_config/test_config_decode.json  
     ```
        "use_omni_placement": false,
        "enable_mc2_v2":false,
     ```
### 3.3 配置 omniinfer/tools/srcipts/pd_run.sh  
    ```
    SERVED_MODEL_NAME="Qwen235B"
    export PROFILING_FORWARD=0
    export PROFILING_SAVE_PATH=/data/PROFILING/0910
    VLLM_ENABLE_MC2=1
    NUM_SPECULATIVE_TOKENS=1     ----删除 
    export SCALE_PARALLEL=1
    export INF_NAN_MODE_FORCE_DISABLE=1
    export MOE_DISPATCH_COMBINE=1
    export DP_SIZE=$NUM_DP
    export PYTORCH_NPU_ALLOC_CONF="expandable_segments: True"

    #pip3 uninstall torch_npu -y     ----删除 
    #pip3 install /data/torch_npu-2.5.1.post1.dev20250731-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.wh1     ----删除 
    #pip3 install /data/mindstudio_probe-8.1.1-py3-none-any.wh1     ----删除 

    print_help(){
        echo " --num-speculative-tokens    vLLM framework: Speculative decoding parameter, number of speculative tokens per step (default $NUM_SPECULATIVE_TOKENS)" ----删除
    }

    parse_long_option(){
            --num-speculative-tokens)     ----删除 
               NUM_SPECULATIVE_TOKENS     ----删除 
               ;;     ----删除 

    }

    KV_TRANSFER_CONFIG=$(cat <<EOF
    {
        "kv_parallel_size": $KV_PARALLEL_SIZE,
        "kv_connector_extra_config": {
            "kv_producer_dp_size": 4
        }
    })

    export PYTHONPATH=/workspace/omniinfer:$PYTHONPATH
    export LD_LIBRARY_PATH=/workspace/omniinfer:$LD_LIBRARY_PATH

    # Turn on these two variables to enable proc_bind     ----删除 
    # export CPU_AFFINITY_CONF=2     ----删除 
    # export PROFILING_NAMELIST=/workspace/omniinfer/omni/tools/profiler/proc_bind/proc_marker_namelist.yml     ----删除 

    export CPU_AFFINITY_CONF=1,npu0:0-1,npu1:40-41,npu2:80-81,npu3:120-121,npu4:160-161,npu5:200-201,npu6:240-241,npu7:280-281
            --enable-mtp \  ---删除
            --number-speculative-tokens "$NUM_SPECULATIVE_TOKENS" \  ---删除

    ```

### 3.4 配置omniinfer/tools/ansible/template/omni_infer_server_template.yml  
     ```
    environment:
        MODEL_LEN_MAX_PREFILL: "40000"
        MODEL_LEN_MAX_DECODE: "40000"

        "KV_CONNECTOR": "AscendHcclConnectorV1"   ----删除

        ASCEND_TOOLKIT_HOME: "/usr/local/Ascend//ascend-toolkit/latest"
        DECODE_TENSOR_PARALLEL_SIZE: "4"
        PREFILL_TENSOR_PARALLEL_SIZE: "4"
    vars：
        run_vllm_server_prefill_cmd: |
            HCCL_BUFFSIZE=500
            tp=4
            dp=4
            KV_PARALLEL_SIZE=$((DECODE_TENSOR_PARALLEL_SIZE + 0))
            EXTRA_ARGS='--max-num-batched-tokens 40000 --enforce-eager --enable-expert-parallel --disable-log-requests --max-num-seqs 32 --no-enable-prefix-caching'

            "kv_role": "kv_producer"    ----删除
            if [ ${KV_CONNECTOR} == "LMCacheConnectorV1" ]; then    ----删除
              export MOONCAKE_CONFIG_PATH=$MOONCAKE_CONFIG_PATH    ----删除
              export LMCACHE_CONFIG_FILE=$LMCACHE_MOONCAKE_CONFIG_PATH    ----删除
              "kv_role": "kv_both"    ----删除
            fi    ----删除

              --num-dp ${dp} \
              --num-servers ${dp} \
              --kv-role  "kv_producer" \              
              --tp ${tp} \

              --kv-connector ${KV_CONNECTOR"} \    ----删除

        run_vllm_server_decode_cmd: |
            HCCL_BUFFSIZE=768
            dp=16
            NUM_SERVERS=4
            KV_PARALLEL_SIZE=$((DECODE_TENSOR_PARALLEL_SIZE + 0))
            EXTRA_ARGS='--max-num-batched-tokens 40000 --enable-expert-parallel --disable-log-requests --max-num-seqs 32 --no-enable-prefix-caching'
            ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1,"use_ge_graph_cached":true}}'

            if [ ${KV_CONNECTOR} == "LMCacheConnectorV1" ]; then    ----删除
              export MOONCAKE_CONFIG_PATH=$MOONCAKE_CONFIG_PATH    ----删除
              export LMCACHE_CONFIG_FILE=$LMCACHE_MOONCAKE_CONFIG_PATH    ----删除
            fi    ----删除
              --kv-connector ${KV_CONNECTOR"} \    ----删除

        run_proxy_cmd: |
            prefill_result=""
            prefill_api_servers="{{ PREFILL_API_SERVER_LIST }}"
            prefill_api_servers=`echo "$prefill_api_servers" | awk '$1=$1'`
            prefill_array=(${prefill_api_servers//,/ })
            for var in ${prefill_array[@]}; do
              address=${var%@*}
              ip=${address%:*}
              port=${address#*:}
              num=${var#*@}
              for ((i=0; i<=$num;i++)); do
              if [[ -z ${prefill_result} ]]; then
                 prefill_result="$ip:$port"
              else
                 prefill_result="${prefill_result},$ip:$port"
              fi
              ((port++))
              done
            done
            echo "Final prefill result: $prefill_result"
              --decode-1b-sdk ${DECODE_LB_SDK} \

              --prefill-max-num-seqs 16 \     ----删除
              --decode-max-num-seqs 32      ----删除

        generate_mooncake_config_json: |      ----删除
          {      ----删除
            "metadata_server": "etcd://{{ ETCD_IP}}:{{ ETCD_PORT}}",      ----删除
            "local_hostname": "{{ HOST_IP }}",      ----删除
            "protocol": "tcp",      ----删除
            "transfer_timeout": 60,      ----删除
            "global_segment_size":17179869184,      ----删除
            "master_server_address": "{{ MOONCAKE_MASTER_IP }}:{{ MOONCAKE_MASTER_PORT }}"      ----删除

          }             ----删除

        generate_lmcache_mooncake_config_yml: |      ----删除
          # 256 Tokens per kV Chunk      ----删除
          chunk_size: 256     ----删除
          # Enable CPU memory backend      ----删除
          local_cpu: false # default     ----删除
          # 5GB of Pinned CPU memory     ----删除
          max_local_cpu_size: 5.0 # default      ----删除

          remote_url: "mooncakestore": //{{ MOONCAKE_MASTER_IP }}:{{ MOONCAKE_MASTER_PORT }}"      ----删除
          remote_serde: "naive"      ----删除
          external_lookup_client: "mooncakestore": //{{ MOONCAKE_MASTER_IP }}:{{ MOONCAKE_MASTER_PORT }}"      ----删除

          extra_config:        ----删除
             remote_enable_mla_worker_id_as0: true        ----删除

        docker_start_vllm_cmd_p: >
          -e KV_CONNECTOR=$KV_CONNECTOR    ----删除
          -e MOONCAKE_CONFIG_PATH=$MOONCAKE_CONFIG_PATH    ----删除
          -e LMCACHE_MOONCAKE_CONFIG_PATH=$LMCACHE_MOONCAKE_CONFIG_PATH    ----删除

          -e ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME
          -e DECODE_TENSOR_PARALLEL_SIZE=$DECODE_TENSOR_PARALLEL_SIZE

        docker_start_vllm_cmd_d: >
          -e KV_CONNECTOR=$KV_CONNECTOR    ----删除
          -e MOONCAKE_CONFIG_PATH=$MOONCAKE_CONFIG_PATH    ----删除
          -e LMCACHE_MOONCAKE_CONFIG_PATH=$LMCACHE_MOONCAKE_CONFIG_PATH    ----删除

          -e ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME

        docker_cp_proxy_code_cmd:>    ----删除
          [ "${KV_CONNECTOR}"!="LMCacheConnectorV1" ] || { docker cp {{ ansible_env.CODE_PATH }}/omniinfer $DOCKER_NAME_C:/workspace/;}    ----删除
        docker_update_proxy_code_cmd: >      ----删除
            {{ docker_exec_cmd }}      ----删除
            -e KV_CONNECTOR=$KV_CONNECTOR      ----删除
            $DOCKER_NAME_C       ----删除
            /bin/bash -c '[ "${KV_CONNECTOR}" != "LMCacheConnectorV1" ] || { source ~/.bashrc && cd /workspace/omniinfer/omni/adaptors/lmcache/script && bash install.sh &> ${LOG_PATH}/install.log;}'       ----删除

        docker_update_prefill_code_cmd: >
            "{{ docker_exec_cmd }} $ DOCKER_NAME_P /bin/bash -c '. ~/.bashrc && cd /workspace/omniinfer/infer_engines && git config --global --add safe.directory /workspace/omniinfer/infer_engines/vllm && cd vllm && git checkout -f && cd .. && bash bash_install_code.sh && pip uninstall vllm -y && pip uninstall omni infer -y && cd vllm && SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e . --no-deps && cd ../../ && pip install -e . --no-deps && pip uninstall numpy -y && pip install numpy==1.26 --no-deps > ${LOG_PATH}/{{ inventory_hostname }}/pip.log'"

        docker_update_decode_code_cmd: >
            "{{ docker_exec_cmd }} $ DOCKER_NAME_D /bin/bash -c '. ~/.bashrc && cd /workspace/omniinfer/infer_engines && git config --global --add safe.directory /workspace/omniinfer/infer_engines/vllm && cd vllm && git checkout -f && cd .. && bash bash_install_code.sh && pip uninstall vllm -y && pip uninstall omni infer -y && cd vllm && SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e . --no-deps && cd ../../ && pip install -e . --no-deps && pip uninstall numpy -y && pip install numpy==1.26 --no-deps > ${LOG_PATH}/{{ inventory_hostname }}/pip.log'"

        docker_start_etcd_cmd: >       ----删除
            {{ docker_exec_cmd }}       ----删除
            -e KV_CONNECTOR=$KV_CONNECTOR          ----删除
            $DOCKER_NAME_C         ----删除
            /bin/bash -c '[ "${KV_CONNECTOR}" != "LMCacheConnectorV1" ] || { nohup etcd --listen-client-urls http:// {{ ETCD_IP}}:{{ ETCD_PORT}} --advertise-client-urls http:// {{ ETCD_IP}}:{{ ETCD_PORT}} &> ${LOG_PATH}/etcd.log &}'          ----删除

        docker_start_mooncake_master_cmd: >       ----删除
            {{ docker_exec_cmd }}       ----删除
            -e KV_CONNECTOR=$KV_CONNECTOR          ----删除
            $DOCKER_NAME_C         ----删除
            /bin/bash -c '[ "${KV_CONNECTOR}" != "LMCacheConnectorV1" ] || { source ~/.bashrc && nohup mooncake_master -rpc_address={{ MOONCAKE_MASTER_IP }} -rpc_port={{ MOONCAKE_MASTER_PORT}} -metrics_port={{ MOONCAKE_MASTER_PORT}} -v 2 -eviction_high_watermark_ratio=0.8 -eviction_ratio=0.1 &> ${LOG_PATH}/mooncake_master.log &}'          ----删除
    
    tasks:
        - name: Check and delete containers used for global proxy server.
            tags:
                - run_proxy

        - name: Run container for global proxy server.
            tags:
                - run_proxy

        - name: Create a directory to store the log.
            tags: 
                - run_proxy

        - name: Copy the code from the host machine into the container (proxy).      ---删除
            command: bash -c "{{ docker_cp_proxy_code_cmd }}"      ---删除
            environment:          ---删除
              DOCKER_NAME_C: "{{ ACTUAL_DOCKER_NAME_C }}"   ---删除
            when: "'C' in group_names"         ---删除
            tags: sync_code         ---删除

        - name: docker_update_proxy_code_cmd.    ---删除
            command: bash -c "{{ docker_update_proxy_code_cmd }}"       ---删除
            environment:        ---删除
              DOCKER_NAME_C: "{{ ACTUAL_DOCKER_NAME_C }}"    ---删除
            when: "'C' in group_names"       ---删除
            tags: pip_install        ---删除

        - name: Delete temporary script files.
            tags:
                - run_proxy

        - name: Register all values.
            set_fact:
                PREFILL_API_SERVER_LIST: >-
                    {% set result = [] %}
                    {% set tp_size = (vars.get('PREFILL_TENSOR_PARALLEL_SIZE', '4') | int) %}
                    {% for host in groups['P'] | default([]) %}
                        {% set h = hostvars[host] %}
                        {% set ip = h.ansible_host | default('') %}
                        {% if ip %}
                          {% set devices = (h.ascend_rt_visible_devices | default('0')) %}
                          {% set count = devices.split(',') | length | int %}
                          {% set instances = (count / tp_size) | round(0, 'ceil') | int - 1%}
                          {% set port = h.api_port | default('9000') %}
                          {% set entry = ip ~ ':' ~ port ~ '@' ~ instances %}
                          {% if entry not in result %}
                            {% set _ = result.append(entry) %}
                          {% endif %}
                        {% endif %}
                    {% endfor %}
                    {{ result | join(',') }}

                DECODE_API_SERVER_LIST: >-
                    {% set result = [] %}
                    {% set tp_size = (vars.get('DECODE_TENSOR_PARALLEL_SIZE', '4') | int) %}
                    {% for host in groups['D'] | default([]) %}
                        {% set h = hostvars.get(host, {}) %}
                        {% set ip = h.ansible_host | default('') %}
                        {% if ip %}
                          {% set devices = (h.ascend_rt_visible_devices | default('0')) %}
                          {% set count = devices.split(',') | length | int %}
                          {% set instances = (count / tp_size) | round(0, 'ceil') | int - 1%}
                          {% set port = h.api_port | default('9100') %}
                          {% set entry = ip ~ ':' ~ port ~ '@' ~ instances %}
                          {% if entry not in result %}
                            {% set _ = result.append(entry) %}

                #DECODE_API_SERVER_LIST: >-    ---删除
                #    {% set result = [] %}    ---删除
                #    {% for host in groups['D'] | default([]) %}    ---删除
                #        {% set h = hostvars.get(host, {}) %}    ---删除
                #        {% set ip = h.ansible_host | default('') %}    ---删除
                #        {% set port = h.api_port | default('9100') %}    ---删除
                #        {% set num = h.ascend_rt_visible_devices.count(',') | default('0') %}    ---删除
                #        {% if ip %}    ---删除
                #          {% set entry = ip ~ ':' ~ port ~ '@' ~ num %}    ---删除
                #          {% if entry not in result %}    ---删除
                #            {% set _ = result.append(entry) %}    ---删除
                #          {% endif %}    ---删除
                #        {% endif %}    ---删除
                #    {% endfor %}    ---删除
                3    {{ result | join(',') }}       ---删除

        - name: Set mooncake & etcd configuration.       ---删除
          set_fact:       ---删除
            ETCD_IP: "{{ hostvars[groups['C'][0]]['ansible_host'] }}"       ---删除
            ETCD_PORT: "{{ etcd_port }}"       ---删除
            MOONCAKE_MASTER_IP: "{{ hostvars[groups['C'][0]]['ansible_host'] }}"             ---删除      
            MOONCAKE_MASTER_PORT: "{{ mooncake_master_port }}"       ---删除
            MOONCAKE_METRICS_PORT: "{{ mooncake_metrics_port }}"       ---删除
            PROXY_PORT: "{{ proxy_port }}"                  ---删除
          when: "'P' in group_names or 'D' in group_names or 'C' in group_names"       ---删除
          tags:       ---删除
            - stop_server       ---删除
            - run_server       ---删除

        - name: Kill Mooncake master pod.       ---删除
          command:  /bin/bash -c "kill -9 $(netstat -tulpn | grep : {{ MOONCAKE_MASTER_PORT }} | awk '{print $7}'  grep -oP '\d+')"       ---删除
          failed_when: false       ---删除
          when: "'C' in group_names"       ---删除
          tags:       ---删除
            - stop_server       ---删除
            - run_server       ---删除

        - name: Kill etcd pod.      ---删除
          command:  /bin/bash -c "kill -9 $(netstat -tulpn | grep : {{ ETCD_PORT }} | awk '{print $7}'  grep -oP '\d+')"       ---删除
          failed_when: false       ---删除
          when: "'C' in group_names"       ---删除
          tags:       ---删除
            - stop_server       ---删除
            - run_server       ---删除

        - name: Generate a mooncake config json.       ---删除
          copy:       ---删除
            content: "{{ generate_mooncake_config_json }}"       ---删除
            dest: "$SCRIPTS_PATH/mooncake_config.json"       ---删除
            mode: '0750'       ---删除
          vars:       ---删除
            HOST_IP: "{{ ansible_host }}"       ---删除
          when: "'P' in group_names or 'D' in group_names"       ---删除
          tags: run_server       ---删除

        - name: Generate a lmcache + mooncake config yaml.      ---删除
          copy:       ---删除
            content: "{{ generate_lmcache_mooncake_config_yaml }}"       ---删除
            dest: "$SCRIPTS_PATH/lmcache_mooncake_config.yaml"       ---删除
            mode: '0750'       ---删除
          when: "'P' in group_names or 'D' in group_names"       ---删除
          tags: run_server       ---删除

        - name: Run etcd for Mooncake.
          command:  /bin/bash -c " {{ docker_start_etcd_cmd }} "       ---删除
          environment:        ---删除
            DOCKER_NAME_C: "{{ ACTUAL_DOCKER_NAME_C }}"       ---删除
          when: "'C' in group_names"       ---删除
          tags: run_server       ---删除

        - name: Run master port of Mooncake.
          command:  /bin/bash -c " {{ docker_start_mooncake_master_cmd }} "       ---删除
          environment:        ---删除
            DOCKER_NAME_C: "{{ ACTUAL_DOCKER_NAME_C }}"       ---删除
          when: "'C' in group_names"       ---删除
          tags: run_server       ---删除

        - name: Run the Omniai service for prefill instances.
          MOONCAKE_CONFIG_PATH: "$SCRIPTS_PATH/mooncake_config.json"     ----删除
          LMCACHE_MOONCAKE_CONFIG_PATH: "$SCRIPTS_PATH/lmcache_mooncake_config.yaml"    ----删除
            
        - name: Run the Omniai service for decode instances.
          MOONCAKE_CONFIG_PATH: "$SCRIPTS_PATH/mooncake_config.json"     ----删除
          LMCACHE_MOONCAKE_CONFIG_PATH: "$SCRIPTS_PATH/lmcache_mooncake_config.yaml"    ----删除
         
        - name: check vLLM is Ready
            shell: |
              timeout=600
              if grep -q "Application startup complete" {{ ansible_env.LOG_PATH }}/server_0.log; then
                exit 0
            delegate_to: "{{ inventory_hostname }}"
            become: yes
            when:
                - "'P' in group_names or 'D' in group_names" 
            
        - name: Ensure bind_cpu.sh is executable in the container     ---删除
          ansible.builtin.shell:      ---删除
            cmd: docker exec $DOCKER_NAME_P chmod +x /workspace/omniinfer/tools/scripts/bind_cpu.sh      ---删除
          environment:        ---删除
            DOCKER_NAME_P: "{{ ACTUAL_DOCKER_NAME_P }}"       ---删除
          when: "'P' in group_names"       ---删除
          tags:          ---删除
            - proc_bind         ---删除
            
        - name: Ensure bind_cpu.sh is executable in the container     ---删除
          ansible.builtin.shell      ---删除
            cmd: docker exec $DOCKER_NAME_D chmod +x /workspace/omniinfer/tools/scripts/bind_cpu.sh      ---删除
          environment:        ---删除
            DOCKER_NAME_D: "{{ ACTUAL_DOCKER_NAME_D }}"       ---删除
          when: "'D' in group_names"       ---删除

        - name: Ensure bind_cpu.sh is executable
          file:
            path: ${CODE_PATH}/omniinfer/tools/scripts/bind_cpu.sh
            mode: '0750'  
          become: yes
          when:
            - "proc_bind_enabled | default(false)"

     ```


# 三、Kimi模型部署  

## 1、kimi-k2-int8-A3-2P8-1D16

### 1.1 配置 omniinfer/tests/test_config/test_config_decode_k2.json:
    ```
    {
        "operator_optimizition_config":{
            "use_super_kernel": true
        }
    }
    ```

### 1.2 配置 omniinfer/tests/tools/ansible/template/omni_infer_server_template_k2.yml:
    ```
    environment:
        MODEL_LEN_MAX_PREFILL: "33008"
        MODEL_LEN_MAX_DECODE: "33008"
        PREFILL_LB_SDK: "least_total_load"
        DECODE_LB_SDK: "weighted_least_active"
    vars:
        docker_run_cmd: |
            docker run -it --shm-size=500g \
                -e CODE_PATH=$CODE_PATH---删除

        run_vllm_server_prefill_cmd: |
            ADDITIONAL_CONFIG='{"enable_omni_attn": false, "multi_rank_pull_kv": true}'

        run_vllm_server_decode_cmd:
            EXTRA_ARGS='--enable-expert-parallel --disable-log-requests --max-num-seqs 48 --no-enable-prefix-caching'
            ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1, "use_ge_graph_cached":true},"enable_omni_attn": false , "multi_rank_pull_kv": true}'
            if [[ -e "/usr/local/Ascend/ascend-toolkit" ]]; then---删除
                python /workspace/omniinfer/tools/scripts/process_nz_config.py /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json---删除
            else---删除
                python /workspace/omniinfer/tools/scripts/process_nz_config.py /usr/local/Ascend/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json---删除
            fi---删除
            python /workspace/omniinfer/tools/scripts/process_nz_config.py /usr/local/Ascend/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json

     ```


## 2、kimi-k2-int4-A3-2P8-1D16

### 2.1 配置 omniinfer/tests/tools/ansible/template/omni_infer_server_template_k2.yml
    ```
    environment:
       CODE_PATH: "/workspace"

    vars:
      -v CODE_PATH=$CODE_PATH \
         
        run_vllm_server_prefill_cmd: |
            export CPU_AFFINITY_CONF=2   ---删除
            export ASCEND_GLOBAL_LOG_LEVEL=3

        run_vllm_server_decode_cmd: |
            export CPU_AFFINITY_CONF=2   ---删除
            export ASCEND_GLOBAL_LOG_LEVEL=3

    docker_update_prefill_code_cmd: >
       {{ docker_exec_cmd }} $ DOCKER_NAME_P /bin/bash -c 'if [[ -e "/usr/local/Ascend/ascend-toolkit" ]]; then export ASCEND_TOOLKIT_HOME: /usr/local/Ascend/ascend-toolkit/lastest; else ASCEND_TOOLKIT_HOME: /usr/local/Ascend/lastest;fi && . ~/.bashrc && cd /workspace/omniinfer/infer_engines && git config --global --add safe.directory /workspace/omniinfer/infer_engines/vllm && cd vllm && git checkout -f && cd .. && bash bash_install_code.sh && pip uninstall vllm -y && pip uninstall omni_infer -y && cd vllm && SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e . --no-deps && cd ../../ && pip install -e . --no-deps && pip uninstall numpy -y && pip install numpy==1.26 --no-deps > ${LOG_PATH}/{{ inventory_hostname }}/pip.log'

    docker_update_decode_code_cmd: >
       {{ docker_exec_cmd }} $ DOCKER_NAME_D /bin/bash -c 'if [[ -e "/usr/local/Ascend/ascend-toolkit" ]]; then export ASCEND_TOOLKIT_HOME: /usr/local/Ascend/ascend-toolkit/lastest; else ASCEND_TOOLKIT_HOME: /usr/local/Ascend/lastest;fi && . ~/.bashrc && cd /workspace/omniinfer/infer_engines && git config --global --add safe.directory /workspace/omniinfer/infer_engines/vllm && cd vllm && git checkout -f && cd .. && bash bash_install_code.sh && pip uninstall vllm -y && pip uninstall omni_infer -y && cd vllm && SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e . --no-deps && cd ../../ && pip install -e . --no-deps && pip uninstall numpy -y && pip install numpy==1.26 --no-deps > ${LOG_PATH}/{{ inventory_hostname }}/pip.log'

    ```