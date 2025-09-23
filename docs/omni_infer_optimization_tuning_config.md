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

## 2、DeepSeek-R1-int8-A2-1P16-1D32

### 2.1 配置 omniinfer/tools/ansible/template/omni_infer_server_template.yml
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
    
## 3、DeepSeek-R1-0528-BF16-A3-2P32-1D32

### 3.1 配置 omniinfer/tests/test_config/config_p.yml
      
    ```
    pattern_path: "../../tests/test_config/placement_pattern_20250715_105711_58_redundant_layers_58_layers_64_ranks_epmaxdeploy_200_prefill.npy"
    Optimizers:
        - expert_balance_optimizer.ExpertsBalanceOptimizer:
            batch_size: 16
    ```
### 3.2 配置 omniinfer/tests/test_config/test_prefill_prefill_bf16.json
    ```
    "model_parallel_config": {
        "dense_mlp_tp_size": 1
    },
    "operator_optimizition_config": {
        "gmm_nz": true,
        "unquant_bmm_nz": true,
        "decode_moe_dispatch_combine": false,
        "use_omni_placement": true,
        "omni_placement_config_path": "../../tests/test_config/config_p.yaml",
        "enable_mc2_v2": false,
    }
    ```
### 3.3 配置 omniinfer/tests/test_config/test_config_decode_bf16.json                 
      
    ```
    "model_parallel_config": {
        "dense_mlp_tp_size": 16,
        "o_proj_tp_size": 8,
    },
    "operator_optimizition_config": {
        "gmm_nz": true,
        "unquant_bmm_nz": true,
        "prefill_enable_mla_alltoall": false,
        "omni_placement_config_path": "../../tests/test_config/config_d.yaml"
    }
    ```
### 3.4 配置 omniinfer/tools/scripts/pd_run.sh 
    ```
    export CPU_AFFINITY_CONF=2
    ```
### 3.5 配置 omniinfer/tools/ansible/template/omni_infer_server_template.yml  
    ```
    environment:
        MODEL_LEN_MAX_PREFILL: "70400"
        MODEL_LEN_MAX_DECODE: "40000"

        PREFILL_LB_SDK: "least_total_load"
        DECODE_LB_SDK: "weighted_least_active"

    vars:
        run_vllm_server_prefill_cmd: |
            HCCL_BUFFSIZE=200
            export HCCL_CONNECT_TIMEOUT=600
            export HCCL_EXEC_TIMEOUT=600
            export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
            MODEL_EXTRA_CFG_PATH="${CODE_PATH}/omniinfer/tests/test_config/test_prefill_prefill_bf16.json"
            EXTRA_ARGS='--max-num-batched-tokens 70400 --enforce-eager --no-enable-prefix-caching --enable-expert-parallel --disable-log-requests --max-num-seqs 128 --scheduler-cls=omni.adaptors.vllm.worker.tfas.tfas_scheduler.TFASScheduler'
            GPU_UTIL=0.90

            export HCCL_OP_EXPANSION_MODE=AIV
            ADDITIONAL_CONFIG='{"tfas_scheduler_config": {"adjust_param": 8.708, "token_budget": 38000}}'

            export OMNI_REUSE_PREFILLED_TOKENS=1
            export OMNI_SKIP_DECODE_TOKENIZE=1
            export TOKENIZER_PROC_POOL=1  

            PROFILING_NAMELIST=/workspace/omniinfer/omni/adaptors/vllm/patches/profiler_patches/proc_bind/proc_marker_namelist.yml bash /workspace/omniinfer/tools/scripts/pd_run.sh \---删除
            bash ${CODE_PATH}/omniinfer/tools/scripts/pd_run.sh \
                --additional-config "$ADDITIONAL_CONFIG" \

        run_vllm_server_decode_cmd: |
            HCCL_BUFFSIZE=200
            export HCCL_CONNECT_TIMEOUT=600
            export HCCL_EXEC_TIMEOUT=600
            export HCCL_DETERMINISTIC=false 
            MODEL_EXTRA_CFG_PATH="${CODE_PATH}/omniinfer/tests/test_config/test_config_decode_bf16.json"
            EXTRA_ARGS='--enable-expert-parallel --disable-log-requests --max-num-seqs 4 --no-enable-prefix-caching'
            ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1, "use_ge_graph_cached":true}}'

            export OMNI_REUSE_PREFILLED_TOKENS=1
            export OMNI_SKIP_DECODE_TOKENIZE=1
            export TOKENIZER_PROC_POOL=1

            PROFILING_NAMELIST=/workspace/omniinfer/omni/adaptors/vllm/patches/profiler_patches/proc_bind/proc_marker_namelist.yml bash /workspace/omniinfer/tools/scripts/pd_run.sh \---删除
            bash ${CODE_PATH}/omniinfer/tools/scripts/pd_run.sh \
    ```

## 4、DeepSeek-V3-INT4-A3-8P4-1D32

### 4.1 配置 omniinfer/tests/test_config/config_p.yml
    ```
    pattern_path: "../../omni/accelerators/placement/patterns/MoE_placement_prefill_node_8_extra_expert_per_node_0_stoptime_1.npy"
    ```
### 4.2 配置 omniinfer/tests/test_config/config_d.yml
    ```
    pattern_path: "../../omni/accelerators/placement/patterns/base_patterns/DSV3_baseline_64_devices_58_MoE_Layers.npy"
    max_redundant_per_expert: 10 # 10
    max_redundant_per_rank: 1 # 1
     ```
### 4.3 配置 omniinfer/tests/test_config/test_config_decode.json
    ```
     {
        "operator_optimizition_config": {
             "enable_mc2_v2": true
        }
     }
    ```     
### 4.4 配置 omniinfer/tools/ansible/template/omni_infer_server_template_a3_1machine2p.yml
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

## 1、Qwen2.5-7B-A2/A3-单机混部  

### 1.2 启动脚本：
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
        --max-model-len 4096 \
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
    "operator_optimizition_config":{
        "use_omni_placement": false
    }
     ```
### 3.2 配置 omniinfer/tests/test_config/test_config_decode.json  
     ```
    "operator_optimizition_config":{
        "use_omni_placement": false
    }
     ```
### 3.3 配置 omniinfer/tools/srcipts/pd_run.sh  
    ```
    KV_TRANSFER_CONFIG=$(cat <<EOF
    {
        "kv_connector_extra_config": {
            "kv_producer_dp_size": 4
        }
    })
    export PYTHONPATH=/workspace/omni_infer:$PYTHONPATH
    export OMNI_USE_DSV3=0
    export MOE_DISPATCH_COMBINE=1
    export DP_SIZE=$NUM_DP
    export PYTORCH_NPU_ALLOC_CONF="expandable_segments: True"
    export SCALE_PARALLEL=1
    export INF_NAN_MODE_FORCE_DISABLE=1
    export CPU_AFFINITY_CONF=1,npu0:0-19,npu1:20-39,npu2:40-59,npu3:60-79,npu4:80-99,npu5:100-119,npu6:120-139,npu7:140-159,npu8:160-179,npu9:180-199,npu10:200-219,npu11:220-239,npu12:240-259,npu13:260-279,npu14:280-299,npu15:300-319
    export ENABLE_OVERWRITE_REQ_IDS=1---删除
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    common_operations() {
        python start_api_servers.py \
            --enable-mtp \---删除
    }
    ```
### 3.4 配置omniinfer/tools/ansible/template/omni_infer_server_template.yml  
     ```
    environment:
        MODEL_LEN_MAX_PREFILL: "8192"
        MODEL_LEN_MAX_DECODE: "8192"
        ASCEND_TOOLKIT_HOME: "/usr/local/Ascend//ascend-toolkit/latest"
        DECODE_TENSOR_PARALLEL_SIZE: "4"
        PREFILL_TENSOR_PARALLEL_SIZE: "4"
    vars：
        run_vllm_server_prefill_cmd: |
            HCCL_BUFFSIZE=500
            tp=4
            dp=4
            KV_PARALLEL_SIZE=$((DECODE_TENSOR_PARALLEL_SIZE + 0))
            EXTRA_ARGS='--max-num-batched-tokens 60000 --enforce-eager --enable-expert-parallel --disable-log-requests --max-num-seqs 48 --no-enable-prefix-caching'
            PROFILING_NAMELIST=/workspace/omniinfer/omni/adaptors/vllm/patches/profiler_patches/proc_bind/proc_marker_namelist.yml bash /workspace/omniinfer/tools/scripts/pd_run.sh \
                --num-dp ${dp} \
                --num-servers ${dp} \
                --tp ${tp} \
            
        run_vllm_server_decode_cmd: |
            HCCL_BUFFSIZE=768
            dp=16
            ((dp++))---删除
            NUM_SERVERS=4
            KV_PARALLEL_SIZE=$((DECODE_TENSOR_PARALLEL_SIZE + 0))
            EXTRA_ARGS='--enable-expert-parallel --disable-log-requests --max-num-seqs 48 --no-enable-prefix-caching'
            ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1,"use_ge_graph_cached":true}}'

        run_proxy_cmd: |
            prefill_result=""
            prefill_result=`echo "$prefill_result" | awk '$1=$1'`---删除
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

        docker_start_vllm_cmd_p: >
            {{ docker_exec_emd }}
            -e ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME
            -e DECODE_TENSOR_PARALLEL_SIZE=$DECODE_TENSOR_PARALLEL_SIZE

        docker_start_vllm_cmd_d: >
            {{ docker_exec_emd }}
            -e ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME
    
    tasks:
        - name: Check and delete containers used for global proxy server.
            tags:
                - run_docker
                - clean_up
                - run_proxy

        - name: Run container for global proxy server.
            tags:
                - run_docker
                - run_proxy

        - name: Create a directory to store the log.
            tags: 
                - run_docker
                - run_proxy

        - name: Delete temporary script files.
            tags:
                - ranktable
                - clean_up
                - run_proxy

        - name: Register all values.
            set_fact:
                #PREFILL_API_SERVER_LIST全部替换为如下：
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
                #DECODE_API_SERVER_LIST全部替换为如下：
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
                        {% endif %}
                        {% endif %}
                    {% endfor %}
                    {{ result | join(',') }}
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