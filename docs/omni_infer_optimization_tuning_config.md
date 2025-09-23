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

## 2、DeepSeek-R1-int8-A3-2P32-1D32
### 2.1 配置 omniinfer/tests/test_config/test_config_decode.json                     
    ```
    "model_parallel_config": {
        "dp_size": 32
       
    }
    ```
### 2.2 配置 omniinfer/tools/ansible/template/omni_infer_server_template.yml  
    ```
    environment:
        MODEL_LEN_MAX_PREFILL: "65536"
        MODEL_LEN_MAX_DECODE: "65536"
        PREFILL_LB_SDK: "least_conn"
        DECODE_LB_SDK: "least_conn"

    vars:
        run_vllm_server_prefill_cmd: |
            EXTRA_ARGS='--max-num-batched-tokens 65536 --enforce-eager --enable-expert-parallel --disable-log-requests --max-num-seqs 16 --no-enable-prefix-caching'

        run_vllm_server_decode_cmd: |
            EXTRA_ARGS='--enable-expert-parallel --disable-log-requests --max-num-seqs 32 --no-enable-prefix-caching --preemption-mode swap'         
    ```

## 3、DeepSeek-R1-int8-A2-1P16-1D32 
### 3.1 配置 omniinfer/tools/ansible/template/omni_infer_server_template.yml
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
            export ASCEND_PROCESS_LOG_PATH="/data/v040_test/log_path"
            pip install pybind11
            cd ${CODE_PATH}/omniinfer/omni/accelerators/placement && python setup.py bdist_wheel >> ${LOG_PATH}/pip.log
            rm -rf *.egg-info/
            pip install ${CODE_PATH}/omniinfer/omni/accelerators/placement/dist/omni_*.whl >> ${LOG_PATH}/pip.log
            cd ${CODE_PATH}/omniinfer/tools/scripts
            PROFILING_NAMELIST=/workspace/omniinfer/omni/adaptors/vllm/patches/profiler_patches/proc_bind/proc_marker_namelist.yml bash /workspace/omniinfer/tools/scripts/pd_run.sh \----删除
            bash /workspace/omniinfer/tools/scripts/pd_run.sh \
                --log-dir "${LOG_PATH}" > ${LOG_PATH}/run_prefill.log 2>&1 &

        run_vllm_server_decode_cmd: |
            HCCL_BUFFSIZE=20
            MODEL_EXTRA_CFG_PATH="${CODE_PATH}/omniinfer/tests/test_config/test_config_decode_a2.json"
            EXTRA_ARGS='--enable-expert-parallel --disable-log-requests --max-num-seqs 60 --no-enable-prefix-caching'
            GPU_UTIL=0.95
            ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1, "use_ge_graph_cached": false, "block_num_floating_range": 50}}'
            VLLM_ENABLE_MC2=0
            export ASCEND_PLATFORM=A2
            export CPU_AFFINITY_CONF=1,npu0:144-167,npu1:0-23,npu2:144-167,npu3:0-23,npu4:96-119,npu5:48-71,npu6:96-119,npu7:48-71
            export ASCEND_PROCESS_LOG_PATH="/data/v040_test/log_path"
            pip install pybind11
            cd ${CODE_PATH}/omniinfer/omni/accelerators/placement && python setup.py bdist_wheel >> ${LOG_PATH}/pip.log
            rm -rf *.egg-info/
            pip install ${CODE_PATH}/omniinfer/omni/accelerators/placement/dist/omni_*.whl >> ${LOG_PATH}/pip.log
            cd ${CODE_PATH}/omniinfer/tools/scripts
            PROFILING_NAMELIST=/workspace/omniinfer/omni/adaptors/vllm/patches/profiler_patches/proc_bind/proc_marker_namelist.yml bash /workspace/omniinfer/tools/scripts/pd_run.sh \----删除
            bash ${CODE_PATH}/omniinfer/tools/scripts/pd_run.sh \
                --log-dir "${LOG_PATH}" > ${LOG_PATH}/run_decode.log 2>&1 &

        run_proxy_cmd: |
            cd ${CODE_PATH}/omniinfer/omni/accelerators/sched/global_proxy
            NGINX_VERSION="${NGINX_VERSION:-1.24.0}"
            if [ ! -f "nginx-${NGINX_VERSION}.tar.gz" ]; then
                wget --no-check-certificate "https://nginx.org/download/nginx-${NGINX_VERSION}.tar.gz" > ${LOG_PATH}/install_nginx.log 2>&1
            fi
            tar -zxf "nginx-${NGINX_VERSION}.tar.gz"
            unset http_proxy
            unset https_proxy
            bash build.sh >> ${LOG_PATH}/install_nginx.log 2>&1
            cd ${CODE_PATH}/omniinfer/tools/scripts
            bash global_proxy.sh \
                --log-file ${LOG_PATH}/nginx_error.log \

        docker_start_proxy_cmd_c: >
            {{ docker_exec_cmd }}
            -e http_proxy=$HTTP_PROXY
            -e https_proxy=$HTTP_PROXY
            docker_cp_prefill_code_cmd: "docker cp {{ ansible_env.CODE_PATH }}/omniinfer $DOCKER_NAME_P:/workspace/"---删除
            docker_cp_decode_code_cmd: "docker cp {{ ansible_env.CODE_PATH }}/omniinfer $DOCKER_NAME_D:/workspace/"---删除  
            docker_update_prefill_code_cmd: "{{ docker_exec_cmd }} $DOCKER_NAME_P /bin/bash -c 'export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/latest && cd {{ ansible_env.CODE_PATH }}/omniinfer/infer_engines && git config --global --add safe.directory {{ ansible_env.CODE_PATH }}/omniinfer/infer_engines/vllm && cd vllm && git checkout -f && cd .. && bash bash_install_code.sh && pip uninstall vllm -y && pip uninstall omniinfer -y && cd vllm && SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e . && cd ../../ && pip install -e . && pip uninstall numpy -y && pip install numpy==1.26 > ${LOG_PATH}/pip.log'" 
            docker_update_decode_code_cmd: "{{ docker_exec_cmd }} $DOCKER_NAME_D /bin/bash -c 'export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/latest && cd {{ ansible_env.CODE_PATH }}/omniinfer/infer_engines && git config --global --add safe.directory {{ ansible_env.CODE_PATH }}/omniinfer/infer_engines/vllm && cd vllm && git checkout -f && cd .. && bash bash_install_code.sh && pip uninstall vllm -y && pip uninstall omniinfer -y && cd vllm && SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e . && cd ../../ && pip install -e . && pip uninstall numpy -y && pip install numpy==1.26 > ${LOG_PATH}/pip.log'"

    tasks:
        - name: Create a directory to store the log.---删除
            ansible.builtin.file:---删除
                path: "{{ ansible_env.LOG_PATH }}/{{ inventory_hostname }}"---删除
                state: directory---删除
            when: "'P' in group_names or 'D' in group_names or 'C' in group_names"---删除
            tags: ---删除
                - run_docker---删除
                - run_proxy---删除

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

        - name: docker_update_prefill_code_cmd.
            tags: sync_code

        - name: docker_update_decode_code_cmd.
            tags: sync_code

        - name: Create a directory on the executor to store ranktable file.
            tags:
                - run_docker
                - clean_up

        - name: Delete temporary script files.
            tags:
                - run_docker
                - clean_up

        - name: Forward the JSON file of the decode instances to the executor.
            register: fetch_result

        - name: The executor synchronizes the global ranktable file to all instances.
            throttle: 1

        - name: Generate a script to kill all Python processes in the container.
            tags: run_server

        - name: Generate a script to kill all Ray processes in the container. 
            tags: run_server

        - name: Kill all Python processes in the container of prefill.
            tags: run_server

        - name: Kill all Ray processes in the container of prefill.
            tags: run_server

        - name: Kill all Python processes in the container of decode.
            tags: stop_server

        - name: Kill all Ray processes in the container of decode.
            tags: stop_server

        - name: Remove proc_trace.txt if it exists (P & D nodes)---删除
            ansible.builtin.file:---删除
                path: /tmp/process/proc_trace.txt---删除
                state: absent---删除
            when: "'P' in group_names or 'D' in group_names"---删除
            tags:---删除
                - run_server---删除

        - name: Create a directory on the executor to store the log.
            when: "'P' in group_names or 'D' in group_names"

        - name: Forward logs from all machines to the executor.
            ansible.builtin.synchronize:
                src: "{{ ansible_env.LOG_PATH }}"
            when: "'P' in group_names or 'D' in group_names"
            delegate_to: localhost
    ```       

# 二、Qwen模型部署
## 1、Qwen3-235B-int8-A3-2P8-1D32
### 1.1 配置 omniinfer/tests/test_config/test_config_prefill.json
     ```
    "operator_optimizition_config":{
        "use_omni_placement": false
    }
     ```
### 1.2 配置 omniinfer/tests/test_config/test_config_decode.json
     ```
    "operator_optimizition_config":{
        "use_omni_placement": false
    }
     ```
### 1.3 配置 omniinfer/tools/scripts/pd_run.sh
    ```
    KV_TRANSFER_CONFIG=$(cat <<EOF
    {
        "kv_connector_extra_config": {
            "kv_producer_dp_size": 4
        }
    }
    )
    export PYTHONPATH=/workspace/omni_infer:$PYTHONPATH
    export OMNI_USE_DSV3=0
    export MOE_DISPATCH_COMBINE=1
    export DP_SIZE=$NUM_DP
    export PYTORCH_NPU_ALLOC_CONF="expandable_segments: True"
    export SCALE_PARALLEL=1
    export INF_NAN_MODE_FORCE_DISABLE=1
    export CPU_AFFINITY_CONF=1,npu0:0-19,npu1:20-39,npu2:40-59,npu3:60-79,npu4:80-99,npu5:100-119,npu6:120-139,npu7:140-159,npu8:160-179,npu9:180-199,npu10:200-219,npu11:220-239,npu12:240-259,npu13:260-279,npu14:280-299,npu15:300-319
    export ENABLE_OVERWRITE_REQ_IDS=1---删除
    export VLLM_ALLOW_LONG_MAX_MODE_LEN=1
    common_operations() {
        python start_api_servers.py \
            --enable-mtp \---删除
    }
    ```
### 1.4 配置omniinfer/tools/ansible/template/omni_infer_server_template.yml
     ```
    environment:
        MODEL_LEN_MAX_PREFILL: "8192"
        MODEL_LEN_MAX_DECODE: "8192"
        ASCEND_TOOLKIT_HOME: "/usr/local/Ascend/ascend-toolkit/latest"
        DECODE_TENSOR_PARALLEL_SIZE: "4"
        PREFILL_TENSOR_PARALLEL_SIZE: "4"
    vars：
        run_vllm_server_prefill_cmd: |
            HCCL_BUFFSIZE=500
            tp=4
            dp=4
            KV_PARALLEL_SIZE=$((DECODE_TENSOR_PARALLEL_SIZE + 0))
            EXTRA_ARGS='--max-num-batched-tokens 8192 --enforce-eager --enable-expert-parallel --disable-log-requests --max-num-seqs 48 --no-enable-prefix-caching'
            PROFILING_NAMELIST=/workspace/omniinfer/omni/adaptors/vllm/patches/profiler_patches/proc_bind/proc_marker_namelist.yml bash /workspace/omniinfer/tools/scripts/pd_run.sh \
                --num-dp ${dp} \
                --num-servers ${dp} \
                --tp ${tp} \
            
        run_vllm_server_decode_cmd: |
            HCCL_BUFFSIZE=768
            dp=16
            NUM_SERVERS=4
            KV_PARALLEL_SIZE=$((DECODE_TENSOR_PARALLEL_SIZE + 0))
            EXTRA_ARGS='--enable-expert-parallel --disable-log-requests --max-num-seqs 48 --no-enable-prefix-caching'
            ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1,"use_ge_graph_cached":true}}'

        run_proxy_cmd: |
            prefill_result="{{ PREFILL_API_SERVER_LIST }}"----删除
            prefill_result=`echo "$prefill_result" | awk '$1=$1'`----删除
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
                PREFILL_API_SERVER_LIST: >-
                    #改为如下内容
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
                    #改为如下内容
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
### 1.2 配置omniinfer/tools/ansible/template/omni_infer_server_template_k2.yml:
    ```
    vars:
        docker_run_cmd: |
            docker run -it
                -v $CODE_PATH:${CODE_PATH}----删除

        run_vllm_server_prefill_cmd: |
            export CPU_AFFINITY_CONF=2----删除
            export ASCEND_GLOBAL_LOG_LEVEL=3
            ADDITIONAL_CONFIG='{"enable_omni_attn": false, "multi_rank_pull_kv": true}'

        run_vllm_server_decode_cmd:
            export CPU_AFFINITY_CONF=2----删除
            export ASCEND_GLOBAL_LOG_LEVEL=3
            EXTRA_ARGS='--enable-expert-parallel --disable-log-requests --max-num-seqs 48 --no-enable-prefix-caching'
            ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1, "use_ge_graph_cached":true},"enable_omni_attn": false , "multi_rank_pull_kv": true}'
    ```