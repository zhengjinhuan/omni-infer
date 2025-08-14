# 一、DeepSeek模型部署 

## 1、DeepSeek-R1-int8-A3-4P1D    

### 1.1 配置 omniinfer/tests/test_config/config_d.yaml:
    '''
	pattern_path: "../../omni/accelerators/placement/patterns/base_patterns/DSV3_baseline_64_devices_58_MoE_Layers.npy"
	max_redundant_per_expert: 10
	max_redundant_per_rank: 1
    '''

### 1.2 配置 omniinfer/tests/test_config/test_config_decode.json:
    '''
	"enable_mc2_v2": true
    '''  

### 1.3 配置 omniinfer/tools/ansible/template/omni_infer_server_template.yml:
    '''
	environment:
		MODEL_LEN_MAX_DECODE: "44000"
		CODE_PATH: "/workspace"
	vars:
	    run_vllm_server_decode_cmd:
			......
			HCCL_BUFFSIZE=800
			......
			GPU_UTIL=0.94
    '''

## 2、DeepSeek-R1-int8-A2-1P1D 

### 2.1 配置 omniinfer/tools/ansible/template/omni_infer_server_template.yml
    ```
    environment:
      MODEL_LEN_MAX_PREFILL: "32000"
      MODEL_LEN_MAX_DECODE: "44000"
      CODE_PATH: "/workspace"
    vars:
      run_vllm_server_prefill_cmd:
        ...
        HCCL_BUFFSIZE=20
        ...
        MODEL_EXTRA_CFG_PATH="${CODE_PATH}/omniinfer/tests/test_config/test_config_prefill_a2.json"
        EXTRA_ARGS='--max-num-batched-tokens 16896 --enforce-eager --enable-expert-parallel --disable-log-requests --max-num-seqs 8 --no-enable-prefix-caching'
        GPU_UTIL=0.95
        VLLM_ENABLE_MC2=0
        ...
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True 的后面补上：
        export ASCEND_PLATFORM=A2
        export CPU_AFFINITY_CONF=1,npu0:144-167,npu1:0-23,npu2:144-167,npu3:0-23,npu4:96-119,npu5:48-71,npu6:96-119,npu7:48-71
        export ASCEND_PROCESS_LOG_PATH="/tmp/v030_test/log_path"
      run_vllm_server_decode_cmd:
        ...
        HCCL_BUFFSIZE=20
        ...
        MODEL_EXTRA_CFG_PATH="${CODE_PATH}/omniinfer/tests/test_config/test_config_decode_a2.json"
        EXTRA_ARGS='--enable-expert-parallel --disable-log-requests --max-num-seqs 60 --no-enable-prefix-caching'
        GPU_UTIL=0.95
        ...
        VLLM_ENABLE_MC2=0
        ...
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True 的后面补上：
        export ASCEND_PLATFORM=A2
        export CPU_AFFINITY_CONF=1,npu0:144-167,npu1:0-23,npu2:144-167,npu3:0-23,npu4:96-119,npu5:48-71,npu6:96-119,npu7:48-71
        export ASCEND_PROCESS_LOG_PATH="/tmp/v030_test/log_path"
    tasks:
      - name: The executor synchronizes the global ranktable file to all instances.
          synchronize:
            src: "{{ ranktable_save_path }}/global/"
            dest: "{{ ranktable_save_path }}/global/"
          throttle: 1
          when: "inventory_hostname != groups['P'][0]"
          tags: ranktable
    ```

## 3、DeepSeek-R1-int8-A3-18P1D    

### 3.1 配置 omniinfer/tools/ansible/templage/omni_infer_server_template.yml
    '''
    environment:
      CODE_PATH: "/workspace"
  
      run_vllm_server_decode_cmd:
        --- 
        HCCL_BUFFSIZE=1600
        ---
        MODEL_EXTRA_CFG_PATH="${CODE_PATH}/omniinfer/tests/test_config/test_config_decode_dp288.json"
        EXTRA_ARGS='--enable-expert-parallel --disable-log-requests --max-num-seqs 48 --no-enable-prefix-caching'
        ---
        ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1, "use_ge_graph_cached":true, "block_num_floating_range":50}}'
    '''

### 3.2 配置 omniinfer/tests/test_config/config_d.yaml
    '''
    pattern_path:  "参考社区文档生成npy文件，填写对应npy文件路径"
  
    enable_dynamic: False
    max_redundant_per_expert: 30
    max_redundant_per_rank: 1
    '''

### 3.3 配置 omniinfer/tests/test_config/test_config_decode_dp288.json  
    '''
    "use_omni_placement": true,
    "omni_placement_config_path": "../../tests/test_config/config_d.yaml",
    "enable_mc2_v2": true,
    '''


## 4、DeepSeek-R1-0528-BF16-A3-4机组P-2P1D

### 4.1 配置 omniinfer/tools/ansible/templage/omni_infer_server_template.yml
    '''
    environment:
        MODEL_LEN_MAX_PREFILL: "70400"  
        MODEL_LEN_MAX_DECODE: "40000"
        CODE_PATH: "/workspace"

        PREFILL_LB_SDK: "least_total_load"
        DECODE_LB_SDK: "weighted_least_active"

    run_vllm_server_prefill_cmd: |
      source /usr/local/Ascend/latest/bin/setenv.bash
      HCCL_BUFFSIZE=200
      export HCCL_CONNECT_TIMEOUT=600
      export HCCL_EXEC_TIMEOUT=600  
      export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"   

      export HCCL_OP_EXPANSION_MODE=AIV   
      export CPU_AFFINITY_CONF=2   
      export OMNI_REUSE_PREFILLED_TOKENS=1   
      export OMNI_SKIP_DECODE_TOKENIZE=1   
      export TOKENIZER_PROC_POOL=1    
      
      EXTRA_ARGS='--max-num-batched-tokens 70400 --enforce-eager --no-enable-prefix-caching --enable-expert-parallel --disable-log-requests --max-num-seqs 128 --scheduler-cls=omni.adaptors.vllm.worker.tfas.tfas_scheduler.TFASScheduler'
      GPU_UTIL=0.90
      ADDITIONAL_CONFIG='{"tfas_scheduler_config": {"adjust_param": 8.708, "token_budget": 38000}}'
      
      bash ${CODE_PATH}/omniinfer/tools/scripts/pd_run.sh \
        --additional-config "$ADDITIONAL_CONFIG" \    

    run_vllm_server_decode_cmd: |

      source /usr/local/Ascend/latest/bin/setenv.bash    
      HCCL_BUFFSIZE=200
      export HCCL_CONNECT_TIMEOUT=600
      export HCCL_EXEC_TIMEOUT=600
      export HCCL_DETERMINISTIC=false  
      export HCCL_OP_EXPANSION_MODE=AIV   
      export CPU_AFFINITY_CONF=2   
      export OMNI_REUSE_PREFILLED_TOKENS=1   
      export OMNI_SKIP_DECODE_TOKENIZE=1   
      export TOKENIZER_PROC_POOL=1   

      EXTRA_ARGS='--enable-expert-parallel --disable-log-requests --max-num-seqs 4 --no-enable-prefix-caching'

      ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1, "use_ge_graph_cached":true}}'
    '''

### 4.2 配置 omniinfer/tests/test_config/config_p.yml
    '''  
    pattern_path: "参考社区文档生成npy文件，填写对应npy文件路径"
    max_moe_layer_num: 58
    max_batch_size: 256  
    max_top_k: 8    
        
    enable_rank_round_robin: False
    
    # Optimizers
    Optimizers:
      - expert_balance_optimizer.ExpertsBalanceOptimizer:
          batch_size: 16
    '''
    
### 4.3 配置 omniinfer/tests/test_config/test_config_decode.json
    '''
    {
        "model_parallel_config": {
            "o_proj_tp_size": 8
        },
        "operator_optimizition_config": {
            "unquant_bmm_nz": true, 
            "use_omni_placement": false,
            "use_super_kernel": false,
            "prefill_enable_mla_alltoall": false, 
            "use_mlaprolog": true,
            "control_accept_rate": -1,  
            "enable_mc2_v2": true,
            "enable_combine_addrmsnorm_fusion": true,
            "decode_gear_list": [
                8
            ]
        }
    }  
    '''

### 4.4 配置 omniinfer/tests/test_config/test_config_prefill.json
    '''
    {
        "model_parallel_config": {
            "o_proj_tp_size": 2
        },
        "operator_optimizition_config": {
            "enable_node_mlp": false,
            "unquant_bmm_nz": true,
            "prefill_enable_mla_alltoall": true,
        }
    }  
    '''

  
# 二、Qwen模型部署  

## 1、Qwen2.5-7B-A2/A3-混部  

### 1.1 权重配置文件 Qwen2.5-7B-Instruct/generation_config.json：  
    "repetition_penalty": 1.05,
    "temperature": 0.6,
    "top_k": 40,
    "top_p": 0.95,

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

## 2、QwQ-32B-BF16-A3-混部  

### 2.1 配置 omniinfer/tools/ansible/template/omni_infer_server_template_qwq.yml

    ```
    environment:
      MODEL_LEN_MAX_PREFILL: "61440"
      MODEL_LEN_MAX_DECODE: "61440"
      CODE_PATH: "/workspace"
    vars:
      run_vllm_server_prefill_cmd:
        ...
        EXTRA_ARGS='--max-num-batched-tokens 61440 --enforce-eager --disable-log-requests --max-num-seqs 64'
        ...
        GPU_UTIL=0.85
      run_vllm_server_decode_cmd:
        ...
        EXTRA_ARGS='--max-num-batched-tokens 61440 --disable-log-requests --max-num-seqs 64'
        ...
        将 export HCCL_OP_EXPANSION_MOD="AIV" 替换为 export HCCL_OP_EXPANSION_MODE=AIV
    tasks:
      - name: The executor synchronizes the global ranktable file to all instances.
          synchronize:
            src: "{{ ranktable_save_path }}/global/"
            dest: "{{ ranktable_save_path }}/global/"
          throttle: 1
          when: "inventory_hostname != groups['P'][0]"
          tags: ranktable
    ```