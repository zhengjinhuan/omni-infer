```

+omni_infer
    +—— accelerators                    # Accelerator package/plugin
        +—— placement                      # MoE experts balancer
        +—— sched                          # Schedulers, request level and instance level
            +—— global_proxy               # global proxy for load balance and PD two phase scheduling
        +—— attention                      # Attention accelerators
        +—— cache                          # Attention accelerators
        +—— pd                             # Prefill-Decode seperation accelerators
        +—— torchair                       # Torch.compile 
        +—— dist                           # Distributed communicators
        +—— quant                          # Quantization
        +—— preprocessing
        +—— postprocessing
    +—— models                          # Models inference scripts
        +—— common
            +—— layers
            	+—— linear
            	+—— activation
            	+—— layernorm
            	+—— moe
            	+—— attention
        +—— pangu 
        +—— deepseek 
        +—— qwen 
    +—— adaptors                        # Adaptors for mainstream inference frameworks, include monkey patches if necessary
        +—— vllm
        +—— vllm-ascend
        +—— sglang
        +—— llamacpp
    +—— infer_engines                   # Supported inference engines source code, simplify build process. Git submodules
        +—— vllm
        +—— vllm-ascend
        +—— sglang
        +—— llamacpp
    +—— tests                           # Unit test entry point, each component need to have its own 
                                          unit test dir and can be orchastrated from here.
    +—— benchmarks                      # Benchmark scripts and datasets
    +—— tools                           # miscellaneous tools

```
