# Multi-LoRA Benchmark

This directory includes Python scripts and configuration files needed to run some benchmarks using Ascend-vLLM in the Multi-LoRA scenario.

# Setup
## Overview
* `no_lora.py`: Benchmarking the base model by reporting the decode, prefill, and end-to-end latency.
* `multilora_inference.py`: Benchmarking the Multi-LoRA sceario by reporting the decode, prefill, and end-to-end latency.
* `config/common.json`: Paths to base model, LoRA adapters, and the configuration files for loading the model.
* `config/adapters.json`: Includes the information for the LoRA adapters, with a specific prompt for each one.
* `config/prompts.json`: Includes the prompts used in the No-LoRA scenario.

## Configuation
Befor running the scripts, you need to specify the model's path and the root path of the LoRA adapters in `common.json` for the Multi-LoRA scenario. Then, in order to create a workload for benchmarking, you need to specify the adapter's path and the prompt you want to use for that specific request in a `adapters.json` file and provide the path to this file in the `common.json`. If you just want to run the base model, just provide the prompts in a `prompts.json` file and put this path in the `common.json`.

Our benchmark can support three different workloads for choosing the LoRA adapters provided in one `adapters.json` and it would generate the reporting results in CSV format redirected to a file that is specified in `common.json`:
* distinct: Choose `batch_size` different LoRA adapters.
* uniform: Randomly choose `batch_size` adapters out of a set of $\sqrt{batch\_size}$ distinct adapters.
* identical: Repeat an identical LoRA adapter `batch_size` .

The others options in `common.json` are the arguments that we use to initialize the engine. There are several arguments specific for muti-LoRA support, inculding:

* `max_loras`: Controls the number of LoRAs that can be used in the same batch. Larger numbers will cause higher memory usage, as each LoRA slot requires its own preallocated tensor.
* `max_lora_rank`: Controls the maximum supported rank of all LoRAs. Larger numbers will cause higher memory usage. If you know that all LoRAs will use the same rank, it is recommended to set this as low as possible.
* `max_cpu_loras`: Controls the size of the CPU LoRA cache.

# Run

After specifying the paths and the engine arguments, it is recommended to run `python no_lora.py` first to make sure that the base model itself can be executed without a problem. Then you can execute `python multilora_inference.py` with the specified setting provided in the `common.json` to utilize the Multi-LoRA support. Also, you can control the batch sizes used for benchmarking, in the main function of each script.

