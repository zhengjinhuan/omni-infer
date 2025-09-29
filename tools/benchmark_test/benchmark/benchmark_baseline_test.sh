#!/bin/bash

set -x

dataset=
output_dir=
providers_path=./providers.yaml
epochs=6


cd ../benchmark_tools

  growth_rate=(6)
 for gr in "${growth_rate[@]}"; do
     echo "Start 8K-1K===================================================="$(date)"===================================================="
     python benchmark_parallel.py \
     --run-method climbing \
     --backend openai-chat \
     --providers-path $providers_path \
     --parallel-num 900 1000 --epochs $epochs \
     --prompt-tokens 8192 --output-tokens 1024 \
     --control-method queue \
     --use-spec-decode --num-speculative-tokens=1 --num-scheduler-steps=1 \
     --growth-rate $gr \
     --dataset-dir  $dataset \
     --benchmark-dir $output_dir
     echo "End 8K-1K===================================================="$(date)"===================================================="
     sleep 60
 done
