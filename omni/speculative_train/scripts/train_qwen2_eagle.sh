NUM_GPUS=${1:-16}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    -m omni.speculative_train.train_qwen2_eagle \
    --target-model-path Qwen/Qwen3-Coder-480B-A35B-Instruct \
    --draft-model-config $ROOT_DIR/configs/qwen3-coder-480B-A35B-instruct-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/opc.jsonl \
    --train-hidden-states-path $ROOT_DIR/cache/hidden_states \
    --output-dir $ROOT_DIR/outputs/Qwen3-Coder-480B-A35B-Instruct \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --resume
