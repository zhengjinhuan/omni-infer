NUM_GPUS=${1:-16}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    train_qwen2_eagle.py \
    --target-model-path /data/model/QwQ-32B \
    --draft-model-config /data/model/qwq-32b-eagle/config.json \
    --train-data-path /data/d00646319/offline-dataset/longbench-2048-2048 \
    --train-hidden-states-path /data/d00646319/offline-dataset/longbench-2048-2048 \
    --num-epochs 10 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template qwen \
    --output-dir ./ \
    --ttt-length 1 \
    --resume
