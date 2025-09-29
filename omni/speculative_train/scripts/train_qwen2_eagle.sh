LOGFILE=re_$(date +%Y-%m%d-%H%M-%S).log

NUM_GPUS=${1:-16}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    train_offline_eagle.py \
    --target-model-path /data/model/QwQ-32B \
    --draft-model-config /data/model/qwq-32b-eagle/config.json \
    --train-data-path /data/d00646319/offline-dataset/death-no-end_512_1024-2048/hidden-states \
    --num-epochs 10 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --output-dir ./train-result/ \
    --ttt-length 3 \
    --checkpoint /data/model/qwq-32b-eagle \
    --resume \
    --profile --profile-record-shapes \
    | tee $LOGFILE
