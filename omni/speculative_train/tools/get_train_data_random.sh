#!/bin/bash
IP=7.150.13.75
PORT=7000
MODEL_NAME=qwen

INPUT_KEY=input
MAX_TOKENS=3072
MAX_CONCURRENCY=64
TEMPERATURE=0.6
BASIC_DIR=/data/d00646319/offline-dataset/random-4096
HIDDENSTATES_OUTPUT=$BASIC_DIR/hidden_states_output.json
KEY_TOKEN=151667
MASK_DELTA=1
VOCAB_SIZE=152064
INPUT_LEN=4096

python get_hidden_states_with_random_input.py \
    --ip $IP \
    --port $PORT \
    --model-name $MODEL_NAME \
    --vocab-size $VOCAB_SIZE \
    --input-len $INPUT_LEN \
    --max-concurrency $MAX_CONCURRENCY \
    --output $HIDDENSTATES_OUTPUT
