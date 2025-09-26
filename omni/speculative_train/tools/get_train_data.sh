#!/bin/bash
IP=7.150.13.75
PORT=7000
MODEL_NAME=qwen

INPUT_KEY=input
MAX_TOKENS=3072
MAX_CONCURRENCY=64
TEMPERATURE=0.6
BASIC_DIR=/data/d00646319/offline-dataset/death-no-end-1024_2048-3072
DATAFILE=$BASIC_DIR/3906.json
TOKEN_OUTPUT=$BASIC_DIR/token_output.json
TOKEN_SAVED_DIR=$BASIC_DIR/tokens
HIDDENSTATES_OUTPUT=$BASIC_DIR/hidden_states_output.json
HIDDENSTATES_SAVED_DIR=$BASIC_DIR/hidden-states
KEY_TOKEN=151667
MASK_DELTA=1


# python get_tokens.py \
#     --ip $IP \
#     --port $PORT \
#     --model-name $MODEL_NAME \
#     --datafile $DATAFILE \
#     --input-key $INPUT_KEY \
#     --max-tokens $MAX_TOKENS \
#     --max-concurrency $MAX_CONCURRENCY \
#     --temperature $TEMPERATURE \
#     --output $TOKEN_OUTPUT

python get_hidden_states.py \
    --ip $IP \
    --port $PORT \
    --model-name $MODEL_NAME \
    --input-dir $TOKEN_SAVED_DIR \
    --max-concurrency $MAX_CONCURRENCY \
    --output $HIDDENSTATES_OUTPUT

# python make_loss_mask.py \
#     --data-dir $HIDDENSTATES_SAVED_DIR \
#     --key-token $KEY_TOKEN \
#     --mask-delta $MASK_DELTA

