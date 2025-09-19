import argparse
import hashlib
import math
import os
import time
from collections import defaultdict

import torch
import torch_npu


import torch.distributed as dist
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from tqdm import tqdm
from transformers import AutoTokenizer


from omni.speculative_train.models.auto import AutoDraftModelConfig, AutoEagleDraftModel

from omni.speculative_train.specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from omni.speculative_train.specforge.utils import (
    create_draft_config_from_target,
    get_last_checkpoint,
    print_on_rank0,
    print_with_rank,
    rank_0_priority,
)

from omni.speculative_train.data.dataset import (
    build_offline_eagle_dataset,
    prepare_dp_dataloaders,
)

from vllm.config import ParallelConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Train Eagle with offline data")

    # add model-related arguments
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument(
        "--draft-model-config",
        type=str,
        required=False,
        help="Draft model config path. If not provided, will auto-generate from target model.",
    )
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="The key of the embedding weight to load from the target model",
    )
    parser.add_argument(
        "--lm-head-key",
        type=str,
        default="lm_head.weight",
        help="The key of the lm head weight to load from the target model",
    )

    # add training-related arguments
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--train-hidden-states-path", type=str, required=True)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--eval-hidden-states-path", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--draft-global-batch-size", type=int, default=16)
    parser.add_argument("--draft-micro-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.015)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Total training steps. If not provided, will be calculated as num_epochs * steps_per_epoch",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--log-steps", type=int, default=50, help="Log training metrics every N steps"
    )
    parser.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="The length for Test-Time Training (TTT).",
    )
    parser.add_argument("--draft-attention-backend", type=str, default="flex_attention")
    # data processing type
    parser.add_argument("--chat-template", type=str, default="llama3")
    parser.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Whether the input data is preformatted text with the chat template already applied to the conversation messages.",
    )

    # distributed training
    parser.add_argument("--tp-size", type=int, default=1)

    # other args
    parser.add_argument("--cache-key", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    # parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=20,
        help="Timeout for collective communication in minutes",
    )
    # resume
    parser.add_argument("--resume", action="store_true")

    # report backend
    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        choices=["wandb", "tensorboard", "swanlab", "mlflow", "none"],
        help="The integration to report results and logs to.",
    )
    # wandb-specific args
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="The project name for W&B."
    )
    parser.add_argument(
        "--wandb-name", type=str, default=None, help="The run name for W&B."
    )
    parser.add_argument("--wandb-key", type=str, default=None, help="W&B API key.")
    # add swanlab-specific args ---
    parser.add_argument(
        "--swanlab-project",
        type=str,
        default=None,
        help="The project name for swanlab.",
    )
    parser.add_argument(
        "--swanlab-name",
        type=str,
        default=None,
        help="The experiment name for swanlab.",
    )
    parser.add_argument(
        "--swanlab-key",
        type=str,
        default=None,
        help="The API key for swanlab non-interactive login.",
    )
    # mlflow-specific args
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="The MLflow tracking URI. If not set, uses MLFLOW_TRACKING_URI environment variable or defaults to local './mlruns'.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=None,
        help="The MLflow experiment name. If not set, uses MLFLOW_EXPERIMENT_NAME environment variable.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="The MLflow run name. If not set, MLflow will auto-generate one.",
    )

    parser.add_argument("--build-dataset-num-proc", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-start-step", type=int, default=30)
    parser.add_argument("--profile-num-steps", type=int, default=4)
    parser.add_argument("--profile-record-shapes", action="store_true")

    args = parser.parse_args()

    return parser, args


# initialize
parser, args = parse_args()
init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
args.dp_size = dist.get_world_size() // args.tp_size


config = AutoDraftModelConfig.from_file("/data/model/qwq-32b-eagle/config.json")
model = AutoEagleDraftModel.from_config(config).npu()
print(model)
names = [item[0] for item in model.named_parameters()]
print(names)


with rank_0_priority():
    train_eagle3_dataset = build_offline_eagle_dataset(
        args.train_hidden_states_path,
        args.max_length,
        'pt',
    )

train_dataloader = prepare_dp_dataloaders(
    train_eagle3_dataset,
    args.draft_micro_batch_size,
    num_workers=4,
    shuffle=True,
    process_group=get_dp_group(),
    pin_memory=True,
)
print_with_rank("Initialized train dataloader")
print(train_dataloader)