import contextlib
import gc
import logging
import os
import pickle
import weakref
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import patch

import torch
import torch.distributed
from torch.distributed import Backend, ProcessGroup
from sglang.srt.distributed.parallel_state import GroupCoordinator as GroupCoordinatorGPU
from sglang.srt.distributed.parallel_state import (
    init_model_parallel_group,
    get_world_group,
)
import logging
logger = logging.getLogger(__file__)

class GroupCoordinator(GroupCoordinatorGPU):
    pass
_MLP_TP: Optional[GroupCoordinator] = None
_O_PROJ_TP: Optional[GroupCoordinator] = None
_O_PROJ_DP: Optional[GroupCoordinator] = None

def get_mlp_tp_group() -> GroupCoordinator:
    assert _MLP_TP is not None, "MLP TP group is not initialized"
    return _MLP_TP

def get_o_proj_tp_group() -> GroupCoordinator:
    assert _O_PROJ_TP is not None, "O PROJ TP group is not initialized"
    return _O_PROJ_TP

def get_o_proj_dp_group() -> GroupCoordinator:
    assert _O_PROJ_DP is not None, "O PROJ DP group is not initialized"
    return _O_PROJ_DP


def initialize_mlp_tp_group(backend, tensor_model_parallel_size, dp_size) -> None:
    """Initialize tensor parallel group for MLP layers.
    
    Args:
        mlp_tp_size (int): TP size used for MLP layers.
        backend (str): torch.distributed backend, e.g. "nccl".
    """

    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    global _MLP_TP
    if _MLP_TP is not None:
        raise RuntimeError("_MLP_TP has already been initialized")

    world_size: int = torch.distributed.get_world_size()
    rank: int = torch.distributed.get_rank()
    mlp_tp_size_str = os.getenv("MLP_TP_SIZE")
    if mlp_tp_size_str is None:
        mlp_tp_size = tensor_model_parallel_size
    else:
        mlp_tp_size = int(mlp_tp_size_str)
    if world_size % mlp_tp_size != 0:
        raise RuntimeError(
            f"MLP TP Size ({mlp_tp_size}) must divide world size ({world_size})"
        )

    num_groups: int = world_size // mlp_tp_size
    group_ranks = []
    for i in range(num_groups):
        ranks = list(range(i * mlp_tp_size, (i + 1) * mlp_tp_size))
        group_ranks.append(ranks)

    if backend is None:
        backend = torch.distributed.get_backend()  # 默认取当前 backend

    _MLP_TP = init_model_parallel_group(
        group_ranks,
        local_rank=get_world_group().local_rank,
        backend=backend,
        use_message_queue_broadcaster=True,
        group_name="mlp_tp_group",
    )


def initialize_o_proj_tp_group(backend, tensor_model_parallel_size, dp_size) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()
    o_proj_tp_size_str = os.getenv("O_PROJ_TP_SIZE")
    if o_proj_tp_size_str is None:
        o_proj_tp_size = tensor_model_parallel_size // dp_size
    else:
        o_proj_tp_size = int(o_proj_tp_size_str)
    if world_size % o_proj_tp_size != 0:
        raise RuntimeError(f"o_proj TP Size ({o_proj_tp_size}) should be divisible by world size ({world_size})")
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    num_local_groups: int = world_size // o_proj_tp_size
    global _O_PROJ_TP
    if _O_PROJ_TP is not None:
        raise RuntimeError("_O_PROJ_TP must be None")
    group_ranks = []
    for i in range(num_local_groups):
        ranks = list(range(i * o_proj_tp_size, (i + 1) * o_proj_tp_size))
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    _O_PROJ_TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=False,
        group_name="o_proj_tp_group",
    )


def initialize_o_proj_dp_group(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()
    o_proj_tp_size = int(os.getenv("O_PROJ_TP_SIZE", "1"))
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    dp_size: int = world_size // o_proj_tp_size
    global _O_PROJ_DP
    if _O_PROJ_DP is not None:
        raise RuntimeError("_O_PROJ_DP must be None")
    all_ranks = torch.arange(world_size).reshape(dp_size, o_proj_tp_size)
    group_ranks = all_ranks.transpose(0, 1)
    group_ranks = [x.tolist() for x in group_ranks]
    # message queue broadcaster is only used in tensor model parallel group
    _O_PROJ_DP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=False,
        group_name="o_proj_dp_group",
    )


def initialize_add_groups(backend, tensor_model_parallel_size, dp_size):

    initialize_mlp_tp_group(backend, tensor_model_parallel_size, dp_size)

    initialize_o_proj_dp_group(backend)
    initialize_o_proj_tp_group(backend, tensor_model_parallel_size, dp_size)

def get_mlp_tp_group_parallel_world_size():
    """Return world size for the mlp tensor parallel group."""
    return get_mlp_tp_group().world_size

def get_mlp_tp_group_parallel_rank():
    """Return my rank for the mlp tensor parallel group."""
    return get_mlp_tp_group().rank_in_group

def get_o_proj_tp_group_parallel_world_size():
    """Return world size for the o_proj tensor parallel group."""
    return get_o_proj_tp_group().world_size

def get_o_proj_tp_group_parallel_rank():
    """Return my rank for the o_proj tensor parallel group."""
    return get_o_proj_tp_group().rank_in_group

def get_o_proj_dp_group_parallel_world_size():
    """Return world size for the o_proj data parallel group."""
    return get_o_proj_dp_group().world_size

def get_o_proj_dp_group_parallel_rank():
    """Return my rank for the o_proj data parallel group."""
    return get_o_proj_dp_group().rank_in_group
