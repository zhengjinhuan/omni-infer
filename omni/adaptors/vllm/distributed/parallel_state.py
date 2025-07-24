# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, List

import torch
import torch.distributed
from vllm.distributed import GroupCoordinator as GroupCoordinatorGPU
from vllm.logger import logger
from vllm.distributed import (
    parallel_state,
    init_model_parallel_group,
    get_world_group,
    get_dp_group,
    get_ep_group
)
from vllm.logger import logger
from omni.models.common.config.model_config import model_extra_config
import os

initialize_model_parallel_default = parallel_state.initialize_model_parallel

_DIE_PER_NODE_910C = 16
_DIE_PER_NODE_910B = 8

def get_npu_device_count():
    if os.getenv("ASCEND_PLATFORM", "A3") == "A2":
        return _DIE_PER_NODE_910B
    else:
        return _DIE_PER_NODE_910C


class GroupCoordinator(GroupCoordinatorGPU):

    def all_to_all(
        self,
        input_: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: Optional[List[int]] = None,
        gather_sizes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        return self.device_communicator.all_to_all(input_, scatter_dim, gather_dim, scatter_sizes, gather_sizes)

    def swap(self, input: torch.Tensor, method="all2allv") -> torch.Tensor:
        if len(self.ranks) != 2:
            return input

        if method == "all2allv":
            rank_0 = self.ranks[0]
            rank_1 = self.ranks[1]
            input_shape = input.shape
            input = input.view(-1)
            output = torch.empty_like(input, dtype=input.dtype, device=input.device)

            if self.rank == rank_0:
                split_sizes = [0, input.shape[0]]
            elif self.rank == rank_1:
                split_sizes = [input.shape[0], 0]

            torch.distributed.all_to_all_single(output, input,
                                                output_split_sizes=split_sizes,
                                                input_split_sizes=split_sizes,
                                                group=self.device_group)
            return output.view(input_shape)

        if method == "allgather":
            rank_0 = self.ranks[0]
            rank_1 = self.ranks[1]
            output = torch.empty_like(input, dtype=input.dtype, device=input.device)
            input_size = input.size()
            output_size= (input_size[0] * 2, ) + input_size[1:]
            output_tensor = torch.empty(output_size, dtype=input.dtype, device=input.device)
            torch.distributed.all_gather_into_tensor(output_tensor, input, group=self.device_group)

            if self.rank == rank_1:
                output, _ = torch.split(output_tensor, output_tensor.shape[0] // 2, dim=0)
            elif self.rank == rank_0:
                _, output = torch.split(output_tensor, output_tensor.shape[0] // 2, dim=0)

            return output
        return input

    def reduce_scatter(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        return self.device_communicator.reduce_scatter(input_)


_NUM_COMM_GROUP = 2
_LOCAL_COMM_LIST = None
_CROSS_COMM_LIST = None
_GLOBAL_COMM_LIST = None
_CROSS_FAR_COMM_LIST = None
_CROSS_NEAR_COMM_LIST = None
_CROSS_ROUND_COMM_LIST = None
# kept for backward compatibility
_LOCAL_WORLD: Optional[GroupCoordinator] = None
_O_PROJ_WORLD: Optional[GroupCoordinator] = None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    backend: Optional[str] = None,
) -> None:
    initialize_model_parallel_default(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        backend,
    )

    initialize_local_world_group(backend)
    if model_extra_config.operator_opt_config.two_stage_comm:
        initialize_cross_comm_group_list(backend)
        initialize_local_comm_group_list(backend)
    else:
        initialize_world_comm_group_list(backend)
        initialize_local_comm_group_list(backend)
        initialize_cross_comm_group_list(backend)

    if model_extra_config.operator_opt_config.enable_round_pipeline_comm:
        num_nodes = torch.distributed.get_world_size() // get_npu_device_count()
        if num_nodes == 4:
            initialize_round_cross_comm_group_list(backend)
            model_extra_config.operator_opt_config.enable_pipeline_comm = 0
        else:
            model_extra_config.operator_opt_config.enable_pipeline_comm = 1
            model_extra_config.operator_opt_config.enable_round_pipeline_comm = 0

    if model_extra_config.operator_opt_config.enable_pipeline_comm:
        initialize_far_cross_comm_group_list(backend)
        initialize_near_cross_comm_group_list(backend)

    if model_extra_config.parall_config.o_proj_tp_size > 1:
        initialize_o_proj_tp_world_group(backend)


def get_mlp_tp_size():
    # Can be enabled
    if model_extra_config.operator_opt_config.enable_node_mlp:
        return get_local_group_world_size_from_list(0)
    else:
        return get_expert_parallel_world_size()


def get_mlp_tp_rank():
    if model_extra_config.operator_opt_config.enable_node_mlp:
        return get_local_group_rank_from_list(0)
    else:
        return get_expert_parallel_rank()


def get_mlp_world_group():
    return get_local_group_from_list(0)


def calculate_effective_local_size(local_size: int, world_size: int) -> int:
    """
    Calculate the effective local size based on available devices and world size.

    Args:
        local_size (int): Number of available NPU devices.
        world_size (int): Total number of processes in the distributed setup.

    Returns:
        int: The effective local size (minimum of local_size and world_size).

    Notes:
        - Logs a warning if not all devices are used.
        - Ensures world_size is divisible by the effective local size (raises AssertionError otherwise).
    """
    effective_local_size = min(local_size, world_size)
    if effective_local_size < local_size:
        logger.info(f"Note: Using only {effective_local_size} of {local_size} available NPU devices")

    if world_size % effective_local_size != 0:
        raise AssertionError(
            f"world_size ({world_size}) must be divisible by effective_local_size ({effective_local_size})"
        )
    return effective_local_size


def initialize_o_proj_tp_world_group(backend) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()
    local_size = model_extra_config.parall_config.o_proj_tp_size
    if local_size < 1:
        raise RuntimeError("model_extra_config.parall_config.o_proj_tp_size must larger than or equal to 1")
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    num_local_groups: int = world_size // local_size
    global _O_PROJ_WORLD
    if _O_PROJ_WORLD is not None:
        raise RuntimeError("_O_PROJ_WORLD must be None")
    group_ranks = []
    for i in range(num_local_groups):
        ranks = list(range(i * local_size, (i + 1) * local_size))
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    _O_PROJ_WORLD = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="o_proj_local",
    )


def get_o_proj_tp_size():
    return get_o_proj_world_group().world_size


def get_o_proj_tp_rank():
    return get_o_proj_world_group().rank_in_group


def get_o_proj_world_group():
    return _O_PROJ_WORLD


def initialize_local_world_group(backend) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()
    local_size = calculate_effective_local_size(torch.npu.device_count() if not int(os.getenv("NO_NPU_MOCK", "0")) \
        else len(os.getenv("ASCEND_RT_VISIBLE_DEVICES").split(",")), world_size)

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    num_local_groups: int = world_size // local_size
    global _LOCAL_WORLD
    if _LOCAL_WORLD is not None:
        raise RuntimeError("_LOCAL_WORLD must be None")
    group_ranks = []
    for i in range(num_local_groups):
        ranks = list(range(i * local_size, (i + 1) * local_size))
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    _LOCAL_WORLD = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="world_local",
    )


def initialize_local_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()
    local_size = calculate_effective_local_size(torch.npu.device_count() if not int(os.getenv("NO_NPU_MOCK", "0")) \
        else len(os.getenv("ASCEND_RT_VISIBLE_DEVICES").split(",")), world_size)

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    num_local_groups: int = world_size // local_size
    global _LOCAL_COMM_LIST
    if _LOCAL_COMM_LIST is not None:
        raise RuntimeError("_LOCAL_COMM_LIST must be None")
    _LOCAL_COMM_LIST = list()
    group_ranks = []
    for i in range(num_local_groups):
        ranks = list(range(i * local_size, (i + 1) * local_size))
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    for i in range(_NUM_COMM_GROUP):
        _LOCAL_COMM_LIST.append(
            init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                use_message_queue_broadcaster=True,
                group_name="world_local",
            )
        )


def initialize_cross_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()
    local_size = calculate_effective_local_size(torch.npu.device_count() if not int(os.getenv("NO_NPU_MOCK", "0")) \
        else len(os.getenv("ASCEND_RT_VISIBLE_DEVICES").split(",")), world_size)

    server_size = world_size // local_size

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    # Build the pipeline model-parallel groups.
    num_cross_groups: int = world_size // server_size
    global _CROSS_COMM_LIST
    if _CROSS_COMM_LIST is not None:
        raise RuntimeError("pipeline model parallel group is already initialized")
    _CROSS_COMM_LIST = list()
    group_ranks = []
    for i in range(num_cross_groups):
        ranks = list(range(i, world_size, num_cross_groups))
        group_ranks.append(ranks)
    # pipeline parallel does not need custom allreduce

    for i in range(_NUM_COMM_GROUP):
        _CROSS_COMM_LIST.append(
            init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                group_name="world_cross",
            )
        )


def initialize_world_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    global _GLOBAL_COMM_LIST
    if _GLOBAL_COMM_LIST is not None:
        raise RuntimeError("_GLOBAL_COMM_LIST must be None")
    _GLOBAL_COMM_LIST = list()
    group_ranks = [range(world_size)]
    for i in range(_NUM_COMM_GROUP):
        _GLOBAL_COMM_LIST.append(
            init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                use_message_queue_broadcaster=True,
                group_name="world_local",
            )
        )


def get_local_world_group() -> GroupCoordinator:
    return _LOCAL_WORLD


def get_local_group_from_list(idx: int) -> GroupCoordinator:
    return _LOCAL_COMM_LIST[idx]


def get_cross_group_from_list(idx: int) -> GroupCoordinator:
    return _CROSS_COMM_LIST[idx]


def get_world_group_from_list(idx: int) -> GroupCoordinator:
    return _GLOBAL_COMM_LIST[idx]


def get_data_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    group = get_dp_group()
    if group is not None:
        return group.world_size
    else:
        return 1


def get_data_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    group = get_dp_group()
    if group is not None:
        return group.rank_in_group
    else:
        return 0


def get_expert_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_ep_group().world_size


def get_expert_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return get_ep_group().rank_in_group


def get_local_group_world_size_from_list(idx: int):
    return _LOCAL_COMM_LIST[idx].world_size


def get_local_group_rank_from_list(idx: int):
    return _LOCAL_COMM_LIST[idx].rank_in_group

def get_near_cross_group_from_list(idx: int) -> GroupCoordinator:
    return _CROSS_NEAR_COMM_LIST[idx]

def get_far_cross_group_from_list(idx: int) -> GroupCoordinator:
    return _CROSS_FAR_COMM_LIST[idx]


def get_local_group_size():
    return get_local_group_from_list(idx=0).world_size

def get_local_group_rank():
    return get_local_group_from_list(idx=0).rank_in_group

def initialize_round_cross_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    local_size = get_npu_device_count()
    assert world_size % local_size == 0

    server_size = world_size // local_size

    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)

    num_cross_groups: int = (world_size // server_size)
    global _CROSS_ROUND_COMM_LIST
    assert _CROSS_ROUND_COMM_LIST is None, (
        "pipeline model parallel group is already initialized")
    _CROSS_ROUND_COMM_LIST = list()

    group_ranks_round0 = []
    group_ranks_round1 = []
    group_ranks_round2 = []
    for i in range(num_cross_groups):
        ranks = [[i + 0 * num_cross_groups, i + 1 * num_cross_groups], \
                [i + 2 * num_cross_groups, i + 3 * num_cross_groups]]
        group_ranks_round0.extend(ranks)

        ranks = [[i + 0 * num_cross_groups, i + 2 * num_cross_groups], \
                [i + 1 * num_cross_groups, i + 3 * num_cross_groups]]
        group_ranks_round1.extend(ranks)

        ranks = [[i + 0 * num_cross_groups, i + 3 * num_cross_groups], \
                [i + 1 * num_cross_groups, i + 2 * num_cross_groups]]
        group_ranks_round2.extend(ranks)

    
    _CROSS_ROUND_COMM_LIST.append(init_model_parallel_group(group_ranks_round0,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="world_round0_cross"))

    _CROSS_ROUND_COMM_LIST.append(init_model_parallel_group(group_ranks_round1,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="world_round1_cross"))

    _CROSS_ROUND_COMM_LIST.append(init_model_parallel_group(group_ranks_round2,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="world_round2_cross"))

def get_round_cross_group_from_list(round: int) -> GroupCoordinator:
    return _CROSS_ROUND_COMM_LIST[round]

def initialize_far_cross_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    local_size  = get_npu_device_count()
    assert world_size % local_size == 0

    server_size = world_size // local_size

    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)
    
    num_cross_groups: int = (world_size // server_size)
    global _CROSS_FAR_COMM_LIST
    assert _CROSS_FAR_COMM_LIST is None, (
        "pipeline model parallel group is already initialized")
    _CROSS_FAR_COMM_LIST = list()
    group_ranks = []
    for i in range(num_cross_groups):
        for j in range(server_size // 2):
            ranks = list(range(i + j * num_cross_groups, world_size, world_size // 2))
            group_ranks.append(ranks)

    for i in range(_NUM_COMM_GROUP):
        _CROSS_FAR_COMM_LIST.append(init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="world_far_cross"))

def initialize_near_cross_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    local_size = get_npu_device_count()
    assert world_size % local_size == 0

    server_size = world_size // local_size

    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)
    
    num_cross_groups: int = (world_size // server_size)
    global _CROSS_NEAR_COMM_LIST
    assert _CROSS_NEAR_COMM_LIST is None, (
        "pipeline model parallel group is already initialized")
    _CROSS_NEAR_COMM_LIST = list()
    group_ranks = []
    for i in range(num_cross_groups):
        ranks = list(range(i, world_size // 2, num_cross_groups))
        group_ranks.append(ranks)

        ranks = list(range(world_size // 2 + i, world_size, num_cross_groups))
        group_ranks.append(ranks)

    for i in range(_NUM_COMM_GROUP):
        _CROSS_NEAR_COMM_LIST.append(init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="world_near_cross"))