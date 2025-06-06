from typing import Optional, List

import torch
from vllm.distributed.parallel_state import GroupCoordinator, get_world_group, init_model_parallel_group


class AscendParallelManager:
    """Manages expert parallel (EP) and expert tensor parallel (ETP) groups for vLLM-Ascend."""

    def __init__(self):
        """Initialize the parallel manager with no groups."""
        self._ep_group: Optional[GroupCoordinator] = None
        self._etp_group: Optional[GroupCoordinator] = None

    def get_ep_group(self) -> GroupCoordinator:
        """Get the expert parallel group.

        Returns:
            GroupCoordinator: The expert parallel group coordinator.

        Raises:
            RuntimeError: If the expert parallel group is not initialized.
        """
        if self._ep_group is None:
            raise RuntimeError("Expert parallel group is not initialized")
        return self._ep_group

    def get_etp_group(self) -> GroupCoordinator:
        """Get the expert tensor parallel group.

        Returns:
            GroupCoordinator: The expert tensor parallel group coordinator.

        Raises:
            RuntimeError: If the expert tensor parallel group is not initialized.
        """
        if self._etp_group is None:
            raise RuntimeError("Expert tensor parallel group is not initialized")
        return self._etp_group

    def initialize(
        self,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        expert_tensor_parallel_size: int = 1,
        backend: Optional[str] = None
    ) -> None:
        """Initialize expert parallel and expert tensor parallel groups.

        Args:
            tensor_model_parallel_size: Size of the tensor model parallel group.
            pipeline_model_parallel_size: Size of the pipeline model parallel group.
            expert_tensor_parallel_size: Size of the expert tensor parallel group.
            backend: The communication backend to use. If None, uses the world group's backend.

        Raises:
            RuntimeError: If torch.distributed is not initialized or if group sizes are invalid.
        """
        if not torch.distributed.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before initializing parallel groups")

        world_size = torch.distributed.get_world_size()
        backend = backend or torch.distributed.get_backend(get_world_group().device_group)

        self._validate_group_sizes(world_size, expert_tensor_parallel_size)

        self._initialize_ep_group(world_size, expert_tensor_parallel_size, backend)
        self._initialize_etp_group(world_size, expert_tensor_parallel_size, backend)

    def _validate_group_sizes(self, world_size: int, expert_tensor_parallel_size: int) -> None:
        """Validate the group sizes against the world size.

        Args:
            world_size: The total number of processes in the world group.
            expert_tensor_parallel_size: Size of the expert tensor parallel group.

        Raises:
            ValueError: If the group sizes are invalid.
        """
        if expert_tensor_parallel_size <= 0 or world_size % expert_tensor_parallel_size != 0:
            raise ValueError(
                f"Expert tensor parallel size {expert_tensor_parallel_size} must be positive and "
                f"divide world size {world_size} evenly"
            )

    def _initialize_ep_group(
        self, world_size: int, expert_tensor_parallel_size: int, backend: str
    ) -> None:
        """Initialize the expert parallel group.

        Args:
            world_size: The total number of processes in the world group.
            expert_tensor_parallel_size: Size of the expert tensor parallel group.
            backend: The communication backend to use.
        """
        num_expert_parallel_groups = world_size // expert_tensor_parallel_size
        group_ranks = [
            list(range(i, world_size, num_expert_parallel_groups))
            for i in range(num_expert_parallel_groups)
        ]
        self._ep_group = init_model_parallel_group(
            group_ranks=group_ranks,
            local_rank=get_world_group().local_rank,
            backend=backend,
            group_name="ep"
        )

    def _initialize_etp_group(
        self, world_size: int, expert_tensor_parallel_size: int, backend: str
    ) -> None:
        """Initialize the expert tensor parallel group.

        Args:
            world_size: The total number of processes in the world group.
            expert_tensor_parallel_size: Size of the expert tensor parallel group.
            backend: The communication backend to use.
        """
        num_expert_tensor_parallel_groups = world_size // expert_tensor_parallel_size
        group_ranks = [
            list(range(i * expert_tensor_parallel_size, (i + 1) * expert_tensor_parallel_size))
            for i in range(num_expert_tensor_parallel_groups)
        ]
        self._etp_group = init_model_parallel_group(
            group_ranks=group_ranks,
            local_rank=get_world_group().local_rank,
            backend=backend,
            group_name="etp"
        )

    def destroy(self) -> None:
        """Destroy the expert parallel and expert tensor parallel groups."""
        if self._ep_group is not None:
            self._ep_group.destroy()
            self._ep_group = None
        if self._etp_group is not None:
            self._etp_group.destroy()
            self._etp_group = None

# Singleton instance for global access
_ascend_parallel_manager = AscendParallelManager()


def get_ep_group() -> GroupCoordinator:
    """Get the expert parallel group from the global manager.

    Returns:
        GroupCoordinator: The expert parallel group coordinator.
    """
    return _ascend_parallel_manager.get_ep_group()


def get_etp_group() -> GroupCoordinator:
    """Get the expert tensor parallel group from the global manager.

    Returns:
        GroupCoordinator: The expert tensor parallel group coordinator.
    """
    return _ascend_parallel_manager.get_etp_group()


def init_ascend_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    backend: Optional[str] = None
) -> None:
    """Initialize Ascend model parallel groups using the global manager.

    Args:
        tensor_model_parallel_size: Size of the tensor model parallel group.
        pipeline_model_parallel_size: Size of the pipeline model parallel group.
        expert_tensor_parallel_size: Size of the expert tensor parallel group.
        backend: The communication backend to use.
    """
    _ascend_parallel_manager.initialize(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        expert_tensor_parallel_size,
        backend
    )


def destroy_ascend_model_parallel() -> None:
    """Destroy Ascend model parallel groups using the global manager."""
    _ascend_parallel_manager.destroy()
