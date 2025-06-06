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

from typing import List, Optional

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase


class NPUCommunicator(DeviceCommunicatorBase):
    """Handles all-to-all communication for NPU devices in vLLM."""

    def __init__(
        self,
        cpu_group: dist.ProcessGroup,
        device: Optional[torch.device] = None,
        device_group: Optional[dist.ProcessGroup] = None,
        unique_name: str = ""
    ) -> None:
        """Initialize the NPU communicator.

        Args:
            cpu_group: The process group for CPU communication.
            device: The NPU device to use. If None, uses the current NPU device.
            device_group: The process group for device communication. If None, uses the default group.
            unique_name: A unique identifier for the communicator.

        Raises:
            RuntimeError: If the device is not an NPU device.
        """
        super().__init__(cpu_group, device, device_group, unique_name)
        self.device = device if device is not None else torch.npu.current_device()
        if self.device.type != "npu":
            raise RuntimeError(f"Expected NPU device, got {self.device.type}")

    def all_to_all(
        self,
        input_: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: Optional[List[int]] = None,
        gather_sizes: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Perform all-to-all communication on the input tensor.

        Args:
            input_: The input tensor to scatter.
            scatter_dim: The dimension along which to scatter the input tensor.
            gather_dim: The dimension along which to gather the output tensor.
            scatter_sizes: Optional list of sizes for scattering the input tensor.
            gather_sizes: Optional list of sizes for gathering the output tensor.

        Returns:
            torch.Tensor: The output tensor after all-to-all communication.

        Raises:
            ValueError: If scatter_dim or gather_dim is invalid, or if scatter_sizes and gather_sizes are inconsistent.
        """
        scatter_dim = self._normalize_dim(scatter_dim, input_.dim())
        gather_dim = self._normalize_dim(gather_dim, input_.dim())
        self._validate_inputs(input_, scatter_dim, gather_sizes, scatter_sizes)

        input_list = self._prepare_input_list(input_, scatter_dim, scatter_sizes)
        output_list = self._prepare_output_list(input_list, gather_dim, gather_sizes)
        
        dist.all_to_all(output_list, input_list, group=self.device_group)
        return torch.cat(output_list, dim=gather_dim).contiguous()

    def _normalize_dim(self, dim: int, tensor_dim: int) -> int:
        """Normalize a dimension index to be non-negative.

        Args:
            dim: The dimension index to normalize.
            tensor_dim: The number of dimensions in the tensor.

        Returns:
            int: The normalized dimension index.

        Raises:
            ValueError: If the dimension is out of bounds.
        """
        if dim < -tensor_dim or dim >= tensor_dim:
            raise ValueError(f"Dimension {dim} is out of bounds for tensor with {tensor_dim} dimensions")
        return dim if dim >= 0 else dim + tensor_dim

    def _validate_inputs(
        self,
        input_: torch.Tensor,
        scatter_dim: int,
        gather_sizes: Optional[List[int]],
        scatter_sizes: Optional[List[int]]
    ) -> None:
        """Validate input parameters for all-to-all communication.

        Args:
            input_: The input tensor.
            scatter_dim: The scatter dimension.
            gather_sizes: Optional list of gather sizes.
            scatter_sizes: Optional list of scatter sizes.

        Raises:
            ValueError: If scatter_sizes or gather_sizes are invalid or inconsistent.
        """
        if scatter_sizes is not None and gather_sizes is not None:
            if len(scatter_sizes) != self.world_size or len(gather_sizes) != self.world_size:
                raise ValueError(
                    f"scatter_sizes and gather_sizes must have length {self.world_size}, "
                    f"got {len(scatter_sizes)} and {len(gather_sizes)}"
                )
            if sum(scatter_sizes) != input_.size(scatter_dim):
                raise ValueError(
                    f"Sum of scatter_sizes ({sum(scatter_sizes)}) does not match "
                    f"input size at scatter_dim ({input_.size(scatter_dim)})"
                )
        elif scatter_sizes is not None or gather_sizes is not None:
            raise ValueError("Both scatter_sizes and gather_sizes must be provided or neither")

    def _prepare_input_list(
        self, input_: torch.Tensor, scatter_dim: int, scatter_sizes: Optional[List[int]]
    ) -> List[torch.Tensor]:
        """Prepare the input tensor list for all-to-all communication.

        Args:
            input_: The input tensor to scatter.
            scatter_dim: The dimension along which to scatter.
            scatter_sizes: Optional list of sizes for scattering.

        Returns:
            List[torch.Tensor]: List of contiguous tensor chunks for scattering.
        """
        if scatter_sizes is not None:
            return [t.contiguous() for t in torch.split(input_, scatter_sizes, scatter_dim)]
        return [t.contiguous() for t in torch.tensor_split(input_, self.world_size, scatter_dim)]

    def _prepare_output_list(
        self, input_list: List[torch.Tensor], gather_dim: int, gather_sizes: Optional[List[int]]
    ) -> List[torch.Tensor]:
        """Prepare the output tensor list for all-to-all communication.

        Args:
            input_list: List of input tensor chunks.
            gather_dim: The dimension along which to gather.
            gather_sizes: Optional list of sizes for gathering.

        Returns:
            List[torch.Tensor]: List of empty tensors for gathering.
        """
        if gather_sizes is not None:
            tensor_shape_base = input_list[self.rank].size()
            output_list = []
            for size in gather_sizes:
                tensor_shape = list(tensor_shape_base)
                tensor_shape[gather_dim] = size
                output_list.append(
                    torch.empty(tensor_shape, dtype=input_list[0].dtype, device=input_list[0].device)
                )
        else:
            output_list = [torch.empty_like(chunk) for chunk in input_list]
        return output_list
