# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import csv
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, cast
import numpy as np
import torch
import torch_npu
import ctypes

from typing import Optional
from omni_planner.cluster_status import ClusterStatus
from omni_planner.placement_handler import create_cluster_activation, create_placement_manager, init_dram_weights
from omni_planner.optim.optimizers import Optimizer
from omni_planner.optim.optimizers_loader import _create_optimizers
from omni_planner.config import Config
from omni_planner.expert_mapping import ExpertMapping
from omni_planner.utils import calculate_time

import time

class OmniPlannerMeta(type):
    """Metaclass to implement singleton pattern for OmniPlanner."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    @classmethod
    def cleanup(cls):
        if cls in cls._instances and not cls._cleanup_called:
            cls._instances[cls].cleanup()
            del cls._instances[cls]
            cls._cleanup_called = True

class OmniPlanner(metaclass=OmniPlannerMeta):
    """
    Optimizes token-to-expert mapping using multiple optimizers.
    Manages expert deployment across distributed systems.

    Attributes:
        config: Configuration object for planner settings
        cluster_status: Cluster status monitor
        optimizers: List of optimization algorithms
        expert_mapping: Expert deployment pattern mapping
    """
    def __init__(self, config_file: str = "/etc/omni/config.yaml", device: str = "npu",
                 rank: int = None, world_size: int = None, num_devices_per_host: int = 16):
        """Initialize OmniPlanner with configuration and distributed settings.

        Args:
            config_file: Path to configuration YAML file
            device: Target device type (e.g., "npu", "cuda")
            rank: Process rank in distributed environment
            world_size: Total number of processes in distributed environment
            num_devices_per_host: Number of devices per host machine (default: 8)
        """
        # Load configuration
        self.config = Config(config_file)
        self.device = torch.device(device)

        # Initialize distributed settings with fallback
        self._init_distributed(rank, world_size, num_devices_per_host)

        # Load and validate placement pattern
        self.expert_mapping = ExpertMapping(self.config.pattern_path, self.device, self.rank, self.num_devices_per_host)
        self.total_deployed_experts = self.expert_mapping.get_total_deployed_experts()

        # Calculate max_num_redundant_expert
        self.max_num_deployed_expert_per_rank = max(max(self.get_deployed_experts_per_layer()) // self.world_size, 1)
        self.max_redundant_num = self.expert_mapping.get_max_redundant_expert_num()

        # Initialize cluster status and optimizers.
        self.cluster_status = ClusterStatus(self.config, self.expert_mapping, self.rank)
        self.optimizers = _create_optimizers(self.config.Optimizers, self.cluster_status)
        self.optimizer = self.optimizers[0]

        # Initialize placement manager
        self._init_placement_manager()

        max_moe_layer_num = getattr(self.config, 'max_moe_layer_num', 58)

        # Initialize selector for each MoE layer
        self.selector = [0] * max_moe_layer_num

        for layer in range(max_moe_layer_num):
            local_expert_mapping = self.cluster_status.expert_mapping.redundant_expert_mapping[layer]
            self.n_routed_experts = local_expert_mapping.shape[1]
            expert_mapping_unique = [ [] for i in range(self.n_routed_experts) ]
            for i in range(self.n_routed_experts):
                expert_mapping_unique[i] = torch.unique(local_expert_mapping[:,i]).tolist()
            
            # Update expert map based on device locality
            expert_mapping_unique = self._init_update_expert_map(expert_mapping_unique)
            
            # Get selector for the updated expert map
            self.selector[layer] = self._init_get_selector(expert_mapping_unique)
        self.enable_dump = getattr(self.config, 'enable_dump', False)

        # redundant_enable_per_layer, True is redundant layer, False is Origin Layer
        self.redundant_enable_per_layer = self.expert_mapping.get_redundant_enable_per_layer()
        self.num_logits_expert_per_rank = max(self.expert_mapping.get_total_num_expert()//self.world_size, 1)

        print("OmniPlanner successfully initialized.")

    @classmethod
    def cleanup(cls):
        if cls in cls._instances:
            del cls._instances[cls]

    def __del__(self):
        # Clean up resources when the object is deleted
        if hasattr(self, 'cluster_activation'):
            self.cluster_activation.stop_thread()
            del self.cluster_activation
            time.sleep(1)

    def _init_distributed(self, rank: int = None, world_size: int = None, num_devices_per_host: int = 16) -> None:
        """Initialize distributed settings with fallback to provided values.

        Args:
            rank: Process rank in distributed environment
            world_size: Total number of processes in distributed environment
            num_devices_per_host: Number of devices per host machine
        """
        # Get rank and world size from distributed environment if not provided
        if rank is None or world_size is None:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank, self.world_size = rank, world_size

        # Get number of devices per host from environment variables or use default
        self.num_devices_per_host = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")  # omni_planner config file
        self.num_devices_per_host = len(self.num_devices_per_host.split(",")) if self.num_devices_per_host else num_devices_per_host

        # Validate that world_size is consistent with num_devices_per_host
        if self.world_size % self.num_devices_per_host != 0:
            print(f"Warning: world_size ({self.world_size}) is not evenly divisible by "
                  f"num_devices_per_host ({self.num_devices_per_host})")

    def _init_placement_manager(self) -> None:
        """Initialize placement handler, and activation tracking."""
        num_layers = self.expert_mapping.get_total_num_layers()

        # Initialize activation count tensor on the specified device
        self.npu_activation_count = torch.zeros(
            (num_layers, self.get_max_num_deployed_expert_per_rank()),
            device=self.device,
            dtype=torch.int64
        )
        torch.npu.synchronize()

        # Create cluster activation tracker
        self.cluster_activation = create_cluster_activation(
            self.rank,
            self.world_size,
            self.expert_mapping.get_total_num_layers(),
            self.get_max_num_deployed_expert_per_rank(),
            self.npu_activation_count
        )

    def _init_update_expert_map(self, expert_mapping_unique):
        """
        Update expert mapping to prioritize local experts based on device proximity.
        It categorizes experts into same-rank, same-host, and distant, then selects the best candidate.
        """
        for i in range(self.n_routed_experts):
            same_rank_candidates = []
            same_host_candidates = []
            distant_candidates = []
            experts_per_device = self.total_deployed_experts // self.world_size
            phy_list = expert_mapping_unique[i]

            # Categorize experts based on their physical location relative to the current rank
            for phy in phy_list:
                phy_device = phy // experts_per_device
                phy_host = phy_device // self.num_devices_per_host
                if phy_device == self.rank:
                    same_rank_candidates.append(phy)
                elif phy_host == self.rank // self.num_devices_per_host:
                    same_host_candidates.append(phy)
                else:
                    distant_candidates.append(phy)

            # Prioritize experts on the same rank, then same host, then distant
            if same_rank_candidates:
                expert_mapping_unique[i] = [same_rank_candidates[self.rank % len(same_rank_candidates)]]
            elif same_host_candidates:
                expert_mapping_unique[i] = [same_host_candidates[self.rank % len(same_host_candidates)]]
            else:
                expert_mapping_unique[i] = [distant_candidates[self.rank % len(distant_candidates)]]

        return expert_mapping_unique

    def _init_get_selector(self, expert_mapping_unique):
        """
        Create a selector tensor from the unique expert mapping.
        This selector is used to route tokens to the appropriate experts.
        """
        selector = torch.tensor(expert_mapping_unique, dtype=torch.int32, device=self.device).view(self.n_routed_experts)
        return selector

    def is_expert_on_current_rank(
        self,
        layer_id: int,
        expert_id: int,
        current_rank: int,
        experts_per_rank: int
    ) -> Tuple[bool, int]:
        """
        Check if expert is deployed on current rank and get its position.

        Args:
            layer_id: ID of the MoE layer
            expert_id: Expert ID within the layer
            current_rank: Target device rank to check
            experts_per_rank: Experts per device in default deployment

        Returns:
            Tuple (exists_on_rank, local_position)
        """
        return self.expert_mapping.is_expert_on_current_rank(layer_id, expert_id, current_rank, experts_per_rank)

    def expert_mapping_on_current_layer(
        self,
        layer_idx_moe: torch.tensor,
        is_prefill=False) -> torch.tensor:
        """
        Get the expert mapping for the current MoE layer.
        It returns the selector for redundant layers or the default mapping for original layers.
        """
        if layer_idx_moe > 57:
            return None

        # For redundant layers, use the pre-calculated selector
        if self.redundant_enable_per_layer[layer_idx_moe]:
            return self.selector[layer_idx_moe]

        # For original layers, use the default expert mapping
        return self.cluster_status.expert_mapping.redundant_expert_mapping[layer_idx_moe][0]

    def plan(
        self,
        layer_idx_moe: Optional[int] = None,
        tokens: Optional[torch.Tensor] = None,
        token_expert_ids: Optional[torch.Tensor] = None,
        token_expert_scores: Optional[torch.Tensor] = None,
        top_k: int = 8,
        expert_mapping: Optional[torch.Tensor] = None,
        is_prefill=True
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Optimize token-to-expert mapping using configured optimizers.

        This method takes input tokens and their initially assigned experts and scores. It computes
        expert loads, updates the cluster status accordingly, and then optimizes the assignment
        of tokens to experts by applying the configured optimization strategies.

        Args:
            layer_idx_moe: Identifier for the current layer (optional)
            tokens: Input tokens tensor with shape [num_tokens, ...]
            token_expert_ids: Initial expert assignments, shape [num_tokens, top_k], -1 indicates unassigned
            token_expert_scores: Importance scores for expert assignments, shape [num_tokens, top_k]

        Returns:
            Tuple containing (original tokens, optimized expert IDs, optimized scores)
        """
        if layer_idx_moe > 57:
            return tokens, token_expert_ids, token_expert_scores

        # Apply the expert mapping to the token expert IDs
        token_expert_ids = expert_mapping[token_expert_ids]

        return tokens, token_expert_ids, token_expert_scores

    def _validate_input(
        self,
        tokens: torch.Tensor,
        expert_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None
    ) -> None:
        """Validate dimensional consistency of input parameters"""
        if expert_ids.ndim != 2:
            raise ValueError("token_expert_ids must be 2-dimensional")
        if scores is None:
            return
        if scores.shape != expert_ids.shape:
            raise ValueError("token_expert_scores must match the shape of token_expert_ids")
        for token_scores in scores:
            for score in token_scores:
                if score.dtype not in (torch.int, torch.float32) or score < 0:
                    raise ValueError("Scores must be non-negative numbers")

    def _compute_expert_loads(
        self,
        layer_idx_moe: Optional[int] = None,
        token_expert_ids: Optional[torch.Tensor] = None
    ) -> None:
        """
        Compute current load distribution across experts.
        Args:
            token_expert_ids (torch.Tensor): optimized expert assignments, shape [num_tokens, top_k], -1 indicates unassigned
        Returns:
            None
        """
        # Update activation counts using a histogram of expert IDs
        self.npu_activation_count[layer_idx_moe] += torch.histc(token_expert_ids,
                                            bins=self.total_deployed_experts, min=0, max=self.total_deployed_experts-1)

    @staticmethod
    def get_deepseek_v3_moe_layer_idx(prefix: str) -> int:
        """
        Calculate the adjusted DeepSeek-V3 MoE layer index from a model layer prefix.

        The function parses a prefix string of format `model.layers.{N}.mlp.experts` to extract the
        layer index `N`, then adjusts this index by subtracting a fixed offset of dense layers
        (FIRST_K_DENSE_REPLACE) as per the DeepSeek-V3 model configuration.

        Args:
            prefix: A layer path string formatted as `model.layers.{N}.mlp.experts`
                (e.g., "model.layers.5.mlp.experts" represents layer 5)

        Returns:
            int: The adjusted layer index after subtracting FIRST_K_DENSE_REPLACE.
                Formula: parsed_layer_id - FIRST_K_DENSE_REPLACE

        Note:
            - LAYER_ID_IDX (2): Indicates layer ID position after splitting the prefix by '.'
            (e.g., ["model", "layers", "5", "mlp", "experts"] -> index 2 is "5")
            - FIRST_K_DENSE_REPLACE (3): Number of initial dense layers from the model's config.json
            that should be excluded when working with MoE layers.

        Example:
            >>> get_deepseek_v3_moe_layer_idx("model.layers.5.mlp.experts")
            2   # 5 (parsed) - 3 (offset) = 2
        """
        # Parses prefix string like 'model.layers.3.mlp.experts'
        LAYER_ID_IDX = 2               # Position of the layer ID after splitting by '.'
        FIRST_K_DENSE_REPLACE = 3      # From config.json: initial dense layers count

        return int(prefix.split(sep='.')[LAYER_ID_IDX]) - FIRST_K_DENSE_REPLACE

    def get_num_of_redundant_experts(self, moe_layer_idx: int, num_expert_per_device_origin=16, rank_device=0) -> int:
        """
        Calculate the number of redundant experts for a specific device and MoE layer.

        Args:
            moe_layer_idx : int
                Index of the MoE layer to query expert distribution.
            num_expert_per_device_origin : int, optional (default=16)
                Original number of experts assigned to this device/layer.
            rank_device : int, optional (default=0)
                Rank identifier of the target device in the distributed system.

        Returns:
            int
                Number of redundant experts, calculated as: (current experts count) - (original experts count).
        """
        if moe_layer_idx > 57:
            return 0
        return self.expert_mapping.get_num_of_redundant_experts(moe_layer_idx, num_expert_per_device_origin, rank_device)

    def get_local_expert_indices_offset(self, layer_idx_moe: int, current_rank: int, default_experts_per_rank: int) -> int:
        return self.expert_mapping.get_local_expert_indices_offset(layer_idx_moe, current_rank, default_experts_per_rank)

    def get_deployed_experts_per_layer(self) -> list:
        return self.expert_mapping.get_deployed_experts_per_layer()

    def get_max_num_deployed_expert_per_rank(self)-> int:
        return self.max_num_deployed_expert_per_rank

    def init_dram_weights(self, param_dict, first_k_dense_replace=3):
        return
        moe_weights = self.placement_manager.get_moe_weights()
        local_rank_pattern = self.expert_mapping.placement_pattern[self.rank].bool()
        init_dram_weights(moe_weights, param_dict, local_rank_pattern, first_k_dense_replace)

    def dump(self,step):
        """
        Dump expert activation counts to a file for analysis.
        This is controlled by `enable_dump` and `dump_dir` in the configuration.
        It handles prefill and decoder steps separately.
        """
        enable_dump = self.enable_dump
        dump_dir = getattr(self.config, 'dump_dir', None)

        # Check if dumping is enabled
        if not enable_dump:
            if dump_dir is not None:
                print(f"Warning: dump_dir is setting to {dump_dir}, If You Want to Dump Experts Activation Pls set enable_dump to True")
            return

        # Ensure dump directory is configured
        if dump_dir is None:
            raise RuntimeError("dump_dir must not be None, Pls Set dump_dir")

        if step==0:
            self.cluster_activation.stopDump()
            if not hasattr(self, "prefill_count"):
                self.prefill_count  = 0
            if not hasattr(self, "last_npu_activation_count"):
                self.last_npu_activation_count = torch.zeros_like(self.npu_activation_count)

        if step==0 or step==1:
            self.prefill_count  += 1
            prefill_dump_dir = os.path.join(dump_dir, "prefill")
            os.makedirs(prefill_dump_dir, exist_ok=True)
            file_path = os.path.join(prefill_dump_dir, f"activation_counts_recordstep_{self.prefill_count}_rank_{self.rank}.txt")
            npu_activation_count = self.npu_activation_count-self.last_npu_activation_count

            with open(file_path, 'w') as f:
                for row in npu_activation_count:
                    row_str = '\t'.join(str(x.item()) for x in row)
                    f.write(row_str + '\n')

        elif step >=32:
            decoder_dump_dir = os.path.join(dump_dir, "decoder")
            os.makedirs(decoder_dump_dir, exist_ok=True)
            self.cluster_activation.setDumpDir(decoder_dump_dir)

        self.last_npu_activation_count = self.npu_activation_count.clone()


# Example usage
if __name__ == "__main__":
    from optimizer.ada_router_optimizer import AdaRouter
    from optimizer.token_balance_optimizer import TokenBalance

    # Example input: 3 tokens, 4 experts each, with importance scores
    input_token = torch.tensor([
        [0, 1, 2, 3],  # Token 1
        [1, 0, 3, 2],  # Token 2
        [3, 2, 1, 0]   # Token 3
    ], dtype=torch.float32).npu()

    input_expert_id = torch.tensor([
        [0, 1, 2, 3],  # Token 1 expert
        [1, 0, 3, 2],  # Token 2 expert
        [3, 2, 1, 0]   # Token 3 expert
    ], dtype=torch.long).npu()

    input_expert_score = torch.tensor([
        [0.9, 0.5, 0.3, 0.7],  # Token 1 expert score
        [0.4, 0.8, 0.6, 0.2],  # Token 2 expert score
        [0.7, 0.3, 0.9, 0.5]   # Token 3 expert score
    ], dtype=torch.float32).npu()

    planner = OmniPlanner("./config.yaml")

    token, token_expert_ids, token_scores = planner.plan(
        layer_id=0,
        tokens=input_token,
        token_expert_ids=input_expert_id,
        token_expert_scores=input_expert_score
    )

    print("Input mapping:")
    print(input_token, input_expert_id, input_expert_score)

    print("\nOptimized mapping:")
    print(token, token_expert_ids, token_scores)