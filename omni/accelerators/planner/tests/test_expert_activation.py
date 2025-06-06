import os
import torch
import torch_npu
import torch.nn as nn
import unittest
import time
import multiprocessing as mp
from omni_planner.placement_handler import create_cluster_activation
import shutil
import pytest


class TestExpertActivation(unittest.TestCase):
    def setUp(self):
        num_layers = 58
        rank = 0
        world_size = 4
        num_deployed_experts = 320//world_size
        self.test_dump_dir = "./test_dump_dir"

        self.npu_activation_count = torch.zeros(
            (num_layers, num_deployed_experts),
            device="npu:0",
            dtype=torch.int64
        )
        self.cluster_activation = create_cluster_activation(
            rank,
            world_size,
            num_layers,
            num_deployed_experts,
            self.npu_activation_count
        )

    def tearDown(self):
        del self.cluster_activation
        if os.path.isdir(self.test_dump_dir):
            shutil.rmtree(self.test_dump_dir)
        

    def test_basic_functionality(self):
        self.cluster_activation.setDumpDir(self.test_dump_dir)
        for i in range(10):
            self.npu_activation_count[i] = i
        time.sleep(5)
    
    def test_dump_prefill(self):
        # Toy Test For PD-Prefill Dump Activation values
        from omni_planner.omni_planner import OmniPlanner
        try:
            planner = OmniPlanner("/home/kww/20250507/omni_infer/config.yaml")
        except:
            return
        planner.dump(0)
        planner.npu_activation_count += 1
        planner.dump(0)
        planner.npu_activation_count += 2
        planner.dump(0)
        # planner.dump(0)


@unittest.skip("Dump Examples")
class TestExpertActivationMultiRank(unittest.TestCase):
    def setUp(self):
        self.world_size = 16
        self.test_dump_dir = "./test_dump_dir"
    def tearDown(self):
        if os.path.isdir(self.test_dump_dir):
            shutil.rmtree(self.test_dump_dir)
    def test_basic_functionality_with_multi_thread(self):
        processes = []
        def run_process(rank):
            print(rank)
            num_layers = 58
            num_deployed_experts = 4
            npu_activation_count = torch.zeros(
                (num_layers, num_deployed_experts),
                device=f"npu:{rank}",
                dtype=torch.int64
            )
            torch.npu.synchronize() # 确保 npu_activatio_count在显存中已经完成初始化
            cluster_activation = create_cluster_activation(
                rank,
                self.world_size,
                num_layers,
                num_deployed_experts,
                npu_activation_count
            )

            cluster_activation.setDumpDir(self.test_dump_dir)
            npu_activation_count += rank
            time.sleep(5)
            del cluster_activation

        for rank in range(self.world_size):
            p = mp.Process(target=run_process, args=(rank,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()