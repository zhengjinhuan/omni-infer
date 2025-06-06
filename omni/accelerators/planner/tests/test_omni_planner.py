# import torch_npu
from pathlib import Path
import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import torch.distributed as dist
from unittest.mock import MagicMock, patch

from omni_planner.omni_planner import OmniPlanner
from omni_planner.config import Config
from omni_planner.cluster_status import ClusterStatus

#torch.set_printoptions(threshold=float("inf"))  # Print all values
class TestOmniPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("start setup")
        # 设置环境变量
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend="hccl", rank=0, world_size=1)
        cls.planner = OmniPlanner("./tests/config_test.yaml", world_size=32)

    @classmethod
    def tearDownClass(cls):
        print("start tearDown")
        dist.destroy_process_group()
        OmniPlanner.cleanup()
        if hasattr(cls, 'planner'):
            del cls.planner

    @unittest.skip("not for our strategy.")
    def test_omni_planner_plan(cls):
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
            [0.4, 0.3, 0.2, 0.1],  # Token 1 expert score
            [0.4, 0.2, 0.2, 0.2],  # Token 2 expert score
            [0.7, 0.11, 0.1, 0.09]   # Token 3 expert score
        ], dtype=torch.float32).npu()

        token, token_expert_ids, token_scores = cls.planner.plan(layer_idx_moe=0,
                                                                  tokens=input_token,
                                                                  token_expert_ids=input_expert_id,
                                                                  token_expert_scores=input_expert_score)

        print("Input mapping:")
        print(input_token, input_expert_id, input_expert_score)

        print("\nOptimized mapping:")
        print(token, token_expert_ids, token_scores)

        # graph plan only run redundant optimizer
        except_expert_ids = torch.tensor([[ 0,  1,  2,  3],
        [ 1,  0,  3, 2],
        [ 3,  2,  1,  0]], dtype=torch.long).npu()
        print(token_expert_ids)

        cls.assertTrue(torch.equal(token_expert_ids, except_expert_ids))

    def test_get_deepseek_v3_moe_layer_idx(cls):
        cls.assertEqual(OmniPlanner.get_deepseek_v3_moe_layer_idx("model.layers.5.mlp.experts"), 2)
        cls.assertEqual(OmniPlanner.get_deepseek_v3_moe_layer_idx("model.layers.10.mlp.experts"), 7)

    def test_initialization(cls):
        cls.assertIsInstance(cls.planner.config, Config)
        cls.assertIsNotNone(cls.planner.expert_mapping)
        cls.assertIsInstance(cls.planner.cluster_status, ClusterStatus)
        cls.assertEqual(len(cls.planner.optimizers), 2)
        cls.assertIn("AdaRouter", [type(instance).__name__ for instance in cls.planner.optimizers])

    def test_total_deployed_experts(cls):
        expected_experts = 256
        np.testing.assert_array_equal(cls.planner.total_deployed_experts, expected_experts)

    def test_is_expert_on_current_rank(cls):
        exists, position = cls.planner.is_expert_on_current_rank(0, 0, 0, 3)
        cls.assertTrue(exists)
        cls.assertEqual(position, 0)

        exists, position = cls.planner.is_expert_on_current_rank(0, 100, 0, 3)
        cls.assertFalse(exists)
        cls.assertEqual(position, 64)

        exists, position = cls.planner.is_expert_on_current_rank(0, 1, 1, 2)
        cls.assertFalse(exists)
        cls.assertEqual(position, 0)

        exists, position = cls.planner.is_expert_on_current_rank(1, 3, 0, 2)
        cls.assertTrue(exists)
        cls.assertEqual(position, 3)

    def test_get_num_of_redundant_experts(cls):
        num_of_redundant_experts = cls.planner.get_num_of_redundant_experts(0, 63, 0)
        cls.assertEqual(num_of_redundant_experts, 1)

@unittest.skip("not for our strategy.")
class TestOmniPlanner_with_none_redundant_optimizer(unittest.TestCase):
    @classmethod
    def setUp(self):
        # 设置环境变量
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29501"
        dist.init_process_group(backend="hccl", rank=0, world_size=1)
        self.planner = OmniPlanner("./tests/config_none_redundant.yaml", world_size=32)

    @classmethod
    def tearDown(self):
        dist.destroy_process_group()
        OmniPlanner.cleanup()
        del self.planner

    def test_none_redundant_optimizer(self):
        # Example input: 3 tokens, 4 experts each, with importance scores
        input_token = torch.tensor([
            [0, 1, 2, 3],  # Token 1
            [1, 0, 3, 2],  # Token 2
            [3, 2, 1, 0]   # Token 3
        ], dtype=torch.float32).npu()

        input_expert_id = torch.tensor([
            [0,1,2,3,4,5,6,7],  # Token 1 expert
            [4,5,6,7,8,9,10,11] # Token 2 expert

        ], dtype=torch.long).npu()#.expand(10000, -1)

        input_expert_score = torch.tensor([
            [0.4, 0.3, 0.2, 0.1],  # Token 1 expert score
            [0.4, 0.2, 0.2, 0.2],  # Token 2 expert score
            [0.7, 0.11, 0.1, 0.09]   # Token 3 expert score
        ], dtype=torch.float32).npu()

        # 测量 self.planner.plan 的运行时间
        # start_time = time.time()  # 记录开始时间

        print("Input mapping:")
        print(input_expert_id)

        token, token_expert_ids, token_scores = self.planner.plan(
            layer_idx_moe=42,
            tokens=input_token,
            token_expert_ids=input_expert_id,
            token_expert_scores=input_expert_score
        )
        # end_time = time.time()  # 记录结束时间
        # elapsed_time = end_time - start_time  # 计算时间差
        # print(f"self.planner.plan execution time: {elapsed_time:.6f} seconds")  # 打印运行时间

        print("\nOptimized mapping:")
        print(token_expert_ids)

        # graph plan only run redundant optimizer
        except_expert_ids = torch.tensor([[0,1,2,3,4,5,6,7],
        [4,5,6,7,9,10,11,12]], dtype=torch.long).npu()

        self.assertTrue(torch.equal(token_expert_ids, except_expert_ids))

class TestOmniPlanner_with_heat_optimizer(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_max_num_deployed_expert_per_rank_0(self):
        # 设置环境变量
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29502"
        dist.init_process_group(backend="hccl", rank=0, world_size=1)
        planner = OmniPlanner("./tests/config_heat.yaml",world_size=64,num_devices_per_host=16)
        self.assertEqual(planner.get_max_num_deployed_expert_per_rank(), 320)
        dist.destroy_process_group()
        planner.cleanup()
        del planner

    @unittest.skip("singleton instance can not change config.")
    def test_get_max_num_deployed_expert_per_rank_1(self):
        # 设置环境变量
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29503"
        dist.init_process_group(backend="hccl", rank=0, world_size=1)

        planner = OmniPlanner("./tests/config_heat.yaml",rank=1,world_size=64,num_devices_per_host=16)
        self.assertEqual(planner.get_max_num_deployed_expert_per_rank(), 5)
        dist.destroy_process_group()
        planner.cleanup()
        del planner

class TestEmptyPattern(unittest.TestCase):
    @classmethod
    def setUp(self):
        # 设置环境变量
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29505"
        dist.init_process_group(backend="hccl", rank=0, world_size=1)
        # 创建一个空pattern的OmniPlanner实例用于测试
        self.planner = OmniPlanner("./tests/config_empty.yaml")

    @classmethod
    def tearDown(self):
        dist.destroy_process_group()
        OmniPlanner.cleanup()
        del self.planner

    @unittest.skip("singleton instance can not change config.")
    def test_empty_pattern(self):
        self.assertEqual(self.planner.placement_pattern, None)
        self.assertEqual(self.planner.is_expert_on_current_rank(0, 16, 1, 16), (True, 0))

if __name__ == '__main__':
    unittest.main()