import unittest
from unittest.mock import patch, MagicMock
import torch.distributed as dist
from omni_planner.atomic_state import State, get_state, set_state
from omni_planner.expert_mapping import ExpertMapping
from omni_planner.distributed_ops import sync

class TestDistributedOps(unittest.TestCase):
    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('torch.distributed.broadcast')
    @patch('torch.npu.synchronize')
    def test_sync_prepared_downgrade(self, mock_synchronize, mock_broadcast, mock_get_rank, mock_is_initialized):
        # 模拟ExpertMapping对象
        expert_mapping = ExpertMapping("")
        expert_mapping.update_working_mapping = MagicMock()
        expert_mapping.get_working_mapping = MagicMock(return_value={'key': 'value'})

        # 设置当前状态为PREPARED_DOWNGRADE
        set_state(State.PREPARED_DOWNGRADE)

        # 调用sync函数
        sync(0, expert_mapping)

        # 验证调用顺序和方法
        expert_mapping.update_working_mapping.assert_called_once()
        expert_mapping.get_working_mapping.assert_called_once()
        mock_broadcast.assert_called_once_with({'key': 'value'}, src=0)
        mock_synchronize.assert_called_once()
        self.assertEqual(get_state(), State.APPLIED_DOWNGRADE)

    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('torch.distributed.broadcast')
    @patch('torch.npu.synchronize')
    def test_sync_weights_updated(self, mock_synchronize, mock_broadcast, mock_get_rank, mock_is_initialized):
        # 模拟ExpertMapping对象
        expert_mapping = ExpertMapping("")
        expert_mapping.update_working_mapping = MagicMock()
        expert_mapping.get_working_mapping = MagicMock(return_value={'key': 'value'})

        # 设置当前状态为WEIGHTS_UPDATED
        set_state(State.WEIGHTS_UPDATED)

        # 调用sync函数
        sync(0, expert_mapping)

        # 验证调用顺序和方法
        expert_mapping.update_working_mapping.assert_called_once()
        expert_mapping.get_working_mapping.assert_called_once()
        mock_broadcast.assert_called_once_with({'key': 'value'}, src=0)
        mock_synchronize.assert_called_once()
        self.assertEqual(get_state(), State.READY)

    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_rank', return_value=0)
    def test_sync_other_states(self, mock_get_rank, mock_is_initialized):
        # 模拟ExpertMapping对象
        expert_mapping = ExpertMapping("")
        expert_mapping.update_working_mapping = MagicMock()
        expert_mapping.get_working_mapping = MagicMock(return_value={'key': 'value'})

        # 设置当前状态为其他状态
        set_state(State.READY)

        # 调用sync函数
        sync(0, expert_mapping)

        # 验证没有调用update_working_mapping
        expert_mapping.update_working_mapping.assert_not_called()
        expert_mapping.get_working_mapping.assert_not_called()

if __name__ == '__main__':
    unittest.main()
