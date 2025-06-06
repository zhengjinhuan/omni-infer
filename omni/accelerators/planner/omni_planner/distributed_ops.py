import logging
import pickle
import torch
import torch_npu
import torch.distributed as dist
from omni_planner.atomic_state import State, get_state, set_state
from omni_planner.expert_mapping import ExpertMapping

# 配置日志
logger = logging.getLogger(__name__)

# 全局函数
def sync_mapping_to_nodes(rank, mapping):
    print(f"Node {rank} before broadcast: mapping = {mapping}")
    # 广播：将 node 0 的mapping广播到所有节点
    dist.broadcast(mapping, src=0)
    # 同步所有节点
    torch.npu.synchronize()
    logger.info("Mapping synced to all nodes")
    print(f"Node {rank} after broadcast: mapping = {mapping}")

def sync(rank: int, expert_mapping: ExpertMapping):
    current_state = get_state()
    if current_state == State.PREPARED_DOWNGRADE:
        expert_mapping.update_working_mapping()
        working_mapping = expert_mapping.get_working_mapping()
        sync_mapping_to_nodes(rank, working_mapping)
        set_state(State.APPLIED_DOWNGRADE)

    elif current_state == State.WEIGHTS_UPDATED:
        expert_mapping.update_working_mapping()
        working_mapping = expert_mapping.get_working_mapping()
        sync_mapping_to_nodes(rank, working_mapping)
        set_state(State.READY)

    # 其他状态正常推理工作

###Demo###
def broadcast_mapping(tensor, src_rank, rank):
    """将mapping从源节点广播到所有节点"""
    print(f"Node {rank} before broadcast: tensor = {tensor}")
    dist.broadcast(tensor, src=src_rank)
    print(f"Node {rank} after broadcast: tensor = {tensor}")

def reduce_activation(dst_rank, rank):
    from omni_planner.omni_planner import OmniPlanner
    planner = OmniPlanner()
    activation_tensor = planner.npu_activation_count

    """将所有节点的activation归约到master节点"""
    print(f"Node {rank} before reduce: tensor = {activation_tensor}")
    dist.reduce(activation_tensor, dst=dst_rank, op=dist.ReduceOp.SUM)
    print(f"Node {rank} after reduce: tensor = {activation_tensor}")

def distribution_warmup():
    tmp = torch.tensor(4018, dtype=torch.int64,device="npu")
    torch.distributed.broadcast(tmp,src=0) # warmup
    torch.npu.synchronize()


def sync_demo(rank: int):
    working_mapping = torch.tensor([1.0, 2.0, 3.0])
    optimized_mapping = torch.tensor([11.0, 12.0, 13.0])

    print("get torch group ", dist.group.WORLD)
    if rank == 0:
        # master节点应用新的专家摆放
        working_mapping = torch.clone(optimized_mapping)

    # 1. 广播：将 node 0 的mapping广播到所有节点
    #broadcast_mapping(working_mapping, src_rank=0, rank=rank)
    broadcast_rootinfo(rank)

    # 同步所有节点
    torch.npu.synchronize()