import os
import numpy as np
import heapq
from typing import List, Tuple, Union

def allocate_expert_deployments_improved(
    loads: Union[List[float], np.ndarray],
    expert_redundant_limit: int,
    budget_limit: int,
    load_normalization: str = 'log'
) -> List[int]:
    # 输入处理
    is_numpy = isinstance(loads, np.ndarray)
    num_experts = loads.size if is_numpy else len(loads)
    loads_list = loads.tolist() if is_numpy else list(loads)
    
    # 负载归一化
    if load_normalization == 'log':
        normalized_loads = [np.log1p(load) for load in loads_list]
    else:
        normalized_loads = loads_list

    # 初始化部署，每个专家部署一次
    deployments = [1] * num_experts  # 每个专家固定部署1次
    return deployments

def distribute_experts_sequentially(
    num_experts: int,
    num_devices: int
) -> Tuple[float, np.ndarray]:
    """
    将专家按顺序均匀分配到设备上，每个设备放置相同数量的专家。
    """
    if num_experts % num_devices != 0:
        raise ValueError(f"专家总数 ({num_experts}) 必须能被设备数 ({num_devices}) 整除。")

    experts_per_device = num_experts // num_devices
    placement_matrix = np.zeros((num_devices, num_experts), dtype=int)

    for device_id in range(num_devices):
        start_idx = device_id * experts_per_device
        end_idx = (device_id + 1) * experts_per_device
        placement_matrix[device_id, start_idx:end_idx] = 1

    return 0.0, placement_matrix

def distribute_experts_to_devices(
    initial_loads: Union[List[float], np.ndarray],
    deployments: List[int],
    num_devices: int
) -> Tuple[float, np.ndarray]:
    """
    将专家实例贪心地分配到设备上，以均衡负载。
    约束：每个专家每层只部署一次，分配到负载最低的设备。
    """
    # 输入校验与处理
    if isinstance(initial_loads, list):
        loads_np = np.array(initial_loads, dtype=float)
    elif isinstance(initial_loads, np.ndarray):
        if initial_loads.ndim != 1:
            raise ValueError("输入 initial_loads 必须是一维的。")
        loads_np = initial_loads.astype(float)
    else:
        raise TypeError("initial_loads 必须是 list 或 numpy.ndarray 类型。")

    if not isinstance(deployments, list) or not all(isinstance(d, int) for d in deployments):
        raise TypeError("deployments 必须是整数列表。")
    if len(loads_np) != len(deployments):
        raise ValueError("initial_loads 和 deployments 的长度必须相等。")
    if num_devices <= 0:
        raise ValueError("num_devices 必须是正整数。")

    num_experts = len(loads_np)
    total_deployments = sum(deployments)
    
    if total_deployments == 0:
        return 0.0, np.zeros((num_devices, num_experts), dtype=int)

    if total_deployments % num_devices != 0:
        raise ValueError(f"总部署次数 ({total_deployments}) 必须能被设备数量 ({num_devices}) 整除。")

    experts_per_device = total_deployments // num_devices
    if experts_per_device == 0 and total_deployments > 0:
        raise ValueError("计算出的每个设备的专家数为0，但总部署数大于0。")

    if experts_per_device > num_experts:
        raise ValueError(f"每个设备需要放置 ({experts_per_device}) 个专家实例，"
                         f"但只有 ({num_experts}) 种不同的专家类型。")

    if any(d > 1 for d in deployments):
        raise ValueError("每个专家每层最多部署一次，deployments 中不应有大于1的值。")

    # 准备数据结构
    expert_instances = []
    for expert_idx, count in enumerate(deployments):
        if count == 1:  # 只添加需要部署的专家
            load = loads_np[expert_idx]
            expert_instances.append((load, expert_idx))

    expert_instances.sort(key=lambda x: x[0], reverse=True)

    device_loads = np.zeros(num_devices, dtype=float)
    placement_matrix = np.zeros((num_devices, num_experts), dtype=int)
    device_expert_counts = np.zeros(num_devices, dtype=int)

    # 贪心分配
    for load, expert_idx in expert_instances:
        # 找到当前负载最低且未放置该专家的设备
        best_device = -1
        min_load = float('inf')
        for device_id in range(num_devices):
            if placement_matrix[device_id, expert_idx] == 0 and device_expert_counts[device_id] < experts_per_device:
                if device_loads[device_id] <= min_load:
                    min_load = device_loads[device_id]
                    best_device = device_id

        if best_device == -1:
            raise RuntimeError(f"无法为专家 {expert_idx} (负载 {load}) 找到合适的设备。")

        placement_matrix[best_device, expert_idx] = 1
        device_loads[best_device] += load
        device_expert_counts[best_device] += 1

    # 验证
    if not np.all(device_expert_counts == experts_per_device):
        print(f"警告：分配后各设备专家数量不完全等于预期值 {experts_per_device}。")
        print(f"实际数量: {device_expert_counts}")

    max_device_load = np.max(device_loads) if total_deployments > 0 else 0.0
    return max_device_load, placement_matrix

def process_expert_deployments(
    input_file: str = './topk_id_count/topk_ids_count_GSM8K_decode_48bs_64die.csv',
    output_dir: str = './',
    num_devices: int = 32,
    num_redundant_layers: int = 10,
    expert_redundant_limit: int = 1,  # 固定为1，因为不允许冗余
    num_moe_layers: int = 58,
    num_eps: int = 256,
    output_file: str = None
) -> np.ndarray:
    # 加载数据
    data = np.genfromtxt(input_file, delimiter=',', skip_header=1)
    ep_activation_counts = data[:, 1:] + 3
    print("ep_activation_counts shape:", ep_activation_counts.shape)
    print("每层最大激活计数:", ep_activation_counts.max(1))

    # 初始化放置模式
    placement_pattern = np.zeros((num_devices, num_moe_layers, num_eps), dtype=np.int32)

    # 计算每层按顺序分配后的设备负载
    layer_max_loads = np.zeros(num_moe_layers)
    for layer_idx in range(num_moe_layers):
        _, base_placement = distribute_experts_sequentially(
            num_experts=num_eps,
            num_devices=num_devices
        )
        device_loads = np.sum(ep_activation_counts[layer_idx] * base_placement, axis=1)
        layer_max_loads[layer_idx] = np.max(device_loads) if device_loads.size > 0 else 0.0

    # 按每层最大负载排序
    sorted_indices = np.argsort(layer_max_loads)
    sorted_values = layer_max_loads[sorted_indices]
    print("基于最大负载排序的索引:", sorted_indices)
    print("排序后的最大负载值:", sorted_values)

    # 选择高负载层
    high_load_layers = set(sorted_indices[-num_redundant_layers:])
    print(f"高负载层索引（前 {num_redundant_layers} 层）:", high_load_layers)

    # 处理每个 MoE 层
    for layer_idx_moe in range(num_moe_layers):
        if layer_idx_moe in high_load_layers:
            # 对高负载层进行优化分配
            expert_allocation_count = allocate_expert_deployments_improved(
                ep_activation_counts[layer_idx_moe],
                expert_redundant_limit=expert_redundant_limit,
                budget_limit=num_eps  # budget_limit 无意义，仅占位
            )
            max_load, placement_matrix = distribute_experts_to_devices(
                initial_loads=ep_activation_counts[layer_idx_moe],
                deployments=expert_allocation_count,
                num_devices=num_devices
            )
            print(f"Layer {layer_idx_moe}: 优化分配，部署专家数 = {sum(expert_allocation_count)}")
        else:
            # 对其他层进行顺序分配
            max_load, placement_matrix = distribute_experts_sequentially(
                num_experts=num_eps,
                num_devices=num_devices
            )
            print(f"Layer {layer_idx_moe}: 顺序分配，部署专家数 = {num_eps}")
        placement_pattern[:, layer_idx_moe, :] += placement_matrix

    # 验证高负载层
    print("高负载层的分配情况:")
    for layer_idx in high_load_layers:
        total_deployments = placement_pattern[:, layer_idx, :].sum()
        print(f"Layer {layer_idx}: Total deployments = {total_deployments}")

    # 保存放置模式
    if output_file is None:
        output_file = (f'DSV3_0418_share_gpt_NoRedundancy_+{num_redundant_layers}'
                       f'_{num_moe_layers}_MoELayers_{num_devices}_dies.npy')
    output_path = os.path.join(output_dir, output_file)
    np.save(output_path, placement_pattern)

    print("Placement pattern shape:", placement_pattern.shape)
    return placement_pattern

if __name__ == "__main__":
    process_expert_deployments()