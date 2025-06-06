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

    # 初始化
    deployments = [1] * num_experts
    remaining_budget = budget_limit
    max_deployments_per_expert = 1 + expert_redundant_limit

    if remaining_budget == 0:
        return deployments

    # 最大堆
    heap = []
    for i in range(num_experts):
        original_load = normalized_loads[i]
        current_deploy_count = deployments[i]
        priority = -original_load / current_deploy_count if original_load > 0 else 0.0
        if current_deploy_count < max_deployments_per_expert:
            heap.append((priority, original_load, i))
    heapq.heapify(heap)

    # 分配部署
    deployments_added = 0
    while deployments_added < remaining_budget and heap:
        neg_load_per_instance, original_load, index = heapq.heappop(heap)
        deployments[index] += 1
        deployments_added += 1
        new_deploy_count = deployments[index]

        if new_deploy_count < max_deployments_per_expert:
            new_priority = -original_load / new_deploy_count if original_load > 0 else 0.0
            heapq.heappush(heap, (new_priority, original_load, index))

    if deployments_added < remaining_budget:
        print(f"Warning: Allocated {deployments_added} of {budget_limit} budget.")
    
    return deployments

def distribute_experts_sequentially(
    num_experts: int,
    num_devices: int
) -> Tuple[float, np.ndarray]:
    """
    将专家按顺序均匀分配到设备上，每个设备放置相同数量的专家。

    Args:
        num_experts: 专家总数。
        num_devices: 设备总数。

    Returns:
        一个元组包含：
        - max_device_load (float): 这里返回0.0，因为不考虑负载。
        - placement_matrix (np.ndarray): (num_devices, num_experts) 的0/1矩阵，
                                        表示专家分配到设备的情况。

    Raises:
        ValueError: 如果专家总数不能被设备数整除。
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
    将专家实例贪心地分配到指定数量的设备上，以均衡负载。

    约束条件：
    1. 每个专家类型在单个设备上最多部署一次。
    2. 每个设备最终放置相同数量的专家实例。
    3. 优先放置原始负载高的专家实例。
    4. 将专家实例放置在当前总负载最低且满足约束的设备上。

    Args:
        initial_loads: 一维列表或NumPy数组，包含每个专家类型的 *原始* 负载。
        deployments: 一维列表，包含每个专家类型需要部署的总次数 (来自上一个函数)。
        num_devices: 目标设备数量。

    Returns:
        一个元组包含：
        - max_device_load (float): 分配后所有设备中的最大总负载。
        - placement_matrix (np.ndarray): 一个 (num_devices, num_experts) 的 0/1 矩阵，
                                          其中 matrix[d][e] = 1 表示专家 e 部署在设备 d 上。

    Raises:
        ValueError: 如果输入不满足约束条件（例如，总部署数不能被设备数整除，
                    单个设备容量不足，或某个专家部署次数超过设备数）。
    """
    # --- 输入校验与处理 ---
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

    # --- 计算和检查约束 ---
    total_deployments = sum(deployments)
    if total_deployments == 0:
        return 0.0, np.zeros((num_devices, num_experts), dtype=int)

    if total_deployments % num_devices != 0:
        raise ValueError(f"总部署次数 ({total_deployments}) 必须能被设备数量 ({num_devices}) 整除。")

    experts_per_device = total_deployments // num_devices
    if experts_per_device == 0 and total_deployments > 0:
        raise ValueError("计算出的每个设备的专家数为0，但总部署数大于0，这不合理。请检查输入。")

    if experts_per_device > num_experts:
        raise ValueError(f"每个设备需要放置 ({experts_per_device}) 个专家实例，"
                         f"但只有 ({num_experts}) 种不同的专家类型。"
                         f"无法满足'每个专家在同一设备上最多部署一次'的约束。")

    if any(d > num_devices for d in deployments):
        max_req_expert_idx = np.argmax(deployments)
        raise ValueError(f"专家 {max_req_expert_idx} 需要部署 {deployments[max_req_expert_idx]} 次，"
                         f"超过了设备总数 {num_devices}。"
                         f"无法满足'每个专家在同一设备上最多部署一次'的约束。")

    # --- 准备数据结构 ---
    expert_instances = []
    for expert_idx, count in enumerate(deployments):
        if count > 0: # 只添加需要部署的
            load = loads_np[expert_idx]
            for _ in range(count):
                expert_instances.append((load, expert_idx))

    expert_instances.sort(key=lambda x: x[0], reverse=True)

    device_loads = np.zeros(num_devices, dtype=float)
    placement_matrix = np.zeros((num_devices, num_experts), dtype=int)
    device_expert_counts = np.zeros(num_devices, dtype=int)

    # --- 贪心分配 ---
    for load, expert_idx in expert_instances:
        best_device = -1
        min_load_for_candidate = float('inf')

        possible_devices = []
        for device_id in range(num_devices):
            can_place_expert = (placement_matrix[device_id, expert_idx] == 0)
            has_space = (device_expert_counts[device_id] < experts_per_device)

            if can_place_expert and has_space:
                possible_devices.append(device_id)

        if not possible_devices:
            raise RuntimeError(f"无法为专家 {expert_idx} (负载 {load}) 找到合适的设备。"
                               "这不应该发生，请检查约束和逻辑。")

        best_device = min(possible_devices, key=lambda dev_id: device_loads[dev_id])

        placement_matrix[best_device, expert_idx] = 1
        device_loads[best_device] += load
        device_expert_counts[best_device] += 1

    # --- 验证与收尾 ---
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
    expert_redundant_limit: int = 11,
    num_moe_layers: int = 58,
    num_eps: int = 256,
    output_file: str = None
) -> np.ndarray:
    """
    Process expert deployments across MoE layers, allocate budgets, and distribute experts to devices.

    Args:
        input_file: Path to the input CSV file containing expert activation counts.
        output_dir: Directory to save the output placement pattern.
        num_devices: Number of devices to distribute experts across.
        num_redundant_layers: Number of layers to assign additional deployment budget.
        expert_redundant_limit: Maximum additional deployments per expert.
        num_moe_layers: Number of MoE layers.
        num_eps: Number of experts per layer.
        output_file: Name of the output .npy file (without path). If None, a default name is generated.

    Returns:
        placement_pattern: A (num_devices, num_moe_layers, num_eps) numpy array representing the deployment pattern.
    """
    # 加载数据
    data = np.genfromtxt(input_file, delimiter=',', skip_header=1)
    ep_activation_counts = data[:, 1:] + 3
    print("ep_activation_counts shape:", ep_activation_counts.shape)
    print("每层最大激活计数:", ep_activation_counts.max(1))

    # 初始化预算和放置模式
    budget_limit = [0 for _ in range(num_moe_layers)]
    placement_pattern = np.zeros((num_devices, num_moe_layers, num_eps), dtype=np.int32)

    # 计算每层按顺序分配后的设备负载
    layer_max_loads = np.zeros(num_moe_layers)
    for layer_idx in range(num_moe_layers):
        # 按顺序分配专家到设备
        _, base_placement = distribute_experts_sequentially(
            num_experts=num_eps,
            num_devices=num_devices
        )
        # 计算每个设备的负载
        device_loads = np.sum(ep_activation_counts[layer_idx] * base_placement, axis=1)
        # 记录该层的最大设备负载
        layer_max_loads[layer_idx] = np.max(device_loads) if device_loads.size > 0 else 0.0

    # 按每层最大负载排序
    sorted_indices = np.argsort(layer_max_loads)
    sorted_values = layer_max_loads[sorted_indices]
    print("基于最大负载排序的索引:", sorted_indices)
    print("排序后的最大负载值:", sorted_values)

    # 为高负载层分配预算
    if num_redundant_layers < 1:
        print('没有冗余预算！')
    else:
        for layer_idx in sorted_indices[-num_redundant_layers:]:
            budget_limit[layer_idx] = num_devices
        print("budget_limit:", budget_limit)

    # Process each MoE layer
    for layer_idx_moe in range(num_moe_layers):
        expert_allocation_count = allocate_expert_deployments_improved(
            ep_activation_counts[layer_idx_moe],
            expert_redundant_limit=expert_redundant_limit,
            budget_limit=budget_limit[layer_idx_moe]
        )
        max_load, placement_matrix = distribute_experts_to_devices(
            initial_loads=ep_activation_counts[layer_idx_moe],
            deployments=expert_allocation_count,
            num_devices=num_devices
        )
        
        placement_pattern[:, layer_idx_moe, :] += placement_matrix
    # Verify high-load layers
    print("高负载层的分配情况:")
    for layer_idx in sorted_indices[-num_redundant_layers:]:
        print(f"Layer {layer_idx}: Total deployments = {placement_pattern[:, layer_idx, :].sum()}")
    # Save the placement pattern
    if output_file is None:
        output_file = (f'DSV3_0428_GSM8K_test_+{num_redundant_layers}'
                       f'_{num_moe_layers}_MoELayers_{num_devices}_dies_epmaxdeploy_{expert_redundant_limit+1}.npy')
    output_path = os.path.join(output_dir, output_file)
    np.save(output_path, placement_pattern)

    print("Placement pattern shape:", placement_pattern.shape)
    return placement_pattern

if __name__ == "__main__":
    process_expert_deployments()
