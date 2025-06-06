import numpy as np
import matplotlib.pyplot as plt

def test_expert_mapping(expert_mapping):
    """
    测试三维矩阵expert_mapping是否满足以下条件：
    1. 对于每个layer，每个ep都被至少分配了一次
    2. 对于每个layer，每个设备拥有数量一样多的ep

    参数:
    expert_mapping: shape为(deviceid, layerid, epid)的三维numpy数组，
                  值为1表示该device上的layer存在该ep，值为0表示不存在

    返回:
    bool: 如果满足条件返回True，否则返回False
    dict: 详细的测试结果信息
    """
    n_devices, n_layers, n_experts = expert_mapping.shape
    result = {"条件1满足": True, "条件2满足": True, "详细信息": {}}

    # 对每个layer进行检查
    for layer_id in range(n_layers):
        layer_result = {}

        # 条件1: 检查每个ep是否至少被分配了一次
        expert_assigned = np.sum(expert_mapping[:, layer_id, :], axis=0)  # 按设备维度求和
        missing_experts = np.where(expert_assigned == 0)[0]

        if len(missing_experts) > 0:
            result["条件1满足"] = False
            layer_result["未分配的EP"] = missing_experts.tolist()

        # 条件2: 检查每个设备上的ep数量是否相同
        experts_per_device = np.sum(expert_mapping[:, layer_id, :], axis=1)  # 每个设备上的ep数量
        if not np.all(experts_per_device == experts_per_device[0]):
            result["条件2满足"] = False
            layer_result["每个设备的EP数量"] = experts_per_device.tolist()
            layer_result["设备EP数量不一致"] = True

        if layer_result:
            result["详细信息"][f"layer_{layer_id}"] = layer_result

    return result["条件1满足"] and result["条件2满足"], result

def view_patterns(placement_pattern, ppname='', fig_save_path=None):
    """
    可视化三维放置模式的求和视图，使用离散颜色映射。

    参数:
    placement_pattern: shape为(deviceid, layerid, epid)的三维numpy数组
    ppname: 可选的图案名称，添加到图表标题中
    fig_save_path: 可选的文件路径，用于保存生成的图像

    返回:
    无（显示图像并可选保存）
    """
    matrix = placement_pattern
    dim_x, dim_y, dim_z = matrix.shape[0], matrix.shape[1], matrix.shape[2]
    sum_axis0 = np.sum(matrix, axis=0)
    sum_axis1 = np.sum(matrix, axis=1)
    sum_axis2 = np.sum(matrix, axis=2)

    # 绘制两个二维求和图，并选择离散型的 cmap
    fig2d, axs = plt.subplots(1, 2, figsize=(18, 5))

    # 为每个图定义离散颜色数
    cmap0 = plt.cm.get_cmap('plasma', dim_x + 1)
    cmap2 = plt.cm.get_cmap('plasma', dim_z + 1)

    # 绘制第一个二维图：对X求和 (Y x Z)
    im0 = axs[0].imshow(sum_axis0, origin='lower', cmap=cmap0, interpolation='nearest')
    axs[0].set_title('Sum over Die_ID dimension' + ppname + '\n(Resulting shape: Layer_ID x EP_ID) : View the number of times each expert is deployed')
    axs[0].set_xlabel('EP_ID')
    axs[0].set_ylabel('Layer_ID')
    fig2d.colorbar(im0, ax=axs[0], ticks=range(dim_x + 1))

    # 绘制第二个二维图：对Z求和 (Y x X)
    im2 = axs[1].imshow(sum_axis2.T, origin='lower', cmap=cmap2, interpolation='nearest')
    axs[1].set_title('Sum over EP_ID dimension' + ppname + '\n(Resulting shape: Layer_ID x Die_ID): View the number of experts deployed per Die')
    axs[1].set_xlabel('Die_ID')
    axs[1].set_ylabel('Layer_ID')
    fig2d.colorbar(im2, ax=axs[1], ticks=range(dim_z + 1))

    plt.tight_layout()
    # 保存二维求和图（可选）
    if fig_save_path is not None:
        plt.savefig(fig_save_path, bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    # 创建一个示例三维矩阵进行测试和可视化
    sample_shape = (64, 58, 256)  # (devices, layers, experts)
    sample_mapping = np.zeros(sample_shape, dtype=np.int32)
    
    # 简单填充示例数据：每个layer的每个ep分配到至少一个设备，每个设备有相同数量的ep
    for layer in range(sample_shape[1]):
        for expert in range(sample_shape[2]):
            device = expert % sample_shape[0]  # 简单分配
            sample_mapping[device, layer, expert] = 1

    # 测试映射
    is_valid, test_result = test_expert_mapping(sample_mapping)
    print("映射是否有效:", is_valid)
    print("测试结果:", test_result)

    # 可视化映射
    view_patterns(sample_mapping, ppname='Sample Pattern', fig_save_path='sample_pattern.png')