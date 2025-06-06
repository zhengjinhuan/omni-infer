import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_device_load(placement_pattern: np.ndarray, load_array: np.ndarray) -> pd.DataFrame:
    """
    分析各层中设备负载，将每个专家的总负载
    按照在placement_pattern中的部署情况均分到部署该专家的设备上。

    参数:
        placement_pattern (np.ndarray): 三维矩阵，形状为 (device_id, layer_id, ep_id)。
                                          每个元素为 1 或 0，表示对应设备在该层是否部署了该专家。
        load_array (np.ndarray): 二维矩阵，形状为 (layer_id, ep_id)。
                                 每个元素表示对应层中该专家的总负载。

    返回:
        pd.DataFrame: 每一行记录一个设备在特定层中承担的负载，包括字段 'layer', 'device_id', 'load'。
    """
    num_devices, num_layers, num_experts = placement_pattern.shape
    load_records = []

    for layer in range(num_layers):
        for device in range(num_devices):
            total_load = 0
            for ep in range(num_experts):
                if placement_pattern[device, layer, ep] == 1:
                    devices_with_ep = np.where(placement_pattern[:, layer, ep] == 1)[0]
                    num_deployment = len(devices_with_ep)
                    if num_deployment > 0:
                        load_for_ep = load_array[layer, ep] / num_deployment
                        total_load += load_for_ep
            load_records.append({
                'layer': layer,
                'device_id': device,
                'load': total_load
            })

    df_load = pd.DataFrame(load_records)
    df_pivot = df_load.pivot(index='layer', columns='device_id', values='load')
    df_pivot.rename(columns=lambda x: f"device_{x}", inplace=True)
    df_pivot = df_pivot.reset_index(drop=True)
    return df_pivot

def calculate_best_ep_per_layer(load_array: np.ndarray, num_devices: int) -> np.ndarray:
    """
    计算每层的平均设备激活值（best_ep_per_layer）。

    对于每层，计算所有专家的总负载（激活值），然后除以设备数，得到平均设备负载。

    参数:
        load_array (np.ndarray): 二维矩阵，形状为 (num_layers, num_experts)。
                                 每个元素表示对应层中某个专家的负载。
        num_devices (int): 设备数量。

    返回:
        np.ndarray: 一维数组，形状为 (num_layers,)，表示每层的平均设备激活值。
    """
    num_layers, num_experts = load_array.shape
    total_load_per_layer = np.sum(load_array, axis=1)
    best_ep_per_layer = total_load_per_layer / num_devices
    return best_ep_per_layer

def analyze_default_deployment_load(load_array: np.ndarray, num_devices: int) -> pd.DataFrame:
    """
    根据默认部署规则计算各层中各设备的负载。

    默认部署规则：
      - 专家按照顺序均匀分布到各个设备上。
      - 例如，如果一层有 32 个专家，4 个设备，则专家索引 0~7 分配给设备 0，8~15 分配给设备 1，依次类推。

    参数:
        load_array (np.ndarray): 二维矩阵，形状为 (num_layers, num_experts)。
                                 每个元素表示对应层中某个专家的负载。
        num_devices (int): 设备数量。

    返回:
        pd.DataFrame: 每一行代表一层，各列依次为 device_0、device_1、…，值为该设备在该层的负载总和。

    异常:
        如果 num_experts 不能被 num_devices 整除，则抛出 ValueError。
    """
    num_layers, num_experts = load_array.shape
    if num_experts % num_devices != 0:
        raise ValueError("Number of experts in load_array must be divisible by num_devices")

    experts_per_device = num_experts // num_devices
    default_load_records = []

    for layer in range(num_layers):
        layer_record = {}
        for device in range(num_devices):
            start = device * experts_per_device
            end = (device + 1) * experts_per_device
            total_load = np.sum(load_array[layer, start:end])
            layer_record[f"device_{device}"] = total_load
        default_load_records.append(layer_record)

    df_default = pd.DataFrame(default_load_records)
    df_default = df_default.reset_index(drop=True)
    return df_default

def plot_load_comparison_heatmaps(optimized_df: pd.DataFrame, default_df: pd.DataFrame, figsize=(18, 8), ppname='') -> None:
    """
    根据两个 DataFrame 绘制设备负载热力图对比图。

    参数:
        optimized_df (pd.DataFrame): 优化后的设备负载 DataFrame，
                                     行代表层（Layer ID），列代表设备（建议列名为 device_0, device_1, ...）。
        default_df (pd.DataFrame): 默认部署的设备负载 DataFrame，
                                   行代表层（Layer ID），列代表设备（建议列名为 device_0, device_1, ...）。

    绘图将展示两个方案的热力图，使用相同的颜色区间 (vmin, vmax) 便于直观对比各层各设备的负载情况，
    同时在每个单元格内标注负载数据，图像大小适中，可展示 layer_id 与 device_id 信息。
    """
    vmin = min(optimized_df.min().min(), default_df.min().min())
    vmax = max(optimized_df.max().max(), default_df.max().max())

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sns.heatmap(
        optimized_df,
        ax=axes[0],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        cbar_kws={'shrink': 0.7}
    )
    axes[0].set_title("Optimized Device Load : " + ppname)
    axes[0].set_xlabel("Device ID")
    axes[0].set_ylabel("Layer ID")

    sns.heatmap(
        default_df,
        ax=axes[1],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        cbar_kws={'shrink': 0.7}
    )
    axes[1].set_title("Default Device Load")
    axes[1].set_xlabel("Device ID")
    axes[1].set_ylabel("Layer ID")

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_load_comparison_heatmaps_multi(optimized_df_lis: list, ppname_lis: list, figsize=(18, 8), num_devices=32, dataset_name='Rand', save_path=None) -> None:
    """
    根据多个 DataFrame 绘制设备负载热力图对比图。

    参数:
        optimized_df_lis (list): 包含多个负载 DataFrame 的列表，
                                每个 DataFrame 行代表层（Layer ID），列代表设备（建议列名为 device_0, device_1, ...）。
        ppname_lis (list): 对应每个 DataFrame 的名称列表。
        figsize (tuple): 图表大小，默认为 (18, 8)。
        num_devices (int): 设备数量，默认为 32。
        dataset_name (str): 数据集名称，用于标题显示，默认为 'Rand'。
        save_path (str, optional): 图片保存路径，如果为 None 则不保存图片。

    绘图将展示多个方案的热力图，使用相同的颜色区间 (vmin, vmax) 便于直观对比各层各设备的负载情况，
    图像大小适中，可展示 layer_id 与 device_id 信息。
    """
    vmin = optimized_df_lis[0].min().min()
    vmax = vmin

    for df in optimized_df_lis:
        vmin = min(vmin, df.min().min())
        vmax = max(vmax, df.max().max())

    num_lis = len(optimized_df_lis)
    fig, axes = plt.subplots(1, num_lis, figsize=figsize)

    for i in range(num_lis):
        # 绘制热力图
        sns.heatmap(
            optimized_df_lis[i],
            ax=axes[i] if num_lis > 1 else axes,
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            cbar_kws={'shrink': 0.7}
        )
        # 设置标题，添加换行符并减小字体大小
        title = ppname_lis[i] + '\n' + dataset_name  # 使用换行符分隔
        (axes[i] if num_lis > 1 else axes).set_title(title, fontsize=10, pad=10)
        (axes[i] if num_lis > 1 else axes).set_xlabel("Die ID", fontsize=10)
        (axes[i] if num_lis > 1 else axes).set_ylabel("Layer ID", fontsize=10)

    # 调整子图间距，防止标题重叠
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3)  # 增加子图之间的水平间距

    # 保存图片
    if save_path is not None:
        filename = f'Heat_{dataset_name}.png'
        save_file_path = os.path.join(save_path, filename)
        plt.savefig(save_file_path, bbox_inches='tight', dpi=100)
    
    plt.show()
    
def plot_max_load_comparison_lis(optimized_df_lis: list, ppname_lis: list, num_devices=32, dataset_name='Rand', save_path=None, load_array=None) -> None:
    """
    根据多个 DataFrame 计算每层设备负载的最大值，并绘制对比柱状图，叠加一条表示所有层最小激活值的水平线。

    每层为一个分组，每组中有多个柱子，表示不同优化方案的最大设备负载。
    水平线表示所有层的最小设备激活值（best_ep_per_layer 的最小值）。

    参数:
        optimized_df_lis (list): 包含多个负载 DataFrame 的列表，
                                每个 DataFrame 行代表层（Layer ID），列代表设备（如 device_0, device_1,...）。
        ppname_lis (list): 对应每个 DataFrame 的名称列表。
        num_devices (int): 设备数量，默认为 32。
        dataset_name (str): 数据集名称，默认为 'Rand'。
        save_path (str, optional): 图片保存路径，如果为 None 则不保存图片。
        load_array (np.ndarray, optional): 二维矩阵，形状为 (num_layers, num_experts)，用于计算 best_ep_per_layer。
                                          如果提供，则绘制水平线。
    """
    max_lis = [df.max(axis=1) for df in optimized_df_lis]

    n_layers = optimized_df_lis[0].shape[0]
    layers = optimized_df_lis[0].index.astype(str)
    indices = np.arange(n_layers)
    bar_width = 0.85

    fig_width = max(12, n_layers * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bar_pos = -bar_width/2 + np.array([i * bar_width/len(optimized_df_lis) for i in range(len(optimized_df_lis))])

    # 绘制柱状图
    for i in range(len(optimized_df_lis)):
        ax.bar(indices + bar_pos[i], max_lis[i], bar_width/len(optimized_df_lis),
               label=ppname_lis[i], color=f'C{i}')

    # 如果提供了 load_array，计算所有层的最小激活值并绘制水平线
    if load_array is not None:
        best_ep_per_layer = calculate_best_ep_per_layer(load_array, num_devices)
        min_best_ep = np.min(best_ep_per_layer)  # 计算所有层的最小激活值
        ax.axhline(
            y=min_best_ep,
            color='gray',  # 使用更柔和的灰色
            linestyle='--',  # 改为虚线
            linewidth=1.9,  # 减小线条粗细
            alpha=0.9,  # 增加透明度
            label='Best EP'  # 简化图例标签
        )
        # 调整 y 轴范围，固定最小值为 0
        max_y = max(max_lis[i].max() for i in range(len(max_lis)))
        line_y = min_best_ep
        ax.set_ylim(0, max(max_y, line_y) * 1.02)  # y 轴下限固定为 0，上限稍放大

    ax.set_xlabel('Layer ID')
    ax.set_ylabel('Maximum Die Load')
    ax.set_title(f'Maximum Die Load Comparison per Layer : {dataset_name}')
    ax.set_xticks(indices)
    ax.set_xticklabels(layers, rotation=45)
    ax.legend()

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    if save_path is not None:
        filename = f'Bars_{dataset_name}.png'
        save_file_path = os.path.join(save_path, filename)
        plt.savefig(save_file_path, bbox_inches='tight', dpi=100)
    plt.show()

def calculate_max_load_reduction(optimized_df_lis: list, ppname_lis: list, save_path: str, dataset_name: str) -> pd.DataFrame:
    """
    Calculate the percentage reduction in total max load for each pattern relative to the default pattern
    and save the results as a CSV file.

    Parameters:
        optimized_df_lis (list): List of DataFrames, each containing load data for devices across layers.
        ppname_lis (list): List of pattern names corresponding to the DataFrames.
        save_path (str): Directory path to save the CSV file.
        dataset_name (str): Dataset name for the CSV filename.

    Returns:
        pd.DataFrame: DataFrame containing pattern names, total max loads, and percentage reductions.
    """
    # Calculate max load per layer for each pattern
    max_loads = [df.max(axis=1) for df in optimized_df_lis]
    
    # Sum max loads across all layers for each pattern
    total_max_loads = [np.sum(max_load) for max_load in max_loads]
    
    # Default pattern is the first in the list
    default_total_max_load = total_max_loads[0]
    
    # Calculate percentage reduction relative to default
    percentage_reductions = [
        ((default_total_max_load - total_max_load) / default_total_max_load) * 100
        for total_max_load in total_max_loads
    ]
    
    # Create DataFrame
    results_df = pd.DataFrame({
        'Pattern': ppname_lis,
        'Total_Max_Load': total_max_loads,
        'Percentage_Reduction': percentage_reductions
    })
    
    # Save to CSV
    filename = f'Max_Load_Reduction_{dataset_name}.csv'
    save_file_path = os.path.join(save_path, filename)
    results_df.to_csv(save_file_path, index=False)
    
    return results_df

def analyze_and_plot_deployments(
    # load_file: str = './topk_id_count/topk_ids_count_random_1k+1k_32_devices_decode_0428night.csv',
    load_file: str = './topk_id_count/topk_ids_count_longbench_3.5k_decode.csv',
    pp_path_lis: list = [
        f'./placement_pattern/DSV3_0430_longbench_1k_decode_rearrangeonly_+58_58_MoELayers_64_dies_epmaxdeploy_12.npy',
        f'./placement_pattern/DSV3_0430_longbench_3.5k_decode_rearrangeonly_+58_58_MoELayers_64_dies_epmaxdeploy_12.npy',
        f'./placement_pattern/DSV3_0430_longbench_6k_decode_+58_58_MoELayers_64_dies_epmaxdeploy_12.npy',
        f'./placement_pattern/DSV3_0506_longbench_1k_decode_redundant+rearrange_only_with_ceiling_+58_58_MoELayers_64_dies_epmaxdeploy_12.npy',
        # f'./placement_pattern/DSV3_0429_longbench_1k_32_devices_decode_1k_threshold_1.7_+10_58_MoELayers_32_dies_epmaxdeploy_12.npy'
        # f'./placement_pattern/DSV3_0425_longbench_prefill_24bs_6kseqlength_4host_RedFullLays_+30_58_MoELayers_64_dies_epmaxdeploy_12.npy',
        # f'./placement_pattern/DSV3_0425_longbench_prefill_24bs_6kseqlength_4host_RedFullLays_+58_58_MoELayers_64_dies_epmaxdeploy_12.npy'
    ],
    # ppname_lis: list = ['Default', 'rearrangeonly_+10', 'rearrangeonly_+20', 'rearrangeonly_+30','rearrangeonly_+58'],
    ppname_lis: list = ['Default', 'longbench_1k', 'longbench_3.5k', 'longbench_6k','longbench_1k_with_ceiling'],
    fig_save_path: str = './',
    num_dies: int = 64,
    dataset_name: str = 'longbench_3.5k_all58_test'
) -> None:
    """
    加载专家负载数据和放置模式，分析设备负载，并绘制热力图和柱状图进行对比。

    参数:
        load_file (str): 专家负载数据的CSV文件路径。
        pp_path_lis (list): 放置模式文件的路径列表（.npy文件）。
        ppname_lis (list): 放置模式对应的名称列表。
        fig_save_path (str): 保存图表的目录路径。
        num_dies (int): 设备数量，默认为64。
        dataset_name (str): 数据集名称，用于图表标题，默认为'sharegpt'。
    """
    load_array = np.genfromtxt(load_file, delimiter=',', skip_header=1)[:, 1:]
    placement_pattern_lis = [np.load(path) for path in pp_path_lis]
    df_lis = [analyze_default_deployment_load(load_array, num_devices=placement_pattern_lis[0].shape[0])]
    for placement_pattern in placement_pattern_lis:
        df_lis.append(analyze_device_load(placement_pattern, load_array))

    plot_load_comparison_heatmaps_multi(
        optimized_df_lis=df_lis,
        ppname_lis=ppname_lis,
        figsize=(23, 10),
        num_devices=num_dies,
        dataset_name=f'{num_dies} Dies_' + dataset_name,
        save_path=fig_save_path
    )

    plot_max_load_comparison_lis(
        optimized_df_lis=df_lis,
        ppname_lis=ppname_lis,
        num_devices=num_dies,
        dataset_name=f'{num_dies} Dies_' + dataset_name,
        save_path=fig_save_path,
        load_array=load_array
    )
    
    calculate_max_load_reduction(
        optimized_df_lis=df_lis,
        ppname_lis=ppname_lis,
        save_path=fig_save_path,
        dataset_name=f'{num_dies} Dies_' + dataset_name
    )

if __name__ == "__main__":
    analyze_and_plot_deployments()
