import os
import argparse
import numpy as np

from step_1_generate_csv_with_ceiling import generate_csv
from step_2_placement_pattern_generation_rearrangeonly import process_expert_deployments
# from step_2_placement_pattern_generation_redundant import process_expert_deployments
from step_3_placement_pattern_checking_and_plot import test_expert_mapping, view_patterns
from step_4_load_analysis_and_plot import analyze_and_plot_deployments

def main():
    # 设置参数解析器
    parser = argparse.ArgumentParser(description="New pipeline for processing expert deployment and load analysis.")
    parser.add_argument('--input_folder', type=str, default='./activation_datas/default',
                        help='Input folder under ./activation_datas containing txt files (default: ./activation_datas/default)')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Name of the output CSV file in topk_id_count folder (e.g., topk_ids_count.csv)')
    parser.add_argument('--num_layers', type=int, default=58,
                        help='Number of layers (default: 58)')
    parser.add_argument('--num_ranks', type=int, required=True,
                        help='Number of rank IDs')
    parser.add_argument('--rank_id_range', type=int, nargs=2, default=None,
                        help='Range of rank IDs to process (min_rank_id, max_rank_id), default: (0, num_ranks-1)')
    parser.add_argument('--num_devices', type=int, default=64,
                        help='Number of devices (default: 64)')
    parser.add_argument('--num_redundant_layers', type=int, nargs='*', default=[0, 10, 20, 30, 58],
                        help='List of redundant layers for batch processing (default: 0 10 20 30 58)')
    parser.add_argument('--expert_redundant_limit', type=int, default=11,
                        help='Maximum additional deployments per expert (default: 11)')
    parser.add_argument('--num_moe_layers', type=int, default=58,
                        help='Number of MoE layers (default: 58)')
    parser.add_argument('--num_eps', type=int, default=256,
                        help='Number of experts per layer (default: 256)')
    parser.add_argument('--dataset_name', type=str, default='sharegpt',
                        help='Dataset name for plotting (default: sharegpt)')
    parser.add_argument('--output_file_prefix', type=str, default='DSV3_0418_share_gpt_RedFullLays',
                        help='Prefix for output placement pattern filenames (default: DSV3_0418_share_gpt_RedFullLays)')

    args = parser.parse_args()

    # 设置默认的 rank_id_range
    if args.rank_id_range is None:
        args.rank_id_range = (0, args.num_ranks - 1)

    # 定义文件夹路径
    base_dir = '.'
    topk_id_count_dir = './topk_id_count'
    placement_pattern_dir = './placement_pattern'
    placement_pattern_view_dir = './placement_pattern_view'
    placement_pattern_analysis_dir = './placement_pattern_analysis'

    # 创建输出目录
    for dir_path in [topk_id_count_dir, placement_pattern_dir, placement_pattern_view_dir, placement_pattern_analysis_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Step 1: 使用 generate_csv.py 生成层-EP计数矩阵
    output_csv_path = os.path.join(topk_id_count_dir, args.output_csv)
    print(f"Step 1: Generating layer-EP count matrix from {args.input_folder} to {output_csv_path}")
    generate_csv(
        input_folder=args.input_folder,
        output_csv=output_csv_path,
        num_layers=args.num_layers,
        num_ranks=args.num_ranks,
        rank_id_range=args.rank_id_range
    )

    # Step 2: 为每个 num_redundant_layers 生成放置模式
    pp_path_lis = []
    ppname_lis = ['Default']  # 从 Default 开始
    for num_redundant_layers in args.num_redundant_layers:
        output_file = (f'{args.output_file_prefix}_+{num_redundant_layers}'
                       f'_{args.num_moe_layers}_MoELayers_{args.num_devices}_dies_epmaxdeploy_{args.expert_redundant_limit + 1}.npy')
        output_path = os.path.join(placement_pattern_dir, output_file)
        pp_path_lis.append(output_path)
        ppname_lis.append(f'Rearrange_{num_redundant_layers}_layers_pattern')

        print(f"Step 2: Generating placement pattern for num_redundant_layers={num_redundant_layers}")
        process_expert_deployments(
            input_file=output_csv_path,
            output_dir=placement_pattern_dir,
            num_devices=args.num_devices,
            num_redundant_layers=num_redundant_layers,
            expert_redundant_limit=args.expert_redundant_limit,
            num_moe_layers=args.num_moe_layers,
            num_eps=args.num_eps,
            output_file=output_file
        )

    # Step 3: 测试和可视化每个放置模式
    for pp_path, ppname in zip(pp_path_lis, ppname_lis[1:]):  # 跳过 Default
        print(f"Step 3: Testing and visualizing pattern {pp_path}")
        placement_pattern = np.load(pp_path)

        # 测试专家映射
        is_valid, test_result = test_expert_mapping(placement_pattern)
        print(f"Pattern {ppname} valid: {is_valid}")
        print(f"Test result: {test_result}")

        # 可视化模式
        fig_save_path = os.path.join(placement_pattern_view_dir, f"{os.path.basename(pp_path)[:-4]}.png")
        view_patterns(placement_pattern, ppname=ppname, fig_save_path=fig_save_path)

    # Step 4: 分析和绘制负载分布
    print(f"Step 4: Analyzing and plotting load distributions")
    analyze_and_plot_deployments(
        load_file=output_csv_path,
        pp_path_lis=pp_path_lis,
        ppname_lis=ppname_lis,
        fig_save_path=placement_pattern_analysis_dir,
        num_dies=args.num_devices,
        dataset_name=args.dataset_name
    )

if __name__ == "__main__":
    main()