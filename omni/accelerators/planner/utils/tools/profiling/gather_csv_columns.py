#!/usr/bin/env python3
"""
用于从CSV文件中收集指定列的数据，并生成汇总和统计报告
"""
import os
import sys
import argparse
import pandas as pd
import glob
import re
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='从CSV文件中收集指定列的数据')
    parser.add_argument('--main-path', required=True, help='包含目标目录的主路径')
    parser.add_argument('--gather', default='Duration(us)', help='要收集的列名')
    parser.add_argument('--round-pattern', default='round_(\d+)$', help='Round目录的正则表达式模式')
    parser.add_argument('--step-pattern', default='step_(\d+)$', help='Step目录的正则表达式模式')
    parser.add_argument('--file-pattern', default='rank_(\d+)\.csv$', help='CSV文件的正则表达式模式')
    parser.add_argument('--analysis-dir', default='analysis', help='Analysis目录名称')
    
    return parser.parse_args()

def find_matching_directories(path, pattern):
    """根据正则表达式模式查找匹配的目录"""
    matching_dirs = []
    pattern_re = re.compile(pattern)
    
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path) and pattern_re.search(item):
            matching_dirs.append(item)
    
    # 如果模式包含数字捕获组，按数字排序
    try:
        matching_dirs.sort(key=lambda x: int(pattern_re.search(x).group(1)))
    except:
        matching_dirs.sort()
    
    return matching_dirs

def find_matching_files(path, pattern):
    """根据正则表达式模式查找匹配的文件"""
    matching_files = []
    pattern_re = re.compile(pattern)
    
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isfile(full_path) and pattern_re.search(item):
            matching_files.append(item)
    
    # 如果模式包含数字捕获组，按数字排序
    try:
        matching_files.sort(key=lambda x: int(pattern_re.search(x).group(1)))
    except:
        matching_files.sort()
    
    return matching_files

def collect_column_data(step_path, file_pattern, column_name):
    """从step目录中的所有匹配文件收集指定列的数据"""
    csv_files = find_matching_files(step_path, file_pattern)
    
    if not csv_files:
        return None, None
    
    # 存储每个rank的完整DataFrame
    rank_data = {}
    row_counts = {}
    
    # 首先读取所有文件并检查行数
    for csv_file in csv_files:
        csv_path = os.path.join(step_path, csv_file)
        try:
            df = pd.read_csv(csv_path)
            
            # 提取rank编号
            rank_match = re.search(r'rank_(\d+)', csv_file)
            rank_num = int(rank_match.group(1)) if rank_match else 0
            
            # 存储行数
            row_counts[rank_num] = len(df)
            
            # 存储数据
            rank_data[rank_num] = df
            
        except Exception as e:
            print(f"错误: 读取文件 {csv_path} 时出错: {e}")
            return None, None
    
    # 检查所有文件是否具有相同的行数
    row_counts_list = list(row_counts.values())
    if len(set(row_counts_list)) > 1:
        print(f"错误: 不同rank的CSV文件行数不同: {row_counts}")
        return None, None
    
    # 检查所有文件是否都包含指定的列
    for rank_num, df in rank_data.items():
        if column_name not in df.columns:
            print(f"错误: rank_{rank_num} 的文件中没有列 '{column_name}'")
            return None, None
    
    # 构建合并的DataFrame
    # 使用第一个文件作为基础（保留行索引信息）
    base_rank = min(rank_data.keys())
    base_df = rank_data[base_rank]
    
    # 创建新的DataFrame，保留原始行的所有信息（如果需要的话）
    # 这里我们只提取指定的列
    result_df = pd.DataFrame()
    
    # 对于每个rank，添加其数据作为新列
    for rank_num in sorted(rank_data.keys()):
        df = rank_data[rank_num]
        result_df[f'{column_name}_rank_{rank_num}'] = df[column_name].values
    
    return result_df, rank_data

def calculate_statistics(data_df, column_name):
    """计算统计信息"""
    stats = {}
    
    # 获取所有包含数据的列（格式为 {column_name}_rank_{rank_num}）
    data_columns = [col for col in data_df.columns if col.startswith(f'{column_name}_rank_')]
    
    if not data_columns:
        return stats
    
    # 计算每个rank的统计信息
    rank_stats = {}
    for col in data_columns:
        # 提取rank编号
        rank_match = re.search(r'_rank_(\d+)$', col)
        if rank_match:
            rank_num = int(rank_match.group(1))
            column_data = data_df[col]
            
            rank_stats[rank_num] = {
                'count': len(column_data),
                'mean': column_data.mean(),
                'std': column_data.std(),
                'min': column_data.min(),
                '25%': column_data.quantile(0.25),
                '50%': column_data.quantile(0.50),
                '75%': column_data.quantile(0.75),
                'max': column_data.max()
            }
    
    # 计算所有数据的整体统计信息
    all_data = pd.concat([data_df[col] for col in data_columns])
    stats['overall'] = {
        'count': len(all_data),
        'mean': all_data.mean(),
        'std': all_data.std(),
        'min': all_data.min(),
        '25%': all_data.quantile(0.25),
        '50%': all_data.quantile(0.50),
        '75%': all_data.quantile(0.75),
        'max': all_data.max()
    }
    
    stats['rank_stats'] = rank_stats
    
    return stats

def process_subfolder(subfolder_path, step_pattern, file_pattern, column_name, output_dir):
    """处理单个子文件夹"""
    subfolder_name = os.path.basename(subfolder_path)
    step_dirs = find_matching_directories(subfolder_path, step_pattern)
    
    for step_dir in step_dirs:
        step_path = os.path.join(subfolder_path, step_dir)
        step_match = re.search(step_pattern, step_dir)
        step_index = step_match.group(1) if step_match else step_dir
        
        # 收集数据
        result_df, rank_data = collect_column_data(step_path, file_pattern, column_name)
        
        if result_df is None:
            print(f"警告: 在 {step_path} 中处理数据时出错")
            continue
        
        # 生成输出文件名
        gather_filename = f"{subfolder_name}_{column_name}_step_{step_index}.csv"
        stats_filename = f"{subfolder_name}_{column_name}_step_{step_index}_statistics.csv"
        
        gather_filepath = os.path.join(output_dir, gather_filename)
        stats_filepath = os.path.join(output_dir, stats_filename)
        
        # 保存汇总数据
        result_df.to_csv(gather_filepath, index=False)
        print(f"已保存汇总数据到: {gather_filepath}")
        
        # 计算并保存统计信息
        stats = calculate_statistics(result_df, column_name)
        
        if stats:
            # 创建统计信息DataFrame
            overall_stats = stats['overall']
            overall_stats_df = pd.DataFrame({
                'metric': ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],
                'value': [overall_stats['count'], overall_stats['mean'], overall_stats['std'], 
                         overall_stats['min'], overall_stats['25%'], overall_stats['50%'], 
                         overall_stats['75%'], overall_stats['max']]
            })
            
            # 创建rank统计信息DataFrame
            rank_stats_data = []
            for rank_num, rank_stat in stats['rank_stats'].items():
                rank_stats_data.append({
                    'rank': rank_num,
                    'count': rank_stat['count'],
                    'mean': rank_stat['mean'],
                    'std': rank_stat['std'],
                    'min': rank_stat['min'],
                    '25%': rank_stat['25%'],
                    '50%': rank_stat['50%'],
                    '75%': rank_stat['75%'],
                    'max': rank_stat['max']
                })
            
            rank_stats_df = pd.DataFrame(rank_stats_data)
            
            # 保存完整的统计信息
            rank_stats_df.to_csv(stats_filepath, index=False)
            print(f"已保存统计信息到: {stats_filepath}")
        else:
            print(f"警告: 无法计算统计信息")

def main():
    args = parse_arguments()
    
    # 查找所有round目录
    round_dirs = find_matching_directories(args.main_path, args.round_pattern)
    
    if not round_dirs:
        print(f"错误: 在 {args.main_path} 中没有找到匹配 '{args.round_pattern}' 的目录")
        sys.exit(1)
    
    print(f"找到 {len(round_dirs)} 个round目录")
    
    # 处理每个round目录
    for round_dir in round_dirs:
        round_path = os.path.join(args.main_path, round_dir)
        analysis_path = os.path.join(round_path, args.analysis_dir)
        
        if not os.path.exists(analysis_path):
            print(f"警告: {analysis_path} 不存在，跳过")
            continue
        
        print(f"\n处理 {round_dir} 中的数据...")
        
        # 获取analysis下的所有子文件夹
        subfolders = [f for f in os.listdir(analysis_path) 
                     if os.path.isdir(os.path.join(analysis_path, f))]
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(analysis_path, subfolder)
            print(f"  处理子文件夹: {subfolder}")
            
            # 处理这个子文件夹
            process_subfolder(subfolder_path, args.step_pattern, 
                            args.file_pattern, args.gather, analysis_path)

if __name__ == "__main__":
    main()