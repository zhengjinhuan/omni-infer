#!/usr/bin/env python3
"""
CSV文件后处理脚本，用于移动文件和提取特定列值的行
"""

import os
import re
import shutil
import pandas as pd
from pathlib import Path
import argparse
import sys
import numpy as np


def extract_sort_value(name, sort_pattern):
    """
    从名称中提取排序值
    
    Args:
        name: 名称字符串
        sort_pattern: 用于提取的正则表达式模式
    
    Returns:
        int: 提取的数字，如果未找到则返回0
    """
    if sort_pattern is None:
        # 默认模式：从后往前的最后一个数字（从最后一个下划线到字符串结束）
        pattern = r'_(\d+)$'
    else:
        pattern = sort_pattern
    
    match = re.search(pattern, str(name))
    if match:
        return int(match.group(1))
    return 0


def identify_ascending_segments(df, sort_column_name, sort_pattern):
    """
    识别数据框中的升序段
    
    Args:
        df: 数据框
        sort_column_name: 用于排序的列名
        sort_pattern: 排序模式的正则表达式
    
    Returns:
        list: 每个元素为一个升序段的索引列表
    """
    # 提取排序值
    sort_values = df[sort_column_name].apply(lambda x: extract_sort_value(x, sort_pattern))
    
    segments = []
    current_segment = []
    prev_value = -float('inf')
    
    for i, value in enumerate(sort_values):
        if value >= prev_value:
            current_segment.append(i)
        else:
            if current_segment:
                segments.append(current_segment)
            current_segment = [i]
        prev_value = value
    
    if current_segment:
        segments.append(current_segment)
    
    return segments


def process_sorted_segments(df, output_dir, value, original_filename, 
                           sort_column_name, sort_pattern, file_pattern):
    """
    处理排序后的数据段
    
    Args:
        df: 过滤后的数据框
        output_dir: 输出目录
        value: 当前处理的值
        original_filename: 原始文件名
        sort_column_name: 用于排序的列名
        sort_pattern: 排序模式的正则表达式
        file_pattern: 文件名模式
    """
    # 识别升序段
    segments = identify_ascending_segments(df, sort_column_name, sort_pattern)
    
    print(f"  找到 {len(segments)} 个升序段")
    
    # 为每个段创建文件
    for step_index, segment_indices in enumerate(segments, 1):
        step_dir = output_dir / f"step_{step_index}"
        step_dir.mkdir(exist_ok=True)
        
        # 提取当前段的数据
        segment_df = df.iloc[segment_indices]
        
        # 从原文件名和file_pattern生成新文件名
        if file_pattern:
            # 尝试从原文件名提取匹配的部分
            match = re.search(file_pattern, original_filename)
            if match:
                step_filename = f"{value}_{original_filename}"
        else:
            # 如果没有file_pattern，使用原始文件名
            step_filename = f"{value}_{original_filename}"
        
        output_file = step_dir / step_filename
        
        # 保存文件
        segment_df.to_csv(output_file, index=False)
        print(f"    段 {step_index}: {len(segment_df)} 行 -> {step_dir.name}/{step_filename}")


def post_process_csv_files(main_path, new_dir_pattern="round_{index}", 
                          column_name="Type", extract_values=None,
                          file_pattern=r'rank_(\d+)\.csv$',
                          sort_column_name="Name", sort_pattern=None):
    """
    后处理CSV文件：移动到all_files目录并提取特定列值的行
    
    Args:
        main_path: 主目录路径
        new_dir_pattern: round目录的命名模式
        column_name: 用于过滤的列名
        extract_values: 要提取的值列表
        file_pattern: 文件名模式
        sort_column_name: 用于排序的列名
        sort_pattern: 排序模式的正则表达式
    """
    main_path = Path(main_path)
    
    # 找到所有round目录
    round_dirs = []
    for item in main_path.iterdir():
        if item.is_dir():
            # 检查是否匹配round目录模式
            match = re.match(new_dir_pattern.replace("{index}", r"(\d+)"), item.name)
            if match:
                round_dirs.append(item)
    
    if not round_dirs:
        print(f"在 {main_path} 中没有找到匹配 {new_dir_pattern} 的目录")
        return
    
    round_dirs.sort(key=lambda x: int(re.match(new_dir_pattern.replace("{index}", r"(\d+)"), x.name).group(1)))
    
    for round_dir in round_dirs:
        print(f"\n处理目录: {round_dir}")
        
        # 创建all_files目录
        all_files_dir = round_dir / "all_files"
        all_files_dir.mkdir(exist_ok=True)
        
        # 移动所有CSV文件到all_files目录
        csv_files = list(round_dir.glob("*.csv"))
        for csv_file in csv_files:
            dest_file = all_files_dir / csv_file.name
            shutil.move(str(csv_file), str(dest_file))
            print(f"  移动: {csv_file.name} -> all_files/")
        
        # 如果指定了要提取的值，进行提取操作
        if extract_values:
            # 创建analysis目录
            analysis_dir = round_dir / "analysis"
            analysis_dir.mkdir(exist_ok=True)
            
            # 为每个要提取的值创建子目录
            for value in extract_values:
                value_dir = analysis_dir / value
                value_dir.mkdir(exist_ok=True)
                
                # 处理all_files目录中的所有CSV文件
                for csv_file in all_files_dir.glob("*.csv"):
                    try:
                        # 读取CSV文件
                        df = pd.read_csv(csv_file)
                        
                        # 检查列是否存在
                        if column_name not in df.columns:
                            print(f"  警告: 文件 {csv_file.name} 中没有列 '{column_name}'")
                            continue
                        
                        # 提取匹配的行
                        filtered_df = df[df[column_name] == value]
                        
                        if not filtered_df.empty:
                            # 检查sort_column_name列是否存在
                            if sort_column_name in filtered_df.columns:
                                # 应用排序逻辑
                                process_sorted_segments(filtered_df, value_dir, value, csv_file.name,
                                                      sort_column_name, sort_pattern, file_pattern)
                            else:
                                # 原始逻辑：直接保存文件
                                new_filename = f"{value}_{csv_file.name}"
                                output_file = value_dir / new_filename
                                filtered_df.to_csv(output_file, index=False)
                                print(f"  提取: {csv_file.name} -> analysis/{value}/{new_filename} ({len(filtered_df)} 行)")
                        else:
                            print(f"  跳过: {csv_file.name} (没有找到 {column_name}='{value}' 的行)")
                    
                    except Exception as e:
                        print(f"  错误: 处理文件 {csv_file.name} 时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description='CSV文件后处理脚本')
    
    parser.add_argument('--main-path', required=True, help='主目录路径')
    parser.add_argument('--new-dir-pattern', default='round_{index}', 
                        help='round目录的命名模式（默认: round_{index}）')
    parser.add_argument('--column-name', default='Type', 
                        help='用于过滤的列名（默认: Type）')
    parser.add_argument('--extract-values', action='append', 
                        help='要提取的值，可以多次指定')
    parser.add_argument('--file-pattern', default=r'rank_(\d+)\.csv$',
                        help='文件名模式（默认: rank_(\\d+)\\.csv$）')
    parser.add_argument('--sort-column-name', default='Name',
                        help='用于排序的列名（默认: Name）')
    parser.add_argument('--sort-pattern', default=None,
                        help='排序模式的正则表达式（默认: 最后一个下划线后的数字）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.main_path):
        print(f"错误: 路径 {args.main_path} 不存在")
        sys.exit(1)
    
    post_process_csv_files(
        main_path=args.main_path,
        new_dir_pattern=args.new_dir_pattern,
        column_name=args.column_name,
        extract_values=args.extract_values,
        file_pattern=args.file_pattern,
        sort_column_name=args.sort_column_name,
        sort_pattern=args.sort_pattern
    )
    
    print("\n后处理完成！")


if __name__ == "__main__":
    main()