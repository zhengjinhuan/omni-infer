import os
import shutil
import glob
import re
import argparse


def main():
    parser = argparse.ArgumentParser(description="Collect files from source directories to target directory.")
    parser.add_argument("-s", "--source", help="The source directory", required=True)
    parser.add_argument("-t", "--target", help="The target directory (default: same as source)", default=None)
    parser.add_argument("--node-rank", type=int, default=0, help="Node rank for target directory suffix")
    parser.add_argument("--path-pattern", default="devserver*/ASCEND_PROFILER_OUTPUT/kernel_details.csv",
                       help="File path pattern relative to source directory")
    parser.add_argument("--output-prefix", default="kernel_details", help="Prefix for output files")
    parser.add_argument("--timestamp-position", type=int, default=-3, 
                       help="Position of timestamp in directory name for sorting (0 to disable)")
    
    args = parser.parse_args()

    source_root = args.source
    # 如果未指定target，使用source作为默认值
    target_dir = args.target if args.target else args.source
    target_dir = target_dir + f"_node_{args.node_rank}"

    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目标目录: {target_dir}")

    # 使用glob查找符合模式的所有文件
    file_pattern = os.path.join(source_root, args.path_pattern)
    matched_files = glob.glob(file_pattern)
    
    # 处理排序
    if args.timestamp_position != 0 and matched_files:
        try:
            # 获取文件的父目录进行排序
            def get_sort_key(filepath):
                # 找到包含时间戳的目录（通常是最顶层的目录）
                rel_path = os.path.relpath(filepath, source_root)
                dirs = rel_path.split(os.sep)
                if dirs:
                    return dirs[0].split('_')[args.timestamp_position]
                return ''
            
            matched_files = sorted(matched_files, key=get_sort_key)
        except IndexError:
            print(f"警告: 无法按位置 {args.timestamp_position} 进行排序，将使用默认排序")
            matched_files = sorted(matched_files)
    else:
        matched_files = sorted(matched_files)

    # 用于跟踪文件编号
    rank_counter = 0

    for file_path in matched_files:
        # 检查文件是否存在
        if os.path.exists(file_path):
            # 构造输出文件名
            output_filename = f"{args.output_prefix}_rank_{rank_counter}.csv"
            output_path = os.path.join(target_dir, output_filename)
            
            # 复制文件
            shutil.copy2(file_path, output_path)
            print(f"复制文件: {file_path} -> {output_path}")
            
            # 增加计数器
            rank_counter += 1
        else:
            print(f"警告: 文件不存在 {file_path}")

    print(f"完成! 共复制了 {rank_counter} 个文件到 {target_dir}")


if __name__ == "__main__":
    main()