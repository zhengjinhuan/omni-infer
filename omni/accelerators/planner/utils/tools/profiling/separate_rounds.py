import os
import re
import shutil
import argparse


def organize_csv_files(main_path, files_per_round, dir_pattern=r'node_(\d+)$', file_pattern=r'rank_(\d+)\.csv$', 
                      new_dir_pattern='round_{index}', preserve_prefix=True):
    """
    组织CSV文件到不同的轮次目录中
    
    Args:
        main_path: 主目录路径
        files_per_round: 每个目录中每轮的文件数量
        dir_pattern: 需要处理的目录名称正则表达式模式，最后一个捕获组必须是数字
        file_pattern: 需要处理的文件名模式，最后一个捕获组必须是数字
        new_dir_pattern: 新目录的命名模式（默认为'round_{index}'）
        preserve_prefix: 是否保留文件名前缀（默认为True）
    """
    try:
        # 检查目录模式是否包含捕获组
        try:
            compiled_dir_pattern = re.compile(dir_pattern)
            if compiled_dir_pattern.groups == 0:
                print(f"错误: 目录模式 '{dir_pattern}' 必须包含至少一个捕获组 ()")
                return False
        except re.error as e:
            print(f"错误: 目录模式 '{dir_pattern}' 不是有效的正则表达式: {e}")
            return False
        
        # 检查文件模式是否包含捕获组
        try:
            compiled_file_pattern = re.compile(file_pattern)
            if compiled_file_pattern.groups == 0:
                print(f"错误: 文件模式 '{file_pattern}' 必须包含至少一个捕获组 ()")
                return False
        except re.error as e:
            print(f"错误: 文件模式 '{file_pattern}' 不是有效的正则表达式: {e}")
            return False
        
        # 确保主目录存在
        if not os.path.exists(main_path):
            print(f"错误: 路径 '{main_path}' 不存在")
            return False
        
        # 获取所有符合模式的目录及其编号
        target_dirs_with_index = []
        for d in os.listdir(main_path):
            if os.path.isdir(os.path.join(main_path, d)):
                match = re.search(dir_pattern, d)
                if match:
                    try:
                        # 获取最后一个捕获组作为目录编号
                        groups = match.groups()
                        if not groups:
                            print(f"错误: 目录 '{d}' 匹配模式但没有捕获组")
                            return False
                        dir_index = int(groups[-1])
                        target_dirs_with_index.append((d, dir_index))
                    except ValueError:
                        print(f"错误: 目录 '{d}' 匹配模式但最后一个捕获组不是数字")
                        return False
        
        if not target_dirs_with_index:
            print(f"在 '{main_path}' 目录下没有找到符合模式 '{dir_pattern}' 的目录")
            return False
        
        # 按目录编号排序
        target_dirs_with_index.sort(key=lambda x: x[1])
        
        # 对每个目录进行处理，获取文件并按轮次分组
        dir_files_by_round = {}  # {round_index: [(dir_index, dir_name, file_info), ...]}
        
        for target_dir, dir_index in target_dirs_with_index:
            target_path = os.path.join(main_path, target_dir)
            
            # 获取该目录中符合模式的文件
            dir_files = []
            for file in os.listdir(target_path):
                match = re.search(file_pattern, file)
                if match:
                    try:
                        groups = match.groups()
                        if not groups:
                            print(f"错误: 文件 '{file}' 匹配模式但没有捕获组")
                            return False
                        
                        # 使用最后一个捕获组作为文件索引
                        file_index = int(groups[-1])
                    except ValueError:
                        print(f"错误: 文件 '{file}' 匹配模式但最后一个捕获组不是数字")
                        return False
                    
                    # 提取前缀（如果需要保留）
                    if preserve_prefix and match.start() > 0:
                        prefix = file[:match.start()]
                    else:
                        prefix = ""
                    
                    dir_files.append({
                        'filename': file,
                        'file_index': file_index,
                        'prefix': prefix,
                        'full_path': os.path.join(target_path, file)
                    })
            
            # 按文件索引排序
            dir_files.sort(key=lambda x: x['file_index'])
            
            # 将该目录的文件按轮次分组
            for i, file_info in enumerate(dir_files):
                round_index = i // files_per_round + 1
                
                if round_index not in dir_files_by_round:
                    dir_files_by_round[round_index] = []
                
                dir_files_by_round[round_index].append((dir_index, target_dir, file_info))
        
        # 统计信息
        total_files = sum(len(files) for files in dir_files_by_round.values())
        print(f"找到 {len(target_dirs_with_index)} 个目录，共 {total_files} 个文件")
        print(f"将创建 {len(dir_files_by_round)} 个round目录")
        
        # 创建round目录并复制文件
        for round_index, files_list in dir_files_by_round.items():
            new_dir_name = new_dir_pattern.format(index=round_index)
            new_dir_path = os.path.join(main_path, new_dir_name)
            
            # 创建新目录(如果不存在)
            if not os.path.exists(new_dir_path):
                os.makedirs(new_dir_path)
                print(f"创建文件夹: {new_dir_path}")
            
            # 按目录编号排序，确保文件顺序
            files_list.sort(key=lambda x: x[0])  # 按dir_index排序
            
            # 为每个round中的文件重新编号
            file_counter = 0
            for dir_index, _, file_info in files_list:
                # 构建新文件名
                new_file_name = re.sub(r'(\d+)', str(file_counter), file_info['filename'], count=1)
                
                src_file = file_info['full_path']
                dst_file = os.path.join(new_dir_path, new_file_name)
                
                # 复制文件(不删除原文件)
                shutil.copy2(src_file, dst_file)
                print(f"复制并重命名文件: {src_file} -> {dst_file}")
                
                file_counter += 1
        
        return True
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="组织CSV文件到不同的轮次目录中")
    parser.add_argument("--main-path", required=True, 
                       help="包含目标目录的主目录路径")
    parser.add_argument("--files-per-round", type=int, required=True, 
                       help="每个目录中每轮的文件数量")
    parser.add_argument("--dir-pattern", default=r"node_(\d+)$", 
                       help="需要处理的目录名称正则表达式模式，最后一个捕获组必须是数字（默认: 'node_(\\d+)$'）")
    parser.add_argument("--file-pattern", default=r"rank_(\d+)\.csv$", 
                       help="需要处理的文件名正则表达式模式，最后一个捕获组必须是数字（默认: 'rank_(\\d+)\\.csv$'）")
    parser.add_argument("--new-dir-pattern", default="round_{index}", 
                       help="新目录的命名模式，{index}会被替换为轮次编号（默认: 'round_{index}'）")
    parser.add_argument("--no-preserve-prefix", action="store_true", 
                       help="不保留文件名前缀")
    
    args = parser.parse_args()
    
    if args.files_per_round <= 0:
        print("错误: files-per-round 必须是正整数")
        exit(1)
    
    print(f"开始整理文件...")
    print(f"主目录: {args.main_path}")
    print(f"每个目录中每轮文件数: {args.files_per_round}")
    print(f"目录模式: {args.dir_pattern}")
    print(f"文件模式: {args.file_pattern}")
    print(f"新目录模式: {args.new_dir_pattern}")
    print(f"保留前缀: {'否' if args.no_preserve_prefix else '是'}")
    
    success = organize_csv_files(
        args.main_path, 
        args.files_per_round,
        dir_pattern=args.dir_pattern,
        file_pattern=args.file_pattern,
        new_dir_pattern=args.new_dir_pattern,
        preserve_prefix=not args.no_preserve_prefix
    )
    
    if success:
        print("文件整理完成！")
    else:
        print("文件整理失败！")
        exit(1)


if __name__ == "__main__":
    main()