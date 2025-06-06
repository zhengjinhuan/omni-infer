import os
import csv
import glob
import numpy as np
import math

def generate_csv(input_folder, output_csv, num_layers=58, num_ranks=32, rank_id_range=(0, 31)):
    """
    Generate a CSV file from txt files in the input folder.
    Each txt file contains num_layers rows, each with numbers_per_rank numbers.
    The CSV will have num_layers rows (layer_0 to layer_{num_layers-1})
    and num_ranks*numbers_per_rank columns (ep_0 to ep_{num_ranks*numbers_per_rank-1}).
    
    Parameters:
        input_folder (str): Folder containing the txt files.
        output_csv (str): Output CSV file name.
        num_layers (int): Number of layers (rows in each txt file).
        num_ranks (int): Number of rank IDs.
        rank_id_range (tuple): (min_rank_id, max_rank_id), inclusive range of rank IDs to process.
    """
    # num_positions = 256 + 32
    num_positions = 256 + 0
    
    # Check if num_positions is divisible by num_ranks
    if num_positions % num_ranks != 0:
        raise ValueError(f"num_positions must be divisible by num_ranks. Got num_ranks={num_ranks}")
    
    numbers_per_rank = num_positions // num_ranks
    
    # Initialize the CSV data matrix
    csv_data = np.zeros((num_layers, num_ranks * numbers_per_rank), dtype=int)
    
    # Find all txt files matching the pattern
    txt_files = glob.glob(os.path.join(input_folder, "activation_counts_recordstep_*.txt"))
    
    # Extract min and max rank_id
    min_rank_id, max_rank_id = rank_id_range
    
    # Process each txt file
    for txt_file in txt_files:
        # Extract rank_id from filename
        filename = os.path.basename(txt_file)
        try:
            rank_id = int(filename.split('_rank_')[1].split('.txt')[0])
        except (IndexError, ValueError):
            print(f"跳过文件名格式不符合预期的文件: {filename}")
            continue
            
        # Check if rank_id is within the specified range
        if not (min_rank_id <= rank_id <= max_rank_id):
            print(f"跳过 rank_id {rank_id}，不在指定范围 [{min_rank_id}, {max_rank_id}] 内")
            continue
            
        # Read the txt file
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            if len(lines) != num_layers:
                print(f"警告: {filename} 有 {len(lines)} 行，预期为 {num_layers} 行")
                continue
                
            # Process each line
            for layer_idx, line in enumerate(lines):
                # Split the line into numbers
                numbers = line.strip().split('\t')
                if len(numbers) != numbers_per_rank:
                    print(f"警告: {filename} 的第 {layer_idx+1} 行有 {len(numbers)} 个数字，预期为 {numbers_per_rank} 个")
                    continue
                    
                # Convert to integers, divide by 128 and floor, then add to csv_data
                try:
                    values = [math.ceil(float(num) / 128) for num in numbers]
                    # Add to corresponding columns: ep_{rank_id*numbers_per_rank} to ep_{rank_id*numbers_per_rank+numbers_per_rank-1}
                    csv_data[layer_idx, rank_id*numbers_per_rank:rank_id*numbers_per_rank+numbers_per_rank] += values
                except ValueError:
                    print(f"警告: {filename} 的第 {layer_idx+1} 行数字格式无效")
                    continue
    
    # Generate CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row
        header = [''] + [f'ep_{i}' for i in range(num_ranks * numbers_per_rank)]
        writer.writerow(header)
        
        # Write data rows
        for layer_idx in range(num_layers):
            row = [f'layer_{layer_idx}'] + csv_data[layer_idx].tolist()
            writer.writerow(row)
            
    print(f"CSV 文件生成成功: {output_csv}")

if __name__ == "__main__":
    # 配置
    input_folder = "./activation_datas/longbench_1k_32die_0428/1step_425"  # 输入文件夹路径
    output_csv = "longbench_1k_32die_0428_recordstep_425.csv"  # 输出 CSV 文件名
    num_layers = 58          # 层数（每个 txt 文件的行数）
    num_ranks = 32           # rank ID 数量
    rank_id_range = (0, 31)  # rank_id 范围，包含边界，例如只处理 rank_0 到 rank_15
    
    # 生成 CSV
    generate_csv(input_folder, output_csv, num_layers, num_ranks, rank_id_range)