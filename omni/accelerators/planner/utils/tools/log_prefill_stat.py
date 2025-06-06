#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import sys
import statistics

def analyze_log(log_path):
    """
    分析日志文件，寻找特定模式的行并统计信息
    
    Args:
        log_path: 日志文件路径
    
    Returns:
        分析结果列表，每个元素是一个段落的统计信息
    """
    # 检查文件是否存在
    if not os.path.exists(log_path):
        print(f"错误：文件 {log_path} 不存在")
        return []
    
    # 定义正则表达式来匹配目标行
    request_pattern = re.compile(r"INFO \d{2}-\d{2} \d{2}:\d{2}:\d{2} engine\.py:271\] Added request .+")
    time_pattern = re.compile(r"\d+ cnt (\d+), model execute time\(ms\): (.+)")
    
    results = []  # 存储所有段落的结果
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    line_index = 0
    total_lines = len(lines)
    
    while line_index < total_lines:
        # 寻找连续5行匹配 "Added request" 的模式
        consecutive_matches = 0
        start_line = line_index
        
        while line_index < total_lines and consecutive_matches < 100:
            if request_pattern.match(lines[line_index]):
                consecutive_matches += 1
            else:
                consecutive_matches = 0
                start_line = line_index + 1
            
            line_index += 1
            
        # 如果没有找到连续5行匹配，则退出循环
        if consecutive_matches < 100:
            break
            
        # 找到了连续5行匹配，现在寻找 "cnt 0" 的行并统计时间
        cnt0_times = []
        cnt_greater_0_times = []
        cnt_values = []  # 记录所有大于0的cnt值
        end_line = line_index
        end_line_cnt0 = line_index  # 专门记录cnt 0部分的结束行
        
        # 标记当前是否已经过了cnt 0的阶段
        passed_cnt0_stage = False
        
        while line_index < total_lines:
            line = lines[line_index]
            time_match = time_pattern.match(line)
            
            if time_match:
                cnt = int(time_match.group(1))
                execute_time = float(time_match.group(2))
                
                if cnt == 0 and not passed_cnt0_stage:
                    # 提取cnt 0的执行时间
                    cnt0_times.append(execute_time)
                    end_line_cnt0 = line_index
                elif cnt > 0:
                    # 已经进入cnt > 0的阶段
                    passed_cnt0_stage = True
                    if cnt == 512:
                        cnt_greater_0_times.append(execute_time)
                        cnt_values.append(cnt)
                    end_line = line_index
                    
                    # 如果遇到cnt 600，则当前段落结束
                    if cnt == 600:
                        break
            
            line_index += 1
        
        # 计算统计信息
        if cnt0_times:
            stats = {
                "start_line": start_line + 1,  # 行号从1开始
                "end_line": end_line + 1,
                "end_line_cnt0": end_line_cnt0 + 1,
                
                # cnt 0的统计
                "cnt0_count": len(cnt0_times),
                "cnt0_min_time": min(cnt0_times) if cnt0_times else 0,
                "cnt0_max_time": max(cnt0_times) if cnt0_times else 0,
                "cnt0_avg_time": statistics.mean(cnt0_times) if cnt0_times else 0,
                "cnt0_median_time": statistics.median(cnt0_times) if cnt0_times else 0,
                
                # cnt > 0的统计
                "cnt_greater_0_count": len(cnt_greater_0_times),
                "cnt_greater_0_min_time": min(cnt_greater_0_times) if cnt_greater_0_times else 0,
                "cnt_greater_0_max_time": max(cnt_greater_0_times) if cnt_greater_0_times else 0,
                "cnt_greater_0_avg_time": statistics.mean(cnt_greater_0_times) if cnt_greater_0_times else 0,
                "cnt_greater_0_median_time": statistics.median(cnt_greater_0_times) if cnt_greater_0_times else 0,
            }
            
            # 添加标准差计算
            if len(cnt0_times) > 1:
                stats["cnt0_std_dev"] = statistics.stdev(cnt0_times)
            else:
                stats["cnt0_std_dev"] = 0
                
            if len(cnt_greater_0_times) > 1:
                stats["cnt_greater_0_std_dev"] = statistics.stdev(cnt_greater_0_times)
            else:
                stats["cnt_greater_0_std_dev"] = 0
                
            # 记录遇到的非零cnt值
            stats["cnt_values"] = cnt_values
                
            results.append(stats)
    
    return results

def generate_markdown(log_path, results):
    """
    生成Markdown报告
    
    Args:
        log_path: 日志文件路径
        results: 分析结果列表
    """
    md_path = os.path.splitext(log_path)[0] + '_cnt_stat.md'
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# 日志文件 {os.path.basename(log_path)} 分析报告\n\n")
        f.write(f"共找到 {len(results)} 个满足条件的段落\n\n")
        
        for i, stat in enumerate(results, 1):
            f.write(f"## 段落 {i}\n\n")
            f.write(f"- 起始行: {stat['start_line']}\n")
            f.write(f"- 结束行: {stat['end_line']}\n")
            
            # cnt 0部分的统计信息
            f.write(f"### cnt 0 统计 (行 {stat['start_line']} - {stat['end_line_cnt0']})\n\n")
            f.write(f"- 包含 cnt 0 的行数: {stat['cnt0_count']}\n")
            f.write(f"- 最小执行时间 (ms): {stat['cnt0_min_time']:.6f}\n")
            f.write(f"- 最大执行时间 (ms): {stat['cnt0_max_time']:.6f}\n")
            f.write(f"- 平均执行时间 (ms): {stat['cnt0_avg_time']:.6f}\n")
            f.write(f"- 中位数执行时间 (ms): {stat['cnt0_median_time']:.6f}\n")
            f.write(f"- 标准差 (ms): {stat['cnt0_std_dev']:.6f}\n\n")
            
            # cnt > 0部分的统计信息
            f.write(f"### cnt > 0 统计 (行 {stat['end_line_cnt0'] + 1} - {stat['end_line']})\n\n")
            f.write(f"- 包含 cnt > 0 的行数: {stat['cnt_greater_0_count']}\n")
            
            if stat['cnt_greater_0_count'] > 0:
                f.write(f"- 最小执行时间 (ms): {stat['cnt_greater_0_min_time']:.6f}\n")
                f.write(f"- 最大执行时间 (ms): {stat['cnt_greater_0_max_time']:.6f}\n")
                f.write(f"- 平均执行时间 (ms): {stat['cnt_greater_0_avg_time']:.6f}\n")
                f.write(f"- 中位数执行时间 (ms): {stat['cnt_greater_0_median_time']:.6f}\n")
                f.write(f"- 标准差 (ms): {stat['cnt_greater_0_std_dev']:.6f}\n")
                
                # 记录遇到的非零cnt值及其数量
                cnt_counts = {}
                for val in stat['cnt_values']:
                    if val in cnt_counts:
                        cnt_counts[val] += 1
                    else:
                        cnt_counts[val] = 1
                
                f.write("\n#### 非零cnt值分布:\n\n")
                for cnt_val, count in sorted(cnt_counts.items()):
                    f.write(f"- cnt {cnt_val}: {count} 次\n")
            else:
                f.write("- 没有找到 cnt > 0 的数据\n")
            
            f.write("\n")
    
    print(f"分析报告已生成: {md_path}")

def main():
    if len(sys.argv) != 2:
        print("用法: python script.py <log文件路径>")
        sys.exit(1)
    
    log_path = sys.argv[1]
    
    if not log_path.endswith('.log'):
        print("错误: 输入文件必须是.log格式")
        sys.exit(1)
    
    results = analyze_log(log_path)
    
    if results:
        generate_markdown(log_path, results)
    else:
        print("未找到满足条件的段落")

if __name__ == "__main__":
    main()