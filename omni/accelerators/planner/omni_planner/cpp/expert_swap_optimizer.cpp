#include "expert_swap_optimizer.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

// 构造函数：初始化优化器的参数并进行合法性检查
ExpertSwapOptimizer::ExpertSwapOptimizer(int num_layers, int world_size, int num_experts, int num_devices_per_host,
                                       int max_changes_per_rank, int load_reduction_threshold)
    : num_layers_(num_layers),
      world_size_(world_size),
      num_experts_(num_experts),
      num_devices_per_host_(num_devices_per_host),
      max_changes_per_rank_(max_changes_per_rank),
      load_reduction_threshold_(load_reduction_threshold) {
    if (num_layers <= 0 || world_size <= 0 || num_experts <= 0 || num_devices_per_host <= 0) {
        throw std::runtime_error("无效的初始化参数");
    }
    if (max_changes_per_rank_ <= 0) {
        throw std::runtime_error("max_changes_per_rank必须为正数");
    }
}

// 优化函数：为指定层生成专家交换指令以平衡设备负载
std::vector<SwapInstruction> ExpertSwapOptimizer::optimize(int layer_id, const std::vector<ExpertInfo>& experts) {
    std::vector<SwapInstruction> swaps;
    if (layer_id < 0 || layer_id >= num_layers_) {
        throw std::out_of_range("无效的layer_id: " + std::to_string(layer_id));
    }

    std::cout << "ExpertSwapOptimizer::optimize: layer_id=" << layer_id << ", experts.size=" << experts.size() << std::endl;

    // 检查是否需要优化
    if (!is_optimized_layer(layer_id, experts)) {
        std::cout << "层 " << layer_id << " 是顺序分配的，无需优化。\n";
        return swaps;
    }

    // 初始化可变专家列表
    std::vector<ExpertInfo> current_experts = experts;

    // 打印专家信息
    std::cout << "优化中的专家信息:\n";
    for (size_t i = 0; i < current_experts.size(); ++i) {
        std::cout << "Expert[" << i << "]: host_id=" << current_experts[i].host_id
                  << ", rank_id=" << current_experts[i].rank_id
                  << ", expert_id=" << current_experts[i].expert_id
                  << ", activations=" << current_experts[i].activations
                  << ", global_position=" << current_experts[i].global_position << std::endl;
    }

    // 计算初始设备负载
    auto device_loads = compute_device_loads(layer_id, current_experts);
    int64_t max_device_load = *std::max_element(device_loads.begin(), device_loads.end());
    double variance = compute_load_variance(device_loads);
    std::cout << "优化层 " << layer_id << ", 初始最大设备负载: " << max_device_load
              << ", 方差: " << variance << "\n";

    // 初始化每个设备的交换次数
    std::vector<int> changes_made_this_rank(world_size_, 0);

    // 计算设备信息（负载和方差）
    std::vector<std::pair<int, double>> device_info; // {device_id, variance}
    for (int device_id = 0; device_id < world_size_; ++device_id) {
        std::vector<int64_t> expert_loads;
        for (const auto& expert : current_experts) {
            if (expert.rank_id == device_id) {
                expert_loads.push_back(expert.activations);
            }
        }
        double mean = 0.0;
        for (int64_t load : expert_loads) mean += load;
        mean /= expert_loads.size();
        double var = 0.0;
        for (int64_t load : expert_loads) {
            var += (load - mean) * (load - mean);
        }
        var = expert_loads.empty() ? 0.0 : std::sqrt(var / expert_loads.size());
        device_info.emplace_back(device_id, var);
    }

    // 按负载和方差排序设备
    std::vector<int> sorted_devices(world_size_);
    for (int i = 0; i < world_size_; ++i) sorted_devices[i] = i;
    std::sort(sorted_devices.begin(), sorted_devices.end(),
              [&](int a, int b) {
                  if (device_loads[a] != device_loads[b]) {
                      return device_loads[a] > device_loads[b];
                  }
                  auto it_a = std::find_if(device_info.begin(), device_info.end(),
                                           [a](const auto& p) { return p.first == a; });
                  auto it_b = std::find_if(device_info.begin(), device_info.end(),
                                           [b](const auto& p) { return p.first == b; });
                  return it_a->second > it_b->second;
              });

    // 选择前一半设备作为hot_devices
    std::vector<int> hot_devices(sorted_devices.begin(), sorted_devices.begin() + world_size_ / 2);
    std::unordered_set<int> hot_set(hot_devices.begin(), hot_devices.end());
    std::vector<int> cold_devices;
    for (int i = 0; i < world_size_; ++i) {
        if (!hot_set.count(i)) cold_devices.push_back(i);
    }

    std::cout << "热设备: ";
    for (int d : hot_devices) std::cout << d << " ";
    std::cout << "\n冷设备: ";
    for (int d : cold_devices) std::cout << d << " ";
    std::cout << "\n";

    // 遍历热设备
    for (int rank_a : hot_devices) {
        if (changes_made_this_rank[rank_a] >= max_changes_per_rank_) continue;

        // 选择未达交换上限的冷设备
        std::vector<int> candidate_cold_devices;
        for (int rank_b : cold_devices) {
            if (changes_made_this_rank[rank_b] < max_changes_per_rank_) {
                candidate_cold_devices.push_back(rank_b);
            }
        }

        if (candidate_cold_devices.empty()) {
            std::cout << "设备 " << rank_a << " 无可用冷设备，跳过。\n";
            continue;
        }

        // 计算当前最大设备负载
        int64_t current_max_load = device_loads[rank_a];
        for (int device_id : candidate_cold_devices) {
            current_max_load = std::max(current_max_load, device_loads[device_id]);
        }

        // 遍历专家寻找最佳交换
        struct SwapCandidate {
            int rank_a, expert_idx_a, expert_position_a;
            int rank_b, expert_idx_b, expert_position_b;
            int64_t load_reduction;
        };
        SwapCandidate best_swap = {-1, -1, -1, -1, -1, -1, -1};
        int64_t max_load_reduction = 0;

        std::vector<int> experts_a;
        for (const auto& expert : current_experts) {
            if (expert.rank_id == rank_a) experts_a.push_back(expert.expert_id);
        }

        for (int expert_idx_a : experts_a) {
            int64_t load_a = 0;
            int expert_position_a = -1;
            for (const auto& expert : current_experts) {
                if (expert.rank_id == rank_a && expert.expert_id == expert_idx_a) {
                    load_a = expert.activations;
                    expert_position_a = expert.global_position;
                    break;
                }
            }

            for (int rank_b : candidate_cold_devices) {
                std::vector<int> experts_b;
                for (const auto& expert : current_experts) {
                    if (expert.rank_id == rank_b) experts_b.push_back(expert.expert_id);
                }

                for (int expert_idx_b : experts_b) {
                    int64_t load_b = 0;
                    int expert_position_b = -1;
                    for (const auto& expert : current_experts) {
                        if (expert.rank_id == rank_b && expert.expert_id == expert_idx_b) {
                            load_b = expert.activations;
                            expert_position_b = expert.global_position;
                            break;
                        }
                    }

                    // 计算交换后的负载
                    auto sim_loads = device_loads;
                    sim_loads[rank_a] = sim_loads[rank_a] - load_a + load_b;
                    sim_loads[rank_b] = sim_loads[rank_b] - load_b + load_a;
                    int64_t new_max_load = sim_loads[rank_a];
                    for (int device_id : candidate_cold_devices) {
                        new_max_load = std::max(new_max_load, sim_loads[device_id]);
                    }
                    int64_t load_reduction = current_max_load - new_max_load;

                    if (load_reduction >= load_reduction_threshold_ &&
                        load_reduction > max_load_reduction) {
                        max_load_reduction = load_reduction;
                        best_swap = {rank_a, expert_idx_a, expert_position_a,
                                     rank_b, expert_idx_b, expert_position_b,
                                     load_reduction};
                    }
                }
            }
        }

        // 应用最佳交换（如果找到）
        if (best_swap.rank_a != -1) {
            changes_made_this_rank[rank_a]++;
            changes_made_this_rank[best_swap.rank_b]++;

            // 更新专家信息
            size_t index_a = 0, index_b = 0;
            for (size_t i = 0; i < current_experts.size(); ++i) {
                if (current_experts[i].rank_id == best_swap.rank_a &&
                    current_experts[i].expert_id == best_swap.expert_idx_a) {
                    index_a = i;
                }
                if (current_experts[i].rank_id == best_swap.rank_b &&
                    current_experts[i].expert_id == best_swap.expert_idx_b) {
                    index_b = i;
                }
            }

            current_experts[index_a].rank_id = best_swap.rank_b;
            current_experts[index_a].global_position = best_swap.expert_position_b;
            current_experts[index_b].rank_id = best_swap.rank_a;
            current_experts[index_b].global_position = best_swap.expert_position_a;

            // 更新设备负载
            int64_t load_a = 0, load_b = 0;
            for (const auto& expert : current_experts) {
                if (expert.rank_id == best_swap.rank_b && expert.expert_id == best_swap.expert_idx_a) {
                    load_a = expert.activations;
                }
                if (expert.rank_id == best_swap.rank_a && expert.expert_id == best_swap.expert_idx_b) {
                    load_b = expert.activations;
                }
            }
            device_loads[best_swap.rank_a] = device_loads[best_swap.rank_a] - load_a + load_b;
            device_loads[best_swap.rank_b] = device_loads[best_swap.rank_b] - load_b + load_a;

            // 添加交换指令
            SwapInstruction instruction;
            instruction.rank_a = best_swap.rank_a;
            instruction.expert_idx_a = best_swap.expert_idx_a;
            instruction.expert_position_a = best_swap.expert_position_a;
            instruction.rank_b = best_swap.rank_b;
            instruction.expert_idx_b = best_swap.expert_idx_b;
            instruction.expert_position_b = best_swap.expert_position_b;
            swaps.push_back(instruction);

            std::cout << "交换: 设备 " << best_swap.rank_a << " 专家 " << best_swap.expert_idx_a
                      << " (位置 " << best_swap.expert_position_a << ") <-> 设备 " << best_swap.rank_b
                      << " 专家 " << best_swap.expert_idx_b << " (位置 " << best_swap.expert_position_b << ")"
                      << ", 负载减少: " << best_swap.load_reduction << "\n";
        }
    }

    // 打印最终负载和方差
    max_device_load = *std::max_element(device_loads.begin(), device_loads.end());
    variance = compute_load_variance(device_loads);
    std::cout << "层 " << layer_id << " 优化完成: 共生成 " << swaps.size() << " 个交换指令，"
              << "最终最大设备负载: " << max_device_load << ", 方差: " << variance << "\n";
    return swaps;
}

// 计算设备负载方差：用于评估负载分布的均匀性
double ExpertSwapOptimizer::compute_load_variance(const std::vector<int64_t>& loads) const {
    double mean = 0.0;
    for (int64_t load : loads) mean += load;
    mean /= world_size_;
    double variance = 0.0;
    for (int64_t load : loads) {
        variance += (load - mean) * (load - mean);
    }
    variance /= world_size_;
    return std::sqrt(variance);
}

// 计算设备负载：统计每个设备的专家激活次数总和
std::vector<int64_t> ExpertSwapOptimizer::compute_device_loads(int layer_id, const std::vector<ExpertInfo>& experts) {
    std::vector<int64_t> device_loads(world_size_, 0);
    for (const auto& expert : experts) {
        if (expert.rank_id >= 0 && expert.rank_id < world_size_) {
            device_loads[expert.rank_id] += expert.activations;
        }
    }
    return device_loads;
}

// 判断是否需要优化：检查层是否为顺序分配或存在重复专家
bool ExpertSwapOptimizer::is_optimized_layer(int layer_id, const std::vector<ExpertInfo>& experts) {
    std::unordered_set<int> expert_ids;
    std::vector<std::vector<int>> device_experts(world_size_);
    for (const auto& expert : experts) {
        if (!expert_ids.insert(expert.expert_id).second) {
            return true; // 存在重复专家，需要优化
        }
        if (expert.rank_id >= 0 && expert.rank_id < world_size_) {
            device_experts[expert.rank_id].push_back(expert.expert_id);
        }
    }

    int experts_per_device = num_experts_ / world_size_;
    bool is_sequential = true;

    for (int device_id = 0; device_id < world_size_; ++device_id) {
        if (device_experts[device_id].size() != experts_per_device) {
            is_sequential = false;
            break;
        }
        std::sort(device_experts[device_id].begin(), device_experts[device_id].end());
        for (int j = 0; j < experts_per_device; ++j) {
            int expected_expert_id = device_id * experts_per_device + j;
            if (device_experts[device_id][j] != expected_expert_id) {
                is_sequential = false;
                break;
            }
        }
        if (!is_sequential) {
            break;
        }
    }

    return !is_sequential;
}

// 获取每设备专家数量：假设专家分布均匀，返回第一个设备的专家数量
int ExpertSwapOptimizer::get_num_experts_per_device(int layer_id, const std::vector<ExpertInfo>& experts) const {
    std::vector<int> device_counts(world_size_, 0);
    for (const auto& expert : experts) {
        if (expert.rank_id >= 0 && expert.rank_id < world_size_) {
            device_counts[expert.rank_id]++;
        }
    }
    return device_counts[0]; // 假设分布均匀
}

// 获取交换指令的 rank_a：返回指定交换指令的第一个设备 ID
int ExpertSwapOptimizer::get_swap_rank_a(size_t swap_index, const std::vector<SwapInstruction>& swaps) const {
    if (swap_index >= swaps.size()) {
        throw std::out_of_range("无效的交换索引: " + std::to_string(swap_index));
    }
    return swaps[swap_index].rank_a;
}

// 获取交换指令的 expert_idx_a：返回指定交换指令的第一个专家索引
int ExpertSwapOptimizer::get_swap_expert_idx_a(size_t swap_index, const std::vector<SwapInstruction>& swaps) const {
    if (swap_index >= swaps.size()) {
        throw std::out_of_range("无效的交换索引: " + std::to_string(swap_index));
    }
    return swaps[swap_index].expert_idx_a;
}

// 获取交换指令的 expert_position_a：返回指定交换指令的第一个专家位置
int ExpertSwapOptimizer::get_swap_expert_position_a(size_t swap_index, const std::vector<SwapInstruction>& swaps) const {
    if (swap_index >= swaps.size()) {
        throw std::out_of_range("无效的交换索引: " + std::to_string(swap_index));
    }
    return swaps[swap_index].expert_position_a;
}

// 获取交换指令的 rank_b：返回指定交换指令的第二个设备 ID
int ExpertSwapOptimizer::get_swap_rank_b(size_t swap_index, const std::vector<SwapInstruction>& swaps) const {
    if (swap_index >= swaps.size()) {
        throw std::out_of_range("无效的交换索引: " + std::to_string(swap_index));
    }
    return swaps[swap_index].rank_b;
}

// 获取交换指令的 expert_idx_b：返回指定交换指令的第二个专家索引
int ExpertSwapOptimizer::get_swap_expert_idx_b(size_t swap_index, const std::vector<SwapInstruction>& swaps) const {
    if (swap_index >= swaps.size()) {
        throw std::out_of_range("无效的交换索引: " + std::to_string(swap_index));
    }
    return swaps[swap_index].expert_idx_b;
}

// 获取交换指令的 expert_position_b：返回指定交换指令的第二个专家位置
int ExpertSwapOptimizer::get_swap_expert_position_b(size_t swap_index, const std::vector<SwapInstruction>& swaps) const {
    if (swap_index >= swaps.size()) {
        throw std::out_of_range("无效的交换索引: " + std::to_string(swap_index));
    }
    return swaps[swap_index].expert_position_b;
}

// 获取完整交换指令：返回指定索引的完整交换指令
SwapInstruction ExpertSwapOptimizer::get_swap_instruction(size_t swap_index, const std::vector<SwapInstruction>& swaps) const {
    if (swap_index >= swaps.size()) {
        throw std::out_of_range("无效的交换索引: " + std::to_string(swap_index));
    }
    return swaps[swap_index];
}