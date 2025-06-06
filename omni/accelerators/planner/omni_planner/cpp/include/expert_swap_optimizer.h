#ifndef EXPERT_SWAP_OPTIMIZER_H
#define EXPERT_SWAP_OPTIMIZER_H

#include <vector>
#include <tuple>
#include <unordered_set>
#include <cstdint>

struct ExpertInfo {
    int host_id;           // 主机标识
    int rank_id;           // 设备标识
    int expert_id;         // 专家标识
    int activations;       // 激活数
    int global_position;   // 专家的全局位置
};

struct SwapInstruction {
    int rank_a;
    int expert_idx_a;
    int expert_position_a;
    int rank_b;
    int expert_idx_b;
    int expert_position_b;
};

class ExpertSwapOptimizer {
public:
    // 构造函数，包含decay_rate和load_gap_factor参数
    ExpertSwapOptimizer(int num_layers, int world_size, int num_experts, int num_devices_per_host,
                       int max_changes_per_rank = 1, int load_reduction_threshold = 1000);

    // 优化函数，返回交换指令
    std::vector<SwapInstruction> optimize(int layer_id, const std::vector<ExpertInfo>& experts);

    // Getter 方法
    int get_num_layers() const { return num_layers_; }
    int get_world_size() const { return world_size_; }
    int get_num_experts() const { return num_experts_; }
    int get_num_devices_per_host() const { return num_devices_per_host_; }
    int get_max_changes_per_rank() const { return max_changes_per_rank_; }
    int get_load_reduction_threshold() const { return load_reduction_threshold_; }

    // Getter 方法：访问交换指令字段
    int get_swap_rank_a(size_t swap_index, const std::vector<SwapInstruction>& swaps) const;
    int get_swap_expert_idx_a(size_t swap_index, const std::vector<SwapInstruction>& swaps) const;
    int get_swap_expert_position_a(size_t swap_index, const std::vector<SwapInstruction>& swaps) const;
    int get_swap_rank_b(size_t swap_index, const std::vector<SwapInstruction>& swaps) const;
    int get_swap_expert_idx_b(size_t swap_index, const std::vector<SwapInstruction>& swaps) const;
    int get_swap_expert_position_b(size_t swap_index, const std::vector<SwapInstruction>& swaps) const;

    // Getter 方法：返回整个交换指令
    SwapInstruction get_swap_instruction(size_t swap_index, const std::vector<SwapInstruction>& swaps) const;

private:
    // 数据成员
    int num_layers_;
    int world_size_;
    int num_experts_;
    int num_devices_per_host_;
    int max_changes_per_rank_;
    int load_reduction_threshold_;


    // 私有成员函数
    std::vector<int64_t> compute_device_loads(int layer_id, const std::vector<ExpertInfo>& experts);
    double compute_load_variance(const std::vector<int64_t>& loads) const;
    bool is_optimized_layer(int layer_id, const std::vector<ExpertInfo>& experts);
    int get_num_experts_per_device(int layer_id, const std::vector<ExpertInfo>& experts) const;
};

#endif // EXPERT_SWAP_OPTIMIZER_H