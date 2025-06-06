#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <unordered_set>
#include <numeric>
#include "expert_activation.h"
#include "placement_optimizer_for_swap.h"
#include "expert_swap_optimizer.h"

/**
 * @brief Constructor for PlacementOptimizerForSwap.
 *
 * Initializes member variables using pointers to PlacementMapping and ClusterActivation objects,
 * and configures the ExpertSwapOptimizer with provided parameters.
 *
 * @param placement_mapping Pointer to a PlacementMapping object containing placement data.
 * @param clusterActivation Pointer to a ClusterActivation object containing activation data.
 * @param max_changes Maximum number of swap changes allowed.
 * @param load_reduction_threshold Threshold for load reduction.
 * @param decay_rate Decay rate for dynamic threshold.
 * @param load_gap_factor Load gap factor for swap decisions.
 * @throws std::runtime_error If placement_mapping or clusterActivation is null.
 */
PlacementOptimizerForSwap::PlacementOptimizerForSwap(
    PlacementMapping* placement_mapping,
    ClusterActivation* clusterActivation,
    int max_changes_per_rank,
    int load_reduction_threshold
)
    : placement_mapping_(placement_mapping),
      clusterActivation_(clusterActivation),
      expert_swap_optimizer_(
          placement_mapping ? placement_mapping->get_num_layers() : 0,
          placement_mapping ? placement_mapping->get_world_size() : 0,
          placement_mapping ? placement_mapping->get_num_experts() : 0,
          placement_mapping ? placement_mapping->get_num_devices_per_host() : 0,
          max_changes_per_rank,
          load_reduction_threshold
      ),
      num_layers_(placement_mapping ? placement_mapping->get_num_layers() : 0),
      rank_(placement_mapping ? placement_mapping->get_rank() : 0),
      world_size_(placement_mapping ? placement_mapping->get_world_size() : 0),
      num_experts_(placement_mapping ? placement_mapping->get_num_experts() : 0),
      num_devices_per_host_(placement_mapping ? placement_mapping->get_num_devices_per_host() : 0) {
    if (!placement_mapping_ || !clusterActivation_) {
        throw std::runtime_error("Invalid initialization parameters");
    }
}

/**
 * @brief Calculates the number of deployed experts per device for a given layer.
 *
 * Computes ceil(get_num_deployed_experts(layer_id) / world_size_).
 *
 * @param layer_id The layer index.
 * @return The number of deployed experts per device for the layer.
 * @throws std::out_of_range If layer_id is invalid.
 */
int PlacementOptimizerForSwap::get_num_experts_per_device(int layer_id) const {
    if (layer_id < 0 || layer_id >= num_layers_) {
        throw std::out_of_range("Invalid layer_id: " + std::to_string(layer_id));
    }

    return num_experts_ / world_size_; // 假设均匀分布
}

/**
 * @brief Retrieves the frequency status of a specific layer across all positions.
 *
 * Queries the activation counts for each position in the specified layer.
 *
 * @param layer The layer index to query.
 * @return std::vector<int64_t> A vector of activation counts for each position in the layer.
 * @throws std::out_of_range If the layer index is invalid.
 */
std::vector<int64_t> PlacementOptimizerForSwap::get_layer_freq_status(int layer) {
    if (layer < 0 || layer >= num_layers_) {
        throw std::out_of_range("Invalid layer: " + std::to_string(layer));
    }

    std::vector<int64_t> layer_freq(num_experts_, 0);

    for (int posid = 0; posid < num_experts_; ++posid) {
        layer_freq[posid] = clusterActivation_->getClusterTotalActivationCount(layer, posid);
    }
    return layer_freq;
}

std::vector<SwapInstruction> PlacementOptimizerForSwap::find_swaps(int layer_idx_moe,
                                                                  const std::vector<int64_t>& layer_freq) {
    int expert_deployed_count = layer_freq.size();
    std::vector<ExpertInfo> experts(expert_deployed_count); // 固定大小初始化

    std::vector<bool> filled(expert_deployed_count, false); // 跟踪已填充的位置
    for (int expert_id = 0; expert_id < num_experts_; ++expert_id) {
        int32_t pos = placement_mapping_->get_rearrange_expert_position_id(layer_idx_moe, expert_id);
        if (pos < 0 || pos >= expert_deployed_count) {
            std::cerr << "Warning: Invalid pos " << pos << " for expert_id " << expert_id
                        << " at layer " << layer_idx_moe << ", index " << 0 << std::endl;
            continue;
        }
        int device_id = pos / get_num_experts_per_device(layer_idx_moe);
        // std::cout << "get_num_experts_per_device(layer_idx_moe)=" << get_num_experts_per_device(layer_idx_moe) << std::endl;
        // std::cout << "num_devices_per_host_=" << num_devices_per_host_ << std::endl;
        int current_host = device_id / num_devices_per_host_;

        experts[pos].host_id = current_host;
        // std::cout << "host_id=" << current_host << std::endl;
        experts[pos].rank_id = device_id;
        // std::cout << "rank_id=" << device_id << std::endl;
        experts[pos].expert_id = expert_id;
        // std::cout << "expert_id=" << expert_id << std::endl;
        experts[pos].activations = static_cast<int>(layer_freq[pos]);
        // std::cout << "activations=" << static_cast<int>(layer_freq[pos]) << std::endl;
        experts[pos].global_position = pos;
        // std::cout << "position=" << pos << std::endl;
        filled[pos] = true;
        // std::cout << "Filled expert: expert_id=" << expert_id << ", pos=" << pos << ", activations=" << experts[pos].activations << std::endl;
    }

    // 检查未填充的位置
    for (int pos = 0; pos < expert_deployed_count; ++pos) {
        if (!filled[pos]) {
            std::cerr << "Warning: Position " << pos << " not filled for layer=" << layer_idx_moe << std::endl;
            // 可选：填充默认值或跳过
            experts[pos].host_id = 0;
            experts[pos].rank_id = 0;
            experts[pos].expert_id = -1; // 标记为无效
            experts[pos].activations = 0;
            experts[pos].global_position = pos;
        }
    }

    // std::cout << "find_swaps: experts.size=" << experts.size() << std::endl;

    // std::cout << "Experts before optimize:" << std::endl;
    // for (size_t i = 0; i < experts.size(); ++i) {
    //     std::cout << "Expert[" << i << "]: host_id=" << experts[i].host_id
    //             << ", rank_id=" << experts[i].rank_id
    //             << ", expert_id=" << experts[i].expert_id
    //             << ", activations=" << experts[i].activations
    //             << ", global_position=" << experts[i].global_position << std::endl;
    // }

    return expert_swap_optimizer_.optimize(layer_idx_moe, experts);
}

/**
 * @brief Optimizes expert placement for the specified layer using swap operations.
 *
 * @param[in] layer_id The layer index to optimize.
 * @return std::vector<SwapInstruction> Vector of swap instructions for optimization.
 * @throws std::out_of_range If layer_id is invalid.
 */
std::vector<SwapInstruction> PlacementOptimizerForSwap::optimize(int layer_id) {
    if (layer_id < 0 || layer_id >= num_layers_) {
        throw std::out_of_range("Invalid layer_id: " + std::to_string(layer_id));
    }

    return find_swaps(layer_id, get_layer_freq_status(layer_id));
}