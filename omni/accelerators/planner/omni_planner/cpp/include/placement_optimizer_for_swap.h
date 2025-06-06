#ifndef PLACEMENT_OPTIMIZER_FOR_SWAP_H
#define PLACEMENT_OPTIMIZER_FOR_SWAP_H

#include <vector>
#include <string>
#include <tuple>
#include <stdexcept>
#include "expert_activation.h"
#include "placement_mapping.h"
#include "expert_swap_optimizer.h"

/**
 * @brief Class for optimizing expert placement across devices using swap operations.
 *
 * Manages the swapping of experts in a distributed system to optimize their distribution
 * based on activation frequencies using ExpertSwapOptimizer.
 */
class PlacementOptimizerForSwap {
private:
    PlacementMapping* placement_mapping_;         ///< Pointer to placement mapping data
    ClusterActivation* clusterActivation_;        ///< Pointer to cluster activation data
    ExpertSwapOptimizer expert_swap_optimizer_;
    int num_layers_;                              ///< Number of layers in the model
    int rank_;                                    ///< Rank of the current process
    int world_size_;                              ///< Total number of processes (world size)
    int num_experts_;                             ///< Total number of experts in the model
    int num_devices_per_host_;                    ///< Number of devices per host

    /**
     * @brief Finds swap instructions for optimizing expert placement in a layer.
     *
     * @param[in] layer_idx_moe Layer index for Mixture of Experts (MoE).
     * @param[in] layer_freq Vector of activation counts for each position.
     * @return std::vector<SwapInstruction> Vector of swap instructions for optimization.
     */
    std::vector<SwapInstruction> find_swaps(
        int layer_idx_moe,
        const std::vector<int64_t>& layer_freq
    );

public:
    /**
     * @brief Constructor for PlacementOptimizerForSwap.
     *
     * Initializes the optimizer with placement and activation data, and configures the ExpertSwapOptimizer.
     *
     * @param[in] placement_mapping Pointer to PlacementMapping object.
     * @param[in] clusterActivation Pointer to ClusterActivation object.
     * @param[in] max_changes Maximum number of swap changes allowed.
     * @param[in] load_reduction_threshold Threshold for load reduction.
     * @param[in] decay_rate Decay rate for dynamic threshold in ExpertSwapOptimizer.
     * @param[in] load_gap_factor Load gap factor for swap decisions.
     * @throws std::runtime_error If either pointer is null.
     */
    PlacementOptimizerForSwap(
        PlacementMapping* placement_mapping,
        ClusterActivation* clusterActivation,
        int max_changes_per_rank = 1,
        int load_reduction_threshold = 50
    );

    /**
     * @brief Default destructor.
     *
     * No special cleanup is needed as the class does not own the pointers.
     */
    ~PlacementOptimizerForSwap() = default;


    /**
     * @brief Calculates the number of deployed experts per device for a given layer.
     *
     *
     * @param layer_id The layer index.
     * @return The number of deployed experts per device for the layer.
     * @throws std::out_of_range If layer_id is invalid.
     */
    int get_num_experts_per_device(int layer_id) const;

    /**
     * @brief Retrieves the frequency status for a specific layer.
     *
     * @param[in] layer Layer index to query.
     * @return std::vector<int64_t> Vector of activation counts for each position in the layer.
     * @throws std::out_of_range If layer index is invalid.
     */
    std::vector<int64_t> get_layer_freq_status(int layer);

    /**
     * @brief Optimizes expert placement for the specified layer using swap operations.
     *
     * @param[in] layer_id Layer index to optimize.
     * @return std::vector<SwapInstruction> Vector of swap instructions for optimization.
     * @throws std::out_of_range If layer_id is invalid.
     */
    std::vector<SwapInstruction> optimize(int layer_id);

    /** @brief Gets the number of layers in the model. @return int Number of layers. */
    int get_num_layers() const { return num_layers_; }
    /** @brief Gets the rank of the current process. @return int Rank. */
    int get_rank() const { return rank_; }
    /** @brief Gets the total number of processes. @return int World size. */
    int get_world_size() const { return world_size_; }
    /** @brief Gets the total number of experts in the model. @return int Number of experts. */
    int get_num_experts() const { return num_experts_; }
    /** @brief Gets the number of devices per host. @return int Number of devices per host. */
    int get_num_devices_per_host() const { return num_devices_per_host_; }
};

#endif // PLACEMENT_OPTIMIZER_FOR_SWAP_H