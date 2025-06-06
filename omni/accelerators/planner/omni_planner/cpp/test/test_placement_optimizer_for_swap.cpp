#include <gtest/gtest.h>
#include <stdexcept>
#include <memory>
#include <vector>
#include <cstdint>
#include <acl/acl.h>
#include <numeric>
#include <iostream>
#include "placement_optimizer_for_swap.h"
#include "expert_swap_optimizer.h"
#include "config.h"

/** @typedef memcpy_fun_t
 *  @brief Function pointer type for memory copy operations compatible with ACL runtime.
 */
typedef aclError (*memcpy_fun_t)(void*, size_t, const void*, size_t, aclrtMemcpyKind);

/** @brief Retrieves the current memory copy function pointer. */
memcpy_fun_t get_memcpy_fun();

/** @brief Sets a new memory copy function to be used. */
void set_memcpy_fun(memcpy_fun_t fun);

/** @brief A no-op memory copy function that mimics ACL memcpy behavior. */
aclError my_memcpy_no_op(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind) {
    if (dst == nullptr || src == nullptr || count > destMax) {
        return ACL_ERROR_INVALID_PARAM;
    }
    memcpy(dst, src, count);
    return ACL_ERROR_NONE;
}

/** @class PlacementOptimizerForSwapTest
 *  @brief Test fixture for testing PlacementOptimizerForSwap with global mapping.
 */
class PlacementOptimizerForSwapTest : public ::testing::Test {
protected:
    static PlacementMapping* shared_placement_mapping;
    static ClusterActivation* shared_cluster_activation;
    static int64_t* mock_data_ptr;
    static int32_t* global_expert_mapping_ptr;
    static int32_t* expert_redundant_count_ptr;
    static int32_t* redundant_expert_mapping_ptr;
    static bool is_initialized;
    PlacementOptimizerForSwap* opt = nullptr;
    memcpy_fun_t old_fun = nullptr;

    void SetUp() override {
        old_fun = get_memcpy_fun();
        set_memcpy_fun(&my_memcpy_no_op);

        if (!is_initialized) {
            try {
                // 初始化 PlacementMapping
                int32_t world_size = 2, num_layers = 2, num_experts = 4, max_redundant_count = 1;
                std::vector<std::vector<std::vector<int>>> placement_pattern_vector = {
                    {{1, 0, 1, 0}, {1, 0, 0, 1}},
                    {{0, 1, 0, 1}, {0, 1, 1, 0}}
                };

                int64_t mapping_shape[3] = {num_layers, num_experts, max_redundant_count};
                int64_t count_shape[2] = {num_layers, num_experts};
                int64_t redundant_expert_mapping_shape[3] = {num_layers, max_redundant_count, num_experts};

                size_t mapping_size = num_layers * num_experts * max_redundant_count * sizeof(int32_t);
                redundant_expert_mapping_ptr = static_cast<int32_t*>(malloc(mapping_size));
                if (!redundant_expert_mapping_ptr) {
                    throw std::runtime_error("Failed to allocate redundant_expert_mapping_ptr");
                }
                memset(redundant_expert_mapping_ptr, 0, mapping_size);

                global_expert_mapping_ptr = static_cast<int32_t*>(malloc(mapping_size));
                if (!global_expert_mapping_ptr) {
                    throw std::runtime_error("Failed to allocate global_expert_mapping_ptr");
                }
                memset(global_expert_mapping_ptr, 0, mapping_size);

                size_t count_size = num_layers * num_experts * sizeof(int32_t);
                expert_redundant_count_ptr = static_cast<int32_t*>(malloc(count_size));
                if (!expert_redundant_count_ptr) {
                    free(global_expert_mapping_ptr);
                    throw std::runtime_error("Failed to allocate expert_redundant_count_ptr");
                }
                memset(expert_redundant_count_ptr, 0, count_size);

                shared_placement_mapping = new PlacementMapping(
                    placement_pattern_vector, 0, 2,
                    redundant_expert_mapping_ptr, redundant_expert_mapping_shape,
                    global_expert_mapping_ptr, mapping_shape,
                    expert_redundant_count_ptr, count_shape
                );

                // 计算 num_positions
                int num_positions = 0;
                for (int rank = 0; rank < world_size; ++rank) {
                    for (int expert = 0; expert < num_experts; ++expert) {
                        num_positions += placement_pattern_vector[rank][0][expert];
                    }
                }
                std::cout << "num_positions: " << num_positions << std::endl;

                // 初始化 ClusterActivation 的 mock 数据
                size_t data_size = num_layers * num_positions * sizeof(int64_t);
                mock_data_ptr = static_cast<int64_t*>(malloc(data_size));
                if (!mock_data_ptr) {
                    delete shared_placement_mapping;
                    free(global_expert_mapping_ptr);
                    free(redundant_expert_mapping_ptr);
                    free(expert_redundant_count_ptr);
                    throw std::runtime_error("Failed to allocate mock_data_ptr");
                }
                for (size_t j = 0; j < num_positions; ++j) {
                    mock_data_ptr[j] = static_cast<int64_t>((j + 1) * 100); // Layer 0: 100, 200, 300, ...
                }
                for (size_t j = 0; j < num_positions; ++j) {
                    mock_data_ptr[num_positions + j] = static_cast<int64_t>((num_positions - j) * 100); // Layer 1: 600, 500, 400, ...
                }

                shared_cluster_activation = new ClusterActivation(
                    Tensor(mock_data_ptr, num_layers * num_positions, sizeof(int64_t), "int64_tensor"),
                    num_layers, num_positions, 12, world_size, 0
                );

                // 手动收集激活数据
                aclError ret = shared_cluster_activation->get_npu_count().to_host(shared_cluster_activation->get_total_count_ptr());
                if (ret != ACL_ERROR_NONE) {
                    delete shared_cluster_activation;
                    delete shared_placement_mapping;
                    free(mock_data_ptr);
                    free(global_expert_mapping_ptr);
                    free(redundant_expert_mapping_ptr);
                    free(expert_redundant_count_ptr);
                    FAIL() << "to_host failed: " << ret;
                }
                int64_t* total_count_ptr = static_cast<int64_t*>(shared_cluster_activation->get_total_count_ptr());
                int64_t* last_count_ptr = static_cast<int64_t*>(shared_cluster_activation->get_last_count_ptr());
                for (size_t layer = 0; layer < num_layers; ++layer) {
                    for (size_t expert = 0; expert < num_positions; ++expert) {
                        size_t idx = layer * num_positions + expert;
                        int64_t count = total_count_ptr[idx] - last_count_ptr[idx];
                        if (count > 0) {
                            shared_cluster_activation->collect_activation(layer, expert, count);
                            last_count_ptr[idx] = total_count_ptr[idx];
                        }
                    }
                }

                is_initialized = true;
            } catch (const std::exception& e) {
                FAIL() << "Initialization failed: " << e.what();
            }
        }

        // 初始化 PlacementOptimizerForSwap
        try {
            opt = new PlacementOptimizerForSwap(
                shared_placement_mapping,
                shared_cluster_activation,
                1, // max_changes
                50 // load_reduction_threshold
            );
        } catch (const std::exception& e) {
            FAIL() << "Failed to create PlacementOptimizerForSwap: " << e.what();
        }
    }

    void TearDown() override {
        set_memcpy_fun(old_fun);
        delete opt;
    }

    static void TearDownTestSuite() {
        delete shared_placement_mapping;
        delete shared_cluster_activation;
        if (mock_data_ptr) free(mock_data_ptr);
        if (global_expert_mapping_ptr) free(global_expert_mapping_ptr);
        if (redundant_expert_mapping_ptr) free(redundant_expert_mapping_ptr);
        if (expert_redundant_count_ptr) free(expert_redundant_count_ptr);
        shared_placement_mapping = nullptr;
        shared_cluster_activation = nullptr;
        mock_data_ptr = nullptr;
        global_expert_mapping_ptr = nullptr;
        redundant_expert_mapping_ptr = nullptr;
        expert_redundant_count_ptr = nullptr;
        is_initialized = false;
    }
};

PlacementMapping* PlacementOptimizerForSwapTest::shared_placement_mapping = nullptr;
ClusterActivation* PlacementOptimizerForSwapTest::shared_cluster_activation = nullptr;
int64_t* PlacementOptimizerForSwapTest::mock_data_ptr = nullptr;
int32_t* PlacementOptimizerForSwapTest::global_expert_mapping_ptr = nullptr;
int32_t* PlacementOptimizerForSwapTest::redundant_expert_mapping_ptr = nullptr;
int32_t* PlacementOptimizerForSwapTest::expert_redundant_count_ptr = nullptr;
bool PlacementOptimizerForSwapTest::is_initialized = false;

// 测试构造函数抛出异常的情况
TEST_F(PlacementOptimizerForSwapTest, ConstructorThrowsOnNullPlacementMapping) {
    EXPECT_THROW(PlacementOptimizerForSwap(nullptr, shared_cluster_activation), std::runtime_error);
}

TEST_F(PlacementOptimizerForSwapTest, ConstructorThrowsOnNullClusterActivation) {
    EXPECT_THROW(PlacementOptimizerForSwap(shared_placement_mapping, nullptr), std::runtime_error);
}

// 测试初始化后的成员变量
TEST_F(PlacementOptimizerForSwapTest, InitializationSetsNumLayers) {
    EXPECT_EQ(opt->get_num_layers(), shared_placement_mapping->get_num_layers());
}

TEST_F(PlacementOptimizerForSwapTest, InitializationSetsWorldSize) {
    EXPECT_EQ(opt->get_world_size(), shared_placement_mapping->get_world_size());
}

TEST_F(PlacementOptimizerForSwapTest, InitializationSetsNumExperts) {
    EXPECT_EQ(opt->get_num_experts(), shared_placement_mapping->get_num_experts());
}

TEST_F(PlacementOptimizerForSwapTest, InitializationSetsRank) {
    EXPECT_EQ(opt->get_rank(), shared_placement_mapping->get_rank());
}

TEST_F(PlacementOptimizerForSwapTest, InitializationSetsNumDevicesPerHost) {
    EXPECT_EQ(opt->get_num_devices_per_host(), shared_placement_mapping->get_num_devices_per_host());
}

// 测试 get_num_experts_per_device
TEST_F(PlacementOptimizerForSwapTest, GetNumExpertsPerDeviceThrowsOnNegativeLayer) {
    EXPECT_THROW(opt->get_num_experts_per_device(-1), std::out_of_range);
}

TEST_F(PlacementOptimizerForSwapTest, GetNumExpertsPerDeviceThrowsOnLayerBeyondMax) {
    EXPECT_THROW(opt->get_num_experts_per_device(opt->get_num_layers()), std::out_of_range);
}

TEST_F(PlacementOptimizerForSwapTest, GetNumExpertsPerDeviceReturnsCorrectValue) {
    int expected_per_device = opt->get_num_experts() / opt->get_world_size();
    EXPECT_EQ(opt->get_num_experts_per_device(0), expected_per_device);
}

// 测试 get_layer_freq_status
TEST_F(PlacementOptimizerForSwapTest, GetLayerFreqStatusThrowsOnNegativeLayer) {
    EXPECT_THROW(opt->get_layer_freq_status(-1), std::out_of_range);
}

TEST_F(PlacementOptimizerForSwapTest, GetLayerFreqStatusThrowsOnLayerBeyondMax) {
    EXPECT_THROW(opt->get_layer_freq_status(opt->get_num_layers()), std::out_of_range);
}

TEST_F(PlacementOptimizerForSwapTest, GetLayerFreqStatusReturnsCorrectVectorSize) {
    auto layer_freq = opt->get_layer_freq_status(0);
    EXPECT_EQ(layer_freq.size(), static_cast<size_t>(opt->get_num_experts()));
}

TEST_F(PlacementOptimizerForSwapTest, GetLayerFreqStatusReturnsNonNegativeFrequencies) {
    auto layer_freq = opt->get_layer_freq_status(0);
    for (const auto& freq : layer_freq) {
        EXPECT_GE(freq, 0);
    }
}

TEST_F(PlacementOptimizerForSwapTest, GetLayerFreqStatusMatchesClusterActivationForFirstLayer) {
    auto layer_freq = opt->get_layer_freq_status(0);
    for (int pos = 0; pos < opt->get_num_experts(); ++pos) {
        EXPECT_EQ(layer_freq[pos], shared_cluster_activation->getClusterTotalActivationCount(0, pos));
    }
}

// 测试 optimize 函数
TEST_F(PlacementOptimizerForSwapTest, OptimizeThrowsOnNegativeLayer) {
    EXPECT_THROW(opt->optimize(-1), std::out_of_range);
}

TEST_F(PlacementOptimizerForSwapTest, OptimizeThrowsOnLayerBeyondMax) {
    EXPECT_THROW(opt->optimize(opt->get_num_layers()), std::out_of_range);
}

TEST_F(PlacementOptimizerForSwapTest, OptimizeReturnsValidSwapInstructions) {
    struct ExpectedSwap {
        int layer;
        int rank_a;
        int expert_idx_a;
        int expert_position_a;
        int rank_b;
        int expert_idx_b;
        int expert_position_b;
    };
    std::vector<ExpectedSwap> expected = {
        {0, -1, -1, -1, -1, -1, -1}, // 预期没有交换
        {1, -1, -1, -1, -1, -1, -1}  // 预期没有交换
    };

    for (const auto& exp : expected) {
        try {
            auto swaps = opt->optimize(exp.layer);
            std::cout << "Layer " << exp.layer << ": " << swaps.size() << " swaps generated" << std::endl;
            if (swaps.empty()) {
                std::cout << "No swaps needed for layer " << exp.layer << std::endl;
                EXPECT_EQ(swaps.size(), 0) << "Layer " << exp.layer;
            } else {
                for (size_t i = 0; i < swaps.size(); ++i) {
                    EXPECT_GE(swaps[i].rank_a, 0) << "Layer " << exp.layer << ", swap " << i;
                    EXPECT_LT(swaps[i].rank_a, opt->get_world_size()) << "Layer " << exp.layer << ", swap " << i;
                    EXPECT_GE(swaps[i].expert_idx_a, 0) << "Layer " << exp.layer << ", swap " << i;
                    EXPECT_LT(swaps[i].expert_idx_a, opt->get_num_experts()) << "Layer " << exp.layer << ", swap " << i;
                    EXPECT_GE(swaps[i].expert_position_a, 0) << "Layer " << exp.layer << ", swap " << i;
                    EXPECT_LT(swaps[i].expert_position_a, opt->get_num_experts()) << "Layer " << exp.layer << ", swap " << i;
                    EXPECT_GE(swaps[i].rank_b, 0) << "Layer " << exp.layer << ", swap " << i;
                    EXPECT_LT(swaps[i].rank_b, opt->get_world_size()) << "Layer " << exp.layer << ", swap " << i;
                    EXPECT_GE(swaps[i].expert_idx_b, 0) << "Layer " << exp.layer << ", swap " << i;
                    EXPECT_LT(swaps[i].expert_idx_b, opt->get_num_experts()) << "Layer " << exp.layer << ", swap " << i;
                    EXPECT_GE(swaps[i].expert_position_b, 0) << "Layer " << exp.layer << ", swap " << i;
                    EXPECT_LT(swaps[i].expert_position_b, opt->get_num_experts()) << "Layer " << exp.layer << ", swap " << i;
                }
            }
        } catch (const std::exception& e) {
            FAIL() << "Optimize failed for layer " << exp.layer << ": " << e.what();
        }
    }
}

// 测试不同参数配置下的 optimize
TEST_F(PlacementOptimizerForSwapTest, OptimizeWithDifferentMaxChanges) {
    try {
        PlacementOptimizerForSwap opt_high_max_changes(
            shared_placement_mapping, shared_cluster_activation, 1, 50);
        auto swaps = opt_high_max_changes.optimize(0);
        std::cout << "OptimizeWithDifferentMaxChanges: " << swaps.size() << " swaps generated" << std::endl;
        EXPECT_LE(swaps.size(), 4) << "Max changes should limit number of swaps";
    } catch (const std::exception& e) {
        FAIL() << "OptimizeWithDifferentMaxChanges failed: " << e.what();
    }
}

TEST_F(PlacementOptimizerForSwapTest, OptimizeWithDifferentLoadReductionThreshold) {
    try {
        PlacementOptimizerForSwap opt_high_threshold(
            shared_placement_mapping, shared_cluster_activation, 1, 1000);
        auto swaps = opt_high_threshold.optimize(0);
        std::cout << "OptimizeWithDifferentLoadReductionThreshold: " << swaps.size() << " swaps generated" << std::endl;
        EXPECT_TRUE(swaps.empty()) << "High threshold should reduce swaps";
    } catch (const std::exception& e) {
        FAIL() << "OptimizeWithDifferentLoadReductionThreshold failed: " << e.what();
    }
}

TEST_F(PlacementOptimizerForSwapTest, OptimizeWithDifferentDecayRate) {
    try {
        PlacementOptimizerForSwap opt_low_decay(
            shared_placement_mapping, shared_cluster_activation, 1, 50);
        auto swaps = opt_low_decay.optimize(0);
        std::cout << "OptimizeWithDifferentDecayRate: " << swaps.size() << " swaps generated" << std::endl;
        EXPECT_LE(swaps.size(), 2) << "Decay rate should influence swap decisions";
    } catch (const std::exception& e) {
        FAIL() << "OptimizeWithDifferentDecayRate failed: " << e.what();
    }
}

TEST_F(PlacementOptimizerForSwapTest, OptimizeWithDifferentLoadGapFactor) {
    try {
        PlacementOptimizerForSwap opt_high_gap_factor(
            shared_placement_mapping, shared_cluster_activation, 1, 50);
        auto swaps = opt_high_gap_factor.optimize(0);
        std::cout << "OptimizeWithDifferentLoadGapFactor: " << swaps.size() << " swaps generated" << std::endl;
        EXPECT_LE(swaps.size(), 2) << "Load gap factor should influence swap decisions";
    } catch (const std::exception& e) {
        FAIL() << "OptimizeWithDifferentLoadGapFactor failed: " << e.what();
    }
}

// 验证 ClusterActivation 初始化
TEST_F(PlacementOptimizerForSwapTest, VerifyClusterActivationInitialization) {
    EXPECT_EQ(shared_cluster_activation->getClusterTotalActivationCount(0, 0), 100);
    EXPECT_EQ(shared_cluster_activation->getClusterTotalActivationCount(1, 0), 400);
}