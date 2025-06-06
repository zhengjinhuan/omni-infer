#include "expert_swap_optimizer.h"
#include <gtest/gtest.h>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <cmath>

class ExpertSwapOptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 默认参数
        num_layers_ = 2;
        world_size_ = 4;
        num_experts_ = 8;
        num_devices_per_host_ = 2;
        max_changes_per_rank_ = 1; // 每个设备最多交换1次
        load_reduction_threshold_ = 10;
    }

    // 运行优化测试的辅助函数
    void RunOptimizeTest(const std::string& test_name,
                         const std::vector<ExpertInfo>& experts,
                         int layer_id,
                         int expected_swap_count,
                         int64_t expected_max_load_reduction) {
        ExpertSwapOptimizer optimizer(num_layers_, world_size_, num_experts_, num_devices_per_host_,
                                     max_changes_per_rank_, load_reduction_threshold_);

        std::cout << "测试: " << test_name << "\n";

        // 验证输入数据一致性
        std::unordered_set<int> expert_ids;
        for (const auto& expert : experts) {
            if (!expert_ids.insert(expert.expert_id).second) {
                FAIL() << "专家ID重复: " << expert.expert_id << " 在 experts 中";
            }
        }
        EXPECT_EQ(expert_ids.size(), num_experts_) << "唯一专家数量与 num_experts_ 不匹配";

        // 计算初始最大设备负载
        std::vector<int64_t> initial_loads(world_size_, 0);
        for (const auto& expert : experts) {
            if (expert.rank_id >= 0 && expert.rank_id < world_size_) {
                initial_loads[expert.rank_id] += expert.activations;
            }
        }
        int64_t initial_max_load = *std::max_element(initial_loads.begin(), initial_loads.end());

        // 执行优化
        auto swaps = optimizer.optimize(layer_id, experts);

        // 验证交换指令数量
        EXPECT_EQ(swaps.size(), expected_swap_count) << "交换指令数量不符合预期: " << test_name;

        // 验证每个设备的交换次数不超过 max_changes_per_rank_
        std::vector<int> swap_counts(world_size_, 0);
        for (const auto& swap : swaps) {
            swap_counts[swap.rank_a]++;
            swap_counts[swap.rank_b]++;
        }
        for (int i = 0; i < world_size_; ++i) {
            EXPECT_LE(swap_counts[i], max_changes_per_rank_)
                << "设备 " << i << " 交换次数超过 max_changes_per_rank_ 在 " << test_name;
        }

        // 模拟优化后的负载
        std::vector<int64_t> final_loads = initial_loads;
        std::vector<ExpertInfo> current_experts = experts;
        for (const auto& swap : swaps) {
            int rank_a = swap.rank_a;
            int expert_idx_a = swap.expert_idx_a;
            int rank_b = swap.rank_b;
            int expert_idx_b = swap.expert_idx_b;

            // 查找对应专家的激活数
            int64_t load_a = 0, load_b = 0;
            size_t index_a = 0, index_b = 0;
            for (size_t i = 0; i < current_experts.size(); ++i) {
                if (current_experts[i].rank_id == rank_a && current_experts[i].expert_id == expert_idx_a) {
                    load_a = current_experts[i].activations;
                    index_a = i;
                }
                if (current_experts[i].rank_id == rank_b && current_experts[i].expert_id == expert_idx_b) {
                    load_b = current_experts[i].activations;
                    index_b = i;
                }
            }

            // 更新负载
            final_loads[rank_a] = final_loads[rank_a] - load_a + load_b;
            final_loads[rank_b] = final_loads[rank_b] - load_b + load_a;

            // 更新 current_experts
            current_experts[index_a].rank_id = rank_b;
            current_experts[index_a].global_position = swap.expert_position_b;
            current_experts[index_b].rank_id = rank_a;
            current_experts[index_b].global_position = swap.expert_position_a;
        }

        // 验证负载减少
        int64_t final_max_load = *std::max_element(final_loads.begin(), final_loads.end());
        int64_t actual_reduction = initial_max_load - final_max_load;
        EXPECT_GE(actual_reduction, expected_max_load_reduction)
            << "负载减少不足: " << test_name;

        // 验证优化后的负载与 current_experts 一致
        std::vector<int64_t> computed_loads(world_size_, 0);
        for (const auto& expert : current_experts) {
            if (expert.rank_id >= 0 && expert.rank_id < world_size_) {
                computed_loads[expert.rank_id] += expert.activations;
            }
        }
        EXPECT_EQ(computed_loads, final_loads) << "计算的负载与模拟负载不匹配: " << test_name;
    }

    // 成员变量
    int num_layers_;
    int world_size_;
    int num_experts_;
    int num_devices_per_host_;
    int max_changes_per_rank_;
    int load_reduction_threshold_;
};

// 测试构造函数：验证有效参数
TEST_F(ExpertSwapOptimizerTest, Constructor_ValidParameters) {
    ExpertSwapOptimizer optimizer(2, 4, 8, 2, 1, 10);
    EXPECT_EQ(optimizer.get_num_layers(), 2);
    EXPECT_EQ(optimizer.get_world_size(), 4);
    EXPECT_EQ(optimizer.get_num_experts(), 8);
    EXPECT_EQ(optimizer.get_num_devices_per_host(), 2);
    EXPECT_EQ(optimizer.get_max_changes_per_rank(), 1);
    EXPECT_EQ(optimizer.get_load_reduction_threshold(), 10);
}

// 测试构造函数：验证无效参数
TEST_F(ExpertSwapOptimizerTest, Constructor_InvalidParameters) {
    EXPECT_THROW(ExpertSwapOptimizer(0, 4, 8, 2, 1, 10), std::runtime_error);
    EXPECT_THROW(ExpertSwapOptimizer(2, 0, 8, 2, 1, 10), std::runtime_error);
    EXPECT_THROW(ExpertSwapOptimizer(2, 4, 0, 2, 1, 10), std::runtime_error);
    EXPECT_THROW(ExpertSwapOptimizer(2, 4, 8, 0, 1, 10), std::runtime_error);
    EXPECT_THROW(ExpertSwapOptimizer(2, 4, 8, 2, 0, 10), std::runtime_error); // max_changes_per_rank <= 0
}

// 测试优化：基本不平衡场景
TEST_F(ExpertSwapOptimizerTest, Optimize_BasicImbalanced) {
    std::vector<ExpertInfo> experts = {
        {0, 0, 0, 100, 0}, {0, 0, 1, 50, 1}, {0, 0, 2, 20, 2}, // Rank 0: 170
        {1, 1, 3, 10, 3}, {1, 1, 4, 5, 4},                     // Rank 1: 15
        {2, 2, 5, 15, 5},                                      // Rank 2: 15
        {3, 3, 6, 30, 6}, {3, 3, 7, 25, 7}                     // Rank 3: 55
    };
    RunOptimizeTest("BasicImbalanced", experts, 0, 2, 20); // 预期1次交换，减少至少20
}

// 测试优化：负载平衡场景
TEST_F(ExpertSwapOptimizerTest, Optimize_Balanced) {
    std::vector<ExpertInfo> experts = {
        {0, 0, 0, 50, 0}, {0, 0, 1, 50, 1}, // Rank 0: 100
        {1, 1, 2, 50, 2}, {1, 1, 3, 50, 3}, // Rank 1: 100
        {2, 2, 4, 50, 4}, {2, 2, 5, 50, 5}, // Rank 2: 100
        {3, 3, 6, 50, 6}, {3, 3, 7, 50, 7}  // Rank 3: 100
    };
    RunOptimizeTest("Balanced", experts, 0, 0, 0); // 已平衡，无交换
}

// 测试优化：无效层
TEST_F(ExpertSwapOptimizerTest, Optimize_InvalidLayer) {
    ExpertSwapOptimizer optimizer(num_layers_, world_size_, num_experts_, num_devices_per_host_,
                                 max_changes_per_rank_, load_reduction_threshold_);
    std::vector<ExpertInfo> experts;
    EXPECT_THROW(optimizer.optimize(-1, experts), std::out_of_range);
    EXPECT_THROW(optimizer.optimize(num_layers_, experts), std::out_of_range);
}

// 测试 Getter 方法
TEST_F(ExpertSwapOptimizerTest, GetterMethods) {
    ExpertSwapOptimizer optimizer(num_layers_, world_size_, num_experts_, num_devices_per_host_,
                                 max_changes_per_rank_, load_reduction_threshold_);
    std::vector<SwapInstruction> swaps = {
        {0, 0, 0, 1, 2, 2}, // Rank 0, Expert 0, Pos 0 <-> Rank 1, Expert 2, Pos 2
        {2, 3, 3, 3, 4, 4}  // Rank 2, Expert 3, Pos 3 <-> Rank 3, Expert 4, Pos 4
    };
    EXPECT_EQ(optimizer.get_swap_rank_a(0, swaps), 0);
    EXPECT_EQ(optimizer.get_swap_expert_idx_a(0, swaps), 0);
    EXPECT_EQ(optimizer.get_swap_expert_position_a(0, swaps), 0);
    EXPECT_EQ(optimizer.get_swap_rank_b(0, swaps), 1);
    EXPECT_EQ(optimizer.get_swap_expert_idx_b(0, swaps), 2);
    EXPECT_EQ(optimizer.get_swap_expert_position_b(0, swaps), 2);

    auto instruction = optimizer.get_swap_instruction(1, swaps);
    EXPECT_EQ(instruction.rank_a, 2);
    EXPECT_EQ(instruction.expert_idx_a, 3);
    EXPECT_EQ(instruction.expert_position_a, 3);
    EXPECT_EQ(instruction.rank_b, 3);
    EXPECT_EQ(instruction.expert_idx_b, 4);
    EXPECT_EQ(instruction.expert_position_b, 4);

    EXPECT_THROW(optimizer.get_swap_rank_a(2, swaps), std::out_of_range);
}

// 测试顺序分配场景
TEST_F(ExpertSwapOptimizerTest, SequentialAllocation) {
    std::vector<ExpertInfo> experts = {
        {0, 0, 0, 50, 0}, {0, 0, 1, 50, 1}, // Rank 0: 100
        {1, 1, 2, 50, 2}, {1, 1, 3, 50, 3}, // Rank 1: 100
        {2, 2, 4, 50, 4}, {2, 2, 5, 50, 5}, // Rank 2: 100
        {3, 3, 6, 50, 6}, {3, 3, 7, 50, 7}  // Rank 3: 100
    };
    RunOptimizeTest("SequentialAllocation", experts, 0, 0, 0); // 顺序分配，无交换
}

// 测试大规模场景
TEST_F(ExpertSwapOptimizerTest, LargeScale) {
    num_layers_ = 4;
    world_size_ = 16;
    num_experts_ = 32;
    num_devices_per_host_ = 4;
    max_changes_per_rank_ = 1;
    load_reduction_threshold_ = 50;

    std::vector<ExpertInfo> experts = {
        {0, 0, 0, 500, 0}, {0, 0, 16, 300, 16},     // Rank 0: 800
        {1, 1, 1, 100, 1}, {1, 1, 17, 50, 17},      // Rank 1: 150
        {2, 2, 2, 150, 2}, {2, 2, 18, 200, 18},     // Rank 2: 350
        {3, 3, 3, 250, 3}, {3, 3, 19, 180, 19},     // Rank 3: 430
        {4, 4, 4, 120, 4}, {4, 4, 20, 90, 20},      // Rank 4: 210
        {5, 5, 5, 110, 5}, {5, 5, 21, 130, 21},     // Rank 5: 240
        {6, 6, 6, 140, 6}, {6, 6, 22, 160, 22},     // Rank 6: 300
        {7, 7, 7, 170, 7}, {7, 7, 23, 190, 23},     // Rank 7: 360
        {8, 8, 8, 210, 8}, {8, 8, 24, 220, 24},     // Rank 8: 430
        {9, 9, 9, 230, 9}, {9, 9, 25, 240, 25},     // Rank 9: 470
        {10, 10, 10, 260, 10}, {10, 10, 26, 270, 26}, // Rank 10: 530
        {11, 11, 11, 280, 11}, {11, 11, 27, 290, 27}, // Rank 11: 570
        {12, 12, 12, 300, 12}, {12, 12, 28, 310, 28}, // Rank 12: 610
        {13, 13, 13, 320, 13}, {13, 13, 29, 330, 29}, // Rank 13: 650
        {14, 14, 14, 340, 14}, {14, 14, 30, 350, 30}, // Rank 14: 690
        {15, 15, 15, 360, 15}, {15, 15, 31, 370, 31}  // Rank 15: 730
    };
    RunOptimizeTest("LargeScale", experts, 0, 7, 50); // 预期4次交换（world_size_/2=8，但受限于负载差）
}

// 测试高冗余场景
TEST_F(ExpertSwapOptimizerTest, HighRedundancy) {
    num_layers_ = 2;
    world_size_ = 4;
    num_experts_ = 8;
    num_devices_per_host_ = 2;
    max_changes_per_rank_ = 1;
    load_reduction_threshold_ = 5;

    std::vector<ExpertInfo> experts = {
        {0, 0, 0, 150, 0}, {0, 0, 4, 100, 4}, // Rank 0: 250
        {1, 1, 1, 120, 1}, {1, 1, 5, 110, 5}, // Rank 1: 230
        {2, 2, 2, 130, 2}, {2, 2, 6, 140, 6}, // Rank 2: 270
        {3, 3, 3, 160, 3}, {3, 3, 7, 170, 7}  // Rank 3: 330
    };
    RunOptimizeTest("HighRedundancy", experts, 0, 1, 30); // 预期1次交换，减少至少30
}

// 测试单层场景
TEST_F(ExpertSwapOptimizerTest, SingleLayer) {
    num_layers_ = 1;
    world_size_ = 4;
    num_experts_ = 8;
    num_devices_per_host_ = 2;
    max_changes_per_rank_ = 1;
    load_reduction_threshold_ = 15;

    std::vector<ExpertInfo> experts = {
        {0, 0, 0, 120, 0}, {0, 0, 4, 80, 4},  // Rank 0: 200
        {1, 1, 1, 100, 1}, {1, 1, 5, 90, 5},  // Rank 1: 190
        {2, 2, 2, 110, 2}, {2, 2, 6, 130, 6}, // Rank 2: 240
        {3, 3, 3, 140, 3}, {3, 3, 7, 150, 7}  // Rank 3: 290
    };
    RunOptimizeTest("SingleLayer", experts, 0, 1, 20); // 预期1次交换，减少至少20
}

// 测试极端不平衡场景
TEST_F(ExpertSwapOptimizerTest, ExtremeImbalance) {
    num_layers_ = 2;
    world_size_ = 4;
    num_experts_ = 8;
    num_devices_per_host_ = 2;
    max_changes_per_rank_ = 1;
    load_reduction_threshold_ = 100;

    std::vector<ExpertInfo> experts = {
        {0, 0, 0, 1000, 0}, {0, 0, 4, 500, 4}, // Rank 0: 1500
        {1, 1, 1, 50, 1}, {1, 1, 5, 25, 5},    // Rank 1: 75
        {2, 2, 2, 100, 2}, {2, 2, 6, 75, 6},   // Rank 2: 175
        {3, 3, 3, 150, 3}, {3, 3, 7, 125, 7}   // Rank 3: 275
    };
    RunOptimizeTest("ExtremeImbalance", experts, 0, 1, 300); // 预期1次交换，减少至少300
}

// 测试多次交换场景
TEST_F(ExpertSwapOptimizerTest, MultipleChanges) {
    num_layers_ = 2;
    world_size_ = 4;
    num_experts_ = 8;
    num_devices_per_host_ = 2;
    max_changes_per_rank_ = 1;
    load_reduction_threshold_ = 20;

    std::vector<ExpertInfo> experts = {
        {0, 0, 0, 300, 0}, {0, 0, 4, 200, 4}, // Rank 0: 500
        {1, 1, 1, 100, 1}, {1, 1, 5, 80, 5},  // Rank 1: 180
        {2, 2, 2, 120, 2}, {2, 2, 6, 110, 6}, // Rank 2: 230
        {3, 3, 3, 130, 3}, {3, 3, 7, 140, 7}  // Rank 3: 270
    };
    RunOptimizeTest("MultipleChanges", experts, 0, 2, 80); // 预期2次交换，减少至少80
}

// 测试两次特定顺序的交换
TEST_F(ExpertSwapOptimizerTest, TwoSwapsWithSpecificOrder) {
    num_layers_ = 2;
    world_size_ = 4;
    num_experts_ = 8;
    num_devices_per_host_ = 2;
    max_changes_per_rank_ = 1;
    load_reduction_threshold_ = 40;

    std::vector<ExpertInfo> experts = {
        {0, 0, 0, 300, 0}, {0, 0, 4, 100, 4}, // Rank 0: 400
        {1, 1, 1, 250, 1}, {1, 1, 5, 140, 5}, // Rank 1: 390
        {2, 2, 2, 120, 2}, {2, 2, 6, 80, 6},  // Rank 2: 200
        {3, 3, 3, 70, 3}, {3, 3, 7, 50, 7}    // Rank 3: 120
    };

    ExpertSwapOptimizer optimizer(num_layers_, world_size_, num_experts_, num_devices_per_host_,
                                 max_changes_per_rank_, load_reduction_threshold_);
    auto swaps = optimizer.optimize(0, experts);

    ASSERT_LE(swaps.size(), 2) << "交换次数应不超过2次";
    RunOptimizeTest("TwoSwapsWithSpecificOrder", experts, 0, 2, 50); // 预期最多2次交换，减少至少50
}

// 测试多个最大负载设备
TEST_F(ExpertSwapOptimizerTest, MultipleMaxLoadDevices) {
    num_layers_ = 2;
    world_size_ = 4;
    num_experts_ = 8;
    num_devices_per_host_ = 2;
    max_changes_per_rank_ = 1;
    load_reduction_threshold_ = 50;

    std::vector<ExpertInfo> experts = {
        {0, 0, 0, 300, 0}, {0, 0, 4, 200, 4}, // Rank 0: 500
        {1, 1, 1, 300, 1}, {1, 1, 5, 200, 5}, // Rank 1: 500
        {2, 2, 2, 60, 2}, {2, 2, 6, 40, 6},   // Rank 2: 100
        {3, 3, 3, 60, 3}, {3, 3, 7, 40, 7}    // Rank 3: 100
    };
    RunOptimizeTest("MultipleMaxLoadDevices", experts, 0, 2, 80); // 预期2次交换，减少至少80
}


// 测试非顺序但负载均匀的场景
TEST_F(ExpertSwapOptimizerTest, NonSequentialButBalanced) {
    std::vector<ExpertInfo> experts = {
        {0, 0, 0, 50, 0}, {0, 0, 4, 50, 4}, // Rank 0: 100
        {1, 1, 1, 50, 1}, {1, 1, 5, 50, 5}, // Rank 1: 100
        {2, 2, 2, 50, 2}, {2, 2, 6, 50, 6}, // Rank 2: 100
        {3, 3, 3, 50, 3}, {3, 3, 7, 50, 7}  // Rank 3: 100
    };
    RunOptimizeTest("NonSequentialButBalanced", experts, 0, 0, 0); // 已平衡，无交换
}

// 测试空专家列表的边界情况
TEST_F(ExpertSwapOptimizerTest, EmptyExperts) {
    ExpertSwapOptimizer optimizer(num_layers_, world_size_, num_experts_, num_devices_per_host_,
                                 max_changes_per_rank_, load_reduction_threshold_);
    std::vector<ExpertInfo> experts;
    auto swaps = optimizer.optimize(0, experts);
    EXPECT_EQ(swaps.size(), 0) << "空专家列表应无交换";
}