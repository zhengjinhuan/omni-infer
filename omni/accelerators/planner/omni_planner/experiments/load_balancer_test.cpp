#include "load_balancer.h"
#include <gtest/gtest.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>

class LoadBalancerTest : public ::testing::Test {
protected:
    // Variation 1: Basic imbalance
    std::vector<Expert> get_initial_experts_basic() {
        return {
            {0, 0, 1, 100}, // Host 0, Rank 0, Expert 1, Load 100
            {0, 0, 2, 10},  // Host 0, Rank 0, Expert 2, Load 10
            {1, 1, 2, 5},   // Host 1, Rank 1, Expert 2, Load 5
            {1, 1, 2, 5}    // Host 1, Rank 1, Expert 2, Load 5
        };
    }

    // Variation 2: Balanced load across hosts
    std::vector<Expert> get_initial_experts_balanced() {
        return {
            {0, 0, 1, 50},  // Host 0, Rank 0, Expert 1, Load 50
            {0, 0, 2, 10},  // Host 0, Rank 0, Expert 2, Load 10
            {1, 1, 1, 50},  // Host 1, Rank 1, Expert 1, Load 50
            {1, 1, 2, 10}   // Host 1, Rank 1, Expert 2, Load 10
        };
    }

    // Variation 3: Multiple experts on one rank
    std::vector<Expert> get_initial_experts_multi_experts() {
        return {
            {0, 0, 1, 150}, // Host 0, Rank 0, Expert 1, Load 150
            {0, 0, 2, 20},  // Host 0, Rank 0, Expert 2, Load 20
            {0, 0, 2, 10},  // Host 0, Rank 0, Expert 2, Load 10
            {1, 1, 2, 5}    // Host 1, Rank 1, Expert 2, Load 5
        };
    }

    // Variation 4: Multiple ranks per host
    std::vector<Expert> get_initial_experts_multi_ranks() {
        return {
            {0, 0, 1, 120}, // Host 0, Rank 0, Expert 1, Load 120
            {0, 0, 2, 10},  // Host 0, Rank 0, Expert 2, Load 10
            {0, 1, 2, 5},   // Host 0, Rank 1, Expert 2, Load 5
            {1, 2, 2, 5},   // Host 1, Rank 2, Expert 2, Load 5
            {1, 3, 2, 5}    // Host 1, Rank 3, Expert 2, Load 5
        };
    }

    // Variation 5: Single expert type (no Expert 1)
    std::vector<Expert> get_initial_experts_single_type() {
        return {
            {0, 0, 2, 20},  // Host 0, Rank 0, Expert 2, Load 20
            {0, 0, 2, 10},  // Host 0, Rank 0, Expert 2, Load 10
            {1, 1, 2, 5},   // Host 1, Rank 1, Expert 2, Load 5
            {1, 1, 2, 5}    // Host 1, Rank 1, Expert 2, Load 5
        };
    }

    // Variation 6: Single host with imbalanced load
    std::vector<Expert> get_initial_experts_single_host_imbalanced() {
        return {
            {0, 0, 1, 150}, // Host 0, Rank 0, Expert 1, Load 150
            {0, 0, 2, 10},  // Host 0, Rank 0, Expert 2, Load 10
            {0, 1, 2, 5},   // Host 0, Rank 1, Expert 2, Load 5
            {0, 2, 2, 5}    // Host 0, Rank 2, Expert 2, Load 5
        };
    }

    // Helper function to run balance_load test
    void RunBalanceTest(const std::string& test_name, const std::vector<Expert>& initial_experts, int threshold,
                        int max_experts, int rank_capacity, int max_changes, int num_ranks,
                        int expected_remove_count, int expected_add_count, int expected_max_load) {
        LoadBalancer balancer;
        std::vector<Expert> experts = initial_experts;
        int expert_count = experts.size();

        Recommendations result = balancer.balance_load(experts.data(), expert_count, max_experts,
                                                       rank_capacity, true, max_changes, num_ranks, threshold);

        std::cout << "Test: " << test_name << "\n" << result << std::endl;

        balancer.ApplyChanges(experts.data(), expert_count, result);

        EXPECT_EQ(result.remove_count, expected_remove_count) << "Unexpected remove_count";
        EXPECT_EQ(result.add_count, expected_add_count) << "Unexpected add_count";
        if (expected_remove_count > 0 && result.remove_count > 0) {
            EXPECT_EQ(result.experts_to_remove[0].expert_id, 2) << "Should remove Expert 2";
        }
        if (expected_add_count > 0 && result.add_count > 0) {
            EXPECT_EQ(result.experts_to_add[0].expert_id, 1) << "Should add Expert 1";
        }

        std::unordered_map<int, int> final_rank_loads;
        for (int j = 0; j < expert_count; ++j) {
            final_rank_loads[experts[j].rank_id] += experts[j].activations;
        }
        int final_max_rank_load = 0;
        for (const auto& pair : final_rank_loads) {
            final_max_rank_load = std::max(final_max_rank_load, pair.second);
        }
        EXPECT_LE(final_max_rank_load, expected_max_load) << "Max rank load should be reduced sufficiently";
    }
};

// ComputeState Tests (unchanged)
TEST_F(LoadBalancerTest, ComputeState_Basic) {
    std::vector<Expert> experts = get_initial_experts_basic();
    int expert_count = experts.size();
    LoadBalancerState state = LoadBalancer::ComputeState(experts.data(), expert_count);

    EXPECT_EQ(state.expert_positions.size(), 4);
    EXPECT_EQ(state.expert1_count, 1);
    EXPECT_EQ(state.expert1_total_activations, 100);
    EXPECT_EQ(state.current_max_host_load, 100);
    EXPECT_EQ(state.host_max_loads[0], 100);
    EXPECT_EQ(state.host_max_loads[1], 5);
    EXPECT_EQ(state.expert_counts[1], 1);
    EXPECT_EQ(state.expert_counts[2], 3);
    EXPECT_EQ(state.expert1_positions.size(), 1);
    EXPECT_EQ(std::get<0>(state.expert1_positions[0]), 0);
}

TEST_F(LoadBalancerTest, ComputeState_Balanced) {
    std::vector<Expert> experts = get_initial_experts_balanced();
    int expert_count = experts.size();
    LoadBalancerState state = LoadBalancer::ComputeState(experts.data(), expert_count);

    EXPECT_EQ(state.expert_positions.size(), 4);
    EXPECT_EQ(state.expert1_count, 2);
    EXPECT_EQ(state.expert1_total_activations, 100);
    EXPECT_EQ(state.current_max_host_load, 50);
    EXPECT_EQ(state.host_max_loads[0], 50);
    EXPECT_EQ(state.host_max_loads[1], 50);
    EXPECT_EQ(state.expert_counts[1], 2);
    EXPECT_EQ(state.expert_counts[2], 2);
    EXPECT_EQ(state.expert1_positions.size(), 2);
}

TEST_F(LoadBalancerTest, ComputeState_MultiExperts) {
    std::vector<Expert> experts = get_initial_experts_multi_experts();
    int expert_count = experts.size();
    LoadBalancerState state = LoadBalancer::ComputeState(experts.data(), expert_count);

    EXPECT_EQ(state.expert_positions.size(), 4);
    EXPECT_EQ(state.expert1_count, 1);
    EXPECT_EQ(state.expert1_total_activations, 150);
    EXPECT_EQ(state.current_max_host_load, 150);
    EXPECT_EQ(state.host_max_loads[0], 150);
    EXPECT_EQ(state.host_max_loads[1], 5);
    EXPECT_EQ(state.expert_counts[1], 1);
    EXPECT_EQ(state.expert_counts[2], 3);
    EXPECT_EQ(state.expert1_positions.size(), 1);
}

TEST_F(LoadBalancerTest, ComputeState_MultiRanks) {
    std::vector<Expert> experts = get_initial_experts_multi_ranks();
    int expert_count = experts.size();
    LoadBalancerState state = LoadBalancer::ComputeState(experts.data(), expert_count);

    EXPECT_EQ(state.expert_positions.size(), 5);
    EXPECT_EQ(state.expert1_count, 1);
    EXPECT_EQ(state.expert1_total_activations, 120);
    EXPECT_EQ(state.current_max_host_load, 120);
    EXPECT_EQ(state.host_max_loads[0], 120);
    EXPECT_EQ(state.host_max_loads[1], 5);
    EXPECT_EQ(state.expert_counts[1], 1);
    EXPECT_EQ(state.expert_counts[2], 4);
    EXPECT_EQ(state.expert1_positions.size(), 1);
}

TEST_F(LoadBalancerTest, ComputeState_SingleType) {
    std::vector<Expert> experts = get_initial_experts_single_type();
    int expert_count = experts.size();
    LoadBalancerState state = LoadBalancer::ComputeState(experts.data(), expert_count);

    EXPECT_EQ(state.expert_positions.size(), 4);
    EXPECT_EQ(state.expert1_count, 0);
    EXPECT_EQ(state.expert1_total_activations, 0);
    EXPECT_EQ(state.current_max_host_load, 20);
    EXPECT_EQ(state.host_max_loads[0], 20);
    EXPECT_EQ(state.host_max_loads[1], 5);
    EXPECT_EQ(state.expert_counts[2], 4);
    EXPECT_EQ(state.expert_counts.count(1), 0);
    EXPECT_EQ(state.expert1_positions.size(), 0);
}

// New Test: Single Host Imbalanced Load
TEST_F(LoadBalancerTest, SingleHostImbalanced_Threshold50) {
    RunBalanceTest("SingleHostImbalanced_Threshold50", get_initial_experts_single_host_imbalanced(), 
                   50, 5, 3, 2, 3, 1, 1, 160 - 50); // 160 - 50 = 110
}

// Balance Load Tests (unchanged)
TEST_F(LoadBalancerTest, Basic_Threshold0) {
    RunBalanceTest("Basic_Threshold0", get_initial_experts_basic(), 0, 4, 3, 2, 2, 1, 1, 110);
}

TEST_F(LoadBalancerTest, Basic_Threshold25) {
    RunBalanceTest("Basic_Threshold25", get_initial_experts_basic(), 25, 4, 3, 2, 2, 1, 1, 110 - 25);
}

TEST_F(LoadBalancerTest, Basic_Threshold50) {
    RunBalanceTest("Basic_Threshold50", get_initial_experts_basic(), 50, 4, 3, 2, 2, 0, 0, 110);
}

TEST_F(LoadBalancerTest, Basic_Threshold75) {
    RunBalanceTest("Basic_Threshold75", get_initial_experts_basic(), 75, 4, 3, 2, 2, 0, 0, 110);
}

TEST_F(LoadBalancerTest, Balanced_Threshold0) {
    RunBalanceTest("Balanced_Threshold0", get_initial_experts_balanced(), 0, 4, 3, 2, 2, 0, 0, 60);
}

TEST_F(LoadBalancerTest, Balanced_Threshold25) {
    RunBalanceTest("Balanced_Threshold25", get_initial_experts_balanced(), 25, 4, 3, 2, 2, 0, 0, 60);
}

TEST_F(LoadBalancerTest, MultiExperts_Threshold0) {
    RunBalanceTest("MultiExperts_Threshold0", get_initial_experts_multi_experts(), 0, 5, 3, 2, 2, 1, 1, 180);
}

TEST_F(LoadBalancerTest, MultiExperts_Threshold50) {
    RunBalanceTest("MultiExperts_Threshold50", get_initial_experts_multi_experts(), 50, 5, 3, 2, 2, 1, 1, 180 - 50);
}

TEST_F(LoadBalancerTest, MultiExperts_Threshold75) {
    RunBalanceTest("MultiExperts_Threshold75", get_initial_experts_multi_experts(), 75, 5, 3, 2, 2, 0, 0, 180);
}

TEST_F(LoadBalancerTest, MultiRanks_Threshold0) {
    RunBalanceTest("MultiRanks_Threshold0", get_initial_experts_multi_ranks(), 0, 6, 3, 2, 4, 1, 1, 130);
}

TEST_F(LoadBalancerTest, MultiRanks_Threshold50) {
    RunBalanceTest("MultiRanks_Threshold50", get_initial_experts_multi_ranks(), 50, 6, 3, 2, 4, 1, 1, 130 - 50);
}

TEST_F(LoadBalancerTest, MultiRanks_Threshold75) {
    RunBalanceTest("MultiRanks_Threshold75", get_initial_experts_multi_ranks(), 75, 6, 3, 2, 4, 0, 0, 130);
}

TEST_F(LoadBalancerTest, SingleType_Threshold0) {
    RunBalanceTest("SingleType_Threshold0", get_initial_experts_single_type(), 0, 5, 3, 2, 2, 0, 0, 30);
}

TEST_F(LoadBalancerTest, SingleType_Threshold25) {
    RunBalanceTest("SingleType_Threshold25", get_initial_experts_single_type(), 25, 5, 3, 2, 2, 0, 0, 30);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}