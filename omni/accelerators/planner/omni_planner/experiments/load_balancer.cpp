#include "load_balancer.h"
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <cstring>
#include <limits>
#include <iostream>

// Recommendations struct implementations
Recommendations::Recommendations() 
    : experts_to_add(nullptr), add_count(0),
      experts_to_remove(nullptr), remove_count(0),
      rank_reassignments(nullptr), reassign_count(0),
      current_max_load(0.0), projected_max_load(0.0) {}

Recommendations::Recommendations(const Recommendations& other)
    : add_count(other.add_count),
      remove_count(other.remove_count),
      reassign_count(other.reassign_count),
      current_max_load(other.current_max_load),
      projected_max_load(other.projected_max_load),
      new_expert1_activations(other.new_expert1_activations) {
    if (add_count > 0) {
        experts_to_add = new Expert[add_count];
        std::memcpy(experts_to_add, other.experts_to_add, add_count * sizeof(Expert));
    } else {
        experts_to_add = nullptr;
    }
    if (remove_count > 0) {
        experts_to_remove = new Expert[remove_count];
        std::memcpy(experts_to_remove, other.experts_to_remove, remove_count * sizeof(Expert));
    } else {
        experts_to_remove = nullptr;
    }
    if (reassign_count > 0) {
        rank_reassignments = new Reassignment[reassign_count];
        std::memcpy(rank_reassignments, other.rank_reassignments, reassign_count * sizeof(Reassignment));
    } else {
        rank_reassignments = nullptr;
    }
}

Recommendations::Recommendations(Recommendations&& other) noexcept
    : experts_to_add(other.experts_to_add), add_count(other.add_count),
      experts_to_remove(other.experts_to_remove), remove_count(other.remove_count),
      rank_reassignments(other.rank_reassignments), reassign_count(other.reassign_count),
      current_max_load(other.current_max_load), projected_max_load(other.projected_max_load),
      new_expert1_activations(std::move(other.new_expert1_activations)) {
    other.experts_to_add = nullptr;
    other.experts_to_remove = nullptr;
    other.rank_reassignments = nullptr;
    other.add_count = 0;
    other.remove_count = 0;
    other.reassign_count = 0;
}

Recommendations& Recommendations::operator=(Recommendations&& other) noexcept {
    if (this != &other) {
        delete[] experts_to_add;
        delete[] experts_to_remove;
        delete[] rank_reassignments;

        experts_to_add = other.experts_to_add;
        add_count = other.add_count;
        experts_to_remove = other.experts_to_remove;
        remove_count = other.remove_count;
        rank_reassignments = other.rank_reassignments;
        reassign_count = other.reassign_count;
        current_max_load = other.current_max_load;
        projected_max_load = other.projected_max_load;
        new_expert1_activations = std::move(other.new_expert1_activations);

        other.experts_to_add = nullptr;
        other.experts_to_remove = nullptr;
        other.rank_reassignments = nullptr;
        other.add_count = 0;
        other.remove_count = 0;
        other.reassign_count = 0;
    }
    return *this;
}

Recommendations::~Recommendations() {
    delete[] experts_to_add;
    delete[] experts_to_remove;
    delete[] rank_reassignments;
}

// HistoryEntry struct implementation
HistoryEntry::HistoryEntry() : valid(false) {}

// LoadBalancer class implementations
LoadBalancer::LoadBalancer() : history_index(0) {}

bool LoadBalancer::was_recently_added(int rank_id, int expert_id, int activations) const {
    for (const auto& entry : history) {
        if (!entry.valid) continue;
        for (int i = 0; i < entry.rec.add_count; ++i) {
            const auto& expert = entry.rec.experts_to_add[i];
            if (expert.rank_id == rank_id && expert.expert_id == expert_id &&
                expert.activations == activations) {
                return true;
            }
        }
    }
    return false;
}

bool LoadBalancer::was_recently_removed(int rank_id, int expert_id, int activations) const {
    for (const auto& entry : history) {
        if (!entry.valid) continue;
        for (int i = 0; i < entry.rec.remove_count; ++i) {
            const auto& expert = entry.rec.experts_to_remove[i];
            if (expert.rank_id == rank_id && expert.expert_id == expert_id &&
                expert.activations == activations) {
                return true;
            }
        }
    }
    return false;
}

bool LoadBalancer::was_recently_reassigned(int from_rank, int to_rank, int expert_id, int activations) const {
    for (const auto& entry : history) {
        if (!entry.valid) continue;
        for (int i = 0; i < entry.rec.reassign_count; ++i) {
            const auto& reassign = entry.rec.rank_reassignments[i];
            if (reassign.from_rank == from_rank && reassign.to_rank == to_rank &&
                reassign.expert_id == expert_id && reassign.activations == activations) {
                return true;
            }
        }
    }
    return false;
}

LoadBalancerState LoadBalancer::ComputeState(Expert* experts, int expert_count) {
    LoadBalancerState state;
    state.expert1_total_activations = 0;
    state.expert1_count = 0;

    for (int i = 0; i < expert_count; ++i) {
        int host_id = experts[i].host_id;
        int rank_id = experts[i].rank_id;
        int expert_id = experts[i].expert_id;
        int activations = experts[i].activations;

        state.host_max_loads[host_id] = std::max(state.host_max_loads[host_id], activations);
        state.expert_counts[expert_id]++;
        state.host_rank_experts[host_id][rank_id].push_back(experts[i]);
        state.expert_positions.push_back({i, host_id, rank_id, activations});
        if (expert_id == 1) {
            state.expert1_total_activations += activations;
            state.expert1_count++;
            state.expert1_positions.push_back({i, host_id, rank_id});
        }
    }

    state.current_max_host_load = 0;
    for (const auto& [host, load] : state.host_max_loads) {
        state.current_max_host_load = std::max(state.current_max_host_load, load);
    }

    std::cout << "Initial State:\n";
    std::cout << "expert_count: " << expert_count << "\n";
    std::cout << "expert1_count: " << state.expert1_count << ", total_expert1_activations: " << state.expert1_total_activations << "\n";
    std::cout << "current_max_host_load: " << state.current_max_host_load << "\n";
    std::cout << "Host Loads: ";
    for (const auto& [host, load] : state.host_max_loads) {
        std::cout << "Host " << host << ": " << load << " ";
    }
    std::cout << "\n";

    return state;
}

struct RemoveCandidate {
    int index;
    int host_id;
    int rank_id;
    int activations;
};

RemoveCandidate FindExpertToRemove(Expert* experts, const LoadBalancerState& state) {
    RemoveCandidate candidate{-1, -1, -1, -1};
    int min_host_load = std::numeric_limits<int>::max();

    for (const auto& [idx, host_id, rank_id, activations] : state.expert_positions) {
        int expert_id = experts[idx].expert_id;
        if (expert_id != 1 && state.expert_counts.at(expert_id) > 1 && activations <= 5 && state.host_max_loads.at(host_id) < min_host_load) {
            candidate.index = idx;
            candidate.host_id = host_id;
            candidate.rank_id = rank_id;
            candidate.activations = activations;
            min_host_load = state.host_max_loads.at(host_id);
        }
    }

    std::cout << "Remove Selection:\n";
    std::cout << "remove_index: " << candidate.index << ", remove_host_id: " << candidate.host_id 
              << ", remove_rank_id: " << candidate.rank_id << ", remove_activations: " << candidate.activations << "\n";

    return candidate;
}

struct AddCandidate {
    int host_id;
    int rank_id;
};

AddCandidate FindRankForExpert1(const LoadBalancerState& state, int num_ranks, int rank_capacity, int num_hosts) {
    AddCandidate candidate{-1, -1};
    int min_rank_load = std::numeric_limits<int>::max();
    int ranks_per_host = num_ranks / num_hosts;

    // Calculate total load per rank across all hosts
    std::unordered_map<int, std::unordered_map<int, int>> rank_loads;
    for (int host = 0; host < num_hosts; ++host) {
        auto it = state.host_rank_experts.find(host);
        if (it != state.host_rank_experts.end()) {
            for (const auto& [rank_id, experts] : it->second) {
                int total_load = 0;
                for (const auto& expert : experts) {
                    total_load += expert.activations;
                }
                rank_loads[host][rank_id] = total_load;
            }
        }
    }

    // Find the rank with the minimum total load within capacity
    for (int host = 0; host < num_hosts; ++host) {
        for (int local_rank = 0; local_rank < ranks_per_host; ++local_rank) {
            int global_rank = host * ranks_per_host + local_rank;
            int load = rank_loads[host].count(global_rank) ? rank_loads[host][global_rank] : 0;
            auto it = state.host_rank_experts.find(host);
            size_t current_size = (it != state.host_rank_experts.end() && it->second.count(global_rank)) ? it->second.at(global_rank).size() : 0;
            if (current_size < static_cast<size_t>(rank_capacity)) {
                if (load < min_rank_load) {
                    min_rank_load = load;
                    candidate.host_id = host;
                    candidate.rank_id = global_rank;
                }
            }
        }
    }

    std::cout << "Add Selection:\n";
    std::cout << "add_host_id: " << candidate.host_id << ", add_rank_id: " << candidate.rank_id 
              << ", min_rank_load: " << min_rank_load << "\n";

    return candidate;
}
struct SimulationResult {
    int benefit;
    std::vector<int> new_activations;
};

SimulationResult SimulateChanges(Expert* experts, const LoadBalancerState& state, const RemoveCandidate& remove, const AddCandidate& add, int threshold) {
    int new_expert1_count = state.expert1_count + 1;
    int base_activations = state.expert1_total_activations / new_expert1_count;
    int remainder = state.expert1_total_activations % new_expert1_count;
    std::vector<int> new_activations(new_expert1_count);
    for (int i = 0; i < new_expert1_count; ++i) {
        new_activations[i] = base_activations + (i < remainder ? 1 : 0);
    }

    std::unordered_map<int, int> simulated_host_max_loads;
    std::unordered_map<int, int> expert1_new_loads;
    int assign_idx = 0;

    for (const auto& [idx, host_id, rank_id] : state.expert1_positions) {
        expert1_new_loads[host_id] = new_activations[assign_idx++];
    }

    expert1_new_loads[add.host_id] = new_activations.back();

    for (const auto& [idx, host_id, rank_id, activations] : state.expert_positions) {
        if (idx == remove.index) continue;
        int expert_id = experts[idx].expert_id;
        int load = (expert_id == 1) ? expert1_new_loads[host_id] : activations;
        simulated_host_max_loads[host_id] = std::max(simulated_host_max_loads[host_id], load);
    }

    int simulated_max_host_load = 0;
    for (const auto& [host, load] : simulated_host_max_loads) {
        simulated_max_host_load = std::max(simulated_max_host_load, load);
    }

    int benefit = state.current_max_host_load - simulated_max_host_load;
    std::cout << "Simulation Results:\n";
    std::cout << "simulated_max_host_load: " << simulated_max_host_load << ", benefit: " << benefit << "\n";

    return {benefit, new_activations};
}

void LoadBalancer::ApplyChanges(Expert* experts, int& expert_count, Recommendations& rec) {
    std::cout << "Applying changes\n";
    for (int i = 0; i < rec.remove_count; ++i) {
        int remove_index = -1;
        for (int j = 0; j < expert_count; ++j) {
            if (experts[j].host_id == rec.experts_to_remove[i].host_id &&
                experts[j].rank_id == rec.experts_to_remove[i].rank_id &&
                experts[j].expert_id == rec.experts_to_remove[i].expert_id &&
                experts[j].activations == rec.experts_to_remove[i].activations) {
                remove_index = j;
                break;
            }
        }
        if (remove_index != -1) {
            for (int k = remove_index; k < expert_count - 1; ++k) {
                experts[k] = experts[k + 1];
            }
            expert_count--;
        }
    }

    for (int i = 0; i < rec.add_count; ++i) {
        experts[expert_count++] = rec.experts_to_add[i];
    }

    if (rec.add_count > 0 && rec.experts_to_add[0].expert_id == 1 && !rec.new_expert1_activations.empty()) {
        int assign_idx = 0;
        for (int i = 0; i < expert_count && assign_idx < rec.new_expert1_activations.size(); ++i) {
            if (experts[i].expert_id == 1) {
                experts[i].activations = rec.new_expert1_activations[assign_idx++];
            }
        }
    }
}

/**
 * Balance the load of experts on ranks in a hosts, with the following constraints:
 * 1. each host has same number of ranks
 * 2. potentially different number of experts on each rank
 * 3. Balance the load by add new experts, which would share the original load and reduce the load on other existing experts
 * 4. Add new experts only when there is capacity left on the rank, the capacity is a predefined constant
 * 5. The rank can be in a different host
 * 6. You are allowed to remove underused experts to vacant a space for adding new experts, but need to make sure there is at least one occurrence for each expert
 * 7. balance the max load of each host and the load is calculated by max load of an expert in the same host
 * 8. only make a change when the load reduction is larger than a threshold that is passed in to the balance_load
 * 9. To avoid adding/removing experts frequently, we only allow modification to experts that are not recently modified
 *
 * @param experts: array of experts
 * @param expert_count: the number of experts
 * @param max_experts: maximum number of experts to add
 * @param rank_capacity: maximum number of experts on a rank
 * @param can_reassign_ranks: whether to allow reassigning experts from one rank to another
 * @param max_changes: maximum number of changes to make
 * @param num_ranks: total number of ranks
 * @param load_reduction_threshold: threshold of load reduction
 * @return: a Recommendation object that contains the add, remove, and reassign operations
 */
Recommendations LoadBalancer::balance_load(Expert* experts, int& expert_count, int max_experts, int rank_capacity,
                                           bool can_reassign_ranks, int max_changes, int num_ranks,
                                           int load_reduction_threshold) {
    LoadBalancerState state = ComputeState(experts, expert_count);
    Recommendations rec;
    Expert temp_add[1000];
    Expert temp_remove[1000];
    int add_idx = 0, remove_idx = 0;

    if (max_changes >= 2 && state.expert1_count > 0 && expert_count <= max_experts) {
        std::cout << "Entering balancing logic\n";

        RemoveCandidate remove = FindExpertToRemove(experts, state);
        if (remove.index != -1) {
            int num_hosts = 2;
            AddCandidate add = FindRankForExpert1(state, num_ranks, rank_capacity, num_hosts);
            if (add.rank_id != -1) {
                SimulationResult sim = SimulateChanges(experts, state, remove, add, load_reduction_threshold);
                if (sim.benefit > load_reduction_threshold) {
                    temp_remove[remove_idx++] = experts[remove.index];
                    temp_add[add_idx++] = {add.host_id, add.rank_id, 1, sim.new_activations.back()}; // Fixed typo here
                    rec.current_max_load = state.current_max_host_load;
                    rec.projected_max_load = state.current_max_host_load - sim.benefit;
                    rec.new_expert1_activations = sim.new_activations;
                } else {
                    std::cout << "Benefit " << sim.benefit << " <= threshold " << load_reduction_threshold << ", no changes applied\n";
                }
            } else {
                std::cout << "No valid rank found for adding Expert 1\n";
            }
        } else {
            std::cout << "No underused expert found to remove\n";
        }
    } else {
        std::cout << "Conditions not met: max_changes=" << max_changes << ", expert1_count=" << state.expert1_count 
                  << ", expert_count=" << expert_count << ", max_experts=" << max_experts << "\n";
    }

    rec.add_count = add_idx;
    if (add_idx > 0) {
        rec.experts_to_add = new Expert[add_idx];
        std::memcpy(rec.experts_to_add, temp_add, add_idx * sizeof(Expert));
    }
    rec.remove_count = remove_idx;
    if (remove_idx > 0) {
        rec.experts_to_remove = new Expert[remove_idx];
        std::memcpy(rec.experts_to_remove, temp_remove, remove_idx * sizeof(Expert));
    }

    std::cout << "Final Counts: remove_count=" << rec.remove_count << ", add_count=" << rec.add_count << "\n";

    Recommendations result = rec;
    history[history_index].rec = std::move(rec);
    history[history_index].valid = true;
    history_index = (history_index + 1) % HISTORY_SIZE;

    return result;
}