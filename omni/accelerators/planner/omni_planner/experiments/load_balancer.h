#ifndef LOAD_BALANCER_H
#define LOAD_BALANCER_H

#include <iostream>
#include <vector>
#include <unordered_map>

struct Expert {
    int host_id;       // Host identifier
    int rank_id;       // Rank identifier
    int expert_id;     // Expert identifier
    int activations;   // Number of activations
};

struct Reassignment {
    int from_rank;
    int to_rank;
    int expert_id;
    int activations;
};

struct Recommendations {
    Recommendations();
    Recommendations(const Recommendations& other);
    Recommendations(Recommendations&& other) noexcept;
    Recommendations& operator=(Recommendations&& other) noexcept;
    ~Recommendations();

    Expert* experts_to_add;
    int add_count;
    Expert* experts_to_remove;
    int remove_count;
    Reassignment* rank_reassignments;
    int reassign_count;
    double current_max_load;
    double projected_max_load;
    std::vector<int> new_expert1_activations;

    friend std::ostream& operator<<(std::ostream& os, const Recommendations& rec);
};

inline std::ostream& operator<<(std::ostream& os, const Recommendations& rec) {
    os << "Recommendations:\n";
    os << "remove_count: " << rec.remove_count << "\n";
    os << "add_count: " << rec.add_count << "\n";
    if (rec.remove_count > 0) {
        os << "Experts to Remove:\n";
        for (int i = 0; i < rec.remove_count; ++i) {
            os << "  Expert " << i << ": host_id=" << rec.experts_to_remove[i].host_id
               << ", rank_id=" << rec.experts_to_remove[i].rank_id
               << ", expert_id=" << rec.experts_to_remove[i].expert_id
               << ", activations=" << rec.experts_to_remove[i].activations << "\n";
        }
    } else {
        os << "No experts to remove\n";
    }
    if (rec.add_count > 0) {
        os << "Experts to Add:\n";
        for (int i = 0; i < rec.add_count; ++i) {
            os << "  Expert " << i << ": host_id=" << rec.experts_to_add[i].host_id
               << ", rank_id=" << rec.experts_to_add[i].rank_id
               << ", expert_id=" << rec.experts_to_add[i].expert_id
               << ", activations=" << rec.experts_to_add[i].activations << "\n";
        }
    } else {
        os << "No experts to add\n";
    }
    os << "current_max_load: " << rec.current_max_load << ", projected_max_load: " << rec.projected_max_load << "\n";
    if (!rec.new_expert1_activations.empty()) {
        os << "new_expert1_activations: ";
        for (int act : rec.new_expert1_activations) {
            os << act << " ";
        }
        os << "\n";
    }
    return os;
}

struct HistoryEntry {
    HistoryEntry();
    Recommendations rec;
    bool valid;
};

// Define LoadBalancerState outside the class for visibility
struct LoadBalancerState {
    std::unordered_map<int, int> host_max_loads;
    std::unordered_map<int, int> expert_counts;
    std::unordered_map<int, std::unordered_map<int, std::vector<Expert>>> host_rank_experts;
    std::vector<std::tuple<int, int, int, int>> expert_positions; // {index, host_id, rank_id, activations}
    int expert1_total_activations;
    int expert1_count;
    std::vector<std::tuple<int, int, int>> expert1_positions; // {index, host_id, rank_id}
    int current_max_host_load;
};

class LoadBalancer {
public:
    LoadBalancer();
    bool was_recently_added(int rank_id, int expert_id, int activations) const;
    bool was_recently_removed(int rank_id, int expert_id, int activations) const;
    bool was_recently_reassigned(int from_rank, int to_rank, int expert_id, int activations) const;

    Recommendations balance_load(Expert* experts, int& expert_count, int max_experts, int rank_capacity,
                                 bool can_reassign_ranks, int max_changes, int num_ranks,
                                 int load_reduction_threshold);
    void ApplyChanges(Expert* experts, int& expert_count, Recommendations& rec);
    static LoadBalancerState ComputeState(Expert* experts, int expert_count); // Made public and static

private:
    static const int HISTORY_SIZE = 10;
    HistoryEntry history[HISTORY_SIZE];
    int history_index;
};

#endif // LOAD_BALANCER_H