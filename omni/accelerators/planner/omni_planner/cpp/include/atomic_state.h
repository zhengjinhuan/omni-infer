#ifndef ATOMIC_STATE_H
#define ATOMIC_STATE_H

#include <atomic>

enum class State {
    PREPARED_DOWNGRADE = 1,  // Downgrade mapping prepared
    APPLIED_DOWNGRADE = 2,   // Downgrade mapping synced
    WEIGHTS_UPDATED = 3,     // Expert weights updated
    READY = 4                // Optimized mapping synced (normal)
};

// Manages a thread-safe atomic state using std::atomic.
class AtomicState {
    public:
        // Returns the singleton instance (thread-safe in C++11).
        static AtomicState& getInstance() noexcept {
            static AtomicState instance;
            return instance;
        }

        // Initializes the state with a given value.
        void initialize_state(int value) noexcept {
            state_.store(value, std::memory_order_seq_cst);
        }

        // Sets the state atomically, validates value (1-4).
        void set_state(int value) noexcept {
            if ((State)value >= State::PREPARED_DOWNGRADE && (State)value <= State::READY) {
                state_.store(value, std::memory_order_seq_cst);
            }
        }

    // Gets the state atomically.
    int get_state() const noexcept {
        return state_.load(std::memory_order_seq_cst);
    }

    private:
        AtomicState() : state_((int)State::READY) {} // Default state is READY (4)
        AtomicState(const AtomicState&) = delete;
        AtomicState& operator=(const AtomicState&) = delete;

        std::atomic<int> state_;
};

// C API for Python binding
extern "C" {
    void initialize_state(int value);
    void set_state(int value);
    int get_state();
}

#endif