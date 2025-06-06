#include "atomic_state.h"

extern "C" {
    void initialize_state(int value) {
        AtomicState::getInstance().initialize_state(value);
    }

    void set_state(int value) {
        AtomicState::getInstance().set_state(value);
    }

    int get_state() {
        return AtomicState::getInstance().get_state();
    }
}