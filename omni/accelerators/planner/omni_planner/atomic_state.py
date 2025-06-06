from enum import IntEnum
import logging
from omni_planner import omni_placement

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 状态枚举
class State(IntEnum):
    PREPARED_DOWNGRADE = 1  # Downgrade mapping prepared
    APPLIED_DOWNGRADE = 2   # Downgrade mapping synced
    WEIGHTS_UPDATED = 3     # Expert weights updated
    READY = 4               # Optimized mapping synced (normal)

def set_state(new_state: State):
    omni_placement.set_state(new_state.value)
    logger.info(f"State changed to {State(omni_placement.get_state()).name}")

def get_state() -> State:
    return State(omni_placement.get_state())