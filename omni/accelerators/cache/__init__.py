from vllm.v1.core import kv_cache_utils
from vllm.v1.engine import core as engine_core
from vllm.v1.worker import block_table
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker import gpu_input_batch
from vllm.v1.core import kv_cache_manager as orig_manager
from vllm.v1.core.sched import scheduler


ENABLED = False


from .kv_cache_interface import (
    get_kv_cache_config_omni_type,
    get_omni_hybrid_kv_cache_spec,
    OmniMultiGroupBlockTable,
    OmniAttentionSpec,
    PATTERN,
)
from .kv_cache_manager import (
    OmniKVCacheBlocks,
    OmniKVCacheManager,
)
from .pd import OmniBiGroupDataDistManager
from .sched import _connector_finished, _update_waiting_for_remote_kv
from .utils import compute_omni_attn_metadata


def apply_omni_patch(enable=False, is_kv_consumer=True):
    if not enable:
        return

    global ENABLED
    ENABLED = True

    if is_kv_consumer:
        # use Omni-related classes and methods only for KV consumers
        kv_cache_utils.get_kv_cache_config = get_kv_cache_config_omni_type
        engine_core.get_kv_cache_config = get_kv_cache_config_omni_type
        GPUModelRunner.get_kv_cache_spec = get_omni_hybrid_kv_cache_spec
        block_table.MultiGroupBlockTable = OmniMultiGroupBlockTable
        gpu_input_batch.MultiGroupBlockTable = OmniMultiGroupBlockTable

        orig_manager.KVCacheBlocks = OmniKVCacheBlocks
        orig_manager.KVCacheManager = OmniKVCacheManager
        scheduler.KVCacheBlocks = OmniKVCacheBlocks
        scheduler.KVCacheManager = OmniKVCacheManager

        scheduler.Scheduler._connector_finished = _connector_finished
        scheduler.Scheduler._update_waiting_for_remote_kv = _update_waiting_for_remote_kv


__all__ = [
    "apply_omni_patch",
]
