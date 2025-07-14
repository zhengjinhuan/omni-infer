from typing import Optional, Any
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request


def _connector_finished(
        self: Scheduler, request: Request) -> tuple[bool, Optional[dict[str, Any]]]:
    """Return two variables. The first is a bool indicating whether the request blocks
    hould be freed now or will be sent asynchronously and freed later.
    For PrefillConnector, the second is a dict containing kv transfer info.
    For DecodeConnector, the passed block_ids is not used and the second is None.
    """
    if self.connector is None:
        return False, None
    block_ids = self.kv_cache_manager.get_block_ids(request.request_id)[0]
    return self.connector.request_finished(request, block_ids)


def _update_waiting_for_remote_kv(self: Scheduler, request: Request) -> bool:
    """
    P/D: check if the request_id is finished_recving.

    The finished_recving_kv_req_ids list is populated
    on the previous steps()'s update_from_output based
    on the worker side connector.

    We do not consider caching the blocks yet.

    The request state will be moved back to WAITING from
    WAITING_FOR_REMOTE_KV.
    """
    if request.request_id not in self.finished_recving_kv_req_ids:
        return False
    if self.vllm_config.speculative_config is not None:
        request.spec_token_ids.append(16426)  # keep the same as mtp patch
    request.num_computed_tokens = request.num_tokens - 1
    # Return that we are ready.
    self.finished_recving_kv_req_ids.remove(request.request_id)
    return True
