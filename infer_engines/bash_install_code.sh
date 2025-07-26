#!/bin/bash
set -e

PATCH_ROOT=../../omni/adaptors/vllm/patches/

cd ./vllm

# Reset local changes to make sure the patches can apply as expected
git reset --hard
git checkout 65334ef3

git apply $PATCH_ROOT/null_value_handling.patch
git apply --whitespace=nowarn $PATCH_ROOT/api_server_keepalive_timeout.patch
git apply --whitespace=nowarn $PATCH_ROOT/manual_apiserver_scaleout.patch
git apply --whitespace=nowarn $PATCH_ROOT/scheduler_kv_cache_manager_partial_kv_transfer.patch
git apply --whitespace=nowarn $PATCH_ROOT/tokenizer_proc_pool.patch
git apply --whitespace=nowarn $PATCH_ROOT/async_schedule_update_output.patch
git apply --whitespace=nowarn $PATCH_ROOT/mtp.patch
git apply --whitespace=nowarn $PATCH_ROOT/num_token_for_proxy.patch
git apply --whitespace=nowarn $PATCH_ROOT/chunked_prefill_disable.patch
git apply --whitespace=nowarn $PATCH_ROOT/multi_step.patch
git apply --whitespace=nowarn $PATCH_ROOT/multiple_mtp.patch
git apply --whitespace=nowarn $PATCH_ROOT/freeze_numpy.patch
git apply --whitespace=nowarn $PATCH_ROOT/npu_ray_adaptor.patch
