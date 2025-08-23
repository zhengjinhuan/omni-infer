#!/bin/bash
set -e

PATCH_ROOT="../../omni/adaptors/sglang/patches/"

cd ./sglang
git reset --hard
git clean -fd
git checkout v0.5.0rc0
git apply --whitespace=nowarn $PATCH_ROOT/npu_support.patch
