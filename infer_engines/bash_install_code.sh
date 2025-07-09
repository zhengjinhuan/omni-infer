#!/bin/bash
set -e

PATCH_ROOT=../../omni/adaptors/vllm/patches/

cd ./vllm
git checkout v0.9.0
git apply --whitespace=nowarn $PATCH_ROOT/manual_apiserver_scaleout.patch