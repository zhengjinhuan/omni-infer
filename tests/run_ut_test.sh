START_DIR=$PWD

SOURCE_ROOT="$(dirname $(dirname "$(realpath "$0")"))"

# test omni-placemnt
cd $SOURCE_ROOT/omni/accelerators/placement
if [ -f "/usr/local/lib/python3.11/site-packages/omni_planner/omni_placement.cpython-311-aarch64-linux-gnu.so" ]; then
    cp "/usr/local/lib/python3.11/site-packages/omni_planner/omni_placement.cpython-311-aarch64-linux-gnu.so" "omni_planner/"
else
    echo "omni_placement.cpython-311-aarch64-linux-gnu.so not exist."
    exit 1
fi
pytest tests/test_moe_weights.py::TestGetExpertIds