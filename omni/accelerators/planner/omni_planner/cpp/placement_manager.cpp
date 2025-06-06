#include <future>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "expert_activation.h"
#include "moe_weights.h"
#include "atomic_state.h"
#include "placement_optimizer.h"
#include "placement_manager.h"
#include "tensor.h"
#include "config.h"
#include <limits.h>
#include <unistd.h>
#include <libgen.h>
#include <stdexcept>
#include <string>
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

OmniConfig config;

namespace py = pybind11;

const int MAX_LAYER = 58;
std::vector<std::vector<SwapInstruction>> g_replacement_candidates(MAX_LAYER);

const char* to_absolute_path(const char* relative_path) {
    if (!relative_path || !*relative_path) {
        throw std::invalid_argument("Relative path is null or empty");
    }

    // 使用 static thread_local 存储结果
    static thread_local std::string result;
    char resolved_path[PATH_MAX];

    // 尝试使用 realpath（文件需存在）
    if (realpath(relative_path, resolved_path)) {
        result = resolved_path;
        return result.c_str();
    }

    // 如果文件不存在，基于当前工作目录拼接
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) {
        throw std::runtime_error("Failed to get current working directory: " + std::string(strerror(errno)));
    }
    result = std::string(cwd) + "/" + relative_path;

    // 尝试规范化
    char* resolved = realpath(result.c_str(), resolved_path);
    if (resolved) {
        result = resolved;
    } else if (errno != ENOENT) {
        throw std::runtime_error("Failed to resolve path: " + std::string(strerror(errno)));
    }

    return result.c_str();
}

/**
 * Constructor for Placement class
 *
 * @param rank Global device ID
 * @param world_size Number of devices in the world
 * @param num_devices_per_host Number of devices per host
 * @param activations Pointer to ClusterActivation object
 * @param expert_mapping_ptr Pointer to expert mapping data
 * @param shape Shape of expert mapping data
 * @param dtype Data type of expert mapping data
 * @param placement_pattern_ptr Pointer to placement pattern data
 * @param placement_shape Shape of placement pattern data
 * @param placement_dtype Data type of placement pattern data
 *
 * Calls initialize_components immediately and starts a separate thread to check shared memory weights
 */
Placement::Placement(int rank, int world_size, int num_devices_per_host, ClusterActivation* activations,
    PlacementMapping* placement_mapping, char* root_info)
    : rank_(rank),
      world_size_(world_size),
      num_devices_per_host_(num_devices_per_host),
      activations_(activations),
      mapping_(placement_mapping) {

    // Initialize components immediately
    initialize_components(root_info);

    // Start a separate thread to check shared memory weights
    init_thread_ = std::thread(&Placement::check_shm_weights, this);
    // init_thread_.detach();
}

void Placement::initialize_components(char* root_info) {
    num_layers_ = mapping_->get_num_layers();
    num_experts_ = mapping_->get_num_experts();
    num_deploy_experts_ = mapping_->get_num_deploy_experts();
    num_deploy_experts_per_rank_ = num_deploy_experts_ / world_size_;

    dist_ptr_ = new Distribution(rank_, world_size_, root_info,HcclCommInitType::RootInfoString);
    moe_weight_ = new MoEWeights(num_experts_, rank_, world_size_);
    optimizer_ = new PlacementOptimizer(mapping_, activations_);
    rearrange_optimizer_ = new PlacementOptimizerForSwap(mapping_, activations_, 20, 50);
}

/**
 * Constructor for Placement class
 *
 * @param rank Global device ID
 * @param world_size Number of devices in the world
 * @param num_devices_per_host Number of devices per host
 * @param activations Pointer to ClusterActivation object
 * @param expert_mapping_ptr Pointer to expert mapping data
 * @param shape Shape of expert mapping data
 * @param dtype Data type of expert mapping data
 * @param placement_pattern_ptr Pointer to placement pattern data
 * @param placement_shape Shape of placement pattern data
 * @param placement_dtype Data type of placement pattern data
 *
 * Calls initialize_components immediately and starts a separate thread to check shared memory weights
 */
Placement::Placement(int rank, int world_size, int num_devices_per_host, ClusterActivation* activations,
    size_t expert_mapping_ptr, std::vector<int64_t> shape, int dtype,
    size_t placement_pattern_ptr, std::vector<int64_t> placement_shape, int placement_dtype)
    : activations_(activations),
      rank_(rank),
      world_size_(world_size),
      num_devices_per_host_(num_devices_per_host) {

    // Initialize components immediately
    initialize_components(expert_mapping_ptr, shape, dtype,
                         placement_pattern_ptr, placement_shape, placement_dtype);

    // Start a separate thread to check shared memory weights
    init_thread_ = std::thread(&Placement::check_shm_weights, this);
    // init_thread_.detach();
}

void Placement::initialize_components(size_t expert_mapping_ptr, std::vector<int64_t> shape, int dtype,
    size_t placement_pattern_ptr, std::vector<int64_t> placement_shape, int placement_dtype) {

    assert(shape.size() == 2);
    int64_t expert_shape[2];
    memcpy(expert_shape, shape.data(), sizeof(int64_t) * 2);

    assert(placement_shape.size() == 3);
    int64_t place_shape[3];
    memcpy(place_shape, placement_shape.data(), sizeof(int64_t) * 3);
    mapping_ = new PlacementMapping(rank_, num_devices_per_host_, (int32_t *)expert_mapping_ptr, expert_shape, dtype,
                                    (int32_t *)placement_pattern_ptr, place_shape, placement_dtype);

    num_layers_ = mapping_->get_num_layers();
    num_experts_ = mapping_->get_num_experts();
    num_deploy_experts_ = mapping_->get_num_deploy_experts();
    num_deploy_experts_per_rank_ = num_deploy_experts_ / world_size_;

    moe_weight_ = new MoEWeights(num_experts_, world_size_);
    optimizer_ = new PlacementOptimizer(mapping_, activations_);
    rearrange_optimizer_ = new PlacementOptimizerForSwap(mapping_, activations_, 20, 50);
}

void Placement::check_shm_weights() {
    std::cout << "check_shm_weights start success." << std::endl;
    while (!should_stop_init_) { // 使用标志控制退出
        if (moe_weight_ && moe_weight_->isShmInitialized()) {
            start_thread();
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(config.activation_quiesce)); // Check every 30s
    }
}

Placement::~Placement() {
    stop_thread();
    delete moe_weight_;
    // delete mapping_;
    delete optimizer_;
    delete rearrange_optimizer_;
    // delete activations_;
    delete dist_ptr_;
}

void Placement::identify_expert_swap_pairs(int layer_id) {
    // optimizer determine the expert to be swaped (source expert -> dst position)
    auto swapInstruction = rearrange_optimizer_->optimize(layer_id);

    std::lock_guard<std::mutex> lock(mtx_);
    g_replacement_candidates[layer_id] = swapInstruction;
}

// 等待合适的时机等待专家权重替换
void quiesce() {
    // wait 5s before move weights to new postion
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // TODO: triger by vLLM when token finish
}

/**
 *
 * This function is called by Placement::placement_manager to manage the placement of experts for a layer.
 * It takes a layer_id as input and performs the following tasks:
 * 1. Determine the expert to be replaced (source expert -> dst position)
 * 2. Change old expert postion to default position
 * 3. Wait 5s before move weights to new postion
 * 4. Move weights to new postion
 * 5. Change source expert postion to new postion
 * 6. Update mapping
 *
 * The output of this function is an array of three elements: src_expert_idx, dst_expert_position, and old_expert_idx.
 * The meaning of these elements is as follows:
 * - src_expert_idx: the idx of the source expert to be replaced
 * - dst_expert_position: the position where the source expert should be moved to
 * - old_expert_idx: the idx of the old expert that was previously at the dst position
 *
 * If the optimizer determines that no replacement is needed, the output will be (-1, -1, -1).
 */
void Placement::replace_redundant_experts(int layer_id) {
    // 1. optimizer determine the expert to be replaced (source expert -> dst position)
    auto [src_expert_idx, dst_expert_position, old_expert_idx] = optimizer_->optimize(layer_id);

    if (src_expert_idx == -1 || dst_expert_position == -1 || old_expert_idx == -1)
    {
        std::cout << "This period no need replace experts," << " src_expert_idx " << src_expert_idx << " dst_expert_position " << dst_expert_position << " old_expert_idx " << old_expert_idx << std::endl;
        return;
    }

    // 2. change old expert postion to default position
    int default_position = mapping_->get_default_mapping_position(layer_id, old_expert_idx);
    mapping_->change_pos_id(layer_id, old_expert_idx, default_position);

    quiesce();

    // 3. move weights to new postion
    if (dst_expert_position / num_deploy_experts_per_rank_ == rank_) {

        int local_pos = dst_expert_position % num_deploy_experts_per_rank_;

        moe_weight_->replacement(layer_id, src_expert_idx, local_pos);
    }


    // 4. change source expert postion to new postion
    mapping_->change_pos_id(layer_id, src_expert_idx, dst_expert_position);

    // 5. update mapping
    mapping_->update_Position_To_Expert_Mapping(layer_id, dst_expert_position, src_expert_idx);
    mapping_->update_Redundant_Expert_Mapping(layer_id, dst_expert_position, src_expert_idx);
}

void Placement::placement_manager() {
    aclInit(NULL); // 初始化 ACL
    aclrtContext context;
    aclrtCreateContext(&context, 0);
    aclrtSetCurrentContext(context);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    std::cout << "placement worker thread started\n";
    while(!should_stop_) {
        activations_->collect_wrapper(dist_ptr_, stream);
        for (int layer = 0; layer < num_layers_; ++layer) {
            if (true) {
                identify_expert_swap_pairs(layer); // rearrange only
            } else {
                replace_redundant_experts(layer);  // with redundant experts
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1 * 60)); // Run every 1 mins
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclFinalize();
    std::cout << "placement worker thread stoped\n";
}

void Placement::start_thread() {
    if (!worker_thread_.joinable()) {
        should_stop_ = false;
        worker_thread_ = std::thread(&Placement::placement_manager, this);
    }
}

void Placement::stop_thread() {
    should_stop_init_ = true; // 通知 init_thread_ 退出
    should_stop_ = true;
    std::this_thread::sleep_for(std::chrono::seconds(1));

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    if (init_thread_.joinable()) {
        init_thread_.join(); // 等待初始化线程完成
    }
}

void update_placement(const Placement& placement, int layer_id) {
    // 1. Identify candidate swap pair
    auto lock = placement.acquire_lock();
    auto swapInstructions = g_replacement_candidates[layer_id];
    lock.unlock();

    PlacementMapping* mapping = placement.get_mapping();
    MoEWeights* moe_weights = placement.get_moe_weights();
    int current_rank = placement.get_rank();
    int num_deploy_experts_per_rank = placement.get_num_deploy_experts_per_rank();
    Distribution* dist_ptr = placement.get_distribution();

    aclInit(NULL); // 初始化 ACL
    aclrtContext context;
    aclrtCreateContext(&context, 0);
    aclrtSetCurrentContext(context);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    std::cout << "Current rank " << current_rank << " on layer " << layer_id << std::endl;
        // 2. Perform expert weights swap in the current rank
    for (const auto& instruction : swapInstructions) {
        if (instruction.rank_a == current_rank) {
            // rank a swap weights
            moe_weights->replacement(dist_ptr, stream, layer_id, instruction.expert_position_a % num_deploy_experts_per_rank, instruction.rank_b);
        } else if (instruction.rank_b == current_rank) {
            // rank b swap weights
            moe_weights->replacement(dist_ptr, stream, layer_id, instruction.expert_position_b % num_deploy_experts_per_rank, instruction.rank_a);
        }

        // change source expert postion to new postion
        mapping->change_pos_id(layer_id, instruction.expert_idx_a, instruction.expert_position_b);
        mapping->change_pos_id(layer_id, instruction.expert_idx_b, instruction.expert_position_a);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclFinalize();
    // 3. Broadcast the updated expert mapping for this layer
    // TODO

    // 4. clear swap instructions for this layer
    lock.lock();
    g_replacement_candidates[layer_id] = std::vector<SwapInstruction>();
    lock.unlock();
}

void do_placement_optimizer(const Placement& placement, int layer_id) {
    auto future = std::async(std::launch::async,
        [](const Placement& p, int layer) {
            update_placement(p, layer);
        },
        std::cref(placement), layer_id);
}

pybind11::bytes GetPDRootInfo()
{
    // 获取root节点，root节点用户可指定，并非只可以设置为0节点
    char * pRootInfo = new char[HCCL_ROOT_INFO_BYTES];
    for (uint32_t i=0; i<HCCL_ROOT_INFO_BYTES; i++) {
        pRootInfo[i] = 0;
    }
    HcclGetRootInfo((HcclRootInfo*)pRootInfo);
    std::cout << "the hccl root info in c++ is " << pRootInfo << std::endl;
    pybind11::bytes results(pRootInfo, HCCL_ROOT_INFO_BYTES);
    delete []pRootInfo;
    return results;
}

PYBIND11_MODULE(omni_placement, m) {
    m.doc() = "MoE weights management with shared memory";

    // 绑定 ut_memcpy_fun 函数
    m.def("set_ut_memcpy_fun", &set_ut_memcpy_fun, "Set the UT memcpy function");
    m.def("unset_ut_memcpy_fun", &unset_ut_memcpy_fun, "Unset the UT memcpy function");

    m.def("do_placement_optimizer", &do_placement_optimizer, py::arg("placement"), py::arg("layer_id"));
    m.def("get_pd_rootinfo", &GetPDRootInfo, "get_pd_rootinfo");

    // 1. 绑定 PlacementMapping 类
    py::class_<PlacementMapping>(m, "PlacementMapping")
        .def(py::init<const std::string& , int , int ,
                        size_t , std::vector<int64_t>,
                        size_t , std::vector<int64_t>,
                        size_t , std::vector<int64_t>,
                        size_t , std::vector<int64_t>>(),
        py::arg("filename"), py::arg("rank"), py::arg("num_devices"),
        py::arg("rendundancy_mapping"), py::arg("rendundancy_mapping_shape"),
        py::arg("global_mapping"), py::arg("global_mapping_shape"),
        py::arg("count"), py::arg("count_shape"),
        py::arg("pattern"), py::arg("pattern_shape"));


    // 3. 绑定 MoEWeights 类
    py::class_<MoEWeights>(m, "MoEWeights")
        // 根据 moe_weights.h 文件修改了 构造函数的入参
        .def(py::init<size_t>(),py::arg("num_experts"))
        .def(py::init<size_t,size_t>(),py::arg("num_experts"),py::arg("world_size"))
        .def(py::init<size_t,size_t,size_t>(),py::arg("num_experts"),py::arg("rank"),py::arg("world_size"))
        .def(py::init<size_t,size_t,size_t,const char*>(),py::arg("num_experts"),py::arg("rank"),py::arg("world_size"),py::arg("rankTableFile"))
        .def("isShmInitialized",&MoEWeights::isShmInitialized)
        .def("init_weights", &MoEWeights::init_weights,
            py::arg("npu_weights"),
            py::arg("expert_ids"),
            py::arg("init_shm"),
            "Initialize with NPU weights")
        ;

    // 4. 绑定 Placement 类
    py::class_<Placement>(m, "Placement")
        .def(py::init<>())
        .def(py::init<int, int, int, ClusterActivation*, PlacementMapping*, char*>(),
                py::arg("rank"),
                py::arg("world_size"),
                py::arg("num_devices_per_host"),
                py::arg("activation"),
                py::arg("placement_mapping"),
                py::arg("root_info"))
        .def(py::init<int, int, int, ClusterActivation*, size_t, std::vector<int64_t>, int,
                        size_t, std::vector<int64_t>, int>(),
                py::arg("rank"),
                py::arg("world_size"),
                py::arg("num_devices_per_host"),
                py::arg("activation"),
                py::arg("expert_mapping_ptr"),
                py::arg("shape"),
                py::arg("dtype"),
                py::arg("placement_pattern_ptr"),
                py::arg("placement_shape"),
                py::arg("placement_dtype"))
        .def("get_moe_weights", &Placement::get_moe_weights, py::return_value_policy::reference);

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<uint64_t, size_t, size_t, const std::string&>(), // 按实际构造函数参数补全
            py::arg("data_ptr"),
            py::arg("length"),
            py::arg("element_size"),
            py::arg("name"))
        .def(py::init<uint64_t, size_t, size_t, const std::string&,const std::string&>(), // 按实际构造函数参数补全
            py::arg("data_ptr"),
            py::arg("length"),
            py::arg("element_size"),
            py::arg("dtype"),
            py::arg("name"));

    py::class_<ClusterActivation>(m, "ClusterActivation")
        .def(py::init<Tensor,size_t, size_t, int, size_t, size_t>(), // 按实际构造函数参数补全
        py::arg("npu_count"),
        py::arg("layer"),
        py::arg("num_expert"),
        py::arg("window_size"),
        py::arg("world_size"),
        py::arg("rank"),
            "Initialize with expert activation")
        .def("collect_activation",
            static_cast<void (ClusterActivation::*)(size_t, size_t, int64_t)>(
                &ClusterActivation::collect_activation),
            py::arg("layer_idx"), py::arg("deploy_expert_idx"), py::arg("count"),
            "Record an expert activation")
        .def("getClusterTotalActivationCount",&ClusterActivation::getClusterTotalActivationCount,
            py::arg("layer"), py::arg("expert"),""
        )
        .def("stop_thread",&ClusterActivation::stop_thread,"")
        .def("stopDump",&ClusterActivation::stopDump,"")
        .def("setDumpDir", &ClusterActivation::setDumpDir,
             py::arg("dump_dir"),
             "Set the dump path for the cluster activation")
        ;

        m.def("initialize_state", &initialize_state, "Initialize the atomic state with default value");
        m.def("set_state", &set_state, "Set the atomic state to a value (enum class State)");
        m.def("get_state", &get_state, "Get the current atomic state");
}