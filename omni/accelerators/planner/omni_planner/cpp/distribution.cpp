#include <string.h>
#include "distribution.h"

const std::map<std::string, HcclDataType> NAME2DATATYPE = {
    {"int", HCCL_DATA_TYPE_INT32  },
    {"int16" , HCCL_DATA_TYPE_INT16},
    {"int8" , HCCL_DATA_TYPE_INT8},
    {"int64", HCCL_DATA_TYPE_INT64  },
    {"float" , HCCL_DATA_TYPE_FP32 },
    {"float16" , HCCL_DATA_TYPE_FP16 },
    {"bfloat16" , HCCL_DATA_TYPE_BFP16 }
};

// void ACLCHECK(aclError ret){
//     if (ret != ACL_SUCCESS) {
//         throw std::runtime_error("acl interface returned error, retcode: " + std::to_string(ret));
//     }
// }

// void HCCLCHECK(HcclResult ret){
//     if (ret != HCCL_SUCCESS) {
//         throw std::runtime_error("hccl interface returned error, retcode: " + std::to_string(ret));
//     }
// }




Distribution::Distribution(size_t  rank, const char* rankTableFile){
    // 构建 HCCL 通信域
    std::cout << "rank TableFile is " << rankTableFile << std::endl;
    HCCLCHECK(HcclCommInitClusterInfo(rankTableFile, rank, &hcclComm_));
    HCCLCHECK(HcclGetRankId(hcclComm_, &rank_));
}

Distribution::Distribution(size_t  rank, size_t world_size, const char* infoStr, HcclCommInitType type){
    // 构建 HCCL 通信域
    if (type == HcclCommInitType::RootInfoString) {
            HcclRootInfo rootInfo;
            memcpy(rootInfo.internal, infoStr, HCCL_ROOT_INFO_BYTES);
            HCCLCHECK(HcclCommInitRootInfo(world_size, &rootInfo, rank, &hcclComm_));
    } else {
        HCCLCHECK(HcclCommInitClusterInfo(infoStr, rank, &hcclComm_));
    }
    HCCLCHECK(HcclGetRankId(hcclComm_, &rank_));
    HCCLCHECK(HcclGetRankSize(hcclComm_, &world_size_));
    assert(world_size == world_size_ && "The world size from rank tables does not correspond with input parameters");
    queues_.resize(world_size_);
    // 判断 queues 是否都是空队列
    for(size_t rank=0; rank<world_size_;++rank){
        assert(queues_[rank].IsEmpty() && "The Queues are not empty for Multi Thread, This may be caused by a wrong Initialization Function");
    }
    start_thread();
}

Distribution::~Distribution() {
    // 销毁HCCL通信域
    stop_thread();
    HCCLCHECK(HcclCommDestroy(hcclComm_));
    
}

void Distribution::enqueue(TransDesc* desc){
    assert(desc != nullptr && "Adding an empty TransDesc ptr to the Queue");
    queues_[desc->instruction.rank_b].Enqueue(desc);
}

void Distribution::start_thread() {

    aclrtContext currentContext;
    ACLCHECK(aclrtGetCurrentContext(&currentContext));
    assert(thread_state_ == ThreadState::INIT);
    thread_state_ = ThreadState::RUNNING;
    threads_.resize(world_size_);
     try {
        for (size_t t_rank = 0; t_rank < world_size_; ++t_rank) {
            threads_.push_back(std::thread(&Distribution::swap_for_thread, this, t_rank,currentContext));
        }
    } catch (...) {
        // If an exception occurs, join all successfully started threads
        thread_state_ = ThreadState::STOPPED;
        // 填一个空指针，让其监听退出线程中的死循环
        for (auto& queue : queues_){
            if (queue.IsEmpty()){
                queue.Enqueue(nullptr);
            }
        }
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        throw; // Re-throw the exception after handling
    }
}

void Distribution::swap_for_thread(size_t t_rank,aclrtContext currentContext){
    // Build Context & Stream
    // ACLCHECK(aclrtSetDevice(rank_));
    ACLCHECK(aclrtSetCurrentContext(currentContext));
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));
    ThreadQueue* queue =  &queues_[t_rank];
    bool send_first = rank_<t_rank;
    bool first_time=true;
    while(thread_state_==ThreadState::RUNNING || first_time){
        first_time = false;
        TransDesc* transDesc = queue->GetFront();
        if (transDesc == nullptr){
            assert(thread_state_ == ThreadState::STOPPED && "Got nullptr from queue but thread_state is not equal to STOP");
            break;
        }
        for (size_t idx=0; idx< transDesc->address.size();++idx){
            void *recv_buf;
            ACLCHECK(aclrtMalloc(&recv_buf, transDesc->sizes[idx], ACL_MEM_MALLOC_HUGE_FIRST));
            swap(transDesc->address[idx], recv_buf, transDesc->lengths[idx], transDesc->dtypes[idx], t_rank, send_first, stream);
            ACLCHECK(aclrtMemcpy(transDesc->address[idx], transDesc->sizes[idx], recv_buf, transDesc->sizes[idx], ACL_MEMCPY_DEVICE_TO_DEVICE));
            ACLCHECK(aclrtFree(recv_buf));
        }
        queue->Dequeue();
        // TODO: CallBack Function to Update projector map
    }
    ACLCHECK(aclrtSynchronizeStream(stream));
    ACLCHECK(aclrtDestroyStream(stream));
    // ACLCHECK(aclrtResetDevice(rank_));
}

void Distribution::stop_thread() {
    thread_state_ = ThreadState::STOPPED;
    // 填一个空指针，让其监听退出线程中的死循环
    for (auto& queue : queues_){
        if (queue.IsEmpty()){
            queue.Enqueue(nullptr);
        }
    }
    for (auto& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}


void Distribution::swap(void * src_addr, void * recv_addr,size_t length, std::string dtype, uint32_t t_rank, bool send_first, aclrtStream stream){
    if (stream==nullptr){
        assert(stream==nullptr && "stream should not be nullptr");
    }
    if(send_first){
        HCCLCHECK(HcclSend(src_addr, length, NAME2DATATYPE.at(dtype), t_rank, hcclComm_, stream));
        HCCLCHECK(HcclRecv(recv_addr, length, NAME2DATATYPE.at(dtype), t_rank, hcclComm_, stream));
    }
    else{
        HCCLCHECK(HcclRecv(recv_addr, length, NAME2DATATYPE.at(dtype), t_rank, hcclComm_, stream));
        HCCLCHECK(HcclSend(src_addr, length, NAME2DATATYPE.at(dtype), t_rank, hcclComm_, stream));
    }
    ACLCHECK(aclrtSynchronizeStream(stream));
}

void Distribution::printCommInfo(){
    // 获取当前进程的秩（Rank）
    uint32_t rank=0;
    HCCLCHECK(HcclGetRankId(hcclComm_, &rank));

    // 获取通信域的大小（Size）
    uint32_t size=0;
    HCCLCHECK(HcclGetRankSize(hcclComm_, &size));

    // 打印通信域信息
    std::cout<<"HCCL Communicator Info:\n";
    std::cout<<"  Rank: "<<rank<<std::endl;
    std::cout<<"  Size: "<<size<<std::endl;
}