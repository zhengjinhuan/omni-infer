#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H
#include <iostream>
#include <fstream>
#include <map>
#include <assert.h>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "expert_swap_optimizer.h"
#include <mutex>
#include <condition_variable>
#include <thread>

const int QUEUE_SIZE = 20000;

// void ACLCHECK(aclError ret);
// void HCCLCHECK(HcclResult ret);
#define ACLCHECK(ret) do {\
    if(ret != ACL_SUCCESS)\
    {\
        printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret);\
    }\
} while(0)

#define HCCLCHECK(ret) do {\
    if(ret != HCCL_SUCCESS)\
    {\
        printf("hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret);\
    }\
} while(0)

typedef struct {
    std::vector<void *> address; // 多个权重的地址
    std::vector<size_t> lengths; // 多个权重的长度
    std::vector<size_t> sizes;
    std::vector<std::string> dtypes; // 多个权重的类型
    size_t layer_idx; 
    SwapInstruction instruction;
    // TODO Call Back Function
} TransDesc;

// 与线程交互的队列, 队头队尾两个计数
class ThreadQueue{
    private:
        TransDesc *mTransDescQueue[QUEUE_SIZE];
        int mCurDescFront = 0;
        int mCurDescRear = 0;
        std::mutex mMutex;
        std::condition_variable mCond;
    public:
        ThreadQueue()
        {
            for (int i = 0; i < QUEUE_SIZE; ++i) {
                mTransDescQueue[i] = new TransDesc;
            }
        }
        ThreadQueue(const ThreadQueue& other)
        {
            mCurDescFront = other.mCurDescFront;
            mCurDescRear = other.mCurDescRear;
            // 深拷贝 mTransDescQueue
            for (int i = 0; i < QUEUE_SIZE; ++i) {
                mTransDescQueue[i] = new TransDesc(*other.mTransDescQueue[i]);
            }
        }
        ~ThreadQueue()
        {
            for (int i = 0; i < QUEUE_SIZE; ++i) {
                delete mTransDescQueue[i];
            }
        }
        bool IsFull()
        {
            return (mCurDescRear + 1) % QUEUE_SIZE == mCurDescFront;
        }

        bool IsEmpty()
        {
            return mCurDescRear == mCurDescFront;
        }
        void Enqueue(TransDesc* desc){
            std::unique_lock<std::mutex> lock(mMutex);

            if (IsFull()) {
                std::cout<<"The Swap Queue is Full, Ignore the Enqueue Operation!";
                return;
            }

            TransDesc* transDesc = mTransDescQueue[mCurDescRear];
            if (desc==nullptr){
                // 用于触发线程退出
                transDesc = nullptr;

            }
            else{ 
                transDesc->address = std::move(desc->address);
                transDesc->lengths = std::move(desc->lengths);
                transDesc->dtypes = std::move(desc->dtypes);
                transDesc->layer_idx = desc->layer_idx;
                transDesc->instruction = desc->instruction;
            }
            mCurDescRear = (mCurDescRear + 1) % QUEUE_SIZE;
            mCond.notify_one();
        }
        void Dequeue()
        {
            std::unique_lock<std::mutex> lock(mMutex);
            mCurDescFront = (mCurDescFront + 1) % QUEUE_SIZE;
            return ;
        }
        TransDesc *GetFront()
        {
            std::unique_lock<std::mutex> lock(mMutex);
            mCond.wait(lock, [this]() { return !IsEmpty(); });
            return mTransDescQueue[mCurDescFront];
        }

};

enum class HcclCommInitType {
    RootInfoString,
    RankTableFile
};

class Distribution{
    private:
        HcclComm hcclComm_;
        uint32_t rank_;
        uint32_t world_size_;
        void initThread();
        std::vector<ThreadQueue> queues_; //  构建world_sizs个队列
        std::vector<std::thread> threads_; // This rank will have world_size Thread

        enum ThreadState{
            INIT,
            RUNNING,
            STOPPING,
            STOPPED
        } thread_state_ = ThreadState::INIT;

        
        void start_thread();
        void swap_for_thread(size_t t_rank,aclrtContext currentContext);
        void stop_thread();
        
    public:
        Distribution(size_t  rank, const char* rankTableFile);
        Distribution(size_t  rank, size_t world_size, const char* infoStr, HcclCommInitType type);
        ~Distribution();
        void enqueue(TransDesc* desc); // 选择放到哪个队列当中
        void swap(void * src_addr, void * recv_addr,size_t length, std::string dtype, uint32_t t_rank, bool send_first, aclrtStream stream);
        void printCommInfo();
        
};
#endif // ACL_CHECK_H