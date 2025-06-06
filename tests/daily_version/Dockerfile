FROM registry-cbu.huawei.com/atelier/build_ep_pytorch_2_1_ascend:pytorch_2.1.0-cann_daily_build-py_3.10-hce_2.0.2412-aarch64-snt9b-subpkgs-20250428110949

USER root

WORKDIR /home/ma-user

ENV GLOO_SOCKET_IFNAME=enp67s0f5 \
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    VLLM_USE_V1=1 \
    VLLM_WORKER_MULTIPROC_METHOD=fork \
    VLLM_ENABLE_MC2=0 \
    USING_LCCL_COM=0 \
    INPUT_PATH=/home/ma-user/modelarts/inputs/data_url_0 \
    OUTPUT_PATH=/home/Omni_infer_v1_Version/output \
    TEST_LEVEL="level0 or level1"

RUN pip config set global.index-url https://mirrors.tools.huawei.com/pypi/simple && \
    pip config set global.trusted-host mirrors.tools.huawei.com && \
    pip config set global.timeout 100 && \
    pip config set global.retries 10

COPY omni_infer /home/ma-user/omni_infer

RUN cd /home/ma-user/omni_infer && \
    source /usr/local/Ascend/CANN-7.7/bin/setenv.bash && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && \
    pip install --upgrade pip && \
    pip uninstall vllm -y && \
    pip uninstall vllm_ascend -y && \
    cd infer_engines && \
    bash bash_install_code.sh && \
    cd /home/ma-user/omni_infer/infer_engines/vllm && \
    VLLM_TARGET_DEVICE=empty pip install -e . -q \
    cd /home/ma-user/omni_infer/infer_engines/vllm_ascend && \
    pip install -e . -q && \
    cd /home/ma-user/omni_infer/infer_engines/vllm && \
    git apply /home/ma-user/omni_infer/tests/reduce_the_num_of_hidden_layers_of_deepseek_v3_w8a8.patch && \
    pip list | grep vllm


RUN cd /home/ma-user && \
    source /usr/local/Ascend/CANN-7.7/bin/setenv.bash && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && \
    pip install -r omni_infer/infer_engines/vllm_ascend/requirements-dev.txt && \
    pip install -r omni_infer/tests/requirements-ci.txt && \
    cd /home/ma-user && \
    pip install urllib3==1.26.7 -q

RUN echo "source /usr/local/Ascend/CANN-7.7/bin/setenv.bash && source /usr/local/Ascend/nnal/atb/set_env.sh" >> /root/.bashrc
