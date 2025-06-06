IMAGE=$1
CONTAINER_NAME=$2
docker run --name ${CONTAINER_NAME} --rm --privileged -idu root \
--shm-size=500g \
-v /root/.ssh/:/root/.ssh/ \
-v /usr/local/sbin:/usr/local/sbin \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/Ascend/toolbox:/usr/local/Ascend/toolbox \
-v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /etc/vnpu.cfg:/etc/vnpu.cfg \
-v /sys/fs/cgroup:/sys/fs/cgroup:ro \
-v /usr/sbin/ifconfig:/usr/sbin/ifconfig \
-v /usr/bin/ping:/usr/sbin/ping \
-v /usr/bin/hostname:/usr/bin/hostname \
-v /home/CI/inputs/Omni_planner_Release_data:/home/ma-user/modelarts/inputs/data_url_0 \
-v /home/Omni_infer_v1_Version/output:/home/Omni_infer_v1_Version/output \
-v /home/CI/models:/home/CI/models \
--network=host \
--device=/dev/davinci0:/dev/davinci0 \
--device=/dev/davinci1:/dev/davinci1 \
--device=/dev/davinci2:/dev/davinci2 \
--device=/dev/davinci3:/dev/davinci3 \
--device=/dev/davinci4:/dev/davinci4 \
--device=/dev/davinci5:/dev/davinci5 \
--device=/dev/davinci6:/dev/davinci6 \
--device=/dev/davinci7:/dev/davinci7 \
--device=/dev/davinci_manager:/dev/davinci_manager \
--device=/dev/devmm_svm:/dev/devmm_svm \
--device=/dev/hisi_hdc:/dev/hisi_hdc \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
-w /home/ma-user \
${IMAGE} \
/bin/bash