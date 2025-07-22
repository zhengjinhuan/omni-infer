# TFAS：基于队列信息感知的动态最优组batch策略的优化调度算法

vLLM原生调度器采用以内存最大化为中心的策略，优化利用NPU内存资源，通过最大化批处理量（batch size）来提升系统吞吐量。这种策略可以实现内存的高效利用，但是原生调度策略主要关注吞吐量（throughput），未充分考虑首token时延（TTFT），即从请求到达系统到生成第一个token的延迟。


## 特性说明

![image-20250715195427224](C:\omniinfer\tools\img.png)

**系统建模**：我们通过实验发现在不影响吞吐的情况下，存在一个 batch size 甜点区，超过/不足都会导致平均首 token 实验的增加。我们对推理系统进行了数学建模，结合队列信息、系统硬件配置以及模型参数等关键因素，推导出最优 batch size 的计算方法，并集成到调度策略中。在 4P1D, TX 爬坡测试场景下，该策略带来了平均首 token 时延降低 14% 的收益，显著提升了用户体验和交互响应能力。



## 快速开始

**首次使用:**   请先以1K定长数据作为输入，在prefill实例起服务时，设置环境变量PREFILL_SCHEDULE_POLICY='tfas_profiler' , 拉起服务生成 prefill_server log 日志。然后，调用 omni/adaptors/vllm/patches/profiler_patches/tfas_profiler.py，生成策略超参：

- `TFAS_REAL_TOKEN_BUDGET`
- `TFAS_INTERCEPT`
- `TFAS_SLOP`

**后续使用**：无需profiling。开启tfas策略，需在prefill实例服务启动时，设置环境变量`PREFILL_SCHEDULE_POLICY='tfas' `，并指定环境变量`TFAS_REAL_TOKEN_BUDGET`，`TFAS_INTERCEPT`，`TFAS_SLOP`为profillng的输出值。建议直接在`/omni/adaptors/vllm/envs.py` 中直接配置策略超参默认值，避免反复设置环。

**注意事项**：由于 `tfas` 在每次组 batch 时会根据队列中的请求信息（请求数量和输入长度）动态计算最优的 `batch_size`，为避免因序列数限制导致越界，我们对 `vllm` 的 `openai/api_server.py` 文件进行了如下补丁处理

当设置`PREFILL_SCHEDULE_POLICY='tfas'` 时，`openai/api_sever.py` 会自动将vllm服务的`max-num-seqs`设置为较大的值 (例如128)，以确保调度过程顺利执行。



当前默认参数是基于 deepseek v3/r1 w8a8 量化权重和华为昇腾芯片 910C 获取的，如果你要测试的配置和我们的默认配置相同，可直接基于默认参数进行测试，否则需要重新进行 profiling 获取策略所需要的超参信息。

