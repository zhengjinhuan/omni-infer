# Omni-Infer：基于昇腾的超大规模MoE模型推理加速技术

中文 | [View English](./README_en.md)

*社区新闻* 🔥
- [2025/07] 社区的PD分离、EP负载均衡、算子与图三个SIG计划在7月的第二周召开**首次SIG例会**，有兴趣与会的开发者请见三个SIG分别的会议安排：[PD分离SIG会议安排](https://gitee.com/omniai/community/blob/master/sigs/sig-pd-seperation.md),[EP负载均衡SIG会议安排](https://gitee.com/omniai/community/blob/master/sigs/sig-ep-placement.md),[算子与图SIG会议安排](https://gitee.com/omniai/community/blob/master/sigs/sig-ops-and-graph.md).
- [2025/07] 2025年7月5日，在苏州举办的OpenInfra基金会Meetup将有Omni-infer议题分享，详情请见[社区活动页面](https://gitee.com/omniai/community/blob/master/events/README.md)

Omni-Infer是一套专为昇腾硬件平台定制的强大推理加速工具集，完全兼容业界目前主流的开源大模型推理框架（比如vLLM等），旨在提供高性能、企业级推理能力，具备原生支持且功能集持续扩展。

## 核心特性

- **企业级低延迟P/D调度**：支持xPyD调度及大规模分布式PD部署的横向扩展，确保最低延迟。详情请参考[全局代理设计](omni/accelerators/sched/global_proxy/README.md)。
- **请求级负载均衡**：针对所有序列长度优化预填充（prefill）和解码（decode）阶段，实现最大吞吐量与低延迟。
- **优化的MoE专家部署**：支持EP144/EP288配置的大规模混合专家（Mixture of Experts, MoE）模型。
- **MoE专家负载均衡**：具备分层非均匀冗余和近实时动态专家放置功能，提升资源利用效率。详情请参考[OmniPlacement设计](omni/accelerators/placement/README.md)。
- **高级注意力机制优化**：专为LLM、MLLM和MoE模型定制，增强性能与可扩展性。

## High-Level 架构图

![image](docs/figures/omni_infer_high_level_arch.png)

## 快速开始

如需将Omni_Infer集成到项目中，请参考[安装指南](docs/omni_infer_installation_guide.md)和[文档](docs/)获取详细的设置说明和API参考。

## 贡献指南

我们欢迎您为Omni_Infer贡献代码！请查看[贡献指南](./CONTRIBUTION.md)，并通过[Gitee Issues](https://gitee.com/omniai/omniinfer/issues/new?issue%5Bassignee_id%5D=0&issue%5Bmilestone_id%5D=0)提交拉取请求或问题。

## 许可证

Omni_Infer基于[MIT许可证](LICENSE)发布。