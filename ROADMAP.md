| Omni-Infer版本  | v0.4.0 | v0.5.0 | v0.6.0 |
|----------|----------|----------|----------|
| Performance & Reliability Optimization   | Chunk prefill支持PD分离【5】,APC亲和调度【4】,W4A8C16量化【7】,精度量化工具【7】  | 支持SGL推理框架【2】,Chunk Prefill混部入图【5】,MTP多头多token投机【6】,Eagle适配【6】,xPyD P侧动态扩缩容【2】,W4A8C8 【7】,支持MoonCake Store【1】  | xPyD D侧动态扩缩容【2】,Omni-attention支持Prefill加速【5】,故障快恢【3】,AFD架构设计   |
| Expanded Model Support   | DS V3/R1,Kimi K2,Qwen3-235B-A22B  | Qwen3-32B（BF16格式）,QwQ-32B （BF16格式）   | GPT-OSS 120B,GPT-OSS 20B,DS V3.1,Qwen-VL  |
| Usability Improvements   | Metrics信息与vLLM接口一致【9】  | PD分离自动部署（裸机、K8S）【2】,环境仿真工具Mock【9】 | 精度分析工具【7】,性能仿真模拟工具【9】  |

### Note
[1] SIG Cache Optimization
[2] SIG PD Disaggregation
[3] SIG EP Placement
[4] SIG Scheduling SDK
[5] SIG Attention
[6] SIG MTP
[7] SIG Quant
[8] SIG Ops and Graph
[9] SIG Tooling and Testing