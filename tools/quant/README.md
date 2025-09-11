参数说明  
--input-bf16-hf-path   原始bf16权重路径  
--output-path  生成量化权重路径  
--device               设备类型，支持cpu和npu  
--model-name           hugginface权重名称，在没有元数据配置时自动根据权重名下载配置文件  
--w4                   int4量化标识, 不加该参数时为int8量化  

操作步骤  
1、拷贝元数据到output路径（注：model.safetensors.index.json需使用fp8权重的对应配置）  
2、执行量化命令  
deepseek/kimi k2
int8量化: python quant_deepseek_kimi2.py --input-bf16-hf-path {bf16权重路径} --output-path {量化权重路径} --device "cpu"  
int4量化: python quant_deepseek_kimi2.py --input-bf16-hf-path {bf16权重路径} --output-path {量化权重路径} --device "cpu" --w4  

qwen  
int8量化: python quant_qwen.py --input-bf16-hf-path {bf16权重路径} --output-path {量化权重路径} --device "cpu"  

编译步骤  
进入python目录下执行： python setup.py bdist_wheel  
