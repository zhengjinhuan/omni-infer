# **一、数据集生成**

 **工具文件:**
 tools\benchmark_test\benchmark_tools\generate_dataset.py

 **执行命令：**

```python generate_dataset.py \
  --dataset ./data/test_dataset.json \
  --tokenizer /path/to/deepseek-ai \
  --min-input 1000 \
  --max-input 1000 \
  --avg-input 1000 \
  --std-input 0 \
  --num-requests 1000
  ```

**需要修改的参数**

|参数|说明|
|---|---|
|dataset	|输出数据集路径|
|min-input	|输入 prompt 最小 token 数|
|max-input	|输入 prompt 最大 token 数|
|avg-input	|平均输入长度|
|num-requests	|生成请求数量（如 1000、5000|
|||



# **二、配置服务端信息（providers.yaml）**

**文件路径:**

tools\benchmark_test\benchmark\providers.yaml

**需要修改的参数**
```    
base_url: 'http://7.0.0.0:0000/v1'
model_name: 'deepseek'
```
|参数|说明|
|---|---|
|base_url	|服务端口|
|model_name	|模型名字|
|||

# **三、配置基准测试脚本（benchmark_baseline_test.sh）**

**脚本路径：**
tools\benchmark_test\benchmark\benchmark_baseline_test.sh
```
cd ..../benchmark_tools  //跳转到benchmark工具路径
dataset=xxx 
output_dir= xxx 
providers_path= xxx 
epochs= xxx 

  growth_rate=(xxx)
 for gr in "${growth_rate[@]}"; do
     echo "Start ===================================================="$(date)"===================================================="
     python benchmark_parallel.py \
     --run-method climbing \
     --backend openai-chat \
     --providers-path $providers_path \
     --parallel-num xxx --epochs $epochs \
     --prompt-tokens xxx --output-tokens xxx \
     --control-method queue \
     --use-spec-decode --num-speculative-tokens=1\
     --num-scheduler-steps=1 \
     --growth-rate $gr \
     --dataset-dir  $dataset \
     --benchmark-dir $output_dir
     echo "End ===================================================="$(date)"===================================================="
     sleep 60
 done
 ```

**需要修改的参数**
|参数|说明|
|---|---|
|dataset|指定数据集路径 |
|output_dir|测试结果输出路径|
|providers_path|providers.yaml路径|
|epochs|测试执行的轮数|
|growth_rate|爬坡速率|
|parallel-num|指定并发数|
|prompt-tokens|输入的 token 数量|
|output-tokens|输出的 token 数量|
|||

**一般growth_rate建议大于等于单die并发数；**

**parallel-num = 期望单die并发 * D侧总die数；**

以8台A3机器组1P1D环境，进行BF16权重的性能测试为例，
测试用例为16K+1K的定长数据集，期望单die并发为4，A3机器单机共有8卡16die，D侧共有4机64die

主要参数可以设置为
growth-rate = 12、parallel-num = 4 * 64 = 256、prompt-tokens = 65536、output-tokens = 1024

# **四、执行测试命令**
```
bash benchmark_baseline_test.sh
```