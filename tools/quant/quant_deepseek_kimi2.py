import optiquant.int8 as qint8
import optiquant.int4 as qint4
import optiquant.faquant as faquant
from argparse import ArgumentParser
import json
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-bf16-hf-path", type=str, required=True, help="bf16 weight path")
    parser.add_argument("--output-path", type=str, required=True, help="quantized weight path")
    parser.add_argument("--device", type=str, required=True, help="support cpu and npu")
    parser.add_argument("--file_count", type=int, default=0, help="File count when loading model")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1", help="Huggingface repo name")

    parser.add_argument("--pangu-mode", default=False, action="store_true", help="pangu mode")
    parser.add_argument("--w4", default=False, action="store_true", help="int4 quantization flag")
    parser.add_argument("--qtype", type=str, default="sszs50g0a0b4sym1", help="quantization config. only support sszs50g0a0b4sym1 now")
    parser.add_argument("--c8-calib-path", type=str, default=None, help="mla c8 calibration data path")
    parser.add_argument("--kvs-safetensor-name", type=str, default=None, help="mla c8 (faquant) safetensor name")

    args = parser.parse_args()

    if args.c8_calib_path is not None:
        faquant.main(args, args.output_path, args.c8_calib_path, args.kvs_safetensor_name)

    if args.w4:
        qint4.main(args, args.input_bf16_hf_path, args.output_path, args.model_name)
        num_bits = {"self_attn.kv_a_proj_with_mqa": 8, "self_attn.q_a_proj": 8, "self_attn.q_b_proj": 8,
                    "self_attn.o_proj": 8, "mlp.down_proj": 8, "mlp.gate_up_proj": 8, "mlp.shared_experts": 8,
                    "mlp.experts": 4}
    else:
        qint8.main(args, args.input_bf16_hf_path, args.output_path, args.model_name)
        num_bits = 8

    ignores = []
    for i in range(62):
        ignore = f"model.layers.{i}.self_attn.kv_b_proj"
        ignores.append(ignore)

    quant_config = {"config_groups": {"group_0": {}}, "format": "int-quantized",
                    "global_compression_ratio": 1.5943962512751309, "ignore": ignores, "kv_cache_scheme": None,
                    "quant_method": "compressed-tensors", "quantization_status": "compressed"}
    quant_config["config_groups"]["group_0"]["input_activations"] = {"actorder": None, "block_structure": None,
                                                                     "dynamic": True, "group_size": None, "num_bits": 8,
                                                                     "observer": "memoryless", "observer_kwargs": {},
                                                                     "strategy": "token", "symmetric": True,
                                                                     "type": "int"}
    quant_config["config_groups"]["group_0"]["output_activations"] = None
    quant_config["config_groups"]["group_0"]["targets"] = ["Linear"]
    quant_config["config_groups"]["group_0"]["weights"] = {"actorder": None, "block_structure": None, "dynamic": False,
                                                           "group_size": None, "num_bits": num_bits,
                                                           "observer": "minmax", "observer_kwargs": {},
                                                           "strategy": "channel", "symmetric": True, "type": "int"}

    config_path = os.path.join(args.output_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    config["quantization_config"] = quant_config

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
