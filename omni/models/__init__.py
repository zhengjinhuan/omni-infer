from vllm import ModelRegistry


def register_model():

    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM",
        "omni.models.deepseek.deepseek_v2:CustomDeepseekV2ForCausalLM")

    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM",
        "omni.models.deepseek.deepseek_v3:CustomDeepseekV3ForCausalLM")
