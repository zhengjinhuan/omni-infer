from vllm.model_executor.models.qwen2 import Qwen2Attention as OriginalQwen2Attention

class Qwen2Attention(OriginalQwen2Attention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if 'prefix' in kwargs:
            self.prefix = kwargs['prefix']
        else:
            self.prefix = ""

    def forward(self, positions, hidden_states):

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k, layer_name=f"{self.prefix}.attn")
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

