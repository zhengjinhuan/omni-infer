import numpy as np
import torch
from vllm.utils import is_pin_memory_available
class PenaltyCache:
    def __init__(self, num_req, vocab_size, device):
        self.cached_req_ids = None
        self.prompt_mask = torch.zeros((num_req + 1, vocab_size),
                                        dtype=torch.bool,
                                        device=device,
                                        pin_memory=is_pin_memory_available())
        self.output_mask = torch.zeros((num_req + 1, vocab_size),
                                        dtype=torch.bool,
                                        device=device,
                                        pin_memory=is_pin_memory_available())
        self.output_bin_counts = torch.zeros((num_req + 1, vocab_size),
                                        dtype=torch.int64,
                                        device=device,
                                        pin_memory=is_pin_memory_available())
        self.ones_cpu = torch.ones(vocab_size, dtype=torch.bool, device="cpu")
    
    def apply_movements(self, src):
        num_reqs = len(src)
        applied = [False] * num_reqs
        last_idx = self.prompt_mask.shape[0] - 1
        for cached_tensor in [self.prompt_mask, self.output_mask, self.output_bin_counts]:
            for i in range(num_reqs):
                cur = i
                while src[cur] != cur and src[cur] != -1 and not applied[cur]:
                    applied[cur] = True
                    if cur == i:
                        cached_tensor[last_idx] = cached_tensor[i]
                    if src[cur] == i:
                        cached_tensor[cur] = cached_tensor[last_idx]
                    else:
                        cached_tensor[cur] = cached_tensor[src[cur]]
                    cur = src[cur]
    
    def permute_cached_reqs(self, new_req_ids):
        if self.cached_req_ids == None:
            self.cached_req_ids = new_req_ids
            return
        num_reqs = len(self.cached_req_ids)
        src = [-1] * num_reqs
        for i in range(len(self.cached_req_ids)):
            try:
                index = new_req_ids.index(self.cached_req_ids[i])
                src[index] = i
            except ValueError:
                pass
        self.apply_movements(src)
        self.cached_req_ids = new_req_ids
    
    def prepare_new_reqs(self, scheduled_new_reqs):
        for i in range(len(scheduled_new_reqs)):
            try:
                index = self.cached_req_ids.index(scheduled_new_reqs[i].req_id)
            except ValueError:
                raise RuntimeError("penalty cache: a scheduled new req is not in req id list")
            self.output_bin_counts[i] = torch.zeros_like(self.output_bin_counts[i])
            self.output_mask[i] = torch.zeros_like(self.output_mask[i])
            prompt_mask_cpu = torch.zeros_like(self.prompt_mask[i], device='cpu')
            prompt_token_ids_tensor = torch.tensor(scheduled_new_reqs[i].prompt_token_ids, dtype=torch.int64)
            prompt_mask_cpu.scatter_(dim=0, index=prompt_token_ids_tensor, src=self.ones_cpu)
            self.prompt_mask[i] = prompt_mask_cpu.to(device=self.prompt_mask.device)
        
    def do_penalty_from_samplinng_metadata(self, input_batch) -> tuple[bool, bool, bool]:
        num_reqs = len(input_batch.req_ids)
        do_frequency_penalties = torch.any(input_batch.frequency_penalties_cpu_tensor[:num_reqs] != 0)
        do_presence_penalties = torch.any(input_batch.presence_penalties_cpu_tensor[:num_reqs] != 0)
        do_repetition_penalties = torch.any(input_batch.repetition_penalties_cpu_tensor[:num_reqs] != 1)
        return (do_frequency_penalties, do_presence_penalties, do_repetition_penalties)
    
    def update_do_penalty(self, sampling_metadata, input_batch):
        (self.do_frequency_penalties, self.do_presence_penalties, self.do_repetition_penalties) = self.do_penalty_from_samplinng_metadata(input_batch)
    
    def prepare_cache(self, scheduled_new_reqs, req_ids, sampling_metadata, input_batch):
        if sampling_metadata.no_penalties:
            self.do_penalties = False
            return
        self.do_penalties = True
        self.update_do_penalty(sampling_metadata, input_batch)
        self.permute_cached_reqs(input_batch.req_ids)
        self.prepare_new_reqs(scheduled_new_reqs)
    
    def save_token_ids(self, sampled_token_ids):
        if not self.do_penalties:
            return
        sampled_token_ids = sampled_token_ids.view(-1)
        self.output_bin_counts.scatter_add_(1, sampled_token_ids.unsqueeze(0).to(dtype=torch.int64),
            torch.ones_like(sampled_token_ids.unsqueeze(0), dtype=torch.int64))
        self.output_mask = self.output_bin_counts > 0
    
    def revert_rejected_tokens(self, accepted_mask, token_ids):
        if not self.do_penalties:
            return
        rejected_mask = -(~accepted_mask).to(dtype=torch.int64)
        token_ids = torch.clone(token_ids).masked_fill_(accepted_mask, 0).to(dtype=torch.int64)
        self.output_bin_counts.scatter_add_(1, token_ids, rejected_mask)
        self.output_mask = self.output_bin_counts > 0

        