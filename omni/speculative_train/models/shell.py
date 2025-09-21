import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def _compute_target_p(target, loss_mask):
    target_head = target
    target_max_token = target_head.argmax(-1)
    target_mask = target_max_token
    target_mask = target_mask[..., None].int()
    position_mask = target_mask * loss_mask
    target_head = target_head.float()
    target_p = nn.Softmax(dim=2)(target_head)
    target_p = target_p.detach()
    return target_p, position_mask

def _compute_loss(logits, target_p, position_mask):
    logits = logits.float()
    out_logp = nn.LogSoftmax(dim=2)(logits)
    plogp = target_p * out_logp
    loss = -torch.sum(position_mask * plogp, 2).mean()
    return loss

class OfflineEagleModel(nn.Module):
    """
    In sgl-spec, we implement offline/online training.
    Offline training means we have the target hidden_states available before training.
    """

    def __init__(
        self, target_head, draft_model, length: int = 7, attention_backend="sdpa"
    ):
        """
        Args:
            target_head: the target head to process the target hidden states.
            draft_model: the draft model to be trained.
            length: TTT length, it means how many turns to unroll during TTT.
        """
        super().__init__()
        self.draft_model = draft_model
        self.target_head = target_head
        # TODO to support TTT
        self.length = length
        self.attention_backend = attention_backend

    def forward(
        self,
        input_ids,
        attention_mask,
        loss_mask,
        hidden_states,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        target = self.target_head(hidden_states)
        # Step 0: handle vocab size
        target_p, position_mask = _compute_target_p(
            target=target,
            loss_mask=loss_mask,
        )
        del target

        batch_size, seq_length, _ = hidden_states.shape
        past_key_values_length = 0

        # Step 5.1: embed the input ids
        inputs_embeds = self.draft_model.embed_input_ids(input_ids)
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)

        # Step 5.2: run the draft model backbone
        hidden_states_out = self.draft_model(
            inputs_embeds=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        # Step 5.4: get logits
        logits = self.compute_logits(hidden_states_out)

        logits = logits.float()
        out_logp = nn.LogSoftmax(dim=2)(logits)
        plogp = target_p * out_logp
        loss = -torch.sum(position_mask * plogp, 2).mean()
        with torch.no_grad():
            acc = ((logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)).sum() / loss_mask.sum().clamp_min(1e-6)

        plosses = [loss]
        acces = [acc]
        return plosses, acces
