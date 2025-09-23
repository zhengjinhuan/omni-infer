import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def _compute_target_p(target, loss_mask, pad_length):
    target_head = target
    target_max_token = target_head.argmax(-1)
    target_mask = target_max_token.int()
    position_mask = target_mask[..., None] * loss_mask
    target_head = target_head.float()
    target_p = nn.Softmax(dim=2)(target_head)

    target_p_padded = F.pad(
        target_p,
        pad=(0, 0, 0, pad_length),
        mode="constant",
        # For bitwise equality with previous code
        value=1 / target_p.shape[-1],
    )

    position_mask_padded = F.pad(
        position_mask,
        pad=(0, 0, 0, pad_length),
        mode="constant",
        value=0,
    )

    target_p_padded = target_p_padded.detach()
    return target_p_padded, position_mask_padded

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
        self, target_head, draft_model, length: int = 1, attention_backend="sdpa"
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

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # norm_hidden_states = self.norm(hidden_states)
        return self.target_head(hidden_states)

    def forward(
        self,
        input_ids,
        attention_mask,
        loss_mask,
        hidden_states,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        target = self.target_head(hidden_states.roll(-1, -2))
        # Step 0: handle vocab size
        with torch.no_grad():
            target_p_padded, position_mask_padded = _compute_target_p(
                target=target,
                loss_mask=loss_mask,
                pad_length=self.length - 1,
            )
        del target

        batch_size, seq_length, _ = hidden_states.shape
        past_key_values_length = 0

        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # make attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self.draft_model.prepare_decoder_attention_mask(
            attention_mask, hidden_states, batch_size, seq_length, 0
        )
        
        plosses = []
        acces = []
        cache_hidden = [[], []]
        for i in range(self.length):
            target_p = target_p_padded[:, i : seq_length + i]
            position_mask = position_mask_padded[:, i : seq_length + i]
            # Step 5.1: embed the input ids
            inputs_embeds = self.draft_model.embed_input_ids(input_ids.roll(-i, -1))

            # Step 5.2: run the draft model backbone
            hidden_states = self.draft_model(
                inputs_embeds=inputs_embeds,
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=cache_hidden,
            )

            # Step 5.4: get logits
            logits = self.compute_logits(hidden_states)

            logits = logits.float()
            out_logp = nn.LogSoftmax(dim=2)(logits)
            plogp = target_p * out_logp
            loss = -torch.sum(position_mask * plogp, 2).mean()
            with torch.no_grad():
                acc = ((logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)).sum() / loss_mask.sum().clamp_min(1e-6)

            plosses.append(loss)
            acces.append(acc)

        return plosses, acces
