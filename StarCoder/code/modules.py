from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RobertaClassificationHead(nn.Module):
    """Binary classification on concatenated CLS vectors of two segments."""
    def __init__(self, hidden_size: int, dropout: float, num_labels: int = 2):
        super().__init__()
        self.dense = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.reshape(-1, features.size(-1) * 2)  # (B, 2H)
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)

class PairClassifier(nn.Module):
    """
    Input: input_ids of shape (B, 2L)
    Process: reshape -> (2B, L) -> encoder -> take CLS (or first token) -> classification head
    """
    def __init__(self, encoder: nn.Module, hidden_size: int, dropout: float,
                 num_labels: int, code_length: int, pad_token_id: int):
        super().__init__()
        self.encoder = encoder
        self.code_length = code_length
        self.pad_token_id = pad_token_id
        self.classifier = RobertaClassificationHead(hidden_size, dropout, num_labels)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        bsz = input_ids.size(0)
        input_ids = input_ids.view(-1, self.code_length)     # (2B, L)
        attention = input_ids.ne(self.pad_token_id)          # pad mask

        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention)
        last_hidden = enc_out[0] if isinstance(enc_out, (list, tuple)) else enc_out.last_hidden_state
        cls_ = last_hidden[:, 0, :]                           # (2B, H)

        logits = self.classifier(cls_)                        # (B, 2)
        probs = F.softmax(logits, dim=-1)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, probs, cls_
        return probs
