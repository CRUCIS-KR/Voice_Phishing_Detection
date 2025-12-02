import torch
import torch.nn as nn

# 자료형 선언 라이브러리
from typing import Optional

# KoBERT 모델 호출 라이브러리
from transformers import AutoModel

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# 양방향 LSTM 구현 (PyTorch nn.LSTM 사용)
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x, mask: Optional[torch.Tensor] = None):
       
        if mask is None:
            out, _ = self.lstm(x)  # (N, T, 2H)
            return out

        N, T, _ = x.size()
        # lengths: number of valid tokens per sequence
        lengths = mask.sum(dim=1).long()  # (N,)

        if (lengths == 0).all():
            return x.new_zeros(N, T, 2 * self.hidden_size)

        x = x * mask.unsqueeze(-1).to(x.dtype)

        lengths_clamped = lengths.clone()
        lengths_clamped[lengths_clamped == 0] = 1

        lengths_sorted, perm_idx = lengths_clamped.sort(0, descending=True)
        x_sorted = x[perm_idx]

        packed = pack_padded_sequence(
            x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_out, _ = self.lstm(packed)
        out_sorted, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=T
        )

        _, unperm_idx = perm_idx.sort(0)
        out = out_sorted[unperm_idx]

        zero_mask = lengths == 0
        if zero_mask.any():
            out[zero_mask] = 0.0

        return out


# KoBERT + LSTM LM (이전 KoBERT_GRU_LM의 GRU를 LSTM으로 대체)
class KoBERT_LSTM_LM(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        classes=2,
        freeze=False,
        dropout=0.1,
        model="skt/kobert-base-v1",
        remove_cls=True,
        lstm_num_layers=1,
    ):
        super().__init__()

        self.KoBERT = AutoModel.from_pretrained(model)
        self.emb_size = self.KoBERT.config.hidden_size
        self.remove_cls = remove_cls

        # BiLSTM 사용
        self.RNN = BiLSTM(
            self.emb_size, hidden_size, num_layers=lstm_num_layers, dropout=0.0
        )

        self.dropout = nn.Dropout(dropout)
        final = 2 * hidden_size
        self.norm = nn.LayerNorm(final)
        self.affine = nn.Linear(final, classes)

        if freeze:
            self.freeze_bert()
        else:
            self.KoBERT_frozen = False

    def freeze_bert(self):
        for p in self.KoBERT.parameters():
            p.requires_grad = False
        self.KoBERT_frozen = True

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.long()

        if self.KoBERT_frozen:
            with torch.no_grad():
                output = self.KoBERT(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=None,
                    return_dict=True,
                )
        else:
            output = self.KoBERT(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=None,
                return_dict=True,
            )

        emb = output.last_hidden_state

        if self.remove_cls:
            emb = emb[:, 1:, :]
            mask = attention_mask[:, 1:]
        else:
            mask = attention_mask

        emb = self.dropout(emb)

        seq_fb = self.RNN(emb, mask=mask)

        mask_reshape = mask.float().unsqueeze(-1)
        seq_sum = (seq_fb * mask_reshape).sum(dim=1)
        token_num = mask_reshape.sum(dim=1).clamp(min=1.0)
        mean = seq_sum / token_num

        mean = self.norm(mean)
        
        mean = self.dropout(mean)
        
        logits = self.affine(mean)

        return {"logits": logits}
