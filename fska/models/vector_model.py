import math

import torch
import torch.nn as nn

from .vector_layers import PositionwiseFeedforwardLayer
from .vector_layers import MultiHeadAttentionLayer


class FSKAVectorModel(nn.Module):
    def __init__(
        self,
        args,
        num_data,
        input_dim,
        output_dim,
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        dropout,
        device,
        freeze_qk=False,
        freeze_v=False,
    ):
        super().__init__()

        self.args = args
        self.num_data = num_data
        self.device = device

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    args,
                    device,
                    hid_dim,
                    n_heads,
                    pf_dim,
                    dropout,
                    None,
                    None,
                    freeze_qk=freeze_qk,
                    freeze_v=freeze_v,
                )
                for _ in range(n_layers)
            ]
        )
        self.fc_in = nn.Linear(input_dim, args.KEY_DIM)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.key_dim = args.KEY_DIM

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.fc_in(x)
        total_kl = x.new_zeros(())
        for layer in self.layers:
            x, attention, KLD = layer(x)
            total_kl = total_kl + KLD
        output = self.fc_out(x)
        return output, total_kl


class EncoderLayer(nn.Module):
    def __init__(self, args, device, hid_dim, n_heads, pf_dim, dropout, w1, w2, freeze_qk=False, freeze_v=False):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(
            args, hid_dim, n_heads, dropout, device, w1, w2, freeze_qk=freeze_qk, freeze_v=freeze_v
        )
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(args, hid_dim, 1, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        _src, attention, KLD = self.self_attention(src, src, src)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src, attention, KLD
