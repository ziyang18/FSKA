import math

import torch
import torch.nn as nn

from .Layer import PositionwiseFeedforwardLayer
from .MHA import MultiHeadAttentionLayer


class baseTransformer(nn.Module):
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

        self.tok_embedding = nn.Embedding(input_dim, 1)
        self.pos_embedding = nn.Embedding(num_data, 1)
        self.pos_embedding.weight.requires_grad = False

        self.w1 = torch.nn.Parameter(torch.empty(hid_dim // n_heads, n_heads, args.M), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.empty(hid_dim // n_heads, n_heads, args.M), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    args,
                    device,
                    hid_dim,
                    n_heads,
                    pf_dim,
                    dropout,
                    self.w1,
                    self.w2,
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
        for layer in self.layers:
            x, attention, KLD = layer(x)
        output = self.fc_out(x)
        return output, KLD


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
