import torch
import torch.nn as nn


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, args, hid_dim, pf_dim, dropout):
        super().__init__()

        self.args = args
        self.fc_1 = nn.Linear(hid_dim, hid_dim)
        self.fc_2 = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = torch.relu(self.fc_2(x))
        return x


from .vector_attention import FRSKAMultiHeadAttention


class MultiHeadAttentionLayer(nn.Module):
    """FSKA-only adapter for the original vector-input backbone."""
    def __init__(self, args, hid_dim, n_heads, dropout, device, w1=None, w2=None,
                 freeze_qk=False, freeze_v=False):
        super().__init__()
        self.frska_layer = FRSKAMultiHeadAttention(
            args, hid_dim, n_heads, dropout, device, w1, w2,
            shared_flow=args.frska_architecture == "shared_full",
            freeze_qk=freeze_qk, freeze_v=freeze_v,
            qk_mode=args.frska_qk_mode, v_mode=args.frska_v_mode,
            positive_feature_map=args.frska_positive_map)

    def forward(self, query, key, value):
        return self.frska_layer(query, key, value)
