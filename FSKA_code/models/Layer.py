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
