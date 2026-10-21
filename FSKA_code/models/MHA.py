import numpy as np
import torch
import torch.distributions as tdist
import torch.nn as nn
from torch.distributions.normal import Normal

from .attsharedw import AttSharedW
from .frska import FRSKAMultiHeadAttention
from .kernel import ika_ns


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, args, hid_dim, n_heads, dropout, device, w1, w2, freeze_qk=False, freeze_v=False):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.p_dist = tdist.Normal(0, args.prior_var)

        self.args = args
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.key_dim = args.KEY_DIM
        self.head_dim = args.KEY_DIM // n_heads
        self.value_head_dim = self.hid_dim // n_heads

        self.fc_q = nn.Linear(args.KEY_DIM, args.KEY_DIM)
        self.fc_k = nn.Linear(args.KEY_DIM, args.KEY_DIM)
        self.fc_v = nn.Linear(args.KEY_DIM, args.KEY_DIM)
        self.fc_o = nn.Linear(args.KEY_DIM, args.KEY_DIM)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        self.device = device

        if args.att_type == "frska":
            shared_flow = getattr(args, "frska_shared_flow", True)
            qk_mode = getattr(args, "frska_qk_mode", "no_qk")
            v_mode = getattr(args, "frska_v_mode", "fixed_orth")
            positive_feature_map = getattr(args, "frska_positive_map", True)
            self.frska_layer = FRSKAMultiHeadAttention(
                args,
                hid_dim,
                n_heads,
                dropout,
                device,
                w1,
                w2,
                shared_flow=shared_flow,
                freeze_qk=freeze_qk,
                freeze_v=freeze_v,
                qk_mode=qk_mode,
                v_mode=v_mode,
                positive_feature_map=positive_feature_map,
            )
            self.attsharedw = None
        else:
            self.frska_layer = None
            self.attsharedw = AttSharedW(args, n_heads, self.key_dim, device, dropout)

        if freeze_qk:
            self._freeze_qk_parameters()
        if freeze_v:
            self._freeze_v_parameters()

        self.standard_normal_dist = Normal(0.0, 1.0)

    def _freeze_qk_parameters(self):
        torch.nn.init.orthogonal_(self.fc_q.weight)
        torch.nn.init.orthogonal_(self.fc_k.weight)
        self.fc_q.weight.requires_grad = False
        self.fc_q.bias.requires_grad = False
        self.fc_k.weight.requires_grad = False
        self.fc_k.bias.requires_grad = False

    def _freeze_v_parameters(self):
        torch.nn.init.orthogonal_(self.fc_v.weight)
        self.fc_v.weight.requires_grad = False
        self.fc_v.bias.requires_grad = False

    def compute_linear_denominator_raw(self, query, key, value):
        if self.args.att_type != "frska" or self.frska_layer is None:
            raise RuntimeError("compute_linear_denominator_raw 仅支持 att_type='frska'")
        return self.frska_layer.compute_linear_denominator_raw(query, key, value)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        maxlen = query.shape[1]

        if self.args.att_type == "frska":
            x, attention, KLD = self.frska_layer(query, key, value)
            return x, attention, KLD

        dot_qk_mode = getattr(self.args, "dot_qk_mode", "normal")
        if self.args.att_type == "dot" and dot_qk_mode == "no_qk":
            Q = query
            K = key
        else:
            Q = self.fc_q(query)
            K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, maxlen, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, maxlen, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, maxlen, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        KLD = torch.tensor(0.0)
        if self.args.att_type == "dot":
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        elif self.args.att_type == "ikandirect":
            w1_proj = self.attsharedw.w1
            w2_proj = self.attsharedw.w2
            scores, norm = ika_ns(Q, K, self.args, self.scale, w1_proj, w2_proj, 2 * np.pi, self.training)
            energy = torch.log(scores + (1e-5)) + norm
        elif self.args.att_type == "mikan":
            mu, logvar, L = self.attsharedw.copulanet(Q, K)
            mu = mu.squeeze(-1)
            logvar = logvar.squeeze(-1)
            var = torch.exp(logvar)

            dim_batch_size, num_head, num_head = L.size()
            dim = int(dim_batch_size / batch_size)

            pos_eps = torch.randn([dim, num_head, self.args.M // 2], device=self.device)
            X_pos = torch.einsum("ijk,ijl->ijl", L, pos_eps)
            X_pos = torch.clamp(X_pos, min=-2.0, max=2.0)
            U_pos = self.standard_normal_dist.cdf(X_pos)

            neg_eps = torch.randn([dim, num_head, self.args.M // 2], device=self.device)
            X_neg = torch.einsum("ijk,ijl->ijl", L, neg_eps)
            X_neg = torch.clamp(X_neg, min=-2.0, max=2.0)
            U_neg = self.standard_normal_dist.cdf(X_neg)

            marginal_pos = Normal(mu.unsqueeze(-1), var.unsqueeze(-1))
            marginal_neg = Normal(-1 * mu.unsqueeze(-1), var.unsqueeze(-1))
            Y_pos = marginal_pos.icdf(U_pos)
            Y_neg = marginal_neg.icdf(U_neg)
            U = torch.cat([U_pos, U_neg])
            ent_copula = -1 * torch.sum(torch.mul(U, torch.log(U + (1e-5))))

            z = torch.cat([Y_pos, Y_neg], -1)
            w1_proj = self.attsharedw.wnet1(z)
            w2_proj = self.attsharedw.wnet2(z)
            scores, norm = ika_ns(Q, K, self.args, self.scale, w1_proj, w2_proj, 2 * np.pi, self.training)
            energy = torch.log(scores + (1e-5)) + norm

            q_dist = tdist.Normal(mu, logvar.exp())
            KLD = torch.distributions.kl_divergence(q_dist, self.p_dist)
            KLD = self.args.kl_lambda * torch.sum(KLD) + self.args.copula_lambda * ent_copula

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.args.KEY_DIM)
        x = self.fc_o(x)
        return x, attention, KLD
