"""
FRSKA (Flow-Regularized Spectral Kernel Attention) Implementation
"""

import numpy as np
import torch
import torch.nn as nn

try:
    import normflows as nf

    HAS_NORMFLOWS = True
except ImportError:
    HAS_NORMFLOWS = False
    print("警告: normflows未安装，将使用简化版本的Flow")


class SpectralFlow(nn.Module):
    def __init__(self, dim, num_flows=3, hidden_dim=128, device="cpu", num_mixtures=4):
        super().__init__()
        self.dim = dim
        self.device = device
        self.num_mixtures = num_mixtures

        if HAS_NORMFLOWS:
            torch.manual_seed(0)
            latent_size = dim
            b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])

            flows = []
            for i in range(num_flows):
                s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
                t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)

                if i % 2 == 0:
                    flows += [nf.flows.MaskedAffineFlow(b, t, s)]
                else:
                    flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
                flows += [nf.flows.ActNorm(latent_size)]

            q0 = nf.distributions.DiagGaussian(latent_size)
            target = nf.distributions.GaussianMixture(
                n_modes=num_mixtures, dim=latent_size, loc=None, scale=None, weights=None, trainable=True
            )
            self.nfm = nf.NormalizingFlow(q0=q0, flows=flows, p=target)
            self.nfm.to(device)
        else:
            self.nfm = None
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim * 2),
            )

    def sample(self, num_samples):
        if HAS_NORMFLOWS and self.nfm is not None:
            samples, _ = self.nfm.sample(num_samples=num_samples)
            return samples
        z = torch.randn(num_samples, self.dim, device=self.device)
        params = self.net(z)
        scale = torch.sigmoid(params[:, : self.dim]) * 2.0 + 0.5
        shift = params[:, self.dim :]
        x = scale * z + shift
        return x

    def log_prob(self, x):
        if HAS_NORMFLOWS and self.nfm is not None:
            return self.nfm.log_prob(x)
        try:
            z = torch.randn(x.shape[0], self.dim, device=x.device)
            params = self.net(z)
            scale = torch.sigmoid(params[:, : self.dim]) * 2.0 + 0.5
            base_log_prob = -0.5 * torch.sum(z**2, dim=-1) - 0.5 * self.dim * np.log(2 * np.pi)
            log_det = torch.sum(torch.log(scale + 1e-6), dim=-1)
            return torch.clamp(base_log_prob + log_det, min=-100, max=100)
        except Exception:
            return torch.full((x.shape[0],), -10.0, device=x.device)

    def reverse_kld(self, num_samples):
        if HAS_NORMFLOWS and self.nfm is not None:
            return self.nfm.reverse_kld(num_samples)
        return torch.tensor(0.0, device=self.device)


class FRSKAAttention(nn.Module):
    def __init__(self, args, head_dim, n_heads, device, shared_flow=False, positive_feature_map=True):
        super().__init__()
        self.args = args
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.M = args.M
        self.device = device
        self.shared_flow = shared_flow
        self.positive_feature_map = positive_feature_map
        self.num_flows = int(getattr(args, "frska_num_flows", 3))
        self.num_mixtures = int(getattr(args, "frska_num_mixtures", 4))

        if shared_flow:
            self.flow = SpectralFlow(
                dim=head_dim * 2,
                num_flows=self.num_flows,
                hidden_dim=128,
                device=device,
                num_mixtures=self.num_mixtures,
            )
            self.flows = None
        else:
            self.flow = None
            self.flows = nn.ModuleList(
                [
                    SpectralFlow(
                        dim=head_dim * 2,
                        num_flows=self.num_flows,
                        hidden_dim=128,
                        device=device,
                        num_mixtures=self.num_mixtures,
                    )
                    for _ in range(n_heads)
                ]
            )

        self.ngsm_prior = None

    def sample_spectral_density(self, Q, K):
        batch_size, n_heads, seq_len, head_dim = Q.shape

        if self.shared_flow:
            num_samples = n_heads * batch_size * self.M
            if HAS_NORMFLOWS and self.flow.nfm is not None:
                w, _ = self.flow.nfm.sample(num_samples=num_samples)
                omega_flat = w
            else:
                omega_flat = self.flow.sample(num_samples)
            omega_flat = torch.clamp(omega_flat, min=-10.0, max=10.0)
            if torch.isnan(omega_flat).any():
                omega_flat = torch.randn_like(omega_flat) * 0.5
            omega = omega_flat.view(batch_size, n_heads, self.M, 2 * head_dim)
            omega1, omega2 = torch.chunk(omega, chunks=2, dim=-1)
            omega_flat_all = omega_flat
        else:
            omega1_list = []
            omega2_list = []
            omega_flat_list = []

            for head_idx in range(n_heads):
                num_samples = batch_size * self.M
                if HAS_NORMFLOWS and self.flows[head_idx].nfm is not None:
                    w, _ = self.flows[head_idx].nfm.sample(num_samples=num_samples)
                    omega_flat_head = w
                else:
                    omega_flat_head = self.flows[head_idx].sample(num_samples)

                omega_flat_head = torch.clamp(omega_flat_head, min=-10.0, max=10.0)
                if torch.isnan(omega_flat_head).any():
                    omega_flat_head = torch.randn_like(omega_flat_head) * 0.5

                omega_head = omega_flat_head.view(batch_size, self.M, 2 * head_dim)
                omega1_head, omega2_head = torch.chunk(omega_head, chunks=2, dim=-1)
                omega1_list.append(omega1_head)
                omega2_list.append(omega2_head)
                omega_flat_list.append(omega_flat_head)

            omega1 = torch.stack(omega1_list, dim=1)
            omega2 = torch.stack(omega2_list, dim=1)
            omega_flat_all = torch.cat(omega_flat_list, dim=0)

        return omega1, omega2, omega_flat_all

    def compute_rff_features(self, x, omega1, omega2):
        x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        omega1_norm = omega1 / (torch.norm(omega1, dim=-1, keepdim=True) + 1e-8)
        omega2_norm = omega2 / (torch.norm(omega2, dim=-1, keepdim=True) + 1e-8)

        x_spectral1 = (2 * np.pi) * torch.einsum("bhnd,bhmd->bhnm", x_norm, omega1_norm)
        x_spectral2 = (2 * np.pi) * torch.einsum("bhnd,bhmd->bhnm", x_norm, omega2_norm)

        x_spectral1 = torch.clamp(x_spectral1, min=-10.0, max=10.0)
        x_spectral2 = torch.clamp(x_spectral2, min=-10.0, max=10.0)

        scale_factor = torch.sqrt(torch.tensor(1.0 / (4.0 * self.M), device=x.device, dtype=x.dtype))
        phi = scale_factor * torch.cat(
            [x_spectral1.cos() + x_spectral2.cos(), x_spectral1.sin() + x_spectral2.sin()], dim=-1
        )

        if torch.isnan(phi).any():
            phi = torch.where(torch.isnan(phi), torch.zeros_like(phi), phi)
        phi = torch.clamp(phi, min=-1.0, max=1.0)
        return phi

    def forward(self, Q, K, V, return_dense_attention=False, return_linear_denominator_raw=False):
        batch_size, n_heads, seq_len, head_dim = Q.shape

        Q = Q / (torch.norm(Q, dim=-1, keepdim=True) + 1e-8)
        K = K / (torch.norm(K, dim=-1, keepdim=True) + 1e-8)

        omega1, omega2, omega_samples = self.sample_spectral_density(Q, K)
        phi_Q = self.compute_rff_features(Q, omega1, omega2)
        phi_K = self.compute_rff_features(K, omega1, omega2)

        if self.positive_feature_map:
            phi_Q_pos = torch.nn.functional.elu(phi_Q) + 1.0
            phi_K_pos = torch.nn.functional.elu(phi_K) + 1.0
        else:
            phi_Q_pos = phi_Q
            phi_K_pos = phi_K

        kv_summary = torch.einsum("bhlm,bhld->bhmd", phi_K_pos, V)
        k_summary = torch.sum(phi_K_pos, dim=2)
        numerator = torch.einsum("bhlm,bhmd->bhld", phi_Q_pos, kv_summary)
        denom_raw = torch.einsum("bhlm,bhm->bhl", phi_Q_pos, k_summary)
        denominator_safe = torch.clamp(denom_raw, min=1e-6).unsqueeze(-1)
        context = numerator / denominator_safe

        kl_div = torch.tensor(0.0, device=Q.device)
        if HAS_NORMFLOWS:
            if self.shared_flow:
                if self.flow.nfm is not None:
                    try:
                        kl_div_raw = self.flow.reverse_kld(self.M)
                        if torch.isnan(kl_div_raw) or torch.isinf(kl_div_raw):
                            kl_div_raw = torch.tensor(0.0, device=Q.device)
                        kl_div = torch.clamp(kl_div_raw, min=0.0, max=100.0) * self.n_heads
                    except Exception:
                        kl_div = torch.tensor(0.0, device=Q.device)
            else:
                for head_idx in range(self.n_heads):
                    if self.flows[head_idx].nfm is not None:
                        try:
                            kl_div_head = self.flows[head_idx].reverse_kld(self.M)
                            if torch.isnan(kl_div_head) or torch.isinf(kl_div_head):
                                kl_div_head = torch.tensor(0.0, device=Q.device)
                            kl_div_head = torch.clamp(kl_div_head, min=0.0, max=100.0)
                            kl_div = kl_div + kl_div_head
                        except Exception:
                            continue
        if torch.isnan(kl_div) or torch.isinf(kl_div):
            kl_div = torch.tensor(0.0, device=Q.device)

        attention = None
        if return_dense_attention:
            dense_scores = torch.matmul(phi_Q, phi_K.transpose(-2, -1))
            dense_scores = torch.nan_to_num(dense_scores, nan=0.0, posinf=0.0, neginf=0.0)
            dense_denom = torch.clamp(torch.sum(dense_scores, dim=-1, keepdim=True), min=1e-6)
            attention = dense_scores / dense_denom

        if return_linear_denominator_raw:
            return context, attention, kl_div, denom_raw
        return context, attention, kl_div


class FRSKAMultiHeadAttention(nn.Module):
    def __init__(
        self,
        args,
        hid_dim,
        n_heads,
        dropout,
        device,
        w1,
        w2,
        shared_flow=False,
        freeze_qk=False,
        freeze_v=False,
        qk_mode="normal",
        v_mode="learnable",
        positive_feature_map=True,
    ):
        super().__init__()
        self.args = args
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.key_dim = args.KEY_DIM
        self.head_dim = args.KEY_DIM // n_heads
        self.device = device
        self.qk_mode = qk_mode
        self.v_mode = v_mode
        self.positive_feature_map = positive_feature_map
        self.return_dense_attention = getattr(args, "frska_return_dense_attention", False)

        if qk_mode == "normal":
            self.fc_q = nn.Linear(args.KEY_DIM, args.KEY_DIM)
            self.fc_k = nn.Linear(args.KEY_DIM, args.KEY_DIM)
        else:
            self.fc_q = None
            self.fc_k = None

        if v_mode == "learnable":
            self.fc_v = nn.Linear(args.KEY_DIM, args.KEY_DIM)
            self.fixed_v_weights = None
        elif v_mode == "fixed_orth":
            self.fc_v = None
            g = torch.Generator(device="cpu")
            g.manual_seed(42)
            rand_mat = torch.randn(self.key_dim, self.key_dim, generator=g)
            q, _ = torch.linalg.qr(rand_mat)
            fixed_v = torch.zeros(n_heads, self.key_dim, self.head_dim)
            for h in range(n_heads):
                start = h * self.head_dim
                end = (h + 1) * self.head_dim
                fixed_v[h] = q[:, start:end]
            self.register_buffer("fixed_v_weights", fixed_v)
        else:
            raise ValueError(f"Unknown v_mode: {v_mode}. Use 'learnable' or 'fixed_orth'")

        self.fc_o = nn.Linear(args.KEY_DIM, args.KEY_DIM)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        self.frska_attention = FRSKAAttention(
            args,
            self.head_dim,
            n_heads,
            device,
            shared_flow=shared_flow,
            positive_feature_map=positive_feature_map,
        )

        if freeze_qk and qk_mode == "normal":
            self._freeze_qk_parameters()
        if freeze_v and v_mode == "learnable":
            self._freeze_v_parameters()

    def _freeze_qk_parameters(self):
        if self.fc_q is None or self.fc_k is None:
            return
        torch.nn.init.orthogonal_(self.fc_q.weight)
        torch.nn.init.orthogonal_(self.fc_k.weight)
        self.fc_q.weight.requires_grad = False
        self.fc_q.bias.requires_grad = False
        self.fc_k.weight.requires_grad = False
        self.fc_k.bias.requires_grad = False

    def _freeze_v_parameters(self):
        if self.fc_v is None:
            return
        torch.nn.init.orthogonal_(self.fc_v.weight)
        self.fc_v.weight.requires_grad = False
        self.fc_v.bias.requires_grad = False

    def compute_linear_denominator_raw(self, query, key, value):
        batch_size = query.shape[0]
        maxlen = query.shape[1]

        if self.qk_mode == "normal":
            Q = self.fc_q(query)
            K = self.fc_k(key)
        else:
            Q = query
            K = key

        if self.v_mode == "learnable":
            V = self.fc_v(value)
        else:
            V_fixed = torch.einsum("bsd,hdk->bshk", value, self.fixed_v_weights)

        Q = Q.view(batch_size, maxlen, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, maxlen, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.v_mode == "learnable":
            V = V.view(batch_size, maxlen, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            V = V_fixed.permute(0, 2, 1, 3).contiguous()

        _, _, _, denom_raw = self.frska_attention(Q, K, V, return_linear_denominator_raw=True)
        return denom_raw

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        maxlen = query.shape[1]

        if self.qk_mode == "normal":
            Q = self.fc_q(query)
            K = self.fc_k(key)
        else:
            Q = query
            K = key

        if self.v_mode == "learnable":
            V = self.fc_v(value)
        else:
            V_fixed = torch.einsum("bsd,hdk->bshk", value, self.fixed_v_weights)

        Q = Q.view(batch_size, maxlen, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, maxlen, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.v_mode == "learnable":
            V = V.view(batch_size, maxlen, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            V = V_fixed.permute(0, 2, 1, 3).contiguous()

        context, attention, kl_div = self.frska_attention(Q, K, V, return_dense_attention=self.return_dense_attention)

        if hasattr(self.args, "kl_lambda_frska"):
            kl_div = kl_div * self.args.kl_lambda_frska
        else:
            kl_div = kl_div * self.args.kl_lambda

        x = self.dropout(context)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.args.KEY_DIM)
        x = self.fc_o(x)
        return x, attention, kl_div
