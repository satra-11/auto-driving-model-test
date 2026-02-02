import torch
import torch.nn as nn
import torch.nn.functional as F


class LTCNLayer(nn.Module):
    def __init__(
        self,
        in_dim: int, # 入力次元(抽出済み特徴量)
        k: int, # 隠れ状態の次元(操作量) 
        num_blocks: int, # LTCNのブロック数
    ):
        super().__init__()
        assert num_blocks >= 1 and k >= 1
        self.in_dim = in_dim
        self.k = k
        self.num_blocks = num_blocks
        self.N = num_blocks * k

        # --- tau: ensure positivity via softplus
        self._tau_raw = nn.Parameter(torch.full((self.N,), 0.0))  # softplus(0)=~0.693
        self.tau_eps = 1e-6

        # --- 外部入力u_tをブロック0に伝える重み
        self.W_in = nn.Linear(in_dim, k)

        # --- 前方ブロックから次のブロックへの接続
        self.W_fwd = nn.ModuleList([nn.Linear(k, k) for _ in range(num_blocks - 1)])

        # --- Recurrent per block: y_j -> k
        self.W_rec = nn.ModuleList([nn.Linear(k, k) for _ in range(num_blocks)])

        # --- E matrices (per block): combine net_out_j and net_recurr_j into drive term
        #     Shapes: (B, k, k) so that (E[j] @ vec_k)
        self.E_l = nn.Parameter(torch.zeros(num_blocks, k, k))
        self.E_l_r = nn.Parameter(torch.zeros(num_blocks, k, k))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.zeros_(self.W_in.bias)
        for lin in self.W_fwd:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        for lin in self.W_rec:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        nn.init.xavier_uniform_(self.E_l)
        nn.init.xavier_uniform_(self.E_l_r)

    def forward(
        self,
        y: torch.Tensor,  # (..., N)
        u_t: torch.Tensor | None,  # (..., in_dim)  (required for block 0 each step)
        dt: float = 5e-2,
    ) -> torch.Tensor:
        assert y.shape[-1] == self.N
        if u_t is not None:
            assert u_t.shape[:-1] == y.shape[:-1] and u_t.shape[-1] == self.in_dim

        # reshape to (..., B, k)
        *batch, _ = y.shape
        B, k = self.num_blocks, self.k
        y = y.view(*batch, B, k)

        # tau positive
        tau = F.softplus(self._tau_raw) + self.tau_eps  # shape (N,)
        tau = tau.view(B, k)
        if len(batch) > 0:
            tau = tau.view(*([1] * len(batch)), B, k).expand(*batch, B, k)

        # ---- net_out per block
        net_out_list = []
        # block 0 from input
        if u_t is None:
            net0 = torch.zeros(*batch, k, device=y.device, dtype=y.dtype)
        else:
            net0 = torch.tanh(self.W_in(u_t))
        net_out_list.append(net0)

        # blocks 1..B-1 from previous block state
        for j in range(1, B):
            prev = y[..., j - 1, :]  # (..., k)
            net_j = torch.tanh(self.W_fwd[j - 1](prev))
            net_out_list.append(net_j)
        net_out = torch.stack(net_out_list, dim=-2)  # (..., B, k)

        # ---- net_recurr per block (from same block state)
        net_recurr_list = []
        for j in range(B):
            cur = y[..., j, :]  # (..., k)
            net_r = torch.tanh(self.W_rec[j](cur))
            net_recurr_list.append(net_r)
        net_recurr = torch.stack(net_recurr_list, dim=-2)  # (..., B, k)

        # ---- decay term: (1/tau) + |net_out| + |net_recurr|
        decay = (1.0 / tau) + net_out.abs() + net_recurr.abs()  # (..., B, k)

        # ---- E-lin combinations: (E_l[j] @ net_out_j) and (E_l_r[j] @ net_recurr_j)
        # reshape for bmm: (..., B, k, k) @ (..., B, k, 1) -> (..., B, k, 1)
        E = self.E_l.view(*([1] * len(batch)), B, k, k).expand(*batch, B, k, k)
        Er = self.E_l_r.view(*([1] * len(batch)), B, k, k).expand(*batch, B, k, k)

        out_term = torch.matmul(E, net_out.unsqueeze(-1)).squeeze(-1)  # (..., B, k)
        recurr_term = torch.matmul(Er, net_recurr.unsqueeze(-1)).squeeze(
            -1
        )  # (..., B, k)

        # semi-implicit Euler integration
        drive = out_term + recurr_term
        y = (y + dt * drive) / (1.0 + dt * decay)

        return y.view(*batch, B * k)
