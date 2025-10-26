import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        assert dim_V % num_heads == 0, "dim_V must be divisible by num_heads"
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, key_padding_mask: torch.Tensor = None):
        """
        Q: [B, Lq, dim_Q]
        K: [B, Lk, dim_K]  (V is computed from K)
        key_padding_mask: [B, Lk] with True=valid, False=pad (optional)
        """
        B, Lq, _ = Q.shape
        _, Lk, _ = K.shape

        Q = self.fc_q(Q)          # [B, Lq, dim_V]
        K = self.fc_k(K)          # [B, Lk, dim_V]
        V = self.fc_v(K)          # [B, Lk, dim_V]

        dim_split = self.dim_V // self.num_heads
        scale = math.sqrt(dim_split)

        # split heads by concatenating on the batch axis
        Q_ = torch.cat(Q.split(dim_split, dim=2), dim=0)   # [B*h, Lq, dim_split]
        K_ = torch.cat(K.split(dim_split, dim=2), dim=0)   # [B*h, Lk, dim_split]
        V_ = torch.cat(V.split(dim_split, dim=2), dim=0)   # [B*h, Lk, dim_split]

        attn_logits = Q_.bmm(K_.transpose(1, 2)) / scale   # [B*h, Lq, Lk]

        if key_padding_mask is not None:
            # mask shape -> [B, 1, Lk] -> expand to [B*h, Lq, Lk]
            m = (~key_padding_mask).unsqueeze(1).to(attn_logits.dtype)  # 1 where pad
            m = m.repeat(self.num_heads, 1, 1)                          # [B*h, 1, Lk]
            attn_logits = attn_logits.masked_fill(m.bool(), float('-inf'))

        A = torch.softmax(attn_logits, dim=2)                            # [B*h, Lq, Lk]
        O_ = Q_ + A.bmm(V_)                                              # residual inside head
        O = torch.cat(O_.split(B, dim=0), dim=2)                         # [B, Lq, dim_V]

        O = O if not hasattr(self, 'ln0') else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if not hasattr(self, 'ln1') else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
    

class FusedMemoryBuilder(nn.Module):
    """
    Builds a small cross-graph memory:
      S_all = concat_g PMA_k(Z_g)
      S_mem = rFF( SAB(S_all) )           # or ISAB_m(S_all) if use_isab=True
    Returns: S_mem [B, K*k, D], valid_mem [B, K*k] (all True), gid_mem [B, K*k] or None
    """
    def __init__(self, dim, num_heads, k_seeds=4, use_isab=False, m_inds=128, ln=False, num_layers: int = 1):
        super().__init__()
        self.dim = dim
        self.k = k_seeds
        self.use_isab = use_isab
        self.num_layers = num_layers

        self.rff_in  = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.rff_out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

        self.pma = PMA(dim=dim, num_heads=num_heads, num_seeds=k_seeds, ln=ln)
        
        self.fusion_layers = nn.ModuleList()
        for _ in range(num_layers):
            attn_block = (
                ISAB(dim_in=dim, dim_out=dim, num_heads=num_heads, num_inds=m_inds, ln=ln)
                if use_isab else
                SAB(dim_in=dim, dim_out=dim, num_heads=num_heads, ln=ln)
            )
            ffn = nn.Sequential(
                nn.Linear(dim, int(dim * 4 / 3)),
                nn.GELU(),
                nn.Linear(int(dim * 4 / 3), dim)
            )
            self.fusion_layers.append(nn.ModuleList([attn_block, ffn]))

    @torch.no_grad()
    def _build_key_masks(self, padded_mask_list):
        # Convert your per-graph [B, M] masks into per-graph key masks (True=valid).
        return padded_mask_list

    def forward(self, padded_list, padded_mask_list, return_gid=False):
        """
        padded_list:      list of [B, M, D] tensors (one per graph)
        padded_mask_list: list of [B, M] bool tensors (True=valid)
        return_gid:       if True, also returns per-summary graph ids
        
        Returns:
            S_mem:    [B, K*k, D] fused memory
            valid_mem:[B, K*k]   all True
            gid_mem:  [B, K*k] or None
        """
        B, _, _ = padded_list[0].shape
        S_chunks = []
        gid_chunks = []

        # Summarize each graph with PMA_k over rFF(Z_g) using key masks
        for g, (H_g, mask_g) in enumerate(zip(padded_list, padded_mask_list)):
            Z_g = self.rff_in(H_g)                                      # [B, M, D]
            # PMA: queries = seeds, keys/values = tokens
            # pass key_padding_mask so padded tokens don't receive attention
            if mask_g is None:
                S_g = self.pma(Z_g)
            else:    
                S_g = self.pma.mab(self.pma.S.repeat(B,1,1), Z_g, key_padding_mask=mask_g)

            S_chunks.append(S_g)                                        # [B, k, D]
            if return_gid:
                gid_chunks.append(torch.full((B, self.k), g, dtype=torch.long, device=H_g.device))

        S_all = torch.cat(S_chunks, dim=1)                               # [B, K*k, D]
        Z = S_all
        # Multi-layer fusion
        for attn, ffn in self.fusion_layers:
            Z_res = attn(Z)                                       # [B, K*k, D]
            Z = Z + ffn(Z_res)                                    # [B, K*k, D]

        S_mem = self.rff_out(Z)
        valid_mem = torch.ones(B, S_mem.size(1), dtype=torch.bool, device=S_mem.device)

        if return_gid:
            gid_mem = torch.cat(gid_chunks, dim=1)                       # [B, K*k]
            return S_mem, valid_mem, gid_mem
        return S_mem, valid_mem, None
