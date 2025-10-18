import math
import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import pylibmgm
import pylibmgm.solver
from dataclasses import dataclass
@dataclass
class MGMConfig:
    """Configuration for MGM solver and synchronization.

    Attributes:
        func: cost transform function ("logit", "cosine", "atanh", "logsig", "softmax")
        tau:  temperature used for softmax/other transforms
        sync: whether to run solution synchronization after solving
        opt_level: pylibmgm optimization level
    """
    func: str = "logit"
    tau: float = 0.05
    sync: bool = False
    opt_level: pylibmgm.solver.OptimizationLevel = pylibmgm.solver.OptimizationLevel.DEFAULT


def sinkhorn_logspace(
    similarity: torch.Tensor,
    epsilon: float = 0.035,
    max_iter: int = 27,
) -> torch.Tensor:
    """Log-space Sinkhorn to convert a batch of similarity matrices into doubly-stochastic matrices.

    Args:
        similarity: [B, N, M] similarities
        epsilon: entropic regularization
        max_iter: number of iterations
    Returns:
        [B, N, M] doubly-stochastic matrix
    """
    log_Q = similarity / epsilon
    for _ in range(max_iter):
        log_Q = log_Q - torch.logsumexp(log_Q, dim=2, keepdim=True)
        log_Q = log_Q - torch.logsumexp(log_Q, dim=1, keepdim=True)
    return torch.exp(log_Q)


def hard_perm_from_sink(P_sink: torch.Tensor) -> torch.Tensor:
    """Project a soft assignment onto a hard permutation via Hungarian.

    Args:
        P_sink: [B, N, M] soft matrix
    Returns:
        [B, N, M] hard permutation (0/1)
    """
    B, N, M = P_sink.shape
    P_pred = torch.zeros_like(P_sink)
    cost = -P_sink.detach().cpu().numpy()
    for b in range(B):
        r, c = linear_sum_assignment(cost[b])
        P_pred[b, r, c] = 1
    return P_pred.to(P_sink.device)


def symmetric_softmax_costs(Fi: torch.Tensor, Fj: torch.Tensor, tau: float = 0.07, eps: float = 1e-9) -> torch.Tensor:
    """Build a symmetric probability and convert to costs.

    Fi: [ni, d], Fj: [nj, d]
    Returns: [ni, nj] costs
    """
    S = Fi @ Fj.t()
    P_row = torch.softmax(tau * S, dim=1)
    P_col = torch.softmax(tau * S.T, dim=1).T
    P_sym = 0.5 * (P_row + P_col)
    C = -torch.log((1 + P_sym) / (1 - P_sym).clamp_min(eps))
    return C


def mgm_model_synchronizing(
    sim_matrices: List[torch.Tensor],
    n_points_gt_list: List[torch.Tensor],
    pairs: List[Tuple[int, int]],
    batch_idx: int,
    embeds: List[torch.Tensor],
    sync: bool = False,
    func: str = "logit",
    tau: float = 0.15,
    mgm_config: MGMConfig | None = None,
):
    """Solve MGM for one batch index and return list of (i,j, match_matrix_ij).

    Args mirror the previous inline implementation in eval.py.
    """
    alpha = 2.0
    eps = 1e-8

    mgm_model = pylibmgm.MgmModel()

    # Resolve configuration precedence: explicit args override config fields if provided.
    if mgm_config is not None:
        sync = mgm_config.sync
        func = mgm_config.func
        tau = mgm_config.tau

    for idx, (i, j) in enumerate(pairs):
        ni = int(n_points_gt_list[i][batch_idx].item())
        nj = int(n_points_gt_list[j][batch_idx].item())

        if func == "softmax":
            Fi = embeds[i][batch_idx]
            Fj = embeds[j][batch_idx]
            mat_np = symmetric_softmax_costs(Fi, Fj, tau=tau, eps=eps)
        else:
            S = sim_matrices[idx][batch_idx][:ni, :nj]
            mu = S.mean(-1, keepdim=True)
            sigma = S.std(-1, keepdim=True) + 1e-8
            S_z = (S - mu) / sigma
            S_np = S_z.cpu().numpy()

            mat_np = np.empty_like(S_np)
            for u in range(ni):
                for v in range(nj):
                    s = float(S_np[u, v])
                    if func == "cosine":
                        cost = -alpha * s
                    elif func == "logit":
                        x = max(-1 + eps, min(1 - eps, alpha * s))
                        cost = -math.log((1 + x) / (1 - x))
                    elif func == "atanh":
                        x = max(-1 + eps, min(1 - eps, alpha * s))
                        cost = -math.atanh(x)
                    elif func == "logsig":
                        cost = F.softplus(-alpha * torch.tensor(s)).item()
                    else:
                        raise ValueError(f"Unknown func '{func}'")
                    mat_np[u, v] = cost

        gm = pylibmgm.GmModel(
            pylibmgm.Graph(i, ni),
            pylibmgm.Graph(j, nj),
            ni * nj,
            0,
        )

        for u in range(ni):
            for v in range(nj):
                gm.add_assignment(u, v, float(mat_np[u, v]) - 100)

        mgm_model.add_model(gm)

    logger = logging.getLogger("libmgm")
    logger.setLevel(logging.WARNING)

    # Choose opt level
    opt_level = (mgm_config.opt_level if mgm_config is not None else pylibmgm.solver.OptimizationLevel.DEFAULT)
    sol = pylibmgm.solver.solve_mgm(mgm_model, opt_level=opt_level)

    if sync:
        sol = pylibmgm.solver.synchronize_solution(
            mgm_model,
            sol,
            feasible=True,
            iterations=3,
            opt_level=opt_level,
        )

    out = []
    for (i, j) in pairs:
        labels = sol[(i, j)]
        ni = int(n_points_gt_list[i][batch_idx].item())
        nj = int(n_points_gt_list[j][batch_idx].item())
        M = np.zeros((ni, nj), dtype=np.int32)
        for u, v in enumerate(labels):
            if 0 <= v < nj:
                M[u, v] = 1
        out.append((i, j, M))
    return out
