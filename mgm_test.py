"""
K      : # graphs in the matching instance
ns     : list with number of keypoints in each graph
"""

import math
import numpy as np
import torch
import pylibmgm
from itertools import combinations

torch.manual_seed(0)

K   = 4                                      # graphs per instance
B   = 2                                      # batch size for test
ns  = [torch.randint(4, 9, (B,)) for _ in range(K)]  # [B] each

# build every pair (i,j) once, in lexicographic order
pairs = list(combinations(range(K), 2))      # [(0,1),(0,2),...]

sim_list = []                                # one tensor per pair
for (i, j) in pairs:
    # maximum sizes across batch (for padding)
    ni_max = ns[i].max().item()
    nj_max = ns[j].max().item()

    # random cosine-sim values in (-1,1)
    S_ij = 2*torch.rand(B, ni_max, nj_max) - 1
    sim_list.append(S_ij)

print(f"created {len(sim_list)} similarity matrices")


def mgm_from_sims(sim_mats, ns_list, pairs, b, parallel=False):
    mgm = pylibmgm.MgmModel()

    for idx, (i, j) in enumerate(pairs):
        S   = sim_mats[idx][b]                      # [Ni_max,Nj_max]
        ni  = int(ns_list[i][b])
        nj  = int(ns_list[j][b])

        gm  = pylibmgm.GmModel(
                pylibmgm.Graph(i, ni),
                pylibmgm.Graph(j, nj),
                ni*nj, 0)

        for u in range(ni):
            for v in range(nj):
                s = float(S[u, v].clamp(-0.999, 0.999))
                cost = -math.log((1+s)/(1-s))       # “logit” unary
                gm.add_assignment(u, v, cost)

        mgm.add_model(gm)

    if parallel:
        sol = pylibmgm.solver.solve_mgm_parallel(
                mgm, opt_level=pylibmgm.solver.OptimizationLevel.DEFAULT)
    else:
        sol = pylibmgm.solver.solve_mgm(
                mgm, opt_level=pylibmgm.solver.OptimizationLevel.DEFAULT)

    # collect permutation matrices for inspection
    perms = {}
    for (i, j) in pairs:
        lab      = sol[(i, j)]
        ni, nj   = int(ns_list[i][b]), int(ns_list[j][b])
        P        = np.zeros((ni, nj), dtype=np.int32)
        for u, v in enumerate(lab):
            if 0 <= v < nj:
                P[u, v] = 1
        perms[(i, j)] = P
    return perms

for b in range(B):
    P_serial  = mgm_from_sims(sim_list, ns, pairs, b, parallel=False)
    P_parallel= mgm_from_sims(sim_list, ns, pairs, b, parallel=True)

    print(f"P_serial: {P_serial}")
    print(f"P_parallel: {P_parallel}")
    ok = all((P_serial[p] == P_parallel[p]).all() for p in pairs)
    print(f"batch {b}: identical solutions? {ok}")
