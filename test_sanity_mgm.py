import numpy as np
import torch
import mgm_py
from scipy.optimize import linear_sum_assignment

def hungarian_matching(sim):
    """
    Given a similarity matrix (Ni x Nj), returns a binary 0/1 assignment matrix
    using the Hungarian algorithm. Maximizes total similarity.
    """
    # Convert to cost matrix by negating similarity
    cost_matrix = -sim
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matching = np.zeros_like(sim)
    matching[row_ind, col_ind] = 1
    return matching

def count_3cycles_from_numpy(all_matches, graph_sizes):
    """
    Count inconsistent 3-cycles in a set of pairwise match matrices.
    all_matches: list of 0/1 numpy arrays in order [(0,1), (0,2), ..., (K-2,K-1)]
    graph_sizes: list of integers [n0, n1, ..., n_{K-1}]
    """
    K = len(graph_sizes)
    pairs = [(i, j) for i in range(K) for j in range(i+1, K)]
    pair2mat = { (i, j): all_matches[p] for p, (i, j) in enumerate(pairs) }

    def get_P(i, j):
        return pair2mat[(i, j)] if i < j else pair2mat[(j, i)].T

    bad = 0
    for i in range(K):
        ni = graph_sizes[i]
        for j in range(i+1, K):
            for k in range(j+1, K):
                Pij = get_P(i, j)
                Pjk = get_P(j, k)
                Pki = get_P(k, i)
                for u in range(ni):
                    v = Pij[u].argmax()
                    w = Pjk[v].argmax()
                    if Pki[w, u] != 1:
                        bad += 1
    return bad

# Graph 0 has 4 nodes, Graph 1 has 3 nodes, Graph 2 has 5 nodes.
K = 3
graph_sizes = [4, 3, 5]   

sim_01 = np.random.rand(4, 3).astype(np.float64)
sim_02 = np.random.rand(4, 5).astype(np.float64)
sim_12 = np.random.rand(3, 5).astype(np.float64)

print("sim(0,1) =\n", sim_01)
print("sim(1,2) =\n", sim_12)
print("sim(0,2) =\n", sim_02)

gt_01 = hungarian_matching(sim_01)
gt_02 = hungarian_matching(sim_02)
gt_12 = hungarian_matching(sim_12)

print("Ground-truth matches (0,1):\n", gt_01)
print("Ground-truth matches (1,2):\n", gt_12)
print("Ground-truth matches (0,2):\n", gt_02)

# Manually inject a blatant cycle‐inconsistency for testing:
sim_01[0, :] = -1.0   
sim_01[0, 0] = +1.0
sim_12[0, :] = -1.0  
sim_12[0, 0] = +1.0
sim_02[0, :] = -1.0  
sim_02[0, 1] = +1.0

# Build the model and solve
mgm_model = mgm_py.build_mgm_model_from_similarity_tensors(
    [sim_01, sim_02, sim_12], 
    graph_sizes
) 
solution = mgm_py.run_mgm_model(
    mgm_model,
    mode="optimal",
    incremental_set_size=0,
    merge_one=False,
    nr_threads=1,
    libmpopt_seed=42
)

matches_01, matches_02, matches_12 = mgm_py.export_all_match_matrices(solution)

print("— After MGM —")
print("matches(0,1) =\n", matches_01)
print("matches(1,2) =\n", matches_12)
print("matches(0,2) =\n", matches_02)

print("Inconsistent-3-cycles (pre-MGM) =",
      count_3cycles_from_numpy([gt_01, gt_02, gt_12], graph_sizes))

print("Inconsistent-3-cycles (post-MGM) =",
      count_3cycles_from_numpy([matches_01, matches_02, matches_12], graph_sizes))
