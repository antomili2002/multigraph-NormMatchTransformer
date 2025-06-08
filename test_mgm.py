import numpy as np
import mgm_py
import torch

def main():
    graph_sizes = [4, 3, 5]
    sim_01 = np.random.rand(4,3).astype(np.float64)
    sim_02 = np.random.rand(4,5).astype(np.float64)
    sim_12 = np.random.rand(3,5).astype(np.float64)

    # build MgmModel from similarity tensors
    model = mgm_py.build_mgm_model_from_similarity_tensors(
        [sim_01, sim_02, sim_12],
        graph_sizes
    )

    # “optimal” solver
    sol = mgm_py.run_mgm_model(
        model,
        mode="optimal",
        incremental_set_size=0,
        merge_one=False,
        nr_threads=2,
        libmpopt_seed=12345
    )

    print("Total cost:", sol.evaluate())
    print("Cycle consistent:", sol.is_cycle_consistent())
    all_matches = mgm_py.export_all_match_matrices(sol)
    
    len_matrices = len(all_matches)
    
    for i in range(len_matrices):
        mat = all_matches[i]
        print(f"Matrix: {i}, shape: {mat.shape}, Matrix:\n {mat}")

if __name__ == "__main__":
    main()