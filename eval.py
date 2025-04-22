import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist  # Add this import
from scipy.optimize import linear_sum_assignment
from itertools import combinations

from utils.config import cfg
from utils.evaluation_metric import calculate_correct_and_valid, calculate_f1_score
from scipy.linalg import eigh

def sinkhorn_logspace(
    similarity: torch.Tensor,
    epsilon: float = 0.035,
    max_iter: int = 27
) -> torch.Tensor:
    """
    Log-space Sinkhorn to convert a batch of similarity matrices into 
    doubly-stochastic matrices.
    
    Args:
        similarity: [batch_size, n, m] matrix of similarities
        epsilon:    Entropic regularization (larger => smoother distribution)
        max_iter:   Number of Sinkhorn iterations in log domain
    
    Returns:
        [batch_size, n, m] doubly-stochastic matrix
    """
    log_Q = similarity / epsilon

    for _ in range(max_iter):
        log_sum_rows = torch.logsumexp(log_Q, dim=2, keepdim=True)
        log_Q = log_Q - log_sum_rows

        log_sum_cols = torch.logsumexp(log_Q, dim=1, keepdim=True)
        # broadcast subtraction
        log_Q = log_Q - log_sum_cols

    Q = torch.exp(log_Q)
    return Q

def permutation_synchronization(pred_perm_mats, pairs):
    """
    pred_perm_mats: List[L] von Tensoren (B x Ni x Nj)
    pairs:         List[L] von Tupeln (i,j) für K Graphen
    return:        List[B] von Dict[(i,j) -> P_syn (Ni x Nj)]
    """
    B = pred_perm_mats[0].size(0)
    # for each graph get K and Ni 
    nodes = set()
    for i,j in pairs:
        nodes.add(i); nodes.add(j)
    K = max(nodes) + 1

    N = [None]*K
    for idx, (i,j) in enumerate(pairs):
        _, Ni, Nj = pred_perm_mats[idx].shape
        if N[i] is None: N[i] = Ni
        if N[j] is None: N[j] = Nj

    offsets = [0]*K
    for i in range(1, K):
        offsets[i] = offsets[i-1] + N[i-1]
    Ntot = sum(N)

    synced_batches = [dict() for _ in range(B)]

    for b in range(B):
        H = torch.zeros(Ntot, Ntot, device=pred_perm_mats[0].device)
        # diag
        for i in range(K):
            o = offsets[i]
            H[o:o+N[i], o:o+N[i]] = torch.eye(N[i], device=H.device)
        # Off-diag
        for idx, (i,j) in enumerate(pairs):
            oi, oj = offsets[i], offsets[j]
            Pij = pred_perm_mats[idx][b]
            H[oi:oi+N[i], oj:oj+N[j]] = Pij
            H[oj:oj+N[j], oi:oi+N[i]] = Pij.t()

        # spectral decomposition 
        H_sym = (H + H.transpose(0, 1)) / 2
        d = max(N)
        # take biggest d eigenvectors
        try:
            e_vals, e_vecs = torch.linalg.eigh(H_sym)
            U = e_vecs[:, -d:]    # Ntot x d
        except RuntimeError:
            # Fallback auf CPU + NumPy
            H_cpu = H_sym.cpu().numpy()
            H_cpu = (H_cpu + H_cpu.T) / 2
            vals, vecs = np.linalg.eigh(H_cpu)
            U = torch.from_numpy(vecs[:, -d:]).to(H_sym.device)

        # 3) Für jedes Paar (i,j) Hungarian
        for (i,j) in pairs:
            oi, oj = offsets[i], offsets[j]
            Ui = U[oi:oi+N[i], :]   # Ni x d
            Uj = U[oj:oj+N[j], :]   # Nj x d
            S = (Ui @ Uj.t()).cpu().numpy()  # Ni x Nj

            # Hungarian max: cost = -S
            row_ind, col_ind = linear_sum_assignment(-S)
            P_syn = torch.zeros(N[i], N[j], device=H.device)
            P_syn[row_ind, col_ind] = 1.0
            synced_batches[b][(i,j)] = P_syn

    return synced_batches

@torch.no_grad()
def check_cycle_consistency(pred_perm_mats,          # List[len(pairs)] of [B, Ni, Nj] tensors
                            pairs,                   # same order as pred_perm_mats
                            n_points_gt,             # list of K tensors [B]  (for masking padded rows)
                            atol=1e-6):              # tolerance for equality test
    """
    Checks 3‑cycle consistency for every batch sample.

    For each triple (i, j, k) we expect:
          P_ij · P_jk  ≈  P_ik

    Counts how many source keypoints violate this equality.

    Returns
    -------
    cycle_err   : int        total #inconsistent keypoints in the batch
    cycle_total : int        total #checked keypoints in the batch
    """

    # Build dict[(i,j)] -> tensor[B, Ni, Nj]  for faster lookup
    Pdict = {pair: P for pair, P in zip(pairs, pred_perm_mats)}

    K = len(n_points_gt)                 # number of graphs in the instance
    triples = list(combinations(range(K), 3))
    B = pred_perm_mats[0].size(0)

    cycle_err = 0
    cycle_tot = 0

    for (i, j, k) in triples:            # all 3‑cycles
        P_ij = Pdict[(i, j)]             # [B, Ni, Nj]
        P_jk = Pdict[(j, k)]             # [B, Nj, Nk]
        P_ik = Pdict[(i, k)]             # [B, Ni, Nk]

        # Matrix product in parallel over batch
        composed = torch.bmm(P_ij, P_jk)            # [B, Ni, Nk]

        # Hard 0/1 comparison -> difference
        diff = (composed - P_ik).abs() > atol       # Bool tensor

        # Mask padded rows ( > ns[i] )
        Ni = P_ij.size(1)
        for b in range(B):
            valid_rows = int(n_points_gt[i][b].item())
            diff[b, valid_rows:, :] = False         # ignore padded

        cycle_err += diff.sum().item()
        cycle_tot += diff.numel() - diff[:,Ni:, :].numel()  # only valid rows

    return cycle_err, cycle_tot

def eval_model(model, dataloader, local_rank, output_rank, eval_epoch=None, verbose=True):
    print("Start evaluation...")
    since = time.time()

    device = next(model.parameters()).device

    if eval_epoch is not None:
        model_path = str(Path(cfg.model_dir) / "params" / "{:04}".format(eval_epoch) / "params.pt")
        if local_rank == output_rank:
            print("Loading model parameters from {}".format(model_path))
        model.load_state_dict(torch.load(model_path))

    was_training = model.training
    model.eval()

    ds = dataloader.dataset
    K = cfg.EVAL.num_graphs_in_matching_instance
    ds.set_num_graphs(K)
    classes = ds.classes
    cls_cache = ds.cls

    accs = torch.zeros(len(classes), device=device)
    #f1_scores = torch.zeros(len(classes), device=device)
    error_dist_dict = {}
    
    pairs = list(combinations(range(K), 2))
    
    for cls_inx, cls in enumerate(classes):
        if local_rank == output_rank:
            if verbose:
                print("Evaluating class {}: {}/{}".format(cls, cls_inx, len(classes)))

        running_since = time.time()
        iter_num = 0
        ds.set_cls(cls)
        
        result_dict = {}
        tp = 0
        fp = 0
        fn = 0
        epoch_f1 = 0.0
        epoch_correct = 0
        epoch_total_valid = 0
        for k, inputs in enumerate(dataloader, 1):
            iter_num = iter_num + 1
            data_list = [_.cuda() for _ in inputs["images"]]
            points_gt = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt = [_.cuda() for _ in inputs["ns"]]
            edges = [_.to("cuda") for _ in inputs["edges"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]

            batch_num = data_list[0].size(0)
            #num_nodes_s = points_gt[0].size(1)
            #num_nodes_t = points_gt[1].size(1)
            
            pred_perm_mats = []
            
            for idx, (g_i,g_j) in enumerate(pairs):
                imgs_pair  = [data_list[g_i], data_list[g_j]]
                pts_pair   = [points_gt[g_i], points_gt[g_j]]
                edges_pair = [edges[g_i], edges[g_j]]
                n_points_gt_pair = [n_points_gt[g_i], n_points_gt[g_j]]
                gt_pair = [perm_mat_list[idx]]
                
                # Debug: Shapes vor forward
                #print(f"\n--- DEBUG PAIR {g_i},{g_j} ---")
                #print(" imgs:", [x.shape for x in imgs_pair])
                #print(" pts:",  [x.shape for x in pts_pair])
                #print(" ns:",   [x.shape for x in n_points_gt_pair])
                #print(" gt_pm:", [x.shape for x in gt_pair])
                
                with torch.no_grad():
                    sim, _, _, _ = model(
                        images            = imgs_pair,
                        points            = pts_pair,
                        graphs            = edges_pair,
                        n_points          = n_points_gt_pair,
                        n_points_sample   = n_points_gt_pair[0],
                        perm_mats         = [perm_mat_list[idx]],
                        eval_pred_points  = None,
                        in_training       = False,
                    )
                
                P_sink = sinkhorn_logspace(sim) # [B, Ni, Nj]
                N_s, N_t = P_sink.size(1), P_sink.size(2)
                
                # Row-wise argmax
                sink_max = torch.argmax(P_sink, dim=-1)  # [B, N_s]

                # Baue harte Permutationsmatrix per Batch
                P_pred = torch.zeros(batch_num, N_s, N_t, device=device)
                for b in range(batch_num):
                    for src in range(N_s):
                        if src < n_points_gt[g_i][b]:
                            tgt = sink_max[b, src].item()
                            P_pred[b, src, tgt] = 1
                pred_perm_mats.append(P_pred)

            # --- 4) Synchronisation falls K>2 ---
            #if K > 2:
            #    synced = permutation_synchronization(pred_perm_mats, pairs)
            #else:
            synced = [
                { pairs[idx]: pred_perm_mats[idx][b] for idx in range(len(pairs)) }
                for b in range(batch_num)
            ]
            
            # --- 5) Accuracy & Error-Distribution ---
            for p_idx, (g_i, g_j) in enumerate(pairs):
                P_gt = perm_mat_list[p_idx]
                for b in range(batch_num):
                    P_syn = synced[b][(g_i, g_j)]
                    valid = (P_gt[b].sum(dim=1) > 0)
                    if valid.sum().item() == 0:
                        continue

                    pred_inds = P_syn.argmax(dim=1).unsqueeze(0)  # [1, N_s]
                    gt_inds   = P_gt[b].argmax(dim=1).unsqueeze(0)
                    batch_corr, batch_valid = calculate_correct_and_valid(pred_inds, gt_inds)

                    epoch_correct     += batch_corr
                    epoch_total_valid += batch_valid

                    # Fehler-Statistik pro n_keypoints von Graph i
                    n_pts = int(n_points_gt[g_i][b].item())
                    err   = (pred_inds[0,:n_pts] != gt_inds[0,:n_pts]).int()
                    if n_pts not in result_dict:
                        result_dict[n_pts] = [0, torch.zeros(n_pts, device=device)]
                    result_dict[n_pts][0] += 1
                    result_dict[n_pts][1] += err

            #cycle_err, cycle_tot = check_cycle_consistency(
            #    pred_perm_mats=pred_perm_mats,
            #    pairs=pairs,
            #    n_points_gt=n_points_gt,
            #    atol=1e-8
            #)
            #if verbose and local_rank == output_rank:
            #    pct = 100 * cycle_err / max(1, cycle_tot)
            #    print(f"[Cycle‑Chk]  batch {iter_num:<4}  "
            #          f"inconsistent: {cycle_err}/{cycle_tot}  "
            #          f"({pct:5.2f} %)")
            
            if iter_num % 40 == 0 and verbose: #cfg.STATISTIC_STEP
                running_speed = 40 * batch_num / (time.time() - running_since) #cfg.STATISTIC_STEP
                print("Class {:<8} Iteration {:<4} {:>4.2f}sample/s".format(cls, iter_num, running_speed))
                running_since = time.time()
        
        
        dataset_size = len(dataloader.dataset)
        
        if epoch_total_valid > 0:
            epoch_acc = epoch_correct / epoch_total_valid
        else:
            epoch_acc = 0.0

        # precision_global = tp / (tp + fp + 1e-8)
        # recall_global = tp / (tp + fn + 1e-8)
        
        # Global F1 score
        # epoch_f1 = 2 * (precision_global * recall_global) / (precision_global + recall_global + 1e-8)
        
        accs[cls_inx] = epoch_acc
        # f1_scores[i] = epoch_f1
        if verbose:
            print("Class {} acc = {:.4f}".format(cls, accs[cls_inx]))
            
        error_dist_dict[cls] = result_dict
        
    # print(error_dist_dict)
    time_elapsed = time.time() - since
    print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)
    ds.cls = cls_cache

    print("Matching accuracy")
    for cls, single_acc in zip(classes, accs):
        print("{} = {:.4f}".format(cls, single_acc))
    print("average = {:.4f}".format(torch.mean(accs)))

    return accs, error_dist_dict