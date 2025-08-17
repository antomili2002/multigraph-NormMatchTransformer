import time
import pylibmgm.solver
import wandb
from pathlib import Path
import numpy as np
import torch
import math
import pylibmgm
import logging
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.distributed as dist  # Add this import
from scipy.optimize import linear_sum_assignment
from itertools import combinations

from utils.config import cfg
from utils.evaluation_metric import calculate_correct_and_valid, matching_accuracy_from_lists, perm_distance_masked
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

def hard_perm_from_sink(P_sink: torch.Tensor) -> torch.Tensor:
    """
    P_sink: Tensor [B, N, N]
    returns P_pred: Tensor [B, N, N] each a valid permutation
    """
    B, N, T = P_sink.shape
    P_pred = torch.zeros_like(P_sink)
    cost = -P_sink.detach().cpu().numpy() # maximize P_sink
    for b in range(B):
        r, c = linear_sum_assignment(cost[b])
        P_pred[b, r, c] = 1
    return P_pred.to(P_sink.device)

def symmetric_softmax_costs(Fi: torch.Tensor,
                            Fj: torch.Tensor,
                            tau: float = 0.07,
                            eps: float = 1e-9) -> torch.Tensor:
    """
    Fi: [ni, d] embeddings for graph i
    Fj: [nj, d] embeddings for graph j
    Return: cost matrix C [ni, nj] where C = -log( 0.5*(softmax_row + softmax_col) )
    """
    S = Fi @ Fj.t()                      # [ni, nj]

    # row-wise and column-wise softmax
    P_row = torch.softmax(tau * S, dim=1)            # row dist
    P_col = torch.softmax(tau * S.T, dim=1).T            # column dist

    P_sym = 0.5 * (P_row + P_col)               # symmetrical probabilities
    #P_sym = P_sym.clamp_min(eps).clamp_max(1 - eps)

    # c = -log((1 + p) / (1 - p))
    C = -torch.log( (1 + P_sym) / (1 - P_sym) )
    return C

def mgm_model_synchronizing(
    sim_matrices,         # list of pairwise similarity tensors
    n_points_gt_list,     # list of [B] #nodes
    pairs,                # list of (i,j)
    batch_idx: int,
    embeds: list,         # list of [B, Ni, D] embedding tensors
    sync: bool = False,
    func: str = "logit",  # "cosine" | "logit" | "atanh" | "logsig" | "softmax"
    tau: float = 0.15,
):
    """Return list[(i,j,match_mat_ij)] for a single Batch index."""
    alpha = 2.0
    eps   = 1e-8

    mgm_model = pylibmgm.MgmModel()

    for idx, (i, j) in enumerate(pairs):
        ni = int(n_points_gt_list[i][batch_idx].item())
        nj = int(n_points_gt_list[j][batch_idx].item())

        if func == "softmax":
            Fi = embeds[i][batch_idx]      # [ni, D]
            Fj = embeds[j][batch_idx]      # [nj, D]
            mat_np = symmetric_softmax_costs(Fi, Fj, tau=tau, eps=eps)  # [ni, nj]
        else:
            # start from cosine similarities
            S  = sim_matrices[idx][batch_idx][:ni, :nj]      # [ni, nj]
            mu = S.mean(-1, keepdim=True)
            sigma = S.std(-1, keepdim=True) + 1e-8
            S_z   = (S - mu) / sigma                         # z‑score per row
            S_np  = S_z.cpu().numpy()

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
            ni * nj,           # #unary
            0                  # no quadratic edges yet
        )

        for u in range(ni):
            for v in range(nj):
                gm.add_assignment(u, v, float(mat_np[u, v]) - 100)

        mgm_model.add_model(gm)
        
    sol = pylibmgm.solver.solve_mgm(
        mgm_model,
        opt_level=pylibmgm.solver.OptimizationLevel.DEFAULT
    )

    if sync:
        sol = pylibmgm.solver.synchronize_solution(
            mgm_model, sol, feasible=True, iterations=3,
            opt_level=pylibmgm.solver.OptimizationLevel.DEFAULT
        )
        
    out = []
    for (i, j) in pairs:
        labels = sol[(i, j)]
        ni = int(n_points_gt_list[i][batch_idx].item())
        nj = int(n_points_gt_list[j][batch_idx].item())
        M  = np.zeros((ni, nj), dtype=np.int32)
        for u, v in enumerate(labels):
            if 0 <= v < nj:
                M[u, v] = 1
        out.append((i, j, M))
    return out

def f1_counts_from_row_argmax(pred_idx: torch.Tensor, gt_idx: torch.Tensor):
    """
    pred_idx, gt_idx: [ni, 1] long tensors with the predicted/true column index per row.
    Returns TP, FP, FN as int64 tensors on the same device.

    With one prediction & one GT per row:
      - TP = #rows where pred == gt
      - Every mistake creates one FP and one FN
    """
    correct = (pred_idx == gt_idx).squeeze(1)
    tp = correct.sum().to(torch.int64)
    wrong = (~correct).sum().to(torch.int64)
    fp = wrong
    fn = wrong
    return tp, fp, fn

def f1_from_counts(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    eps = torch.tensor(1e-8, device=tp.device, dtype=torch.float32)
    tp = tp.to(torch.float32); fp = fp.to(torch.float32); fn = fn.to(torch.float32)
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    return 2 * prec * rec / (prec + rec + eps)

def f1_counts_from_perm(P_pred: torch.Tensor, P_gt: torch.Tensor):
    # P_pred, P_gt: [ni, nj], binary (0/1) – or threshold P_pred if soft.
    pred = (P_pred > 0.5).to(torch.int64)
    gt   = (P_gt   > 0.5).to(torch.int64)

    tp = (pred & gt).sum().to(torch.int64)
    fp = (pred.sum() - tp).to(torch.int64)
    fn = (gt.sum()   - tp).to(torch.int64)
    return tp, fp, fn

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

    accs_pre_sync = torch.zeros(len(classes), device=device)
    accs_post_sync = torch.zeros(len(classes), device=device)
    
    f1_scores_pre = torch.zeros(len(classes), device=device)
    f1_scores_post = torch.zeros(len(classes), device=device)
    
    tp_pre_global  = torch.tensor(0, dtype=torch.int64, device=device)
    fp_pre_global  = torch.tensor(0, dtype=torch.int64, device=device)
    fn_pre_global  = torch.tensor(0, dtype=torch.int64, device=device)
    tp_post_global = torch.tensor(0, dtype=torch.int64, device=device)
    fp_post_global = torch.tensor(0, dtype=torch.int64, device=device)
    fn_post_global = torch.tensor(0, dtype=torch.int64, device=device)
    
    error_dist_dict = {}
    pairs = list(combinations(range(K), 2))
    
    for cls_inx, cls in enumerate(classes):
        if local_rank == output_rank:
            if verbose:
                print("Evaluating class {}: {}/{}".format(cls, cls_inx, len(classes)))

        running_since = time.time()
        iter_num = 0
        ds.set_cls(cls)
        
        sum_pre_acc = 0.0    # sum over batches of (correct_pre/valid_pre)
        sum_post_acc = 0.0   # sum over batches of (correct_post/valid_post)
        
        tp_pre_cls  = torch.tensor(0, dtype=torch.int64, device=device)
        fp_pre_cls  = torch.tensor(0, dtype=torch.int64, device=device)
        fn_pre_cls  = torch.tensor(0, dtype=torch.int64, device=device)
        tp_post_cls = torch.tensor(0, dtype=torch.int64, device=device)
        fp_post_cls = torch.tensor(0, dtype=torch.int64, device=device)
        fn_post_cls = torch.tensor(0, dtype=torch.int64, device=device)
        
        result_dict = {}
        num_batches = 0

        for k, inputs in enumerate(dataloader, 1):
            iter_num = iter_num + 1
            data_list = [_.cuda() for _ in inputs["images"]]
            points_gt = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt = [_.cuda() for _ in inputs["ns"]]
            edges = [_.to("cuda") for _ in inputs["edges"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]

            batch_num = data_list[0].size(0)    # batch size
            
            with torch.no_grad():
                sim_list, embeds, _ = model(
                    images            = data_list,
                    points            = points_gt,
                    graphs            = edges,          # list of K Data
                    n_points          = n_points_gt,
                    n_points_sample   = n_points_gt[0],
                    perm_mats         = perm_mat_list,
                    in_training       = False,
                )

            pred_perm_mats = []
            for sim in sim_list:
                P_sink = sinkhorn_logspace(sim)      # [B, Mi_max, Mj_max]
                P_pred = hard_perm_from_sink(P_sink) # [B, Mi_max, Mj_max]
                pred_perm_mats.append(P_pred)
            
            # slice out invalid padding of perm_mats and pred_mats
            # Accumulate correct vs valid counts for pre-sync
            idx_c = 0
            correct_pre, valid_pre = 0.0, 0.0
            for g_i in range(K):
                n_i = n_points_gt[g_i]
                for g_j in range(g_i+1, K):
                    n_j = n_points_gt[g_j]
                    for b in range(batch_num):
                        ni, nj = n_i[b].item(), n_j[b].item()
                        Pp = pred_perm_mats[idx_c][b, :ni, :nj]       # [ni×nj]
                        Pg = perm_mat_list[idx_c][b, :ni, :nj]       # [ni×nj]
                        # row-wise argmax → predict & gt indices
                        pred_idx = Pp.argmax(dim=1, keepdim=True)  # [ni,1]
                        gt_idx   = Pg.argmax(dim=1, keepdim=True)  # [ni,1]
                        c, v = calculate_correct_and_valid(pred_idx, gt_idx)
                        correct_pre += c
                        valid_pre   += v
                        
                        tp_, fp_, fn_ = f1_counts_from_perm(Pp, Pg)
                        tp_pre_cls  += tp_; fp_pre_cls  += fp_; fn_pre_cls  += fn_
                        tp_pre_global += tp_; fp_pre_global += fp_; fn_pre_global += fn_
                        
                    idx_c += 1
            pre_acc_batch = (correct_pre / valid_pre)
            sum_pre_acc += pre_acc_batch
            
            correct_post = 0.0
            valid_post   = 0.0
            for b in range(batch_num):
                graph_sizes_b = [n_points_gt[g][b].item() for g in range(K)] # graph-sizes for this example
                sim_list_b = []
                for idx, (i, j) in enumerate(pairs):
                    ni, nj = graph_sizes_b[i], graph_sizes_b[j]
                    mat_b_full = sim_list[idx][b]
                    mat_b = mat_b_full[:ni, :nj].detach().cpu().numpy()
                    sim_list_b.append(mat_b.astype(np.float64))
                
                logger = logging.getLogger("libmgm")
                logger.setLevel(logging.WARNING) # set log level of mgm model
                
                # create MgmModel, solve mgm and extract labelings
                all_matches_b = mgm_model_synchronizing(
                    sim_matrices     = sim_list,
                    n_points_gt_list = n_points_gt,
                    pairs            = pairs,
                    batch_idx        = b,
                    embeds           = embeds,   
                    sync             = False,     
                    func             = "logit",
                    tau              = 0.05
                )
                
                # accumulate post-sync just like pre-sync
                for idx, (g_i, g_j) in enumerate(pairs):
                    ni, nj = graph_sizes_b[g_i], graph_sizes_b[g_j]
                    
                    _, _, Ps = all_matches_b[idx]
                    Pg = perm_mat_list[idx][b, :ni, :nj]
                    P_post = torch.Tensor(Ps).to(device)
                    
                    pred_idx = P_post.argmax(dim=1, keepdim=True)
                    gt_idx   = Pg.argmax(dim=1, keepdim=True)
                    c, v = calculate_correct_and_valid(pred_idx, gt_idx)
                    correct_post += c
                    valid_post += v

                    tp_, fp_, fn_ = f1_counts_from_perm(P_post, Pg)
                    tp_post_cls  += tp_; fp_post_cls  += fp_; fn_post_cls  += fn_
                    tp_post_global += tp_; fp_post_global += fp_; fn_post_global += fn_
            
            
                for idx, (g_i, g_j) in enumerate(pairs):
                    ni = graph_sizes_b[g_i]
                    _, _, Ppost_np = all_matches_b[idx]  # shape (ni, nj)
                    pred_idx = torch.Tensor(Ppost_np.argmax(axis=1).reshape(ni, 1)).to(device)
                    Pg = perm_mat_list[idx][b, :ni, :nj]
                    gt_idx = Pg.argmax(dim=1, keepdim=True)  # [ni,1]

                    err_vec = (pred_idx.squeeze(1) != gt_idx.squeeze(1)).int()
                    n_pts = ni
                    if n_pts not in result_dict:
                        result_dict[n_pts] = [0, torch.zeros(n_pts, device=device, dtype=torch.int32)]
                    result_dict[n_pts][0] += 1
                    result_dict[n_pts][1] += err_vec
            
            num_batches += 1
            post_acc_batch = (correct_post / valid_post)
            sum_post_acc += post_acc_batch
            
            # progress print every 40 batches
            if iter_num % 40 == 0 and verbose: #cfg.STATISTIC_STEP
                running_speed = 40 * batch_num / (time.time() - running_since) #cfg.STATISTIC_STEP
                print("Class {:<8} Iteration {:<4} {:>4.2f}sample/s".format(cls, iter_num, running_speed))
                running_since = time.time()
        
        acc_pre_cls  = sum_pre_acc  / iter_num
        acc_post_cls = sum_post_acc / iter_num
        
        accs_pre_sync[cls_inx] = acc_pre_cls
        accs_post_sync[cls_inx] = acc_post_cls
        
        f1_pre_cls  = f1_from_counts(tp_pre_cls,  fp_pre_cls,  fn_pre_cls)
        f1_post_cls = f1_from_counts(tp_post_cls, fp_post_cls, fn_post_cls)
        f1_scores_pre[cls_inx]  = f1_pre_cls
        f1_scores_post[cls_inx] = f1_post_cls
        
        # f1_scores[i] = epoch_f1
        if verbose:
            print(f"Class {cls} acc_pre_sync = {acc_pre_cls:.4f}, acc_post_sync = {acc_post_cls:.4f}")
            #print(f"Avg inconsistent 3-cycles before sync: {avg_bad_pre:.2f}, after sync: {avg_bad_post:.2f}")
            
        error_dist_dict[cls] = result_dict
        
    # print(error_dist_dict)
    time_elapsed = time.time() - since
    print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    # wandb logging
    avg_pre = torch.mean(accs_pre_sync).item()
    avg_post = torch.mean(accs_post_sync).item()
    
    f1_macro_pre  = torch.mean(f1_scores_pre).item()
    f1_macro_post = torch.mean(f1_scores_post).item()
    f1_micro_pre  = f1_from_counts(tp_pre_global,  fp_pre_global,  fn_pre_global).item()
    f1_micro_post = f1_from_counts(tp_post_global, fp_post_global, fn_post_global).item()
    
    if local_rank == output_rank:
        wandb.log({
                "eval/avg_pre_sync":  avg_pre,
                "eval/avg_post_sync": avg_post,
                "eval/time_s":       time_elapsed
        }, step=(eval_epoch or wandb.run.step))
        
        wandb.log({
            "eval/f1_micro_pre":  f1_micro_pre,
            "eval/f1_micro_post": f1_micro_post,
            "eval/f1_macro_pre":  f1_macro_pre,
            "eval/f1_macro_post": f1_macro_post,
        }, step=(eval_epoch or wandb.run.step))
        
        # log per-class accuracies too
        for cls, pre, post in zip(classes, accs_pre_sync, accs_post_sync):
            wandb.log({
                    f"eval/pre_sync/{cls}":  pre.item(),
                    f"eval/post_sync/{cls}": post.item()
            }, step=(eval_epoch or wandb.run.step))

    model.train(mode=was_training)
    ds.cls = cls_cache

    print("Matching accuracy")
    for cls, pre_acc, post_acc, f1pre, f1post in zip(classes, accs_pre_sync, accs_post_sync, f1_scores_pre, f1_scores_post):
        print("{}: pre sync = {:.4f}, post sync = {:.4f} | f1_pre = {:.4f}, f1_post = {:.4f}".format(cls, pre_acc, post_acc, f1pre, f1post))
    print("average pre sync = {:.4f}, average after sync = {:.4f}".format(torch.mean(accs_pre_sync), torch.mean(accs_post_sync)))
    print("Macro F1      pre = {:.4f}, post = {:.4f}".format(torch.mean(f1_scores_pre), torch.mean(f1_scores_post)))
    print("Micro (global) F1 pre = {:.4f}, post = {:.4f}".format(f1_micro_pre, f1_micro_post))

    return accs_pre_sync, accs_post_sync, error_dist_dict