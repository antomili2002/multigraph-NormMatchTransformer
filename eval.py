import time
import pylibmgm.solver
import wandb
from pathlib import Path
import numpy as np
import torch
import math
import pylibmgm
import logging
import warnings
import torch.nn.functional as F
import torch.distributed as dist  # Add this import
from scipy.optimize import linear_sum_assignment
from itertools import combinations

from utils.config import cfg
from utils.evaluation_metric import calculate_correct_and_valid, count_cycle_inconsistencies
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

def print_mgm(mgm: pylibmgm.MgmModel, *, max_costs_to_print: int = 20):
    """
    Pretty-print the content of an `MgmModel`.

    Parameters
    ----------
    mgm : pylibmgm.MgmModel
    max_costs_to_print : int
        For very large graphs, printing the full cost matrix can flood stdout.
        If either dimension > max_costs_to_print we print only the shape and
        a count of NaNs instead of the full matrix.
    """
    print("=" * 72)
    print(f"MgmModel  |  #graphs = {mgm.no_graphs}   #pair-models = {len(mgm.models)}")
    print("-" * 72)

    # show the individual Graph objects first
    for g in mgm.graphs:
        print(f"Graph id={g.id:<3}  |  #nodes = {g.no_nodes}")

    print("=" * 72)

    for (i, j), gm in sorted(mgm.models.items()):          # <-- idx_key is (id_i, id_j)
        ni, nj = gm.graph1.no_nodes, gm.graph2.no_nodes
        nA, nE = gm.no_assignments(), gm.no_edges()
        print(f"pair ({i},{j})  {ni}x{nj}  |  assignments={nA:<3}  edges={nE}")

        # rebuild unary-cost matrix
        M = np.full((ni, nj), np.nan)
        costmap = gm.costs()
        for (u, v) in gm.assignment_list:
            c = costmap.unary(u, v)
            M[u, v] = c

        big = max(ni, nj) > max_costs_to_print
        if big:
            nan_cnt = np.isnan(M).sum()
            print(f"cost-matrix shape={M.shape}, NaNs={nan_cnt}")
        else:
            with np.printoptions(precision=3, suppress=True):
                print(np.array2string(M, prefix="    "))
        print("-" * 80)

def mgm_model_synchronizing(
    sim_matrices,       # List[torch.Tensor], length C = K*(K-1)/2, each [B, Ni_max, Nj_max]
    n_points_gt_list,   # List[torch.Tensor], length K, each [B]
    pairs,              # List of (i,j) in lex order
    batch_idx,          # index of batch
    sync=False,         # whether to post-sync
    func = "logit"      # function for the unary costs
    ):
    
    K = len(n_points_gt_list)
    alpha = 2.0 # try 0.25, 0.5, 1, 2.0
    
    mgm_model = pylibmgm.MgmModel()
    
    for idx,(i,j) in enumerate(pairs):
        ni = int(n_points_gt_list[i][batch_idx].item())
        nj = int(n_points_gt_list[j][batch_idx].item())
        Si = sim_matrices[idx][batch_idx][:ni, :nj]             # [Ni_max, Nj_max]
        
        mu = Si.mean(-1, keepdim=True)
        sigma = Si.std(-1, keepdim=True) + 1e-8
        S_z = (Si - mu) / sigma
        mat = S_z.detach().cpu().numpy()

        # reserve exactly ni*nj unary costs, no quadratics
        gm = pylibmgm.GmModel(pylibmgm.Graph(i, ni), pylibmgm.Graph(j, nj), ni*nj, 0)

        # fill in unary costs with function
        eps = 1e-8
        for u in range(ni):
            for v in range(nj):
                s = float(mat[u,v])
                
                if func == "cosine":
                    cost = -alpha * s
                elif func == "logit":
                    x = max(-1 + eps, min(1 - eps, alpha * s))
                    cost = -math.log((1+x)/(1-x))
                elif func == "atanh":
                    x = max(-1 + eps, min(1 - eps, alpha * s))
                    cost = -math.atanh(x)                
                elif func == "logsig":
                    # –log sigmoid(alpha s)  ==  softplus( –alpha s )
                    cost = F.softplus(-alpha * torch.tensor(s)).item()
                else:
                    raise ValueError(f"unknown cost type '{func}'")
                
                gm.add_assignment(u, v, cost)

        mgm_model.add_model(gm)
    
    #print_mgm(mgm_model)
    sol = pylibmgm.solver.solve_mgm(mgm_model, opt_level=pylibmgm.solver.OptimizationLevel.DEFAULT)

    if sync:
        sol = pylibmgm.solver.synchronize_solution(mgm_model, sol, feasible=True, iterations=3, opt_level=pylibmgm.solver.OptimizationLevel.DEFAULT)
    
    out = []
    for (i, j) in pairs:
        labels = sol[(i, j)]          # pylibmgm returns a GmSolution via __getitem__
        #lab   = gmsol.labeling     
        ni    = int(n_points_gt_list[i][batch_idx].item())
        nj    = int(n_points_gt_list[j][batch_idx].item())
        mat_ij = np.zeros((ni, nj), dtype=np.int32)
        for u, v in enumerate(labels):
            if 0 <= v < nj:
                mat_ij[u, v] = 1
        out.append((i, j, mat_ij))
    
    return out

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
    
    # per-class cycle inconsistency accumulators
    bad_pre_cls   = torch.zeros(len(classes), dtype=torch.long)
    rows_pre_cls  = torch.zeros(len(classes), dtype=torch.long)
    bad_post_cls  = torch.zeros(len(classes), dtype=torch.long)
    rows_post_cls = torch.zeros(len(classes), dtype=torch.long)
    
    error_dist_dict = {}
    pairs = list(combinations(range(K), 2))
    
    for cls_inx, cls in enumerate(classes):
        if local_rank == output_rank:
            if verbose:
                print("Evaluating class {}: {}/{}".format(cls, cls_inx, len(classes)))

        running_since = time.time()
        iter_num = 0
        ds.set_cls(cls)
        
        correct_pre_total  = 0.0    # running total of correct rows
        valid_pre_total    = 0.0    # running total of valid   rows
        correct_post_total = 0.0
        valid_post_total   = 0.0
        
        result_dict = {}
        
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
                    idx_c += 1
            correct_pre_total += correct_pre
            valid_pre_total += valid_pre
            
            correct_post = 0.0
            valid_post   = 0.0
            for b in range(batch_num):
                graph_sizes_b = [n_points_gt[g][b].item() for g in range(K)] # graph-sizes for this example
                
                logger = logging.getLogger("libmgm")
                logger.setLevel(logging.WARNING) # set log level of mgm model
                
                # create MgmModel, solve mgm and extract labelings
                all_matches_b = mgm_model_synchronizing(
                    sim_matrices     = sim_list,
                    n_points_gt_list = n_points_gt,
                    pairs            = pairs,
                    batch_idx        = b,  
                    sync             = True,     
                    func             = "cosine"
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
            
                # build pre-sync dict
                P_dict_pre = {}
                idx_tmp = 0
                for gi in range(K):
                    for gj in range(gi+1, K):
                        ni, nj = graph_sizes_b[gi], graph_sizes_b[gj]
                        Pp = pred_perm_mats[idx_tmp][b, :ni, :nj]
                        P_dict_pre[(gi, gj)] = Pp
                        idx_tmp += 1
                        
                # build post-sync dict
                P_dict_post = {}
                for idx2, (gi, gj) in enumerate(pairs):
                    ni, nj = graph_sizes_b[gi], graph_sizes_b[gj]
                    _, _, P_np = all_matches_b[idx2]
                    P_dict_post[(gi, gj)] = torch.tensor(P_np, dtype=torch.float32, device=device)

                bad_p, rows_p = count_cycle_inconsistencies(P_dict_pre,  graph_sizes_b)
                bad_q, rows_q = count_cycle_inconsistencies(P_dict_post, graph_sizes_b)

                bad_pre_cls[cls_inx]   += bad_p
                rows_pre_cls[cls_inx]  += rows_p
                bad_post_cls[cls_inx]  += bad_q
                rows_post_cls[cls_inx] += rows_q
            
            correct_post_total += correct_post
            valid_post_total += valid_post
            
            # progress print every 40 batches
            if iter_num % 40 == 0 and verbose: #cfg.STATISTIC_STEP
                running_speed = 40 * batch_num / (time.time() - running_since) #cfg.STATISTIC_STEP
                print("Class {:<8} Iteration {:<4} {:>4.2f}sample/s".format(cls, iter_num, running_speed))
                running_since = time.time()
        
        acc_pre_cls  = correct_pre_total  / valid_pre_total
        acc_post_cls = correct_post_total / valid_post_total
        
        accs_pre_sync[cls_inx] = acc_pre_cls
        accs_post_sync[cls_inx] = acc_post_cls
        
        # f1_scores[i] = epoch_f1
        if verbose:
            print(f"Class {cls} acc_pre_sync = {acc_pre_cls:.4f}, acc_post_sync = {acc_post_cls:.4f}")
            
        error_dist_dict[cls] = result_dict
        
    # print(error_dist_dict)
    time_elapsed = time.time() - since
    print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    # wandb logging
    avg_pre = torch.mean(accs_pre_sync).item()
    avg_post = torch.mean(accs_post_sync).item()
    
    # per-class cycle inconsistency % (pre & post)
    cycle_pre_pct_cls  = 100.0 * bad_pre_cls.float()  / torch.clamp(rows_pre_cls.float(),  min=1)
    cycle_post_pct_cls = 100.0 * bad_post_cls.float() / torch.clamp(rows_post_cls.float(), min=1)
    
    # global averages over classes
    avg_cycle_pre_pct  = cycle_pre_pct_cls.mean().item()
    avg_cycle_post_pct = cycle_post_pct_cls.mean().item()
    
    if local_rank == output_rank:
        wandb.log({
                "eval/avg_pre_sync":  avg_pre,
                "eval/avg_post_sync": avg_post,
                "eval/avg_cycle_inconsistency_pre_pct":  avg_cycle_pre_pct,
                "eval/avg_cycle_inconsistency_post_pct": avg_cycle_post_pct,
                "eval/time_s":       time_elapsed
        }, step=(eval_epoch or wandb.run.step))
        # log per-class accuracies too
        for cls, pre, post, cyc_pre, cyc_post in zip(classes, accs_pre_sync, accs_post_sync, cycle_pre_pct_cls, cycle_post_pct_cls):
            wandb.log({
                    f"eval/pre_sync/{cls}":  pre.item(),
                    f"eval/post_sync/{cls}": post.item(),
                    f"eval/cycle_pre_pct/{cls}":  cyc_pre.item(),
                    f"eval/cycle_post_pct/{cls}": cyc_post.item()
            }, step=(eval_epoch or wandb.run.step))

    model.train(mode=was_training)
    ds.cls = cls_cache

    print("Matching accuracy")
    for cls, pre_acc, post_acc, cyc_pre, cyc_post in zip(classes, accs_pre_sync, accs_post_sync, cycle_pre_pct_cls, cycle_post_pct_cls):
        print(f"{cls}: pre = {pre_acc:.4f}, post = {post_acc:.4f} | "f"cycle_pre = {cyc_pre:.2f}%, cycle_post = {cyc_post:.2f}%")
    print("average pre sync = {:.4f}, average after sync = {:.4f}".format(torch.mean(accs_pre_sync), torch.mean(accs_post_sync)))
    print(f"avg cycle inconsistency pre =  {avg_cycle_pre_pct:.2f}%, "f"post = {avg_cycle_post_pct:.2f}%")

    return accs_pre_sync, accs_post_sync, error_dist_dict