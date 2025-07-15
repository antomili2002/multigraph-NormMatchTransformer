import time
import pylibmgm.solver
import wandb
from pathlib import Path
import numpy as np
import torch
import math
import pylibmgm
import logging
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

def mgm_model_synchronizing(
    sim_matrices,       # List[torch.Tensor], length C = K*(K-1)/2, each [B, Ni_max, Nj_max]
    n_points_gt_list,   # List[torch.Tensor], length K, each [B]
    pairs,              # List of (i,j) in lex order
    batch_idx,          # index of batch
    parallel=False,     # whether to use solve_mgm_parallel
    sync=False,         # whether to post-sync
    func = "logit"      # function for the unary costs
    ):
    
    K = len(n_points_gt_list)
    device = sim_matrices[0].device
    
    mgm_model = pylibmgm.MgmModel()
    
    for idx,(i,j) in enumerate(pairs):
        Si = sim_matrices[idx][batch_idx]             # [Ni_max, Nj_max]
        ni = int(n_points_gt_list[i][batch_idx].item())
        nj = int(n_points_gt_list[j][batch_idx].item())
        mat = Si[:ni,:nj].detach().cpu().numpy()

        # reserve exactly ni*nj unary costs, no quadratics
        gm = pylibmgm.GmModel(pylibmgm.Graph(i, ni), pylibmgm.Graph(j, nj), ni*nj, 0)

        # fill in unary costs with function
        eps = 1e-8
        #alpha = 1.0
        for u in range(ni):
            for v in range(nj):
                s = float(mat[u,v])
                
                # clamp to avoid + /- 1
                if func in ("atanh", "logit"):
                    s = max(-1+eps, min(1-eps, s))
                if func == "logit":
                    cost = float(-math.log((1+s)/(1-s)))
                elif func == "atanh":
                    cost = float(-math.atanh(s))
                else:  # "cosine"
                    cost = float(-s)
                
                gm.add_assignment(u, v, cost)

        mgm_model.add_model(gm)
    
    if parallel: # does not work because of std::out_of_range
        sol = pylibmgm.solver.solve_mgm_parallel(mgm_model, opt_level=pylibmgm.solver.OptimizationLevel.DEFAULT)
    else:
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
    
    error_dist_dict = {}
    pairs = list(combinations(range(K), 2))
    
    for cls_inx, cls in enumerate(classes):
        if local_rank == output_rank:
            if verbose:
                print("Evaluating class {}: {}/{}".format(cls, cls_inx, len(classes)))

        running_since = time.time()
        iter_num = 0
        ds.set_cls(cls)
        
        #sum_pre_acc = 0.0    # sum over batches of (correct_pre/valid_pre)
        #sum_post_acc = 0.0   # sum over batches of (correct_post/valid_post)
        
        correct_pre_total  = 0.0    # running total of correct rows
        valid_pre_total    = 0.0    # running total of valid   rows
        correct_post_total = 0.0
        valid_post_total   = 0.0
        
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
                    parallel         = False,     
                    sync             = True,     
                    func             = "logit"
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
            
            correct_post_total += correct_post
            valid_post_total += valid_post
            
            #num_batches += 1
            #post_acc_batch = (correct_post / valid_post)
            #sum_post_acc += post_acc_batch
            
            # progress print every 40 batches
            if iter_num % 40 == 0 and verbose: #cfg.STATISTIC_STEP
                running_speed = 40 * batch_num / (time.time() - running_since) #cfg.STATISTIC_STEP
                print("Class {:<8} Iteration {:<4} {:>4.2f}sample/s".format(cls, iter_num, running_speed))
                running_since = time.time()
        
        #acc_pre_cls  = sum_pre_acc  / iter_num
        #acc_post_cls = sum_post_acc / iter_num
        
        acc_pre_cls  = correct_pre_total  / valid_pre_total
        acc_post_cls = correct_post_total / valid_post_total
        
        accs_pre_sync[cls_inx] = acc_pre_cls
        accs_post_sync[cls_inx] = acc_post_cls
        
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
    
    if local_rank == output_rank:
        wandb.log({
                "eval/avg_pre_sync":  avg_pre,
                "eval/avg_post_sync": avg_post,
                "eval/time_s":       time_elapsed
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
    for cls, pre_acc, post_acc in zip(classes, accs_pre_sync, accs_post_sync):
        print("{}: pre sync = {:.4f}, after sync {:.4f}".format(cls, pre_acc, post_acc))
    print("average pre sync = {:.4f}, average after sync = {:.4f}".format(torch.mean(accs_pre_sync), torch.mean(accs_post_sync)))

    return accs_pre_sync, accs_post_sync, error_dist_dict