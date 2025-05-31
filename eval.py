import time
import wandb
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist  # Add this import
from scipy.optimize import linear_sum_assignment
from itertools import combinations
from permsync import perm_sync_batched, error_against_ground_truth_batched

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

def count_inconsistent_cycles(pairwise: torch.Tensor):
    """
    pairwise: Tensor of shape (K, K, n, n) giving P_{ij} for one batch item.
    Returns total number of inconsistent triples (i<j<k).
    """
    K, _, n, _ = pairwise.shape
    total_bad = 0
    for i in range(K):
        for j in range(i+1, K):
            for k in range(j+1, K):
                # Compose P_ij * P_jk * P_ki
                C = pairwise[i,j] @ pairwise[j,k] @ pairwise[k,i]
                deviation = torch.abs(C - torch.eye(n, device=C.device))
                if (deviation > 1e-4).any():
                    total_bad += 1
    return total_bad

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
        
        pre_acc_tot, post_acc_tot = 0.0, 0.0        # accurancy acc. before and after sync
        result_dict = {}
        bad_pre_tot, bad_post_tot = 0.0, 0.0        # inconsistent cycles acc.
        num_batches = 0
        
        for k, inputs in enumerate(dataloader, 1):
            iter_num = iter_num + 1
            data_list = [_.cuda() for _ in inputs["images"]]
            points_gt = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt = [_.cuda() for _ in inputs["ns"]]
            edges = [_.to("cuda") for _ in inputs["edges"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]

            batch_num = data_list[0].size(0)    # batch size
            pred_perm_mats = []                 # pairewise prediction
            
            for idx, (g_i,g_j) in enumerate(pairs):     # pairewise inference loop
                imgs_pair  = [data_list[g_i], data_list[g_j]]
                pts_pair   = [points_gt[g_i], points_gt[g_j]]
                edges_pair = [edges[g_i], edges[g_j]]
                n_points_gt_pair = [n_points_gt[g_i], n_points_gt[g_j]]
                
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
                #sink_max = torch.argmax(P_sink, dim=-1)  # [B, N_s]

                # Baue harte Permutationsmatrix per Batch
                #P_pred = torch.zeros(batch_num, N_s, N_t, device=device)
                #for b in range(batch_num):
                #    for src in range(N_s):
                #        if src < n_points_gt[g_i][b]:
                #            tgt = sink_max[b, src].item()
                #            P_pred[b, src, tgt] = 1
                P_pred = hard_perm_from_sink(P_sink)
                pred_perm_mats.append(P_pred)

            #perm_mats_masked, pred_mats_masked = [], []
            # slice out invalid padding of perm_mats and pred_mats
            # Accumulate correct vs valid counts for pre-sync
            correct_pre, valid_pre = 0.0, 0.0
            for idx, (g_i, g_j) in enumerate(pairs):
                n_i = n_points_gt[g_i]
                n_j = n_points_gt[g_j]
                for b in range(batch_num):
                    ni, nj = n_i[b].item(), n_j[b].item()
                    Pp = pred_perm_mats[idx][b, :ni, :nj]       # [ni×nj]
                    Pg = perm_mat_list[idx][b, :ni, :nj]       # [ni×nj]
                    # row-wise argmax → predict & gt indices
                    pred_idx = Pp.argmax(dim=1, keepdim=True)  # [ni,1]
                    gt_idx   = Pg.argmax(dim=1, keepdim=True)  # [ni,1]
                    c, v = calculate_correct_and_valid(pred_idx, gt_idx)
                    correct_pre += c
                    valid_pre   += v
            pre_acc_tot += (correct_pre / valid_pre)
            
            # permutation synchronization if K > 2
            N_s = pred_perm_mats[0].shape[1]  # assumes square permutations: [B, N_s, N_t]
            T_tensor = torch.eye(N_s, device=device).repeat(batch_num, K, K, 1, 1)
            for idx, (i, j) in enumerate(pairs):
                T_tensor[:, i, j] = pred_perm_mats[idx]
                T_tensor[:, j, i] = pred_perm_mats[idx].transpose(1, 2)
                
            tau = perm_sync_batched(T_tensor) if K > 2 else T_tensor
            
            # calculation of inconsistent cycles
            for b in range(batch_num):
                P_pre = T_tensor[b]
                P_post = tau[b]
                bad_pre_tot  += count_inconsistent_cycles(P_pre)
                bad_post_tot += count_inconsistent_cycles(P_post)
            num_batches += batch_num
            
            correct_post, valid_post = 0.0, 0.0
            for idx, (i, j) in enumerate(pairs):
                n_i = n_points_gt[i]
                n_j = n_points_gt[j]
                for b in range(batch_num):
                    ni, nj = n_i[b].item(), n_j[b].item()
                    Ps = tau[b, i, j][:ni, :nj]
                    Pg = perm_mat_list[idx][b, :ni, :nj]
                    pred_idx = Ps.argmax(dim=1, keepdim=True)
                    gt_idx   = Pg.argmax(dim=1, keepdim=True)
                    c, v = calculate_correct_and_valid(pred_idx, gt_idx)
                    correct_post += c
                    valid_post   += v
                    
                    # ---------- per-keypoint error distribution ----------- #
                    n_pts = int(n_points_gt[i][b])
                    err_vec = (pred_idx[0, :n_pts] != gt_idx[0, :n_pts]).int()
                    if n_pts not in result_dict:
                        result_dict[n_pts] = [0, torch.zeros(n_pts, device=device)]
                    result_dict[n_pts][0] += 1
                    result_dict[n_pts][1] += err_vec
                    
            post_acc_tot += (correct_post / valid_post)
            
            # progress print every 40 batches
            if iter_num % 40 == 0 and verbose: #cfg.STATISTIC_STEP
                running_speed = 40 * batch_num / (time.time() - running_since) #cfg.STATISTIC_STEP
                print("Class {:<8} Iteration {:<4} {:>4.2f}sample/s".format(cls, iter_num, running_speed))
                running_since = time.time()
        
        acc_pre_cls  = pre_acc_tot  / iter_num
        acc_post_cls = post_acc_tot / iter_num
        
        accs_pre_sync[cls_inx] = acc_pre_cls
        accs_post_sync[cls_inx] = acc_post_cls
        
        avg_bad_pre = bad_pre_tot / num_batches
        avg_bad_post = bad_post_tot / num_batches
        
        # f1_scores[i] = epoch_f1
        if verbose:
            print(f"Class {cls} acc_pre_sync = {acc_pre_cls:.4f} acc_post_sync = {acc_post_cls:.4f}")
            print(f"Avg inconsistent 3-cycles before sync: {avg_bad_pre:.2f}, after sync: {avg_bad_post:.2f}")
            
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