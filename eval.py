import time
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
                # how far from identity?
                # If it were perfect, C would equal I exactly.
                # Count any off-diagonal or missing diagonal entries:
                deviation = torch.abs(C - torch.eye(n, device=C.device))
                # If you want just a binary flag:
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
        #pre_dist_tot, post_dist_tot = 0.0, 0.0      # distance acc.
        pair_cnt        = 0
        result_dict = {}
        epoch_correct, epoch_total_valid = 0.0, 0.0 # epoch acc.
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

            perm_mats_masked, pred_mats_masked = [], []
            # slice out invalid padding of perm_mats and pred_mats
            for idx, (g_i, g_j) in enumerate(pairs):
                n_i = n_points_gt[g_i]  # shape [B]
                n_j = n_points_gt[g_j]  # shape [B]
                for b in range(batch_num):
                    ni = n_i[b].item()
                    nj = n_j[b].item()
                    # slice away all padded rows/cols
                    P_pred_valid = pred_perm_mats[idx][b, :ni, :nj]      # [ni × nj]
                    P_gt_valid = perm_mat_list[idx][b][:ni, :nj]
                    pred_mats_masked.append(P_pred_valid)
                    perm_mats_masked.append(P_gt_valid)
            
            acc_pre, _, _ = matching_accuracy_from_lists(pred_mats_masked, perm_mats_masked)
            pre_acc_tot += acc_pre.item()
            
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
            
            sync_pred, sync_gt = [], []
            for idx, (i, j) in enumerate(pairs):
                sync_pred.extend(tau[:, i, j])
                sync_gt.extend (perm_mat_list[idx])
            acc_post, _, _ = matching_accuracy_from_lists(sync_pred, sync_gt)   
            post_acc_tot += acc_post.item()
            
            # distance before and after (masked)
            #for idx, (i, j) in enumerate(pairs):
            #    n_valid = n_points_gt[i]              # [B]
            #    pre_dist  = perm_distance_masked(T_tensor[:,   i, j], perm_mat_list[idx], n_valid)
            #    post_dist = perm_distance_masked(tau[:, i, j], perm_mat_list[idx], n_valid)

            #    pre_dist_tot  += pre_dist.item()  * batch_num
            #    post_dist_tot += post_dist.item() * batch_num
            #    pair_cnt      += batch_num
            
           # ------------- per-batch row/col accuracy for logging --------- #
            for idx, (i, j) in enumerate(pairs):
                P_gt = perm_mat_list[idx]     # [B,n,n]
                for b in range(batch_num):
                    P_syn  = tau[b, i, j]
                    if P_gt[b].sum() == 0:    # empty GT -> skip
                        continue

                    pred_idx = P_syn.argmax(dim=1).unsqueeze(0)
                    gt_idx   = P_gt[b].argmax(dim=1).unsqueeze(0)
                    corr, valid = calculate_correct_and_valid(pred_idx, gt_idx)
                    epoch_correct     += corr
                    epoch_total_valid += valid

                    # ---------- per-keypoint error distribution ----------- #
                    n_pts = int(n_points_gt[i][b])
                    err_vec = (pred_idx[0, :n_pts] != gt_idx[0, :n_pts]).int()
                    if n_pts not in result_dict:
                        result_dict[n_pts] = [0, torch.zeros(n_pts, device=device)]
                    result_dict[n_pts][0] += 1
                    result_dict[n_pts][1] += err_vec
            
            # progress print every 40 batches
            if iter_num % 40 == 0 and verbose: #cfg.STATISTIC_STEP
                running_speed = 40 * batch_num / (time.time() - running_since) #cfg.STATISTIC_STEP
                print("Class {:<8} Iteration {:<4} {:>4.2f}sample/s".format(cls, iter_num, running_speed))
                running_since = time.time()
        
        #dist_b = pre_dist_tot  / pair_cnt
        #dist_a = post_dist_tot / pair_cnt
        
        acc_pre_cls  = pre_acc_tot  / iter_num
        acc_post_cls = post_acc_tot / iter_num
        
        accs_pre_sync[cls_inx] = acc_pre_cls
        accs_post_sync[cls_inx] = acc_post_cls
        
        avg_bad_pre = bad_pre_tot / num_batches
        avg_bad_post = bad_post_tot / num_batches
        
        # f1_scores[i] = epoch_f1
        if verbose:
            print(f"Class {cls} acc_pre_sync = {acc_pre_cls:.4f} acc_post_sync = {acc_post_cls:.4f}")
            #print(f"Avg distance  : before={dist_b:.3f} | after={dist_a:.3f}")
            print(f"Avg inconsistent 3-cycles before sync: {avg_bad_pre:.2f}, after sync: {avg_bad_post:.2f}")
            
        error_dist_dict[cls] = result_dict
        
    # print(error_dist_dict)
    time_elapsed = time.time() - since
    print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)
    ds.cls = cls_cache

    print("Matching accuracy")
    for cls, pre_acc, post_acc in zip(classes, accs_pre_sync, accs_post_sync):
        print("{}: pre sync = {:.4f}, after sync {:.4f}".format(cls, pre_acc, post_acc))
    print("average pre sync = {:.4f}, average after sync = {:.4f}".format(torch.mean(accs_pre_sync), torch.mean(accs_post_sync)))

    return accs_pre_sync, accs_post_sync, error_dist_dict