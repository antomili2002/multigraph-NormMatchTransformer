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


def reshape_perm_matrices(perm_mat_list: list, data_list: list):
    """
    Reshapes Ground Truth Matrices List from a List of Tensors of shape (B, K, K)
    to List[List[torch.Tensor]] with shape (B, K, K) for indexing using perm_mats[i][j]

    Args:
        perm_mat_list (list): Ground Truth Matrices List of shape List[Tensor] (B, K, K)
        data_list (list): List of data input

    Returns:
        torch.Tensor: List of List of toch.Tensor with shape (B, K, K)
    """
    G = len(data_list)  # number of graphs
    perm_mats = [[None for _ in range(G)] for _ in range(G)]

    idx = 0
    for i in range(G):
        for j in range(i + 1, G):  # upper triangle
            P_ij = perm_mat_list[idx]
            perm_mats[i][j] = P_ij
            perm_mats[j][i] = P_ij.transpose(-1, -2)
            idx += 1
        perm_mats[i][i] = torch.eye(P_ij.size(-1), device=P_ij.device).expand(P_ij.size(0), -1, -1)
    return perm_mats

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

def sinkhorn_cosine(
    cosine_sim: torch.Tensor,
    max_iter: int = 15,
    eps: float = 1e-9
) -> torch.Tensor:
    """
    Converts a batch of cosine similarity matrices into doubly-stochastic matrices 
    using the Sinkhorn algorithm.

    Args:
        cosine_sim: Tensor of shape (batch_size, n, m).
        max_iter:   Number of Sinkhorn iterations.
        eps:        Small numerical stabilizer to avoid division by zero.

    Returns:
        Doubly-stochastic matrices of shape (batch_size, n, m).
    """

    # 1) Exponentiate to ensure entries are positive
    #    (you can also add a temperature scale if needed).
    # cosine_sim[cosine_sim <= 0] = eps
    # input_tensor = cosine_sim
    Q = torch.exp(cosine_sim)

    for _ in range(max_iter):
        # 2) Row normalization
        row_sums = Q.sum(dim=2, keepdim=True) + eps
        Q = Q / row_sums

        # 3) Column normalization
        col_sums = Q.sum(dim=1, keepdim=True) + eps
        Q = Q / col_sums
        
    return Q

def cosine_norm(x, dim=-1):
        """
        Places vectors onto the unit-hypersphere

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        # calculate the magnitude of the vectors
        norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min=1e-6)
        # divide by the magnitude to place on the unit hypersphere
        return x / norm
    
def permutation_synchronization(pairwise_perm_matrices, num_graphs, keypoint_sizes):
    """
    Synchroizes pairewise permutation matrices into absolute permutation using spectral methods

    Args:
        pairwise_perm_matrices: Dict[(i, j)] -> P_ij where P_ij is [ni, nj] (torch.Tensor)
        num_graphs: int, number of graphs (G)
        keypoint_sizes: List[int], number of keypoints per graph [n1, n2, ..., nG]

    Returns:
        List of tensors [P_0, ..., P_G-1], where each P_i is [ni, n0]
        These represent permutations from graph i to the reference graph 0.
    """
    total_kpts = sum(keypoint_sizes)  # total keypoints across all graphs
    device = next(iter(pairwise_perm_matrices.values())).device  # get correct device

    # Initialize full block matrix [N, N] with N = total number of keypoints
    block_matrix = torch.zeros((total_kpts, total_kpts), device=device)

    # Build mapping of (i, j) to proper slice indices
    offsets = [0]
    for size in keypoint_sizes:
        offsets.append(offsets[-1] + size)  # offsets[i] is the start index of graph i

    for (i, j), P_ij in pairwise_perm_matrices.items():
        ni, nj = keypoint_sizes[i], keypoint_sizes[j]
        i_start, i_end = offsets[i], offsets[i + 1]
        j_start, j_end = offsets[j], offsets[j + 1]

        # Insert P_ij and its transpose
        block_matrix[i_start:i_end, j_start:j_end] = P_ij
        block_matrix[j_start:j_end, i_start:i_end] = P_ij.T

    # Spectral decomposition: get top-n0 eigenvectors (corresponds to reference graph 0)
    _, eigvecs = torch.linalg.eigh(block_matrix)  # [N, N]
    n0 = keypoint_sizes[0]
    embeddings = eigvecs[:, -n0:]  # Take last n0 components

    # Split into individual graph embeddings: List[G] of [ni, n0]
    node_embeddings = []
    for i in range(num_graphs):
        start, end = offsets[i], offsets[i + 1]
        node_embeddings.append(embeddings[start:end])  # [ni, n0]

    # Align each embedding to base embedding (graph 0)
    base = node_embeddings[0]  # [n0, n0]
    perms = []
    for emb in node_embeddings:
        sim = -emb @ base.T  # [ni, n0]
        row, col = linear_sum_assignment(sim.cpu().numpy())
        P = torch.zeros_like(sim)
        P[row, col] = 1
        perms.append(P)  # each P is [ni, n0]

    return perms  # List[G] of [ni, n0]

def multi_graph_inference(similarity_matrices, epsilon=0.035, max_iter=27):
    B = similarity_matrices[0][0].shape[0]
    G = len(similarity_matrices)
    n = similarity_matrices[0][0].shape[1]
    all_abs_perms = []
    
    for b in range(B):
        pairwise_perm_matrices = {}
        keypoint_sizes = []
        
        # Determine number of keypoints per graph for sample b
        for i in range(G):
            n_i = similarity_matrices[i][i].shape[1]
            keypoint_sizes.append(n_i)

        for i, j in combinations(range(G), 2):
            sim = similarity_matrices[i][j][b]  # [ni, nj]
            P_ij = sinkhorn_logspace(sim[None], epsilon, max_iter)[0]  # [ni, nj]
            pairwise_perm_matrices[(i, j)] = P_ij
            pairwise_perm_matrices[(j, i)] = P_ij.T

        abs_perms = permutation_synchronization(pairwise_perm_matrices, G, keypoint_sizes)  # List[G] of [ni, n0]

        # Pad to square shape [max_n, max_n] for stacking
        max_n = max(keypoint_sizes)
        padded = []
        for P in abs_perms:
            P_padded = torch.zeros((max_n, max_n), device=P.device)
            P_padded[:P.shape[0], :P.shape[1]] = P
            padded.append(P_padded)
        all_abs_perms.append(torch.stack(padded, dim=0))  # [G, max_n, max_n]
        
    return torch.stack(all_abs_perms, dim=0)  # [B, G, n, n]
    
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
    ds.set_num_graphs(cfg.EVAL.num_graphs_in_matching_instance)
    classes = ds.classes
    cls_cache = ds.cls

    accs = torch.zeros(len(classes), device=device)
    f1_scores = torch.zeros(len(classes), device=device)
    error_dist_dict = {}
    

    for i, cls in enumerate(classes):
        if local_rank == output_rank:
            if verbose:
                print("Evaluating class {}: {}/{}".format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0
        ds.set_cls(cls)

        # for analysis of each step accuracy
        result_dict = {}
        tp, fp, fn = 0, 0, 0
        epoch_f1, epoch_correct, epoch_total_valid = 0, 0, 0
        for k, inputs in enumerate(dataloader):
            data_list = [_.cuda() for _ in inputs["images"]]
            points_gt = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt = [_.cuda() for _ in inputs["ns"]]
            edges = [_.to("cuda") for _ in inputs["edges"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]

            # reshape gt_perm_mat
            perm_mat_list = reshape_perm_matrices(perm_mat_list, data_list)
            
            n_points_gt_sample = n_points_gt[0]
            batch_num = data_list[0].size(0)
            iter_num = iter_num + 1

            with torch.no_grad():
                
                decoded_graphs, similarity_matrices = model(
                    data_list, points_gt, edges, n_points_gt, n_points_gt_sample, perm_mat_list, in_training=False
                )

                pred_perm = multi_graph_inference(similarity_matrices)

                for b in range(pred_perm.size(0)):
                    for gi in range(len(perm_mat_list)):
                        pred = torch.argmax(pred_perm[b, gi], dim=-1)
                        target = torch.argmax(perm_mat_list[gi][0][b], dim=-1)

                        #print(f"pred shape: {pred.shape}")
                        #print(f"target shape: {target.shape}")
                        
                        K = min(pred.shape[0], target.shape[0])
                        prediction_tensor = pred[:K].to(target.device)
                        y_values_matching = target[:K]
                        
                        #print("pred device:", pred.device)
                        #print("target device:", target.device)

                        
                        correct, valid = calculate_correct_and_valid(prediction_tensor, y_values_matching)
                        _tp, _fp, _fn = calculate_f1_score(prediction_tensor, y_values_matching)

                        epoch_correct += correct
                        epoch_total_valid += valid
                        tp += _tp
                        fp += _fp
                        fn += _fn
                
                
                #error_list = (prediction_tensor != y_values_matching).int()
            
            if iter_num % 40 == 0 and verbose: #cfg.STATISTIC_STEP
                running_speed = 40 * batch_num / (time.time() - running_since) #cfg.STATISTIC_STEP
                print("Class {:<8} Iteration {:<4} {:>4.2f}sample/s".format(cls, iter_num, running_speed))
                running_since = time.time()
        
        acc = epoch_correct / epoch_total_valid if epoch_total_valid else 0
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        accs[i] = acc
        f1_scores[i] = f1

        if verbose:
            print("Class {} Acc = {:.4f}, F1 = {:.4f}".format(cls, acc, f1))
        
    # print(error_dist_dict)
    time_elapsed = time.time() - since
    print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)
    ds.cls = cls_cache

    print("Matching accuracy")
    for cls, single_acc, f1_sc in zip(classes, accs, f1_scores):
        print("{} = {:.4f}, {:.4f}".format(cls, single_acc, f1_sc))
    print("average = {:.4f}, {:.4f}".format(torch.mean(accs), torch.mean(f1_scores)))

    return accs, f1_scores, error_dist_dict
