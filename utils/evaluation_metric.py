import torch
from sklearn.metrics import f1_score

def make_perm_mat_pred(matching_vec, num_nodes_t):

    device = matching_vec.device

    batch_size = matching_vec.size()[0]
    nodes = matching_vec.size()[1]
    perm_mat_pred = []
    for i in range(batch_size):
        row_idx = torch.arange(nodes)
        one_hot_pred = torch.zeros(nodes, num_nodes_t)
        index = matching_vec[i, :]
        one_hot_pred[row_idx, index] = 1
        perm_mat_pred.append(one_hot_pred)
    
    return torch.stack(perm_mat_pred)

def make_sampled_perm_mat_pred(matching_vec, n_sampled_list):

    device = matching_vec.device

    batch_size = matching_vec.size()[0]
    nodes = matching_vec.size()[1]
    perm_mat_pred = []
    for i in range(batch_size):
        row_idx = torch.arange(nodes)
        one_hot_pred = torch.zeros(nodes, n_sampled_list[i])
        index = matching_vec[i, :]
        one_hot_pred[row_idx, index] = 1
        perm_mat_pred.append(one_hot_pred)
    
    return torch.stack(perm_mat_pred)

# def make_perm_mat_pred(matching_vec, num_nodes_t, n_points_gt_list):

#     device = matching_vec.device

#     batch_size = matching_vec.size()[0]
#     nodes = matching_vec.size()[1]
#     n_point_gt = n_points_gt_list[0]
#     perm_mat_pred = []
#     for i in range(batch_size):
#         n_points_in_img = n_point_gt[i]
#         row_idx = torch.arange(n_points_in_img)
#         one_hot_pred = torch.zeros(nodes, num_nodes_t)
#         index = matching_vec[i, :n_points_in_img]
#         one_hot_pred[row_idx, index] = 1
#         perm_mat_pred.append(one_hot_pred)
    
#     return torch.stack(perm_mat_pred)


# def f1_score(tp, fp, fn):
#     """
#     F1 score (harmonic mix of precision and recall) between predicted permutation matrix and ground truth permutation matrix.
#     :param tp: number of true positives
#     :param fp: number of false positives
#     :param fn: number of false negatives
#     :return: F1 score
#     """
#     device = tp.device

#     const = torch.tensor(1e-7, device=device)
#     precision = tp / (tp + fp + const)
#     recall = tp / (tp + fn + const)
#     f1 = 2 * precision * recall / (precision + recall + const)
#     return f1


def calculate_correct_and_valid(prediction_tensor, y_values_matching):
    valid_mask = (prediction_tensor != -1) & (y_values_matching != -1)
    batch_correct = (prediction_tensor[valid_mask] == y_values_matching[valid_mask]).sum().item()
    batch_total_valid = valid_mask.sum().item()
    return batch_correct, batch_total_valid


def calculate_f1_score(prediction_tensor, y_values_matching):
    
    # Mask to filter out invalid predictions/labels
    valid_mask = (prediction_tensor != -1) & (y_values_matching != -1)
    valid_preds = prediction_tensor[valid_mask]
    valid_labels = y_values_matching[valid_mask]
    

    valid_preds = valid_preds.cpu().numpy()
    valid_labels = valid_labels.cpu().numpy()
    
    f1_score_ = f1_score(valid_labels, valid_preds, average='micro')

    return f1_score_


def get_pos_neg(pmat_pred, pmat_gt):
    """
    Calculates number of true positives, false positives and false negatives
    :param pmat_pred: predicted permutation matrix
    :param pmat_gt: ground truth permutation matrix
    :return: tp, fp, fn
    """
    device = pmat_pred.device
    pmat_gt = pmat_gt.to(device)

    tp = torch.sum(pmat_pred * pmat_gt).float()
    fp = torch.sum(pmat_pred * (1 - pmat_gt)).float()
    fn = torch.sum((1 - pmat_pred) * pmat_gt).float()
    return tp, fp, fn


def get_pos_neg_from_lists(pmat_pred_list, pmat_gt_list):
    device = pmat_pred_list[0].device
    tp = torch.zeros(1, device=device)
    fp = torch.zeros(1, device=device)
    fn = torch.zeros(1, device=device)
    for pmat_pred, pmat_gt in zip(pmat_pred_list, pmat_gt_list):
        _tp, _fp, _fn = get_pos_neg(pmat_pred, pmat_gt)
        tp += _tp
        fp += _fp
        fn += _fn
    return tp, fp, fn


def matching_accuracy_from_lists(pmat_pred_list, pmat_gt_list):
    device = pmat_pred_list[0].device
    match_num = torch.zeros(1, device=device)
    total_num = torch.zeros(1, device=device)
    for pmat_pred, pmat_gt in zip(pmat_pred_list, pmat_gt_list):
        _, _match_num, _total_num = matching_accuracy(pmat_pred, pmat_gt)
        match_num += _match_num
        total_num += _total_num
    return match_num / total_num, match_num, total_num


def matching_accuracy(pmat_pred, pmat_gt):
    """
    Matching Accuracy between predicted permutation matrix and ground truth permutation matrix.
    :param pmat_pred: predicted permutation matrix
    :param pmat_gt: ground truth permutation matrix
    :param ns: number of exact pairs
    :return: matching accuracy, matched num of pairs, total num of pairs
    """
    device = pmat_pred.device
    batch_num = pmat_pred.shape[0]

    pmat_gt = pmat_gt.to(device)

    assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), "pmat_pred can noly contain 0/1 elements."
    assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), "pmat_gt should noly contain 0/1 elements."
    assert torch.all(torch.sum(pmat_pred, dim=-1) <= 1) and torch.all(torch.sum(pmat_pred, dim=-2) <= 1)
    assert torch.all(torch.sum(pmat_gt, dim=-1) <= 1) and torch.all(torch.sum(pmat_gt, dim=-2) <= 1)

    match_num = 0
    total_num = 0

    for b in range(batch_num):
        match_num += torch.sum(pmat_pred[b] * pmat_gt[b])
        total_num += torch.sum(pmat_gt[b])

    return match_num / total_num, match_num, total_num
