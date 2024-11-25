import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from utils.config import cfg
from utils.evaluation_metric import calculate_correct_and_valid, calculate_f1_score, get_pos_neg

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
def eval_model(model, dataloader, local_rank, eval_epoch=None, verbose=True):
    print("Start evaluation...")
    since = time.time()

    device = next(model.parameters()).device

    if eval_epoch is not None:
        model_path = str(Path(cfg.model_dir) / "params" / "{:04}".format(eval_epoch) / "params.pt")
        if local_rank == 0:
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
        if local_rank == 0:
            if verbose:
                print("Evaluating class {}: {}/{}".format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0

        
        ds.set_cls(cls)
        acc_match_num = torch.zeros(1, device=device)
        acc_total_num = torch.zeros(1, device=device)
        # tp = torch.zeros(1, device=device)
        # fp = torch.zeros(1, device=device)
        # fn = torch.zeros(1, device=device)

        # for analysis of each step accuracy
        
        result_dict = {}
        tp = 0
        fp = 0
        fn = 0
        epoch_f1 = 0.0
        epoch_correct = 0
        epoch_total_valid = 0
        for k, inputs in enumerate(dataloader):
            data_list = [_.cuda() for _ in inputs["images"]]

            points_gt = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt = [_.cuda() for _ in inputs["ns"]]
            edges = [_.to("cuda") for _ in inputs["edges"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]

            batch_num = data_list[0].size(0)
            num_nodes_s = points_gt[0].size(1)
            num_nodes_t = points_gt[1].size(1)

            iter_num = iter_num + 1

            visualize = k == 0 and cfg.visualize
            visualization_params = {**cfg.visualization_params, **dict(string_info=cls, true_matchings=perm_mat_list)}

            matched_instances_at_step = []
            with torch.set_grad_enabled(False):
                
                matchings = []
                B, N_s, N_t = perm_mat_list[0].size()
                n_points_sample = torch.zeros(B, dtype=torch.int).to(device)
                # perm_mat_dec_list = [torch.zeros(B, N_s, N_t, dtype=torch.int).to(device)]
                # cost_mask = torch.ones(B, N_s, N_t, dtype=torch.int).to(device)
                # batch_idx = torch.arange(cfg.BATCH_SIZE)
            
                # set matching score for padded to zero
                # for batch in batch_idx:
                #     n_point = n_points_gt[0][batch]
                #     cost_mask[batch, n_point:, :] = -1
                #     cost_mask[batch, :, n_point:] = -1
                
                eval_pred_points = 0
                j_pred = 0
                predictions_list = []
                for _ in range(B):
                    predictions_list.append([])
                    
                for np in range(N_t):
                    
                    # target_points, model_output = model(data_list, points_gt, edges, n_points_gt,  perm_mat_list, n_points_sample, eval_pred_points, in_training= False)
                    similarity_scores = model(data_list, points_gt, edges, n_points_gt,  perm_mat_list, n_points_sample, eval_pred_points, in_training= False)
                    # target_points = cosine_norm(target_points)
                    
                    # batch_size = model_output.size()[0]
                    # num_points1 = model_output.size()[1]
                    batch_size = similarity_scores.shape[0]
                    num_points1 = similarity_scores.shape[1]
                    keypoint_preds = F.softmax(similarity_scores, dim=-1)
                    keypoint_preds = torch.argmax(keypoint_preds, dim=-1)
                    for b in range(batch_size):
                        # cosine_similarities = F.cosine_similarity(model_output[b, eval_pred_points].unsqueeze(0), target_points[b])
                        # cosine_similarities = torch.atanh(cosine_similarities)
                        # cosine_scores = F.softmax(cosine_similarities, dim=-1)
                        # cosine_matchings = torch.argmax(cosine_scores, dim=-1)
                        if eval_pred_points < n_points_gt[0][b]:
                            predictions_list[b].append(keypoint_preds[b][eval_pred_points].item())
                        else:
                            predictions_list[b].append(-1)
                    
                    eval_pred_points +=1
                prediction_tensor = torch.tensor(predictions_list).to(perm_mat_list[0].device)
                y_values_matching = torch.argmax(perm_mat_list[0], dim=-1)
                batch_correct, batch_total_valid = calculate_correct_and_valid(prediction_tensor, y_values_matching)
                
                
                
                error_list = (prediction_tensor != y_values_matching).int()
            
                for idx, e in enumerate(n_points_gt[0]):
                    if e.item() not in result_dict:
                        result_dict[e.item()] = [1, error_list[idx,:e.item()]]
                    result_dict[e.item()][0] += 1
                    result_dict[e.item()][1] += error_list[idx,:e.item()]
                # Iterate through the batch
                _tp, _fp, _fn = calculate_f1_score(prediction_tensor, y_values_matching)

                
                epoch_correct += batch_correct
                epoch_total_valid += batch_total_valid
                tp += _tp
                fp += _fp
                fn += _fn
            
            bs = perm_mat_list[0].size(0)
            
            # evaluation metrics
            # _, _acc_match_num, _acc_total_num = matching_accuracy_from_lists(s_pred_mat_list, perm_mat_gt_list)
            # _tp, _fp, _fn = get_pos_neg_from_lists(s_pred_mat_list, perm_mat_gt_list)

            # acc_match_num += _acc_match_num
            # acc_total_num += _acc_total_num
            # tp += _tp
            # fp += _fp
            # fn += _fn

            if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
                running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                print("Class {:<8} Iteration {:<4} {:>4.2f}sample/s".format(cls, iter_num, running_speed))
                running_since = time.time()
        
        
        dataset_size = len(dataloader.dataset)
        
        if epoch_total_valid > 0:
            epoch_acc = epoch_correct / epoch_total_valid
        else:
            epoch_acc = 0.0

        precision_global = tp / (tp + fp + 1e-8)
        recall_global = tp / (tp + fn + 1e-8)
        
        # Global F1 score
        epoch_f1 = 2 * (precision_global * recall_global) / (precision_global + recall_global + 1e-8)
        
        accs[i] = epoch_acc
        f1_scores[i] = epoch_f1
        if verbose:
            print("Class {} acc = {:.4f} F1 = {:.4f}".format(cls, accs[i], f1_scores[i]))
            
        error_dist_dict[cls] = result_dict
        
    # print(error_dist_dict)
    time_elapsed = time.time() - since
    print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)
    ds.cls = cls_cache

    print("Matching accuracy")
    for cls, single_acc, f1_sc in zip(classes, accs, f1_scores):
        print("{} = {:.4f}, {:.4f}".format(cls, single_acc, f1_sc))
    print("average = {:.4f}, {:.4f}".format(torch.mean(accs), torch.mean(f1_scores)))

    # error distribution
    # err_dist={}
    # num_bins = 5
    # for cls, v in error_dist_dict.items():
    #     matched_instances_size = max([tensor.size(0) for tensor in v ])
    #     n_matched_instances = torch.zeros(matched_instances_size)

    #     n_possible_matches  = [torch.ones(tensor.size(0)) for tensor in v]
    #     total_instances_to_match = torch.zeros(matched_instances_size)
    #     for tensor in v:
    #         n_matched_instances[:tensor.size(0)] += tensor
    #     for tensor in n_possible_matches:
    #         total_instances_to_match[:tensor.size(0)] += tensor
        
    #     indices = torch.arange(matched_instances_size)
    #     bin_edges = torch.linspace(0, matched_instances_size, num_bins)
    #     binned_indices = torch.bucketize(indices, bin_edges)
        
    #     binned_matches = torch.bincount(binned_indices, weights=n_matched_instances)
    #     binned_counts = torch.bincount(binned_indices, weights=total_instances_to_match)
        
    #     cls_error_distribution = binned_matches / binned_counts
    #     err_dist[cls] = cls_error_distribution

    return accs, f1_scores, error_dist_dict
