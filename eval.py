import time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from utils.config import cfg
from utils.evaluation_metric import matching_accuracy, f1_score, get_pos_neg, make_perm_mat_pred, matching_accuracy_from_lists, get_pos_neg_from_lists


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
        tp = torch.zeros(1, device=device)
        fp = torch.zeros(1, device=device)
        fn = torch.zeros(1, device=device)

        # for analysis of each step accuracy
        error_dist_dict[cls] = []

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
                perm_mat_dec_list = [torch.zeros(B, N_s, N_t, dtype=torch.int).to(device)]
                cost_mask = torch.ones(B, N_s, N_t, dtype=torch.int).to(device)
                batch_idx = torch.arange(cfg.BATCH_SIZE)
            
                # set matching score for padded to zero
                for batch in batch_idx:
                    n_point = n_points_gt[0][batch]
                    cost_mask[batch, n_point:, :] = -1
                    cost_mask[batch, :, n_point:] = -1
                
                eval_pred_points = 0
                j_pred = 0
                predictions_list = []
                for i in range(B):
                    predictions_list.append([])
                for np in range(N_t):
                    
                    print()
                    model(data_list, points_gt, edges, n_points_gt,  perm_mat_list, n_points_sample, eval_pred_points, in_training= False)
                    # model prediction
                    # s_pred_list = model(
                    #     data_list,
                    #     points_gt,
                    #     edges,
                    #     n_points_gt,
                    #     n_points_sample,
                    #     perm_mat_dec_list,
                    #     in_training= False
                    # )
                    scores = s_pred_list.view(B, N_s, N_t)
                    
                    #apply matched mask to matching scores
                    scores[cost_mask == -1] = -torch.inf
                    
                    scores_per_batch = [scores[x,:,:] for x in range(B)]
                    argmax_idx = [torch.argmax(sc) for sc in scores_per_batch]
                    pair_idx = torch.tensor([(x // N_s, x % N_s) for x in  argmax_idx])
                    
                    # update mask of matched nodes
                    cost_mask[batch_idx, pair_idx[:,0], :] = -1
                    cost_mask[batch_idx, :, pair_idx[:,1]] = -1
                    
                    #update permutation matrix
                    perm_mat_dec_list[0][batch_idx, pair_idx[:,0], pair_idx[:,1]] = 1
                    #update numnber of points sampled
                    n_points_sample += 1 

                    # ground truth label and pred at each step: distribution of errors in AR
                    gt = torch.nonzero(perm_mat_list[0][batch_idx, pair_idx[:,0], :] == 1)[:,1].to(pair_idx.device)
                    pred = pair_idx[:,1]
                    matched_instances_at_step.append(gt == pred)

                    matchings.append(pair_idx)
                
                matched_instance_indicator = torch.stack(matched_instances_at_step, dim=1)
                for example in range(len(matched_instance_indicator)):
                    n_point= n_points_gt[0][example]
                    error_dist_dict[cls].append(matched_instance_indicator[example, :n_point].long())
                    
                
                matchings = torch.stack(matchings, dim=2)

                matches_list = []
                s_pred_mat_list = []
                perm_mat_gt_list = []
                for batch in batch_idx:
                    n_point = n_points_gt[0][batch]
                    matched_idxes  = matchings[batch,:, :n_point]
                    matches_list.append(matched_idxes)
                    s_pred_mat = torch.zeros(n_point, n_point).to(perm_mat_list[0].device)
                    s_pred_mat[matched_idxes[0,:], matched_idxes[1,:]] = 1
                    s_pred_mat_list.append(s_pred_mat)
                    perm_mat_gt_list.append(perm_mat_list[0][batch,:n_point, :n_point])
                
            # evaluation metrics
            _, _acc_match_num, _acc_total_num = matching_accuracy_from_lists(s_pred_mat_list, perm_mat_gt_list)
            _tp, _fp, _fn = get_pos_neg_from_lists(s_pred_mat_list, perm_mat_gt_list)

            acc_match_num += _acc_match_num
            acc_total_num += _acc_total_num
            tp += _tp
            fp += _fp
            fn += _fn

            if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
                running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                print("Class {:<8} Iteration {:<4} {:>4.2f}sample/s".format(cls, iter_num, running_speed))
                running_since = time.time()

        accs[i] = acc_match_num / acc_total_num
        f1_scores[i] = f1_score(tp, fp, fn)
        if verbose:
            print("Class {} acc = {:.4f} F1 = {:.4f}".format(cls, accs[i], f1_scores[i]))
        

    time_elapsed = time.time() - since
    print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)
    ds.cls = cls_cache

    print("Matching accuracy")
    for cls, single_acc, f1_sc in zip(classes, accs, f1_scores):
        print("{} = {:.4f}, {:.4f}".format(cls, single_acc, f1_sc))
    print("average = {:.4f}, {:.4f}".format(torch.mean(accs), torch.mean(f1_scores)))

    # error distribution
    err_dist={}
    num_bins = 5
    for cls, v in error_dist_dict.items():
        matched_instances_size = max([tensor.size(0) for tensor in v ])
        n_matched_instances = torch.zeros(matched_instances_size)

        n_possible_matches  = [torch.ones(tensor.size(0)) for tensor in v]
        total_instances_to_match = torch.zeros(matched_instances_size)
        for tensor in v:
            n_matched_instances[:tensor.size(0)] += tensor
        for tensor in n_possible_matches:
            total_instances_to_match[:tensor.size(0)] += tensor
        
        indices = torch.arange(matched_instances_size)
        bin_edges = torch.linspace(0, matched_instances_size, num_bins)
        binned_indices = torch.bucketize(indices, bin_edges)
        
        binned_matches = torch.bincount(binned_indices, weights=n_matched_instances)
        binned_counts = torch.bincount(binned_indices, weights=total_instances_to_match)
        
        cls_error_distribution = binned_matches / binned_counts
        err_dist[cls] = cls_error_distribution

    return accs, f1_scores, err_dist
