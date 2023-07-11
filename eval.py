import time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


from utils.config import cfg
from utils.evaluation_metric import matching_accuracy, f1_score, get_pos_neg, make_perm_mat_pred


def eval_model(model, dataloader, eval_epoch=None, verbose=False):
    print("Start evaluation...")
    since = time.time()

    device = next(model.parameters()).device

    if eval_epoch is not None:
        model_path = str(Path(cfg.model_dir) / "params" / "{:04}".format(eval_epoch) / "params.pt")
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

    for i, cls in enumerate(classes):
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
            
            with torch.set_grad_enabled(False):
                
                matchings = []
                B, N_s, N_t = perm_mat_list[0].size()
                n_points_sample = torch.zeros(B, dtype=torch.int).to(device)
                perm_mat_dec_list = [torch.zeros(B, N_s, N_t, dtype=torch.int).to(device)]
                cost_mask = torch.ones(B, N_s, N_t, dtype=torch.int).to(device)
                
                batch_idx = torch.arange(8)
                for np in range(N_t):
                    s_pred_list = model(
                        data_list,
                        points_gt,
                        edges,
                        n_points_gt,
                        n_points_sample,
                        perm_mat_dec_list,
                        in_training= False
                    )
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
                    
                    matchings.append(pair_idx)
                    
                                    
                matchings = torch.stack(matchings, dim=2)
                matches = []
                for example in range(B):
                    matching = matchings[example,:,:]
                    sorted_values, sorted_indices = torch.sort(matching[0,:], dim=0)
                    sorted_matches = matching[1, sorted_indices]
                    matches.append(sorted_matches)
                matches = torch.stack(matches, dim=0)

            # evaluation metrics
            s_pred_mat_list = [make_perm_mat_pred(matches, num_nodes_t).to(device)]
            _, _acc_match_num, _acc_total_num = matching_accuracy(s_pred_mat_list[0], perm_mat_list[0])
            _tp, _fp, _fn = get_pos_neg(s_pred_mat_list[0], perm_mat_list[0])

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

    return accs, f1_scores
