import math
import torch
import numpy as np
import random
import torch.optim as optim
import torch.nn.functional as F
import wandb

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data import DataLoader, Subset, DistributedSampler
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment
import time
from pathlib import Path
import os
import pandas as pd
import matplotlib
from datetime import timedelta

from sklearn.metrics import f1_score
from data.data_loader_multigraph import GMDataset, get_dataloader
import eval
from model import NMT
from utils.config import cfg
from utils.utils import update_params_from_cmdline, compute_grad_norm
from utils.evaluation_metric import calculate_correct_and_valid, calculate_f1_score, get_pos_neg, get_pos_neg_from_lists
    
class GenericInfoNCELoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.1, margin: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.tau = temperature
        self.eps = eps
        self.m = margin
        
    def forward(self, 
                sim_list: list,      
                perm_list: list,       # [B, M, M]
                points_embedding: list) -> torch.Tensor:
        
        device = sim_list[0].device
        
        # hyperspherical loss
        hyperspherical_loss = 0.0
        for points in points_embedding:
            sim_number = torch.bmm(points, points.transpose(1, 2))
            sim_normed1 = torch.norm(points, p=2, dim=-1).clamp(min=1e-8).unsqueeze(2)
            sim_normed2 = torch.norm(points, p=2, dim=-1).clamp(min=1e-8).unsqueeze(1)
            sim_denominator = torch.bmm(sim_normed1, sim_normed2)
            cosine_sim_ = sim_number / sim_denominator
            
            ident_mat = torch.eye(cosine_sim_.shape[1]).to(device)
            cosine_sim_ = cosine_sim_ - 2 * ident_mat
            
            prot_score_max, _ = torch.max(cosine_sim_, dim=-1)
            prot_score_mean = torch.mean(prot_score_max, dim=-1)
            prot_score_mean = torch.mean(prot_score_mean)
            hyperspherical_loss += prot_score_mean
        hyperspherical_loss = hyperspherical_loss / len(points_embedding)
        
        total_loss = 0.0
        pair_count = 0
        for S_ij, P_ij in zip(sim_list, perm_list):
            B, N_i, Nj = S_ij.shape
            
            has_one = P_ij.sum(dim=2) != 0
            expanded_mask = has_one.unsqueeze(-1).expand_as(P_ij)
            
            S_ij_2 = S_ij.clone().transpose(-2, -1)
            P_ij_2 = P_ij.clone().transpose(-2, -1)
            
            S_ij = S_ij.masked_select(expanded_mask).view(-1, P_ij.size(2))
            y_values = P_ij.masked_select(expanded_mask).view(-1, P_ij.size(2))
            pos_indices = torch.argmax(y_values, dim=1)
            
            S_ij_2 = S_ij_2.masked_select(expanded_mask).view(-1, P_ij_2.size(2))
            y_values_2 = P_ij_2.masked_select(expanded_mask).view(-1, P_ij_2.size(2))
            pos_indices_2 = torch.argmax(y_values_2, dim=1)
            
            logits = S_ij / self.tau
            
            # add cost margin like in BBGM Section 3.3
            rows = torch.arange(logits.size(0), device=logits.device)
            logits[rows, pos_indices] -= self.m
            
            loss_1 = F.cross_entropy(logits, pos_indices)
            
            
            logits_2 = S_ij_2 / self.tau
            
            # add cost margin like in BBGM Section 3.3
            rows_2 = torch.arange(logits_2.size(0), device=logits_2.device)
            logits_2[rows_2, pos_indices_2] -= self.m
            
            loss_2 = F.cross_entropy(logits_2, pos_indices_2)
            
            total_loss += loss_1 + loss_2
            pair_count += 1
        
        loss_nce = total_loss / pair_count
        return loss_nce + hyperspherical_loss
    
class SoftNearestNeighborSimLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.tau = temperature
        self.eps = eps
    
    def forward(self, 
                sims_block: torch.Tensor,      # [B, M, M] 
                labels: torch.Tensor,       # [B, M] int labels in [0, .., M-1) 
                points_embedding: list
                ):
        N, M = sims_block.shape
        device = sims_block.device
        
        # hyperspherical loss
        hyperspherical_loss = 0.0
        for points in points_embedding:
            sim_number = torch.bmm(points, points.transpose(1, 2))
            sim_normed1 = torch.norm(points, p=2, dim=-1).clamp(min=1e-8).unsqueeze(2)
            sim_normed2 = torch.norm(points, p=2, dim=-1).clamp(min=1e-8).unsqueeze(1)
            sim_denominator = torch.bmm(sim_normed1, sim_normed2)
            cosine_sim_ = sim_number / sim_denominator
            
            ident_mat = torch.eye(cosine_sim_.shape[1]).to(device)
            cosine_sim_ = cosine_sim_ - 2 * ident_mat
            
            prot_score_max, _ = torch.max(cosine_sim_, dim=-1)
            prot_score_mean = torch.mean(prot_score_max, dim=-1)
            prot_score_mean = torch.mean(prot_score_mean)
            hyperspherical_loss += prot_score_mean
        hyperspherical_loss = hyperspherical_loss / len(points_embedding)
        
        dist = 1.0 - sims_block
        logits = -dist / self.tau

        exp_logits = logits.exp()
        
        pos_mask = torch.zeros_like(exp_logits)
        pos_mask[torch.arange(N, device=device), labels] = 1.0

        neg_mask = 1.0 - pos_mask
        
        numer = (exp_logits * pos_mask).sum(dim=1)
        denom = (exp_logits * neg_mask).sum(dim=1)

        loss_snn = -torch.log((numer + self.eps) / (denom + self.eps)).mean()                   # [B,M]

        return loss_snn + hyperspherical_loss

lr_schedules = {
    #TODO: CHANGE BACK TO 10
    "long_halving1": (32, (3, 8, 13, 20), 0.3),
    "long_halving2": (32, (10, 15, 30), 0.1),
    "long_halving3": (32, (3, 5,), 0.1),
    "long_halving4": (32, (2, 3), 0.1),
    # "long_halving": (30, (3, 6, 12, 26), 0.25),
    # "long_halving": (50, (40,), 0.1),
    "short_halving": (2, (1,), 0.5),
    "long_nodrop": (10, (10,), 1.0),
    "minirun": (1, (10,), 1.0),
}

def swap_src_tgt_order(data_list, i):
    # edge features
    if data_list[0].__class__.__name__ == 'DataBatch':
        tmp = data_list[1]
        data_list[1] = data_list[0]
        data_list[0] = tmp
    else:
        tmp = data_list[1][i].clone()
        data_list[1][i] = data_list[0][i]
        data_list[0][i] = tmp
    return data_list

def swap_permutation_matrix(perm_mat_list, i):
    transposed_slice = torch.transpose(perm_mat_list[0][i, :, :], 1, 0)
    output_tensor = perm_mat_list[0].clone()
    output_tensor[i, :, :] = transposed_slice

    return [output_tensor]


def train_eval_model(model, criterion, optimizer, dataloader, max_norm, num_epochs, local_rank, output_rank, resume=False, start_epoch=0):
    since = time.time()
    
    K = cfg.TRAIN.num_graphs_in_matching_instance
    dataloader["train"].dataset.set_num_graphs(K)
    
    dataset_size = len(dataloader["train"].dataset)
    all_error_dict = {}

    device = next(model.parameters()).device
    if local_rank == output_rank:
        print("Start training...")
        print("NMT model on device: {}".format(device))
        print("Graphs per training sample: {}".format(K))
        print("Graphs per evaluation sample: {}".format(cfg.EVAL.num_graphs_in_matching_instance))

    checkpoint_path = Path(cfg.model_dir) / "params"
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    if resume:
        params_path = os.path.join(cfg.warmstart_path, f"params.pt")
        print("Loading model parameters from {}".format(params_path))
        model.load_state_dict(torch.load(params_path, map_location=f'cuda:{local_rank}'))

        optim_path = os.path.join(cfg.warmstart_path, f"optim.pt")
        print("Loading optimizer state from {}".format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path, map_location=f'cuda:{local_rank}'))

    # Evaluation only
    if cfg.evaluate_only:
        # assert resume
        if local_rank == output_rank:
            print(f"Evaluating without training...")
            evaluation_epoch = 31
            accs_pre, accs_post, error_dict = eval.eval_model(model, dataloader["test"], local_rank, output_rank, eval_epoch=evaluation_epoch)
            all_error_dict[evaluation_epoch] = error_dict
            acc_dict = {
                "acc_{}".format(cls): single_acc for cls, single_acc in zip(dataloader["train"].dataset.classes, accs_post)
            }
            acc_dict["matching_accuracy"] = torch.mean(accs_post)

            time_elapsed = time.time() - since
            print(
                "Evaluation complete in {:.0f}h {:.0f}m {:.0f}s".format(
                    time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60
                )
            )
        
        return model, all_error_dict

    _, lr_milestones, lr_decay = lr_schedules[cfg.TRAIN.lr_schedule]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=lr_decay
    )
    torch.autograd.set_detect_anomaly(True)
    result_dict = {}
    
    iter_num = 0
    
    for epoch in range(start_epoch, num_epochs):
        if local_rank == output_rank:
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)
        model.train()  # Set model to training mode

        if local_rank == output_rank:
            print("lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer.param_groups]))

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_f1 = 0.0
        running_since = time.time()
        
        epoch_correct = 0
        epoch_total_valid = 0
        
        for inputs in dataloader["train"]:
            data_list = [_.cuda() for _ in inputs["images"]]
            points_gt_list = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt_list = [_.cuda() for _ in inputs["ns"]]
            edges_list = [_.cuda() for _ in inputs["edges"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]
            
            n_points_gt_sample = n_points_gt_list[0]
            iter_num = iter_num + 1

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                sim_list, points_embeddings, edges, layer_loss = model(data_list, points_gt_list, edges_list, n_points_gt_list, n_points_gt_sample, perm_mat_list)
                eval_similarity_scores = sim_list[0].clone().detach()
                
                batch_size = sim_list[0].shape[0]
                
                #idx = 0
                #for i in range(K):
                #    ni = n_points_gt_list[i]   # [B]
                #    for j in range(i+1, K):
                #        Pij = perm_mat_list[idx]    # [B, Mi, Mj]
                #        for b, e in enumerate(ni):
                #            Pij[b, e:, :] = 0
                #        idx += 1
                
                all_sims   = []
                all_labels = []

                c = 0
                for i in range(K):
                    Mi = points_embeddings[i].shape[1]
                    ni = n_points_gt_list[i]   # [B]
                    for j in range(i+1, K):
                        Mj = points_embeddings[j].shape[1]
                        nj = n_points_gt_list[j]  # [B]

                        # forward: i -> j
                        Sij = sim_list[c]     # [B, Mi, Mj]
                        Pij = perm_mat_list[c]    # [B, Mi, Mj]
                        # zero‐out padded rows in source i
                        for b in range(batch_size):
                            Pij[b, ni[b]:, :] = 0
                        has_one   = (Pij.sum(dim=2) > 0)                       # [B, Mi]
                        mask3d    = has_one.unsqueeze(-1).expand_as(Pij)       # [B, Mi, Mj]
                        sims_flat = Sij.masked_select(mask3d).view(-1, Mj)     # [N_ij, Mj]
                        y_flat    = Pij.masked_select(mask3d).view(-1, Mj)     # [N_ij, Mj]
                        labels_f  = y_flat.argmax(dim=1)                       # [N_ij]
                        all_sims.append(sims_flat)
                        all_labels.append(labels_f)

                        # reverse: j -> i
                        Sji = Sij.transpose(1,2)   # [B, Mj, Mi]
                        Pji = Pij.transpose(1,2)   # [B, Mj, Mi]
                        # zero‐out padded rows in source j (now rows of Sji)
                        for b in range(batch_size):
                            Pji[b, nj[b]:, :] = 0
                        has_one_t   = (Pji.sum(dim=2) > 0)                      # [B, Mj]
                        mask3d_t    = has_one_t.unsqueeze(-1).expand_as(Pji)    # [B, Mj, Mi]
                        sims_flat_t = Sji.masked_select(mask3d_t).view(-1, Mi)  # [N_ji, Mi]
                        y_flat_t    = Pji.masked_select(mask3d_t).view(-1, Mi)  # [N_ji, Mi]
                        labels_r    = y_flat_t.argmax(dim=1)                    # [N_ji]
                        all_sims.append(sims_flat_t)
                        all_labels.append(labels_r)

                        c += 1

                # concatenate across all pairs & directions
                sims_block   = torch.cat(all_sims,   dim=0)  # [N_total, M_max]
                labels_block = torch.cat(all_labels, dim=0)  # [N_total]
            
                loss = criterion(sims_block, labels_block, points_embeddings) # prototype_score
                loss = loss + layer_loss
                loss.backward()
                
                if max_norm > 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            torch.nn.utils.clip_grad_norm_(param, max_norm)
                        
                optimizer.step()
                model.module.enforce_constraints()
            
            print_interval = 50
            
            if iter_num % print_interval == 0:
                    print(f"[Epoch {epoch}][Iter {iter_num}] "
                          f"loss: {loss.item():.4f}")
                          
            with torch.no_grad():
                B, N_s, N_t = perm_mat_list[0].size()
                
                eval_pred_points = 0
                predictions_list = []
                for i in range(B):
                    predictions_list.append([])
                
                batch_size = eval_similarity_scores.shape[0]
                keypoint_preds = F.softmax(eval_similarity_scores, dim=-1)
                keypoint_preds = torch.argmax(keypoint_preds, dim=-1)
                for np in range(N_t):
                    for b in range(batch_size):
                        if eval_pred_points < n_points_gt_sample[b]:
                            predictions_list[b].append(keypoint_preds[b][eval_pred_points].item())
                        else:
                            predictions_list[b].append(-1)
                    eval_pred_points +=1
                prediction_tensor = torch.tensor(predictions_list).to(perm_mat_list[0].device)
                y_values_matching = torch.argmax(perm_mat_list[0], dim=-1)
                
                error_list = (prediction_tensor != y_values_matching).int()
            
                for idx, e in enumerate(n_points_gt_sample):
                    if e.item() not in result_dict:
                        result_dict[e.item()] = [1, error_list[idx,:e.item()]]
                    result_dict[e.item()][0] += 1
                    result_dict[e.item()][1] += error_list[idx,:e.item()]
                
                
                has_one = perm_mat_list[0].sum(dim=2) != 0
                expanded_mask = has_one.unsqueeze(-1).expand_as(perm_mat_list[0])
                y_values = perm_mat_list[0].masked_select(expanded_mask).view(-1, perm_mat_list[0].size(2))
                
                batch_correct, batch_total_valid = calculate_correct_and_valid(prediction_tensor, y_values_matching)
                # _tp, _fp, _fn = calculate_f1_score(prediction_tensor, y_values_matching)

                # Accumulate batch statistics
                epoch_correct += batch_correct
                epoch_total_valid += batch_total_valid
                # tp += _tp
                # fp += _fp
                # fn += _fn
                
            bs = perm_mat_list[0].size(0)
            epoch_loss += loss.item() * bs
        
        if epoch_total_valid > 0:
            epoch_acc = epoch_correct / epoch_total_valid
        else:
            epoch_acc = 0.0
        
        epoch_loss = epoch_loss / dataset_size
        epoch_time = time.time() - running_since
        if local_rank == output_rank:
            wandb.log({"ep_loss": epoch_loss, "ep_acc": epoch_acc})
            print(f'epoch loss: {epoch_loss}, epoch accuracy: {epoch_acc}')
            print(f'completed in {epoch_time:.2f}s ({epoch_time/60:.2f}m)')
        if (epoch+1) % cfg.STATISTIC_STEP == 0:
            if local_rank == output_rank:
                accs_pre, accs_post, error_dict = eval.eval_model(model, dataloader["test"], local_rank, output_rank)
                all_error_dict[epoch+1] = error_dict
                wandb.log({"ep_loss": epoch_loss, "ep_acc": epoch_acc, "mean test_acc_pre_sync": torch.mean(accs_pre), "mean test_acc_post_sync": torch.mean(accs_post)})
        
        
        if cfg.save_checkpoint and local_rank == output_rank:
            base_path = Path(checkpoint_path / "{:04}".format(epoch + 1))
            Path(base_path).mkdir(parents=True, exist_ok=True)
            path = str(base_path / "params.pt")
            torch.save(model.state_dict(), path)
            torch.save(optimizer.state_dict(), str(base_path / "optim.pt"))
        scheduler.step()
    
    return model, all_error_dict


if __name__ == "__main__":
    # print('Using config file from: ', os.sys.argv[1])
    cfg = update_params_from_cmdline(default_params=cfg)
    
    #windows
    # dist.init_process_group(backend='gloo', init_method='env://')
    
    #linux
    dist.init_process_group(backend='nccl', init_method='env://')
    
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ['LOCAL_RANK']) 
    output_rank = 0
    
    import json
    import os


    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.makedirs(cfg.model_dir, exist_ok=True)
    with open(os.path.join(cfg.model_dir, "settings.json"), "w") as f:
        json.dump(cfg, f)
    
    if local_rank == output_rank:
        wandb.init(
        # set the wandb project where this run will be logged
        project="NMT",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": cfg.TRAIN.LR,
        "architecture": cfg.MODEL_ARCH,
        "dataset": cfg.DATASET_NAME,
        "epochs": lr_schedules[cfg.TRAIN.lr_schedule][0],
        "batch_size": cfg.BATCH_SIZE,
        "cfg_full": cfg
        }
        )

    torch.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    
    dataset_len = {"train": cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, "test": cfg.EVAL.SAMPLES * world_size} # 
    image_dataset = {
        x: GMDataset(cfg.DATASET_NAME, sets=x, length=dataset_len[x], obj_resize=(256, 256)) for x in ("train", "test")
    }

    sampler = {
    "train": DistributedSampler(image_dataset["train"]),
    "test": DistributedSampler(image_dataset["test"])
    }
    
    dataloader = {x: get_dataloader(image_dataset[x],sampler[x], fix_seed=(x == "test")) for x in ("train", "test")}

    model = NMT()    
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    print("Using device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name())
    
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    criterion = SoftNearestNeighborSimLoss(temperature=cfg.TRAIN.temperature)
    backbone_params = list(model.module.node_layers.parameters()) + list(model.module.edge_layers.parameters())

    backbone_ids = [id(item) for item in backbone_params]

    new_params = [param for param in model.parameters() if id(param) not in backbone_ids]
    opt_params = [
        dict(params=backbone_params, lr=cfg.TRAIN.LR * 0.01 ),
        dict(params=new_params, lr=cfg.TRAIN.LR ),
    ]
    optimizer = optim.Adam(opt_params, weight_decay=cfg.TRAIN.weight_decay)
    

    if not Path(cfg.model_dir).exists():
        Path(cfg.model_dir).mkdir(parents=True)

    num_epochs, _, __ = lr_schedules[cfg.TRAIN.lr_schedule]
    model, all_error_dict = train_eval_model(model, 
                                   criterion, 
                                   optimizer,
                                   dataloader,
                                   cfg.TRAIN.clip_norm, 
                                   num_epochs=num_epochs,
                                   local_rank=local_rank,
                                   output_rank = output_rank,
                                   resume=cfg.warmstart_path is not None, 
                                   start_epoch=0,
                                   )
    
    if local_rank == output_rank:
        if all_error_dict is not None:
            output_folder = "errors"
            os.makedirs(output_folder, exist_ok=True)
            for epoch, class_dict in all_error_dict.items():
                save_dict = {}
                for class_, e_dict in class_dict.items():
                    e_dict_ = sorted(e_dict.items())
                    e_len, e_idx = e_dict_[-1]
                    result_tensor = torch.zeros(e_len, dtype=torch.float).to(device)
                    e_num = 0
                    for errors in e_dict_:
                        e_len, e_idx = errors
                        
                        e_ten = e_idx[1]
                        t1_resized = torch.cat((e_ten, torch.zeros(result_tensor.size(0) - e_ten.size(0), dtype=result_tensor.dtype).to(device))).to(device)
                        
                        result_tensor += t1_resized
                        e_num += e_idx[0]
                    # e_num = e_idx[0]
                    # e_tensor = e_idx[1]
                    
                    e_avg = (result_tensor/e_num).cpu().detach().tolist()
                    
                    save_dict[class_] = e_avg
                    
                file_name = f"{output_folder}/epoch_{epoch}_save_dict.json"
                with open(file_name, "w") as json_file:
                    json.dump(save_dict, json_file)
            
                
    dist.destroy_process_group()
