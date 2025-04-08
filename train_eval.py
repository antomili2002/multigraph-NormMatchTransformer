import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data import DataLoader, Subset, DistributedSampler
from sklearn.model_selection import train_test_split
import numpy as np
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
from model import MNMT, CycleContrastiveLoss
from utils.config import cfg
from utils.utils import update_params_from_cmdline, compute_grad_norm
from utils.evaluation_metric import calculate_correct_and_valid, calculate_f1_score, get_pos_neg, get_pos_neg_from_lists

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
    return perm_mats

def train_eval_model(model, criterion, optimizer, dataloader, max_norm, num_epochs, local_rank, output_rank, resume=False, start_epoch=0):
    
    since = time.time()
    dataloader["train"].dataset.set_num_graphs(cfg.TRAIN.num_graphs_in_matching_instance)
    dataset_size = len(dataloader["train"].dataset)
    

    device = next(model.parameters()).device
    if local_rank == output_rank:
        print("Start training...")
        print("{} model on device: {}".format(cfg.MODEL_ARCH , device))

    checkpoint_path = Path(cfg.model_dir) / "params"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

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
            evaluation_epoch = 13

            all_error_dict = {}
            accs, f1_scores, error_dict = eval.eval_model(model, dataloader["test"], local_rank, output_rank, eval_epoch=evaluation_epoch)
            all_error_dict[evaluation_epoch] = error_dict
            acc_dict = {
                "acc_{}".format(cls): single_acc for cls, single_acc in zip(dataloader["train"].dataset.classes, accs)
            }
            f1_dict = {
                "f1_{}".format(cls): single_f1_score
                for cls, single_f1_score in zip(dataloader["train"].dataset.classes, f1_scores)
            }
            acc_dict.update(f1_dict)
            acc_dict["matching_accuracy"] = torch.mean(accs)
            acc_dict["f1_score"] = torch.mean(f1_scores)

            time_elapsed = time.time() - since
            print(
                "Evaluation complete in {:.0f}h {:.0f}m {:.0f}s".format(
                    time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60
                )
            )
        
        return model, all_error_dict

    _, lr_milestones, lr_decay = lr_schedules[cfg.TRAIN.lr_schedule]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay)
    torch.autograd.set_detect_anomaly(True)
    
    all_error_dict = {}
    result_dict = {}
    
    for epoch in range(start_epoch, num_epochs):
        if local_rank == output_rank:
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)
            print("lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer.param_groups]))
        
        model.train()  # Set model to training mode

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_f1 = 0.0

        tp = 0
        fp = 0
        fn = 0
        
        epoch_correct = 0
        epoch_total_valid = 0
        
        for batch_idx, inputs in enumerate(dataloader["train"]):
            data_list = [_.cuda() for _ in inputs["images"]]
            points_gt_list = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt_list = [_.cuda() for _ in inputs["ns"]]
            edges_list = [_.cuda() for _ in inputs["edges"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]
            
            # reshape gt_perm_mat
            perm_matrices = reshape_perm_matrices(perm_mat_list, data_list)
            
            n_points_gt_sample = n_points_gt_list[0]

            #print("TRAIN")
            #print("---------------------------------------------")
            #print(f"Data List: {data_list[0].shape}")
            #print(f"points_gt_list: {points_gt_list[0].shape}")
            #print(f"n_points_gt_list: {n_points_gt_list[0].shape}")
            #print(f"perm_mat_list: {perm_mat_list[0][1].shape}")
            #print(f"n_points_gt_sample: {n_points_gt_sample.shape}")
            #print("---------------------------------------------")
            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # reshape perm_mats_list -> List[List[Tensor]] with [B, K, K]
                
                        
                # forward
                decoded_graphs, similarity_matrices = model(data_list, points_gt_list, edges_list, n_points_gt_list, n_points_gt_sample, perm_matrices)

                loss = criterion(similarity_matrices, perm_matrices)                 
                loss.backward()
                
                if max_norm > 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            torch.nn.utils.clip_grad_norm_(param, max_norm)
                        
            
                optimizer.step()
                model.module.enforce_constraints()
                
            bs = perm_mat_list[0].size(0)
            epoch_loss += loss.item() * bs
            
            #if local_rank == output_rank and batch_idx % 10 == 0:
            #    print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}] Loss: {loss.item():.4f}")

            # Inference-style evaluation (no autoregression yet)
            with torch.no_grad():
                pred_tensor = []
                for i in range(len(similarity_matrices)):
                    pred = torch.argmax(similarity_matrices[i][(i + 1) % len(similarity_matrices)], dim=-1)  # [B, K]
                    pred_tensor.append(pred)

                y_true = torch.argmax(perm_mat_list[0], dim=-1)  # [B, K]
                pred_tensor = pred_tensor[0].to(y_true.device)

                correct, total_valid = calculate_correct_and_valid(pred_tensor, y_true)
                _tp, _fp, _fn = calculate_f1_score(pred_tensor, y_true)

                # Log accuracy and F1 components for monitoring
                #if local_rank == output_rank and batch_idx % 10 == 0:
                #    print(f"Batch [{batch_idx}] Correct: {correct}, Total Valid: {total_valid}, TP: {_tp}, FP: {_fp}, FN: {_fn}")

        epoch_loss = epoch_loss / dataset_size

        if (epoch + 1) % cfg.STATISTIC_STEP == 0:
            if local_rank == output_rank:
                accs, f1_scores, error_dict = eval.eval_model(model, dataloader["test"], local_rank, output_rank)
                all_error_dict[epoch+1] = error_dict
                wandb.log({"ep_loss": epoch_loss, "mean test_acc": torch.mean(accs), "mean test_f1": torch.mean(f1_scores)})
        else:
            if local_rank == output_rank:
                wandb.log({"ep_loss": epoch_loss})
                print(f'epoch loss: {epoch_loss:.4f}')

        if cfg.save_checkpoint and local_rank == output_rank:
            base_path = checkpoint_path / "{:04}".format(epoch + 1)
            base_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), base_path / "params.pt")
            torch.save(optimizer.state_dict(), base_path / "optim.pt")

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
        project="MNMT",
        
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
    dataset_len = {"train": cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, "test": cfg.EVAL.SAMPLES * world_size} # 
    image_dataset = {
        x: GMDataset(cfg.DATASET_NAME, sets=x, length=dataset_len[x], obj_resize=(256, 256)) for x in ("train", "test")
    }

    sampler = {
    "train": DistributedSampler(image_dataset["train"]),
    "test": DistributedSampler(image_dataset["test"])
    }
    
    dataloader = {x: get_dataloader(image_dataset[x],sampler[x], fix_seed=(x == "test")) for x in ("train", "test")}

    model = MNMT(cfg.TRAIN.num_graphs_in_matching_instance)    
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    print("Using device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name())
    
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    criterion = CycleContrastiveLoss(init_temperature=cfg.TRAIN.temperature)
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
