import torch
import torch.optim as optim
import wandb

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
from pathlib import Path
import os

from data.data_loader_multigraph import GMDataset, get_dataloader
from utils.evaluation_metric import matching_accuracy_from_lists, f1_score, get_pos_neg_from_lists, make_perm_mat_pred
import eval
from matchAR import Net, SimpleNet, EncoderNet, ResMatcherNet
from utils.config import cfg
from utils.utils import update_params_from_cmdline, compute_grad_norm

class HammingLoss(torch.nn.Module):
    def forward(self, suggested, target):
        errors = suggested * (1.0 - target) + (1.0 - suggested) * target
        return errors.mean(dim=0).sum()


lr_schedules = {
    #TODO: CHANGE BACK TO 10
    "long_halving": (30, (2, 4, 6, 9, 10, 13, 16, 18, 20, 23, 26, 29), 0.5),
    # "long_halving": (30, (5,10, 15, 20, 25), 0.5),
    "short_halving": (2, (1,), 0.5),
    "long_nodrop": (10, (10,), 1.0),
    "minirun": (1, (10,), 1.0),
}

def train_val_dataset(dataset, val_split=0.1):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets['train'], datasets['val']

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



def train_eval_model(model, criterion, optimizer, dataloader, max_norm, num_epochs, resume=False, start_epoch=0):
    print("Start training...")

    since = time.time()
    dataloader["train"].dataset.set_num_graphs(cfg.TRAIN.num_graphs_in_matching_instance)
    dataset_size = len(dataloader["train"].dataset)


    device = next(model.parameters()).device
    print("{} model on device: {}".format(cfg.MODEL_ARCH , device))

    checkpoint_path = Path(cfg.model_dir) / "params"
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    if resume:
        params_path = os.path.join(cfg.warmstart_path, f"params.pt")
        print("Loading model parameters from {}".format(params_path))
        model.load_state_dict(torch.load(params_path))

        optim_path = os.path.join(cfg.warmstart_path, f"optim.pt")
        print("Loading optimizer state from {}".format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))

    # Evaluation only
    if cfg.evaluate_only:
        assert resume
        print(f"Evaluating without training...")
        accs, f1_scores = eval.eval_model(model, dataloader["test"], eval_epoch=5)
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
        return model, acc_dict

    _, lr_milestones, lr_decay = lr_schedules[cfg.TRAIN.lr_schedule]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=lr_decay
    )

    for epoch in range(start_epoch, num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        model.train()  # Set model to training mode

        print("lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer.param_groups]))

        epoch_loss = 0.0
        running_loss = 0.0
        running_acc = 0.0
        epoch_acc = 0.0
        running_f1 = 0.0
        epoch_f1 = 0.0
        running_since = time.time()
        iter_num = 0

        # Iterate over data.
        for inputs in dataloader["train"]:
            data_list = [_.cuda() for _ in inputs["images"]]
            points_gt_list = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt_list = [_.cuda() for _ in inputs["ns"]]
            edges_list = [_.to("cuda") for _ in inputs["edges"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]

            # # randomly swap source and target images
            if cfg.TRAIN.random_swap:
                for i in range(data_list[0].shape[0]):
                    # with 0.5 probability
                    swap_flag = torch.bernoulli(torch.Tensor([0.5]))
                    swap_flag = int(swap_flag.item())

                    if swap_flag:
                        # swap edge list
                        # swap everything else
                        perm_mat_list = swap_permutation_matrix(perm_mat_list, i)
                        data_list = swap_src_tgt_order(data_list, i)
                        points_gt_list = swap_src_tgt_order(points_gt_list, i)
                        n_points_gt_list = swap_src_tgt_order(n_points_gt_list, i)
                        edges_list = swap_src_tgt_order(edges_list, i)


            num_graphs = points_gt_list[0].size(0)
            num_nodes_s = points_gt_list[0].size(1)
            num_nodes_t = points_gt_list[1].size(1)
            iter_num = iter_num + 1

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                s_pred_list = model(data_list, points_gt_list, edges_list, n_points_gt_list, perm_mat_list)
                y_gt = torch.flatten(perm_mat_list[0], 1, 2)
                loss = criterion(s_pred_list, y_gt)

                # loss = sum([criterion(s_pred, perm_mat) for s_pred, perm_mat in zip(s_pred_list, perm_mat_list)])
                loss /= len(s_pred_list)

                # backward + optimize
                loss.backward()
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

                # if epoch == 2:
                #     print("here for debugging")
                # train metrics
                C = - s_pred_list.view(num_graphs, num_nodes_s, num_nodes_t)
                y_pred = torch.tensor(np.array([linear_sum_assignment(C[x,:,:].detach().cpu().numpy()) 
                                    for x in range(num_graphs)])).to(device)
                s_pred_mat_list = [make_perm_mat_pred(y_pred[:,1,:], num_nodes_t).to(device)]
                tp, fp, fn = get_pos_neg_from_lists(s_pred_mat_list, perm_mat_list)
                f1 = f1_score(tp, fp, fn)
                acc, _, __ = matching_accuracy_from_lists(s_pred_mat_list, perm_mat_list)

                # statistics
                bs = perm_mat_list[0].size(0)
                running_loss += loss.item() * bs  # multiply with batch size
                epoch_loss += loss.item() * bs
                running_acc += acc.item() * bs
                epoch_acc += acc.item() * bs
                running_f1 += f1.item() * bs
                epoch_f1 += f1.item() * bs

                if iter_num % cfg.STATISTIC_STEP == 0:
                    running_speed = cfg.STATISTIC_STEP * bs / (time.time() - running_since)
                    loss_avg = running_loss / cfg.STATISTIC_STEP / bs
                    acc_avg = running_acc / cfg.STATISTIC_STEP / bs
                    f1_avg = running_f1 / cfg.STATISTIC_STEP / bs
                    print(
                        "Epoch {:<4} Iter {:<4} {:>4.2f}sample/s Loss={:<8.4f} Accuracy={:<2.3} F1={:<2.3}".format(
                            epoch, iter_num, running_speed, loss_avg, acc_avg, f1_avg
                        )
                    )
                    """
                    if cfg.MODEL_ARCH == 'tf':
                        grad_norm_model = compute_grad_norm(model.parameters())
                        grad_norm_splineCNN = compute_grad_norm(model.psi.parameters())
                        grad_norm_encoder = compute_grad_norm(model.transformer.encoder.parameters())
                        grad_norm_decoder = compute_grad_norm(model.transformer.decoder.parameters())
                        grad_mlp_query = compute_grad_norm(model.mlp.parameters())
                        grad_mlp_out = compute_grad_norm(model.mlp_out.parameters())

                        wandb.log({"train_loss": loss_avg, "train_acc": acc_avg, "train_f1": f1_avg, 
                                "grad_model": grad_norm_model, "grad_splineCNN": grad_norm_splineCNN,  
                                    "grad_mlp_out": grad_mlp_out, "grad_mlp_query": grad_mlp_query, 
                                    "grad_encoder": grad_norm_encoder, "grad_decoder": grad_norm_decoder})
                    else:
                        grad_norm_model = compute_grad_norm(model.parameters())
                        grad_norm_splineCNN = compute_grad_norm(model.psi.parameters())
                        grad_norm_encoder = compute_grad_norm(model.encoder.parameters())
                        grad_mlp_query = compute_grad_norm(model.mlp.parameters())
                        grad_mlp_out = compute_grad_norm(model.mlp_out.parameters())

                        wandb.log({"train_loss": loss_avg, "train_acc": acc_avg, "train_f1": f1_avg, 
                                "grad_model": grad_norm_model, "grad_splineCNN": grad_norm_splineCNN,  
                                "grad_mlp_out": grad_mlp_out, "grad_mlp_query": grad_mlp_query, 
                                "grad_encoder": grad_norm_encoder})
                    """
                    wandb.log({"train_loss": loss_avg, "train_acc": acc_avg, "train_f1": f1_avg})

                    running_acc = 0.0
                    running_f1 = 0.0
                    running_loss = 0.0
                    running_since = time.time()

        epoch_loss = epoch_loss / dataset_size
        epoch_acc = epoch_acc / dataset_size
        epoch_f1 = epoch_f1 / dataset_size

        wandb.log({"ep_loss": epoch_loss, "ep_acc": epoch_acc, "ep_f1": epoch_f1})


        if cfg.save_checkpoint:
            base_path = Path(checkpoint_path / "{:04}".format(epoch + 1))
            Path(base_path).mkdir(parents=True, exist_ok=True)
            path = str(base_path / "params.pt")
            torch.save(model.state_dict(), path)
            torch.save(optimizer.state_dict(), str(base_path / "optim.pt"))

        print(
            "Over whole epoch {:<4} -------- Loss: {:.4f} Accuracy: {:.3f} F1: {:.3f}".format(
                epoch, epoch_loss, epoch_acc, epoch_f1
            )
        )

        print()
        # Eval in each epoch
        accs, f1_scores = eval.eval_model(model, dataloader["test"])
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

        # acc_table = [[x,y] for (x,y) in acc_dict.items()]
        # f1_table = [[x,y] for (x,y) in f1_dict.items()]
        # test_acc_table = wandb.Table(data= acc_table, columns = ["Class", "Accuracy"])
        # test_f1_table = wandb.Table(data= f1_table, columns = ["Class", "F1-score"])
        
        # wandb.log({"Test Accuracy" :test_acc_table })
        # wandb.log({"Test F1-score" :test_f1_table })
        wandb.log({"mean test_acc": torch.mean(accs), "mean test_f1": torch.mean(f1_scores)})

        scheduler.step()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60
        )
    )

    return model, acc_dict


if __name__ == "__main__":
    print('Using config file from: ', os.sys.argv[1])
    cfg = update_params_from_cmdline(default_params=cfg)
    import json
    import os

    os.makedirs(cfg.model_dir, exist_ok=True)
    with open(os.path.join(cfg.model_dir, "settings.json"), "w") as f:
        json.dump(cfg, f)
    
    
    wandb.init(
    # set the wandb project where this run will be logged
    project="matchAR",
    
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

    dataset_len = {"train": cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, "test": cfg.EVAL.SAMPLES}
    image_dataset = {
        x: GMDataset(cfg.DATASET_NAME, sets=x, length=dataset_len[x], obj_resize=(256, 256)) for x in ("train", "test")
    }
    dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == "test")) for x in ("train", "test")}

    torch.cuda.set_device(6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.MODEL_ARCH == 'tf':
        model = Net()
    elif cfg.MODEL_ARCH == 'mlp':
        model = SimpleNet()
    elif cfg.MODEL_ARCH == 'enc':
        model = EncoderNet()
    elif cfg.MODEL_ARCH == 'res':
        model = ResMatcherNet()
    
    model = model.cuda()


    criterion = torch.nn.BCEWithLogitsLoss()

    backbone_params = list(model.node_layers.parameters()) + list(model.edge_layers.parameters())
    # backbone_params += list(model.final_layers.parameters())

    backbone_ids = [id(item) for item in backbone_params]

    new_params = [param for param in model.parameters() if id(param) not in backbone_ids]
    opt_params = [
        dict(params=backbone_params, lr=cfg.TRAIN.LR * 0.01),
        dict(params=new_params, lr=cfg.TRAIN.LR),
    ]
    optimizer = optim.RAdam(opt_params)

    if not Path(cfg.model_dir).exists():
        Path(cfg.model_dir).mkdir(parents=True)

    num_epochs, _, __ = lr_schedules[cfg.TRAIN.lr_schedule]
    model, accs = train_eval_model(model, 
                                   criterion, 
                                   optimizer,
                                   dataloader,
                                   cfg.TRAIN.clip_norm, 
                                   num_epochs=num_epochs,
                                   resume=cfg.warmstart_path is not None, 
                                   start_epoch=0,
                                   )