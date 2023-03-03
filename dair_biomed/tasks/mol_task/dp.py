import logging
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from datasets.dp_dataset import SUPPORTED_DP_DATASETS, Task
from models.drug_encoder import SUPPORTED_DRUG_ENCODER
from utils import DPCollator, roc_auc, EarlyStopping, AverageMeter, ToDevice

activation = {
    "sigmoid": nn.Sigmoid(),
    "softplus": nn.Softplus(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
}

class DPModel(nn.Module):
    def __init__(self, encoder, config, out_dim):
        super(DPModel, self).__init__()
        self.encoder = encoder
        pred_head = nn.Sequential()
        hidden_dims = [encoder.output_dim] + config["hidden_size"] + [out_dim]
        for i in range(len(hidden_dims) - 1):
            pred_head.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if i != len(hidden_dims) - 2:
                pred_head.append(nn.Dropout(config["dropout"]))
                if config["activation"] != "none":
                    pred_head.append(activation[config["activation"]])
                if config["batch_norm"]:
                    pred_head.append(nn.BatchNorm1d())
        self.proj_head = nn.Sequential(pred_head)

    def forward(self, drug):
        h, _ = self.encoder(drug)
        return self.proj_head(h)

def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument('--dataset', type=str, default='MoleculeNet')
    parser.add_argument("--dataset_path", type=str, default='../datasets/dp/MoleculeNet/')
    parser.add_argument("--dataset_name", type=str, default='BBBP')
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--output_path", type=str, default="../ckpts/finetune_ckpts/dp/finetune.pth")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser

def get_metric(task, name):
    if task == Task.CLASSFICATION:
        metric_name = "roc_auc"
        metric = roc_auc
    elif task == Task.REGRESSION:
        if name in ["qm7", "qm8", "qm9"]:
            metric_name = "MAE"
            metric = mean_absolute_error
        else:
            metric_name = "MSE"
            metric = mean_squared_error
    return metric_name, metric

def train_dp(train_loader, val_loader, model, task, target, label_index, normalizer, args):
    device = torch.device(args.device)
    if task == Task.CLASSFICATION:
        loss_fn = nn.CrossEntropyLoss()
        mode = "higher"
    elif task == Task.REGRESSION:
        if args.dataset_name in ["qm7", "qm8", "qm9"]:
            loss_fn = nn.L1Loss()
            mode = "lower"
        else:
            loss_fn = nn.MSELoss()
            mode = "lower"
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(mode=mode, patience=args.patience, filename=args.output_path)
    metric_name, _ = get_metric(task, args.dataset_name)
    running_loss = AverageMeter()
    
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        model.train()
        running_loss.reset()

        for step, (drug, label) in enumerate(tqdm(train_loader)):
            drug = ToDevice(drug, device)
            pred = model(drug)
            label = label[:, label_index].to(device)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss.update(loss.detach().cpu().item())
            if (step + 1) % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
                running_loss.reset()
        val_metrics = val_dp(val_loader, model, task, target, label_index, normalizer, args)
        logger.info("%s val %s=%.4lf" % (target, metric_name, val_metrics[metric_name]))
        if stopper.step((val_metrics[metric_name]), model):
            break
    model.load_state_dict(torch.load(args.output_path)["model_state_dict"])
    return model

def val_dp(val_loader, model, task, target, label_index, normalizer, args):
    device = torch.device(args.device)
    metric_name, metric = get_metric(task, args.dataset_name)
    model.eval()
    all_preds, all_y = [], []
    for drug, label in tqdm(val_loader):
        drug = ToDevice(drug, device)
        pred = model(drug).detach().cpu()
        if task == Task.CLASSFICATION:
            pred = F.softmax(pred, dim=-1)[:, 1]
        label = label[:, label_index]
        
        all_preds.append(pred)
        all_y.append(label)
    all_preds = torch.cat(all_preds, dim=0)
    all_y = torch.cat(all_y, dim=0)
    if normalizer is not None:
        all_preds = normalizer.denorm(all_preds)
        all_y = normalizer.denorm(all_y)
    return {metric_name: metric(all_y, all_preds)}

def main(args, config):
    # prepare dataset
    dataset = SUPPORTED_DP_DATASETS[args.dataset](args.dataset_path, config["data"], args.dataset_name)
    task = dataset.task
    targets = dataset.targets
    normalizer = dataset.normalizer

    train_dataset = dataset.index_select(dataset.train_index)
    val_dataset = dataset.index_select(dataset.val_index)
    test_dataset = dataset.index_select(dataset.test_index)
    collator = DPCollator(config["data"]["drug"])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)

    # prepare model
    if config["model"] == "DeepEIK":
        encoder = SUPPORTED_DRUG_ENCODER[config["model"]](config["network"])
    else:
        encoder = SUPPORTED_DRUG_ENCODER[config["model"]](**config["network"]["structure"])
    encoder_ckpt = config["network"]["structure"]["init_checkpoint"]
    if encoder_ckpt != "":
        ckpt = torch.load(encoder_ckpt, map_location="cpu")
        param_key = config["network"]["structure"]["param_key"]
        if param_key != "":
            ckpt = ckpt[param_key]
        encoder.load_state_dict(ckpt)
    model = DPModel(encoder, config["network"]["pred_head"], 2 if task == Task.CLASSFICATION else 1)
    if args.init_checkpoint != "":
        ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        if args.param_key != "":
            ckpt = ckpt[args.param_key]
        model.load_state_dict(ckpt)
    model.to(args.device)

    # configure metric
    metric_name, _ = get_metric(task, args.dataset_name)

    if args.mode == "train":
        for i, target in enumerate(targets):
            train_dp(train_loader, val_loader, model, task, target, i, normalizer[i], args)
            results = val_dp(test_loader, model, task, target, i, normalizer[i], args)
            logger.info("%s test %s=%.4lf" % (target, metric_name, results[metric_name]))
    elif args.mode == "test":
        for i, target in enumerate(targets):
            results = val_dp(test_loader, model, task, target, i, normalizer[i], args)
            logger.info("%s test %s=%.4lf" % (target, metric_name, results[metric_name]))

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    config = json.load(open(args.config_path, "r"))
    main(args, config)