import logging
logger = logging.getLogger(__name__)

import math
import argparse
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

from datasets.drp_dataset import *
from models.tgsa import TGDRP
from utils import EarlyStopping, AverageMeter, roc_auc

SUPPORTED_DRP_MODEL = {
    "TGDRP": TGDRP,
}

def add_arguments(parser):
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--task', type=int, default='regression', help='task type: classification or regression')
    parser.add_argument('--dataset', type=str, default='GDSC', help='dataset')
    parser.add_argument("--dataset_path", type=str, default='../datasets/drp/GDSC/', help='path to the dataset')
    parser.add_argument('--config_path', type=str, help='path to the configuration file')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--epochs', type=int, default=300, help='maximum number of epochs (default: 300)')
    parser.add_argument('--patience', type=int, default=10, help='patience for earlystopping (default: 10)')
    parser.add_argument('--setup', type=str, default='known', help='experimental setup')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--weight_path', type=str, default='', help='filepath for pretrained weights')
    parser.add_argument('--mode', type=str, default='test', help='train or test')

def train_drp(train_loader, val_loader, model, args):
    if args.task == "classification":
        loss_fn = nn.BCEWithLogitsLoss()
        metric = "roc_auc"
    elif args.task == "regression":
        loss_fn = nn.MSELoss()
        metric = "rmse"
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(mode="lower", patience=args.patience)

    running_loss = AverageMeter()
    for i in range(args.epochs):
        logger.info("========Epoch %d========" % (i + 1))
        logger.info("Training...")
        model.train()
        step_loss = 0
        running_loss.reset()
        for drug, cell, label in tqdm(train_loader, desc="Loss=%.4lf" % step_loss):
            if isinstance(cell, list):
                drug, cell, label = drug.to(args.device), [feat.to(args.device) for feat in cell], label.to(args.device)
            else:
                drug, cell, label = drug.to(args.device), cell.to(args.device), label.to(args.device)
            pred = model(drug, cell)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_loss = loss.item()
            running_loss.update(step_loss)
        logger.info("Average training loss %.4lf" % (running_loss.get_average()))

        val_metrics = val_drp(val_loader, model, args)
        if stopper.step(val_metrics[metric], model):
            break
    return model

def val_drp(val_loader, model, args):
    model.eval()
    y_true, y_pred = [], []

    logger.info("Validating...")
    for drug, cell, label in tqdm(val_loader):
        if isinstance(cell, list):
            drug, cell, label = drug.to(args.device), [feat.to(args.device) for feat in cell], label.to(args.device)
        else:
            drug, cell, label = drug.to(args.device), cell.to(args.device), label.to(args.device)
        pred = model(drug, cell)
        if args.task == "classification":
            pred = F.sigmoid(pred)
        y_true.append(label.view(-1, 1).cpu())
        y_pred.append(pred.cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    
    if args.task == "classification":
        results = {
            "roc_auc": roc_auc(y_true, y_pred)
        }
    elif args.task == "regression":
        results = {
            "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "pearson": pearsonr(y_true, y_pred)[0]
        }
    logger.info(" ".join(["%s: %.4lf" % (key, results[key]) for key in results]))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    config = json.load(open(args.config_path, "r"))

    # build dataset
    train_dataset = SUPPORTED_DRP_DATASET[args.dataset](args.dataset_path, config["data"], split="train")
    test_dataset = SUPPORTED_DRP_DATASET[args.dataset](args.dataset_path, config["data"], split="test")

    collate_fn = SUPPORTED_DRP_COLLATE_FN[config["model"]]
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # build model
    config["network"]["input_dim_cell"] = len(config["cell"]["gene_feature"])
    model = SUPPORTED_DRP_MODEL[config["model"]](config["network"])
    if config["model"] == "TGSA":
        model.cluster_predefine = train_dataset.predifined_cluster
        model._build()
    model = model.to(args.device)

    if args.mode == "train":
        if args.pretrain:
            model.GNN_drug.load_state_dict(torch.load(args.weight_path)['model_state_dict'])
        model = train_drp(train_loader, test_loader, model, args)
        val_drp(train_loader, model, args)
        val_drp(test_loader, model, args)
    elif args.mode == "test":
        model.load_state_dict(torch.load(args.weight_path, map_location=args.device)['model_state_dict'])
        val_drp(test_loader, model, args)